"""HTTP client for the Engram REST API with automatic authentication."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any

import httpx


logger = logging.getLogger(__name__)


class EngramClient:
    """Manages HTTP communication and JWT auth with the Engram backend."""

    def __init__(self, api_url: str, username: str, password: str) -> None:
        self._api_url = api_url
        self._username = username
        self._password = password

        self._http: httpx.AsyncClient | None = None
        self._timeout_seconds = float(os.environ.get("ENGRAM_HTTP_TIMEOUT", "30"))
        # Safe by default: avoid ambiguous second writes after timeout/5xx.
        self._unsafe_process_turn_fallback = self._env_bool(
            "ENGRAM_UNSAFE_PROCESS_TURN_FALLBACK", False
        )
        self._process_turn_recovery_attempts = max(
            self._env_int("ENGRAM_PROCESS_TURN_RECOVERY_ATTEMPTS", 3),
            1,
        )
        self._process_turn_recovery_delay_seconds = max(
            self._env_float("ENGRAM_PROCESS_TURN_RECOVERY_DELAY_SECONDS", 1.0),
            0.0,
        )
        self._llm_provider = os.environ.get("LLM_PROVIDER", "ollama").strip().lower()
        self._ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self._ollama_llm_model = os.environ.get("OLLAMA_LLM_MODEL", "gemma3:270m")
        self._openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        self._openai_base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self._openai_llm_model = os.environ.get("OPENAI_LLM_MODEL", "gpt-5-nano")
        self._token: str | None = None
        self._token_expires_at: float = 0.0  # epoch seconds
        self._user_id: str | None = None
        self._conversation_map: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._http = httpx.AsyncClient(base_url=self._api_url, timeout=self._timeout_seconds)
        await self._authenticate()

    async def stop(self) -> None:
        if self._http:
            await self._http.aclose()

    # ------------------------------------------------------------------
    # Auth internals
    # ------------------------------------------------------------------

    async def _authenticate(self) -> None:
        """Login (auto-register on first use) and fetch user_id."""
        try:
            await self._login()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401:
                await self._register()
                await self._login()
            else:
                raise
        await self._fetch_user_id()

    async def _login(self) -> None:
        """POST /auth/login with form-encoded data (OAuth2PasswordRequestForm)."""
        resp = await self._http.post(
            "/auth/login",
            data={"username": self._username, "password": self._password},
        )
        resp.raise_for_status()
        body = resp.json()
        self._token = body["access_token"]
        self._token_expires_at = time.time() + body["expires_in"]

    async def _register(self) -> None:
        """POST /auth/register with JSON body. Silently ignores 'already exists'."""
        resp = await self._http.post(
            "/auth/register",
            json={
                "username": self._username,
                "email": f"{self._username}@example.com",
                "password": self._password,
            },
        )
        if resp.status_code == 400:
            return  # already registered
        resp.raise_for_status()

    async def _fetch_user_id(self) -> None:
        """GET /auth/me to obtain the UUID user_id."""
        resp = await self._http.get("/auth/me", headers=self._auth_headers())
        resp.raise_for_status()
        self._user_id = resp.json()["id"]

    async def _ensure_token(self) -> None:
        """Refresh or re-login if the token is about to expire (60s buffer)."""
        if time.time() < self._token_expires_at - 60:
            return
        try:
            resp = await self._http.post("/auth/refresh", headers=self._auth_headers())
            resp.raise_for_status()
            body = resp.json()
            self._token = body["access_token"]
            self._token_expires_at = time.time() + body["expires_in"]
        except (httpx.HTTPStatusError, httpx.HTTPError):
            await self._login()

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    async def _resolve_conversation_id(self, external_conversation_id: str) -> str:
        """Map an external conversation id to a valid backend conversation id."""
        mapped = self._conversation_map.get(external_conversation_id)
        if mapped is not None:
            return mapped

        create_resp = await self._http.post(
            "/conversation/",
            json={
                "user_id": self._user_id,
                "title": f"MCP Session {external_conversation_id[:8]}",
                "metadata": {
                    "source": "engram_mcp",
                    "external_conversation_id": external_conversation_id,
                },
            },
            headers=self._auth_headers(),
        )
        create_resp.raise_for_status()

        backend_conversation_id = create_resp.json()["id"]
        self._conversation_map[external_conversation_id] = backend_conversation_id
        return backend_conversation_id

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    async def process_turn(
        self, user_message: str, conversation_id: str | None = None
    ) -> dict[str, Any]:
        """POST /memory/process-turn — auto-classify and store a memory."""
        await self._ensure_token()
        mapped_conversation_id = (
            self._conversation_map.get(conversation_id, conversation_id)
            if conversation_id is not None
            else None
        )
        body: dict[str, Any] = {
            "user_message": user_message,
            "user_id": self._user_id,
        }
        if mapped_conversation_id is not None:
            body["conversation_id"] = mapped_conversation_id

        try:
            resp = await self._http.post(
                "/memory/process-turn",
                json=body,
                headers=self._auth_headers(),
            )
            resp.raise_for_status()
        except httpx.ReadTimeout as exc:
            return await self._handle_ambiguous_process_turn_failure(
                error=exc,
                user_message=user_message,
                conversation_id=conversation_id,
                mapped_conversation_id=mapped_conversation_id,
            )
        except httpx.HTTPStatusError as exc:
            if conversation_id is None or exc.response.status_code not in {400, 404, 422, 500}:
                if exc.response.status_code >= 500:
                    return await self._handle_ambiguous_process_turn_failure(
                        error=exc,
                        user_message=user_message,
                        conversation_id=conversation_id,
                        mapped_conversation_id=mapped_conversation_id,
                    )
                raise

            # Conversation ids from external runtimes may not exist in Engram yet.
            # Auto-create/map once, then retry the write.
            self._conversation_map.pop(conversation_id, None)
            resolved_conversation_id = await self._resolve_conversation_id(conversation_id)
            body["conversation_id"] = resolved_conversation_id
            try:
                resp = await self._http.post(
                    "/memory/process-turn",
                    json=body,
                    headers=self._auth_headers(),
                )
                resp.raise_for_status()
            except httpx.ReadTimeout as retry_exc:
                return await self._handle_ambiguous_process_turn_failure(
                    error=retry_exc,
                    user_message=user_message,
                    conversation_id=conversation_id,
                    mapped_conversation_id=resolved_conversation_id,
                )
            except httpx.HTTPStatusError as retry_exc:
                if retry_exc.response.status_code >= 500:
                    return await self._handle_ambiguous_process_turn_failure(
                        error=retry_exc,
                        user_message=user_message,
                        conversation_id=conversation_id,
                        mapped_conversation_id=resolved_conversation_id,
                    )
                raise
        return resp.json()

    async def _handle_ambiguous_process_turn_failure(
        self,
        *,
        error: Exception,
        user_message: str,
        conversation_id: str | None,
        mapped_conversation_id: str | None,
    ) -> dict[str, Any]:
        """Handle timeout/5xx failures where original write outcome is uncertain."""
        recovered_memory_id = await self._recover_memory_id_after_ambiguous_process_turn(
            user_message=user_message,
            conversation_id=mapped_conversation_id,
        )
        if recovered_memory_id is not None:
            return {
                "operation_performed": "ADD",
                "memory_id": recovered_memory_id,
                "memories_affected": 1,
                "processing_time_ms": 0.0,
                "recovered_from_ambiguous_failure": True,
            }

        if self._unsafe_process_turn_fallback:
            logger.warning(
                "Using unsafe process_turn fallback after ambiguous failure; "
                "this may create duplicate memories."
            )
            return await self._fallback_process_turn(user_message, conversation_id)

        raise error

    async def _recover_memory_id_after_ambiguous_process_turn(
        self,
        *,
        user_message: str,
        conversation_id: str | None,
    ) -> int | None:
        """Attempt to discover whether process-turn already created the memory."""
        params: dict[str, Any] = {"limit": 100, "offset": 0}
        if conversation_id is not None:
            params["conversation_id"] = conversation_id

        for attempt in range(self._process_turn_recovery_attempts):
            if attempt > 0 and self._process_turn_recovery_delay_seconds > 0:
                await asyncio.sleep(self._process_turn_recovery_delay_seconds)

            try:
                response = await self._http.get(
                    "/memory/",
                    params=params,
                    headers=self._auth_headers(),
                )
                response.raise_for_status()
            except httpx.HTTPError:
                continue

            memories = response.json()
            for mem in memories:
                if mem.get("text") == user_message:
                    memory_id = mem.get("id")
                    if isinstance(memory_id, int):
                        return memory_id
                    if isinstance(memory_id, str) and memory_id.isdigit():
                        return int(memory_id)

        return None

    async def _fallback_process_turn(
        self, user_message: str, conversation_id: str | None
    ) -> dict[str, Any]:
        """Fallback when process-turn is unavailable: store memory directly."""
        created = await self.create_memory(
            text=user_message,
            importance_score=5.0,
            conversation_id=conversation_id,
            metadata={"source": "process_turn_fallback"},
        )
        return {
            "operation_performed": "ADD",
            "memory_id": created.get("id"),
            "memories_affected": 1,
            "processing_time_ms": 0.0,
            "fallback_used": True,
        }

    async def query_memories(
        self,
        query: str,
        top_k: int = 5,
        scoring_profile: str | None = None,
    ) -> dict[str, Any]:
        """POST /memory/query — semantic search via ACAN retrieval."""
        await self._ensure_token()
        payload: dict[str, Any] = {
            "query": query,
            "user_id": self._user_id,
            "top_k": top_k,
        }
        if scoring_profile is not None:
            payload["scoring_profile"] = scoring_profile

        try:
            resp = await self._http.post(
                "/memory/query",
                json=payload,
                headers=self._auth_headers(),
            )
            resp.raise_for_status()
            return resp.json()
        except httpx.ReadTimeout:
            return await self._fallback_query_memories(query, top_k)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code >= 500:
                return await self._fallback_query_memories(query, top_k)
            raise

    async def _fallback_query_memories(self, query: str, top_k: int) -> dict[str, Any]:
        """Fallback when semantic query endpoint is unavailable."""
        resp = await self._http.get(
            "/memory/",
            params={"limit": min(max(top_k * 10, top_k), 100), "offset": 0},
            headers=self._auth_headers(),
        )
        resp.raise_for_status()
        memories = resp.json()

        tokens = [tok for tok in re.findall(r"\w+", query.lower()) if len(tok) > 2]
        if tokens:
            memories.sort(
                key=lambda mem: (
                    sum(mem.get("text", "").lower().count(tok) for tok in tokens),
                    mem.get("importance_score", 0.0),
                ),
                reverse=True,
            )

        return {
            "query": query,
            "memories": memories[:top_k],
            "total_found": len(memories),
            "processing_time_ms": 0.0,
            "fallback_used": True,
        }

    async def generate_answer(self, question: str, context_memories: list[str]) -> str:
        """Generate concise factual answer from retrieved memory snippets."""
        if not context_memories:
            return "No information available."

        snippets = [text.strip() for text in context_memories if text and text.strip()]
        if not snippets:
            return "No information available."

        context = "\n".join(f"- {snippet}" for snippet in snippets[:5])
        prompt = (
            "Based only on the memories below, answer the question with a brief factual answer.\n"
            "If the answer is not present, respond exactly: No information available.\n\n"
            f"Memories:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        if self._llm_provider == "openai":
            return await self._generate_answer_openai(prompt)
        return await self._generate_answer_ollama(prompt)

    async def _generate_answer_ollama(self, prompt: str) -> str:
        """Generate answer with Ollama Chat API."""
        payload = {
            "model": self._ollama_llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a factual extractor. Return only the answer text.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(f"{self._ollama_base_url}/api/chat", json=payload)
            response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        answer = str(content).strip()
        return answer or "No information available."

    async def _generate_answer_openai(self, prompt: str) -> str:
        """Generate answer with OpenAI Chat Completions API."""
        if not self._openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")

        payload = {
            "model": self._openai_llm_model,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a factual extractor. Return only the answer text.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
            response = await client.post(
                f"{self._openai_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._openai_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
        body = response.json()
        content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        answer = str(content).strip()
        return answer or "No information available."

    async def create_memory(
        self,
        text: str,
        importance_score: float = 5.0,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST /memory/ — direct memory creation."""
        await self._ensure_token()
        mapped_conversation_id = (
            self._conversation_map.get(conversation_id, conversation_id)
            if conversation_id is not None
            else None
        )
        body: dict[str, Any] = {
            "text": text,
            "user_id": self._user_id,
            "importance_score": importance_score,
        }
        if mapped_conversation_id is not None:
            body["conversation_id"] = mapped_conversation_id
        if metadata is not None:
            body["metadata"] = metadata

        try:
            resp = await self._http.post("/memory/", json=body, headers=self._auth_headers())
            resp.raise_for_status()
        except httpx.ReadTimeout:
            raise
        except httpx.HTTPStatusError as exc:
            if conversation_id is None or exc.response.status_code not in {400, 404, 422, 500}:
                raise

            self._conversation_map.pop(conversation_id, None)
            resolved_conversation_id = await self._resolve_conversation_id(conversation_id)
            body["conversation_id"] = resolved_conversation_id
            resp = await self._http.post("/memory/", json=body, headers=self._auth_headers())
            resp.raise_for_status()
        return resp.json()

    async def delete_memory(self, memory_id: int) -> dict[str, Any]:
        """DELETE /memory/{id}."""
        await self._ensure_token()
        resp = await self._http.delete(
            f"/memory/{memory_id}",
            headers=self._auth_headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def get_stats(self) -> dict[str, Any]:
        """GET /memory/stats/overview."""
        await self._ensure_token()
        resp = await self._http.get(
            "/memory/stats/overview",
            headers=self._auth_headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def health_check(self) -> dict[str, Any]:
        """GET /health/ — no auth required."""
        resp = await self._http.get("/health/")
        resp.raise_for_status()
        return resp.json()
