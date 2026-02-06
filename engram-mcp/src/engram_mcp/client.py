"""HTTP client for the Engram REST API with automatic authentication."""

from __future__ import annotations

import time
from typing import Any

import httpx


class EngramClient:
    """Manages HTTP communication and JWT auth with the Engram backend."""

    def __init__(self, api_url: str, username: str, password: str) -> None:
        self._api_url = api_url
        self._username = username
        self._password = password

        self._http: httpx.AsyncClient | None = None
        self._token: str | None = None
        self._token_expires_at: float = 0.0  # epoch seconds
        self._user_id: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._http = httpx.AsyncClient(base_url=self._api_url, timeout=30.0)
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
                "email": f"{self._username}@engram.local",
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

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    async def process_turn(
        self, user_message: str, conversation_id: str = "default"
    ) -> dict[str, Any]:
        """POST /memory/process-turn — auto-classify and store a memory."""
        await self._ensure_token()
        resp = await self._http.post(
            "/memory/process-turn",
            json={
                "user_message": user_message,
                "user_id": self._user_id,
                "conversation_id": conversation_id,
            },
            headers=self._auth_headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def query_memories(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """POST /memory/query — semantic search via ACAN retrieval."""
        await self._ensure_token()
        resp = await self._http.post(
            "/memory/query",
            json={
                "query": query,
                "user_id": self._user_id,
                "top_k": top_k,
            },
            headers=self._auth_headers(),
        )
        resp.raise_for_status()
        return resp.json()

    async def create_memory(
        self,
        text: str,
        importance_score: float = 5.0,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST /memory/ — direct memory creation."""
        await self._ensure_token()
        body: dict[str, Any] = {
            "text": text,
            "user_id": self._user_id,
            "importance_score": importance_score,
        }
        if conversation_id is not None:
            body["conversation_id"] = conversation_id
        if metadata is not None:
            body["metadata"] = metadata
        resp = await self._http.post(
            "/memory/",
            json=body,
            headers=self._auth_headers(),
        )
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
