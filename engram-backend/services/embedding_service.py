"""Embedding service for generating and managing text embeddings."""

import asyncio
import hashlib
from collections import OrderedDict
from dataclasses import dataclass

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import settings
from core.logging import logger
from models.memory import EmbeddingRequest, EmbeddingResponse


@dataclass
class EmbeddingResult:
    """Container for embedding response metadata."""

    embedding: np.ndarray
    model: str
    tokens_used: int | None = None


class EmbeddingService:
    """Service for generating text embeddings using multiple providers."""

    def __init__(self):
        self.embedding_provider = settings.embedding_provider.lower()

        self.ollama_base_url = settings.ollama_base_url
        self.ollama_embedding_model = settings.ollama_embedding_model

        self.openai_api_key = settings.openai_api_key
        self.openai_embedding_model = settings.openai_embedding_model
        self.openai_base_url = "https://api.openai.com/v1"

        self.google_api_key = settings.google_api_key
        self.google_embedding_model = settings.google_embedding_model
        self.google_base_url = "https://generativelanguage.googleapis.com/v1beta"

        self.target_dimension = settings.embedding_dimension
        self.local_model_name = "all-MiniLM-L6-v2"
        self.local_model: SentenceTransformer | None = None
        self._cache_max_entries = 1024
        self._embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_lock = asyncio.Lock()

        # Backwards-compatible model field for callers that inspect this value.
        self.embedding_model = self._default_model_for_provider(self.embedding_provider)

        # Initialize local model once for local mode/fallbacks.
        try:
            self.local_model = SentenceTransformer(self.local_model_name)
            logger.info("Local embedding model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load local embedding model: {e}")

        logger.info(
            "Embedding Service initialized with provider: "
            f"{self.embedding_provider} ({self.embedding_model})"
        )

    def _default_model_for_provider(self, provider: str) -> str:
        """Return the configured default model for a provider."""
        if provider == "openai":
            return self.openai_embedding_model
        if provider == "google":
            return self.google_embedding_model
        if provider == "local":
            return self.local_model_name
        return self.ollama_embedding_model

    def _truncate_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Adjust embedding to target dimension while preserving cosine behavior."""
        if not self.target_dimension:
            return embedding

        if len(embedding) > self.target_dimension:
            embedding = embedding[: self.target_dimension]
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        elif len(embedding) < self.target_dimension:
            padding = np.zeros(self.target_dimension - len(embedding), dtype=np.float32)
            embedding = np.concatenate([embedding, padding])
        return embedding

    async def get_embedding(
        self, text: str, model: str = "", use_local: bool = False
    ) -> np.ndarray:
        """Get embedding for text using configured provider."""
        cache_key = self._build_cache_key(text=text, model=model, use_local=use_local)
        cached = await self._get_cached_embedding(cache_key)
        if cached is not None:
            return cached

        result = await self._get_embedding_result(text, model, use_local)
        await self._set_cached_embedding(cache_key, result.embedding)
        return np.array(result.embedding, copy=True)

    def _build_cache_key(self, text: str, model: str, use_local: bool) -> str:
        """Build stable cache key for embedding requests."""
        selected_model = model or self.embedding_model
        provider = "local" if use_local else self.embedding_provider
        raw_key = f"{provider}|{selected_model}|{self.target_dimension}|{text}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    async def _get_cached_embedding(self, cache_key: str) -> np.ndarray | None:
        """Lookup embedding from LRU cache."""
        async with self._cache_lock:
            cached = self._embedding_cache.get(cache_key)
            if cached is None:
                return None
            self._embedding_cache.move_to_end(cache_key)
            return np.array(cached, copy=True)

    async def _set_cached_embedding(self, cache_key: str, embedding: np.ndarray) -> None:
        """Store embedding in LRU cache."""
        async with self._cache_lock:
            self._embedding_cache[cache_key] = np.array(embedding, copy=True)
            self._embedding_cache.move_to_end(cache_key)
            if len(self._embedding_cache) > self._cache_max_entries:
                self._embedding_cache.popitem(last=False)

    async def _get_embedding_result(
        self, text: str, model: str = "", use_local: bool = False
    ) -> EmbeddingResult:
        """Get embedding and metadata for text using configured provider."""
        if use_local:
            return await self._get_local_embedding_result(text)

        provider_name = self.embedding_provider

        if provider_name == "openai":
            try:
                return await self._get_openai_embedding_result(text, model)
            except Exception as e:
                logger.error(f"OpenAI embedding failed, falling back to Ollama/local: {e}")
                return await self._get_ollama_with_local_fallback(text, model)

        if provider_name == "google":
            try:
                return await self._get_google_embedding_result(text, model)
            except Exception as e:
                logger.error(f"Google AI embedding failed, falling back to Ollama/local: {e}")
                return await self._get_ollama_with_local_fallback(text, model)

        if provider_name == "local":
            return await self._get_local_embedding_result(text)

        if provider_name == "ollama":
            return await self._get_ollama_with_local_fallback(text, model)

        logger.warning(f"Unknown embedding provider '{provider_name}', defaulting to Ollama")
        return await self._get_ollama_with_local_fallback(text, model)

    async def _get_ollama_with_local_fallback(self, text: str, model: str = "") -> EmbeddingResult:
        """Try Ollama first, then fall back to local embeddings."""
        try:
            return await self._get_ollama_embedding_result(text, model)
        except Exception as e:
            logger.error(f"Ollama embedding failed: {e}")
            if self._ensure_local_model():
                logger.info("Falling back to local embedding model")
                return await self._get_local_embedding_result(text)
            raise

    def _ensure_local_model(self) -> bool:
        """Ensure local model is loaded, retrying if initial load failed."""
        if self.local_model is not None:
            return True
        try:
            self.local_model = SentenceTransformer(self.local_model_name)
            logger.info("Local embedding model loaded successfully (retry)")
            return True
        except Exception as e:
            logger.warning(f"Failed to load local embedding model (retry): {e}")
            return False

    async def _get_ollama_embedding_result(self, text: str, model: str = "") -> EmbeddingResult:
        """Get embedding from Ollama API."""
        selected_model = model or self.ollama_embedding_model
        payload: dict[str, object] = {
            "model": selected_model,
            "input": text,
        }

        if self.target_dimension:
            payload["dimensions"] = self.target_dimension

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.ollama_base_url}/api/embed",
                json=payload,
                timeout=30.0,
            )

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if payload.get("dimensions") and exc.response.status_code in {400, 422}:
                    logger.warning(
                        "Ollama model rejected dimensions parameter; retrying without dimensions"
                    )
                    payload.pop("dimensions", None)
                    response = await client.post(
                        f"{self.ollama_base_url}/api/embed",
                        json=payload,
                        timeout=30.0,
                    )
                    response.raise_for_status()
                else:
                    raise

        result = response.json()
        raw_embedding = None

        embeddings = result.get("embeddings")
        if isinstance(embeddings, list) and embeddings:
            raw_embedding = embeddings[0]
        elif "embedding" in result:
            raw_embedding = result.get("embedding")

        if raw_embedding is None:
            raise ValueError("Ollama embedding response missing embedding data")

        embedding = self._truncate_embedding(np.array(raw_embedding, dtype=np.float32))
        usage = result.get("usage", {})
        tokens_used = usage.get("prompt_tokens") if isinstance(usage, dict) else None

        logger.debug(f"Generated Ollama embedding with dimension {len(embedding)}")
        return EmbeddingResult(
            embedding=embedding,
            model=selected_model,
            tokens_used=tokens_used,
        )

    async def _get_openai_embedding_result(self, text: str, model: str = "") -> EmbeddingResult:
        """Get embedding from OpenAI Embeddings API."""
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")

        selected_model = model or self.openai_embedding_model
        payload = {
            "model": selected_model,
            "input": text,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.openai_base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30.0,
            )
            response.raise_for_status()

        result = response.json()
        data = result.get("data", [])

        if not data:
            raise ValueError("OpenAI embedding response missing data")

        raw_embedding = data[0].get("embedding")
        if raw_embedding is None:
            raise ValueError("OpenAI embedding response missing embedding vector")

        embedding = self._truncate_embedding(np.array(raw_embedding, dtype=np.float32))
        usage = result.get("usage", {})
        tokens_used = usage.get("total_tokens") if isinstance(usage, dict) else None

        logger.debug(f"Generated OpenAI embedding with dimension {len(embedding)}")
        return EmbeddingResult(
            embedding=embedding,
            model=selected_model,
            tokens_used=tokens_used,
        )

    async def _get_google_embedding_result(self, text: str, model: str = "") -> EmbeddingResult:
        """Get embedding from Google AI (Gemini) Embeddings API."""
        if not self.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is required when EMBEDDING_PROVIDER=google")

        selected_model = model or self.google_embedding_model
        payload = {
            "content": {"parts": [{"text": text}]},
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.google_base_url}/models/{selected_model}:embedContent",
                headers={"Content-Type": "application/json"},
                params={"key": self.google_api_key},
                json=payload,
                timeout=30.0,
            )
            response.raise_for_status()

        result = response.json()
        embedding_data = result.get("embedding", {})
        raw_embedding = embedding_data.get("values")

        if raw_embedding is None:
            raise ValueError("Google AI embedding response missing embedding values")

        embedding = self._truncate_embedding(np.array(raw_embedding, dtype=np.float32))

        logger.debug(f"Generated Google AI embedding with dimension {len(embedding)}")
        return EmbeddingResult(
            embedding=embedding,
            model=selected_model,
            tokens_used=self.count_tokens(text),
        )

    async def _get_local_embedding_result(self, text: str) -> EmbeddingResult:
        """Get embedding from local sentence-transformers model."""
        if not self._ensure_local_model() or self.local_model is None:
            raise RuntimeError("Local embedding model unavailable")

        loop = asyncio.get_running_loop()
        vector = await loop.run_in_executor(None, self.local_model.encode, text)
        embedding = self._truncate_embedding(np.array(vector, dtype=np.float32))

        logger.debug(f"Generated local embedding with dimension {len(embedding)}")
        return EmbeddingResult(
            embedding=embedding,
            model=self.local_model_name,
            tokens_used=self.count_tokens(text),
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using word-based approximation."""
        try:
            word_count = len(text.split())
            return int(word_count * 1.3)
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            return len(text.split())

    async def get_embeddings_batch(
        self, texts: list[str], model: str = "", batch_size: int = 100
    ) -> list[np.ndarray]:
        """Get embeddings for multiple texts in batches."""
        embeddings: list[np.ndarray] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            if (
                self.embedding_provider == "local"
                and self._ensure_local_model()
                and self.local_model
            ):
                loop = asyncio.get_running_loop()
                batch_embeddings = await loop.run_in_executor(None, self.local_model.encode, batch)
                embeddings.extend(
                    [
                        self._truncate_embedding(np.array(emb, dtype=np.float32))
                        for emb in batch_embeddings
                    ]
                )
            else:
                batch_embeddings = await asyncio.gather(
                    *[self.get_embedding(text, model=model) for text in batch]
                )
                embeddings.extend(batch_embeddings)

            logger.debug(
                "Processed batch "
                f"{i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}"
            )

        return embeddings

    async def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0

    async def find_most_similar(
        self, query_embedding: np.ndarray, candidate_embeddings: list[np.ndarray], top_k: int = 5
    ) -> list[tuple]:
        """Find most similar embeddings to query."""
        similarities = []

        for i, candidate in enumerate(candidate_embeddings):
            similarity = await self.calculate_similarity(query_embedding, candidate)
            similarities.append((i, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    async def process_embedding_request(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Process embedding request and return response."""
        try:
            result = await self._get_embedding_result(request.text, request.model or "")
            tokens_used = (
                result.tokens_used
                if result.tokens_used is not None
                else self.count_tokens(request.text)
            )

            return EmbeddingResponse(
                text=request.text,
                embedding=result.embedding.tolist(),
                model=result.model,
                dimension=len(result.embedding),
                tokens_used=tokens_used,
            )

        except Exception as e:
            logger.error(f"Embedding request processing failed: {e}")
            raise


# Global embedding service instance
embedding_service = EmbeddingService()
