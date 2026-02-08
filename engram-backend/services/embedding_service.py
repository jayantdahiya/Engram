"""Embedding service for generating and managing text embeddings"""

import asyncio
import json

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer

from core.config import settings
from core.logging import logger
from models.memory import EmbeddingRequest, EmbeddingResponse


class EmbeddingService:
    """Service for generating text embeddings using multiple providers"""

    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.embedding_model = settings.ollama_embedding_model
        self.target_dimension = settings.embedding_dimension
        self.local_model: SentenceTransformer | None = None

        # Initialize local model for fallback
        try:
            self.local_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Local embedding model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load local embedding model: {e}")

    def _truncate_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Truncate embedding to target dimension (Matryoshka-style) and re-normalize."""
        if self.target_dimension and len(embedding) > self.target_dimension:
            embedding = embedding[: self.target_dimension]
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        return embedding

    async def get_embedding(
        self, text: str, model: str = "", use_local: bool = False
    ) -> np.ndarray:
        """Get embedding for text using specified model"""

        if use_local and self.local_model:
            return self._truncate_embedding(await self._get_local_embedding(text))
        else:
            return self._truncate_embedding(await self._get_ollama_embedding(text))

    def _ensure_local_model(self) -> bool:
        """Ensure local model is loaded, retrying if initial load failed."""
        if self.local_model is not None:
            return True
        try:
            self.local_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Local embedding model loaded successfully (retry)")
            return True
        except Exception as e:
            logger.warning(f"Failed to load local embedding model (retry): {e}")
            return False

    async def _get_ollama_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Ollama API"""
        try:
            payload = {
                "model": self.embedding_model,
                "prompt": text
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                result = response.json()
                embedding = np.array(result["embedding"], dtype=np.float32)
                logger.debug(f"Generated Ollama embedding with dimension {len(embedding)}")
                return embedding

        except Exception as e:
            logger.error(f"Ollama embedding failed: {e}")
            if self._ensure_local_model():
                logger.info("Falling back to local embedding model")
                return await self._get_local_embedding(text)
            else:
                raise

    async def _get_local_embedding(self, text: str) -> np.ndarray:
        """Get embedding from local model"""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(None, self.local_model.encode, text)

            embedding = np.array(embedding, dtype=np.float32)
            logger.debug(f"Generated local embedding with dimension {len(embedding)}")
            return embedding

        except Exception as e:
            logger.error(f"Local embedding failed: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using word-based approximation"""
        try:
            # Ollama uses a different tokenization approach
            # Using word-based approximation for now
            word_count = len(text.split())
            # Approximate tokens: ~1.3 words per token for English text
            return int(word_count * 1.3)
        except Exception as e:
            logger.error(f"Token counting failed: {e}")
            # Fallback to simple word count
            return len(text.split())

    async def get_embeddings_batch(
        self, texts: list[str], model: str = "", batch_size: int = 100
    ) -> list[np.ndarray]:
        """Get embeddings for multiple texts in batches"""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            if self.local_model:
                # Use local model for batch processing
                loop = asyncio.get_event_loop()
                batch_embeddings = await loop.run_in_executor(None, self.local_model.encode, batch)
                embeddings.extend([np.array(emb, dtype=np.float32) for emb in batch_embeddings])
            else:
                # Use Ollama API for batch processing
                batch_embeddings = await asyncio.gather(
                    *[self.get_embedding(text) for text in batch]
                )
                embeddings.extend(batch_embeddings)

            logger.debug(
                f"Processed batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size}"
            )

        return embeddings

    async def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Ensure embeddings are normalized
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
        """Find most similar embeddings to query"""
        similarities = []

        for i, candidate in enumerate(candidate_embeddings):
            similarity = await self.calculate_similarity(query_embedding, candidate)
            similarities.append((i, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    async def process_embedding_request(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Process embedding request and return response"""
        start_time = asyncio.get_event_loop().time()

        try:
            embedding = await self.get_embedding(request.text)
            tokens_used = self.count_tokens(request.text)

            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000

            return EmbeddingResponse(
                text=request.text,
                embedding=embedding.tolist(),
                model=self.embedding_model,
                dimension=len(embedding),
                tokens_used=tokens_used,
            )

        except Exception as e:
            logger.error(f"Embedding request processing failed: {e}")
            raise


# Global embedding service instance
embedding_service = EmbeddingService()
