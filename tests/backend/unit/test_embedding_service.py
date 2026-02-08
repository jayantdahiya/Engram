"""Unit tests for EmbeddingService."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from services.embedding_service import EmbeddingService


class TestEmbeddingService:
    """Test cases for EmbeddingService."""

    @pytest.fixture
    def embedding_service(self):
        """Create embedding service instance with mocked local model."""
        with patch("services.embedding_service.SentenceTransformer") as mock_st:
            mock_st.return_value = MagicMock()
            service = EmbeddingService()
            return service

    @pytest.fixture
    def sample_embedding(self):
        """Create sample embedding vector."""
        return np.random.rand(1536).astype(np.float32)

    @pytest.mark.asyncio
    async def test_get_embedding_with_ollama(self, embedding_service):
        """Test embedding generation via Ollama API."""
        mock_response = {"embeddings": [np.random.rand(1536).tolist()]}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = MagicMock(
                json=lambda: mock_response, raise_for_status=MagicMock()
            )

            result = await embedding_service.get_embedding("test text")

            assert isinstance(result, np.ndarray)
            assert result.dtype == np.float32

    @pytest.mark.asyncio
    async def test_get_embedding_with_local_model(self, embedding_service):
        """Test embedding generation with local model fallback."""
        embedding_service.local_model = MagicMock()
        embedding_service.local_model.encode.return_value = np.random.rand(384)

        result = await embedding_service.get_embedding("test text", use_local=True)

        assert isinstance(result, np.ndarray)
        embedding_service.local_model.encode.assert_called_once_with("test text")

    def test_truncate_embedding(self, embedding_service):
        """Test Matryoshka-style dimension reduction."""
        embedding_service.target_dimension = 512
        full_embedding = np.random.rand(1536).astype(np.float32)

        result = embedding_service._truncate_embedding(full_embedding)

        assert len(result) == 512
        # Check normalization
        norm = np.linalg.norm(result)
        assert np.isclose(norm, 1.0, atol=1e-5)

    def test_truncate_embedding_no_truncation_needed(self, embedding_service):
        """Test no truncation when embedding is already correct size."""
        embedding_service.target_dimension = 1536
        full_embedding = np.random.rand(1536).astype(np.float32)

        result = embedding_service._truncate_embedding(full_embedding)

        assert len(result) == 1536

    @pytest.mark.asyncio
    async def test_calculate_similarity(self, embedding_service):
        """Test cosine similarity calculation."""
        embedding1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        embedding2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        result = await embedding_service.calculate_similarity(embedding1, embedding2)

        assert result == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_calculate_similarity_orthogonal(self, embedding_service):
        """Test similarity of orthogonal vectors."""
        embedding1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        embedding2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        result = await embedding_service.calculate_similarity(embedding1, embedding2)

        assert result == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.asyncio
    async def test_calculate_similarity_zero_vector(self, embedding_service):
        """Test similarity with zero vector returns 0."""
        embedding1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        embedding2 = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        result = await embedding_service.calculate_similarity(embedding1, embedding2)

        assert result == 0.0

    @pytest.mark.asyncio
    async def test_find_most_similar(self, embedding_service):
        """Test finding most similar embeddings."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = [
            np.array([1.0, 0.0, 0.0], dtype=np.float32),  # Most similar
            np.array([0.5, 0.5, 0.0], dtype=np.float32),  # Medium
            np.array([0.0, 1.0, 0.0], dtype=np.float32),  # Least similar
        ]

        result = await embedding_service.find_most_similar(query, candidates, top_k=2)

        assert len(result) == 2
        assert result[0][0] == 0  # First candidate is most similar
        assert result[0][1] > result[1][1]  # Scores are descending

    def test_count_tokens(self, embedding_service):
        """Test token counting approximation."""
        text = "This is a test text with multiple words"

        result = embedding_service.count_tokens(text)

        # Approximate: ~1.3 tokens per word
        assert result > len(text.split())
        assert result < len(text.split()) * 2

    @pytest.mark.asyncio
    async def test_get_embeddings_batch_with_local_model(self, embedding_service):
        """Test batch embedding generation."""
        embedding_service.local_model = MagicMock()
        embedding_service.local_model.encode.return_value = [
            np.random.rand(384) for _ in range(3)
        ]

        texts = ["text 1", "text 2", "text 3"]
        result = await embedding_service.get_embeddings_batch(texts, batch_size=10)

        assert len(result) == 3
        for emb in result:
            assert isinstance(emb, np.ndarray)

    @pytest.mark.asyncio
    async def test_process_embedding_request(self, embedding_service):
        """Test embedding request processing."""
        from models.memory import EmbeddingRequest

        with patch.object(
            embedding_service,
            "get_embedding",
            return_value=np.random.rand(1536).astype(np.float32),
        ):
            request = EmbeddingRequest(text="test text")
            result = await embedding_service.process_embedding_request(request)

            assert result.text == "test text"
            assert len(result.embedding) == 1536
            assert result.dimension == 1536
            assert result.tokens_used > 0

    @pytest.mark.asyncio
    async def test_ollama_fallback_to_local(self, embedding_service):
        """Test fallback to local model when Ollama fails."""
        embedding_service.local_model = MagicMock()
        embedding_service.local_model.encode.return_value = np.random.rand(384)

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.side_effect = Exception("Ollama not available")

            result = await embedding_service.get_embedding("test text")

            assert isinstance(result, np.ndarray)
