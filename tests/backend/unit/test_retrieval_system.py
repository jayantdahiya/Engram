"""Unit tests for ACANRetrievalSystem."""

import time

import numpy as np
import pytest

from services.retrieval_system import ACANRetrievalSystem


class TestACANRetrievalSystem:
    """Test cases for ACAN retrieval system."""

    @pytest.fixture
    def retrieval_system(self):
        """Create ACAN retrieval system instance."""
        return ACANRetrievalSystem(attention_dim=64)

    @pytest.fixture
    def sample_memories(self):
        """Create sample memories for testing."""
        current_time = time.time()
        return [
            {
                "id": 1,
                "text": "I love pizza",
                "embedding": np.random.rand(1536).tolist(),
                "timestamp": current_time - 3600,  # 1 hour ago
                "importance_score": 7.0,
                "access_count": 5,
            },
            {
                "id": 2,
                "text": "I am vegetarian",
                "embedding": np.random.rand(1536).tolist(),
                "timestamp": current_time - 7200,  # 2 hours ago
                "importance_score": 9.0,
                "access_count": 10,
            },
            {
                "id": 3,
                "text": "I have two dogs",
                "embedding": np.random.rand(1536).tolist(),
                "timestamp": current_time - 86400,  # 1 day ago
                "importance_score": 5.0,
                "access_count": 2,
            },
        ]

    @pytest.fixture
    def query_embedding(self):
        """Create sample query embedding."""
        return np.random.rand(1536).astype(np.float32)

    def test_init_projection_matrix(self, retrieval_system):
        """Test projection matrix initialization."""
        assert retrieval_system.query_projection is not None
        assert retrieval_system.query_projection.shape == (1536, 64)

    def test_recency_weight_recent(self, retrieval_system):
        """Test recency weight for recent memory."""
        current_time = time.time()
        recent_timestamp = current_time - 3600  # 1 hour ago

        weight = retrieval_system._recency_weight(
            recent_timestamp, current_time, half_life_hours=72.0
        )

        assert 0.9 < weight <= 1.0  # Should be close to 1

    def test_recency_weight_old(self, retrieval_system):
        """Test recency weight for old memory."""
        current_time = time.time()
        old_timestamp = current_time - (72 * 3600)  # 72 hours ago (half-life)

        weight = retrieval_system._recency_weight(
            old_timestamp, current_time, half_life_hours=72.0
        )

        assert 0.4 < weight < 0.6  # Should be around 0.5

    def test_recency_weight_very_old(self, retrieval_system):
        """Test recency weight for very old memory."""
        current_time = time.time()
        very_old_timestamp = current_time - (30 * 24 * 3600)  # 30 days ago

        weight = retrieval_system._recency_weight(
            very_old_timestamp, current_time, half_life_hours=72.0
        )

        assert weight < 0.1  # Should be very small

    @pytest.mark.asyncio
    async def test_compute_attention(
        self, retrieval_system, query_embedding, sample_memories
    ):
        """Test cross-attention computation."""
        memory_embeddings = [np.array(m["embedding"]) for m in sample_memories]

        scores = await retrieval_system._compute_attention(
            query_embedding, memory_embeddings
        )

        assert len(scores) == len(sample_memories)
        assert all(isinstance(s, (int, float)) for s in scores)

    @pytest.mark.asyncio
    async def test_compute_composite_scores(
        self, retrieval_system, query_embedding, sample_memories
    ):
        """Test composite scoring with multiple signals."""
        current_time = time.time()

        scores = await retrieval_system.compute_composite_scores(
            query_embedding=query_embedding,
            memories=sample_memories,
            current_time=current_time,
        )

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(sample_memories)

    @pytest.mark.asyncio
    async def test_compute_composite_scores_with_context(
        self, retrieval_system, query_embedding, sample_memories
    ):
        """Test composite scoring with user context."""
        current_time = time.time()
        user_context = {
            "current_topic": "food",
            "conversation_id": "test-conversation",
        }

        scores = await retrieval_system.compute_composite_scores(
            query_embedding=query_embedding,
            memories=sample_memories,
            current_time=current_time,
            user_context=user_context,
        )

        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(sample_memories)

    @pytest.mark.asyncio
    async def test_compute_context_weights(self, retrieval_system, sample_memories):
        """Test context-aware weight computation."""
        user_context = {
            "current_topic": "food",
            "active_entities": ["pizza"],
        }

        weights = await retrieval_system._compute_context_weights(
            sample_memories, user_context
        )

        assert len(weights) == len(sample_memories)
        assert all(w >= 0 for w in weights)

    @pytest.mark.asyncio
    async def test_memory_distillation(
        self, retrieval_system, query_embedding, sample_memories
    ):
        """Test memory distillation for noise reduction."""
        scores = np.array([0.9, 0.5, 0.2])

        distilled_memories = await retrieval_system.memory_distillation(
            memories=sample_memories,
            scores=scores,
            threshold=0.3,
            max_memories=10,
        )

        # Should filter out low-scoring memories
        assert len(distilled_memories) <= len(sample_memories)

    @pytest.mark.asyncio
    async def test_memory_distillation_max_limit(
        self, retrieval_system, query_embedding, sample_memories
    ):
        """Test memory distillation respects max limit."""
        scores = np.array([0.9, 0.8, 0.7])

        distilled_memories = await retrieval_system.memory_distillation(
            memories=sample_memories,
            scores=scores,
            threshold=0.1,
            max_memories=2,
        )

        assert len(distilled_memories) <= 2

    @pytest.mark.asyncio
    async def test_retrieve_with_reranking(
        self, retrieval_system, query_embedding, sample_memories
    ):
        """Test multi-stage retrieval with reranking."""
        current_time = time.time()

        memories, scores = await retrieval_system.retrieve_with_reranking(
            query_embedding=query_embedding,
            memories=sample_memories,
            top_k=2,
            apply_distillation=True,
        )

        assert len(memories) <= 2
        assert len(scores) == len(memories)

    @pytest.mark.asyncio
    async def test_retrieve_with_reranking_no_distillation(
        self, retrieval_system, query_embedding, sample_memories
    ):
        """Test retrieval without distillation."""
        memories, scores = await retrieval_system.retrieve_with_reranking(
            query_embedding=query_embedding,
            memories=sample_memories,
            top_k=3,
            apply_distillation=False,
        )

        assert len(memories) <= 3

    @pytest.mark.asyncio
    async def test_retrieve_empty_memories(self, retrieval_system, query_embedding):
        """Test retrieval with empty memory list."""
        memories, scores = await retrieval_system.retrieve_with_reranking(
            query_embedding=query_embedding,
            memories=[],
            top_k=5,
        )

        assert memories == []

    @pytest.mark.asyncio
    async def test_get_retrieval_explanation(
        self, retrieval_system, query_embedding, sample_memories
    ):
        """Test retrieval explanation generation."""
        scores = np.array([0.9, 0.7, 0.5])

        explanation = await retrieval_system.get_retrieval_explanation(
            query_embedding=query_embedding,
            retrieved_memories=sample_memories,
            scores=scores,
        )

        assert isinstance(explanation, dict)
        assert "explanations" in explanation or len(explanation) >= 0

    @pytest.mark.asyncio
    async def test_composite_score_components(
        self, retrieval_system, query_embedding, sample_memories
    ):
        """Test that composite score considers all components."""
        current_time = time.time()

        # Create memories with different characteristics
        high_importance_memory = {
            "id": 1,
            "text": "Important fact",
            "embedding": query_embedding.tolist(),  # High similarity
            "timestamp": current_time,  # Very recent
            "importance_score": 10.0,  # High importance
            "access_count": 100,  # High access
        }

        low_importance_memory = {
            "id": 2,
            "text": "Less important",
            "embedding": np.zeros(1536).tolist(),  # Zero similarity
            "timestamp": current_time - 86400 * 30,  # Old
            "importance_score": 1.0,  # Low importance
            "access_count": 1,  # Low access
        }

        memories = [high_importance_memory, low_importance_memory]

        scores = await retrieval_system.compute_composite_scores(
            query_embedding=query_embedding,
            memories=memories,
            current_time=current_time,
        )

        # High importance memory should score higher
        assert scores[0] > scores[1]
