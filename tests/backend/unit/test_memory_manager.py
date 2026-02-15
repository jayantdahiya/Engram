"""Unit tests for memory manager"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from models.memory import ConversationTurn, MemoryQuery
from services.memory_manager import MemoryManager


class TestMemoryManager:
    """Test cases for MemoryManager"""

    @pytest.fixture
    def memory_manager(self):
        """Create memory manager instance"""
        return MemoryManager()

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session"""
        session = AsyncMock()
        # db_session.execute() returns a result whose .scalar() gives an int
        mock_result = Mock()
        mock_result.scalar = Mock(return_value=0)
        session.execute = AsyncMock(return_value=mock_result)
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session

    @pytest.fixture
    def sample_embedding(self):
        """Create sample embedding"""
        return np.random.rand(1536).astype(np.float32)

    @pytest.fixture
    def sample_turn(self):
        """Create sample conversation turn"""
        return ConversationTurn(
            user_message="I am vegetarian and avoid dairy",
            user_id="test-user",
            conversation_id="test-conversation",
        )

    @pytest.mark.asyncio
    async def test_process_conversation_turn_add_operation(
        self, memory_manager, mock_db_session, sample_turn
    ):
        """Test processing conversation turn with ADD operation"""

        with (
            patch("services.memory_manager.embedding_service") as mock_embedding,
            patch("services.memory_manager.llm_service"),
            patch.object(memory_manager, "_get_user_memories", return_value=[]),
            patch.object(memory_manager, "_add_memory", return_value=1),
            patch.object(memory_manager, "_extract_and_store_entities"),
        ):
            mock_embedding.get_embedding = AsyncMock(return_value=np.random.rand(1536))

            result = await memory_manager.process_conversation_turn(
                sample_turn, mock_db_session
            )

            assert result.operation_performed == "ADD"
            assert result.memory_id == 1
            assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_process_conversation_turn_update_operation(
        self, memory_manager, mock_db_session, sample_turn
    ):
        """Test processing conversation turn with UPDATE operation"""

        # Mock existing memory with embedding + metadata for the cosine pre-check
        mock_memory = Mock()
        mock_memory.text = "I am vegetarian"
        mock_memory.id = 1
        mock_memory.embedding = np.random.rand(1536).tolist()
        mock_memory.metadata = "{}"

        with (
            patch("services.memory_manager.embedding_service") as mock_embedding,
            patch("services.memory_manager.llm_service") as mock_llm,
            patch.object(
                memory_manager, "_get_user_memories", return_value=[mock_memory]
            ),
            patch.object(memory_manager, "_update_memory"),
            patch.object(memory_manager, "_extract_and_store_entities"),
        ):
            mock_embedding.get_embedding = AsyncMock(return_value=np.random.rand(1536))
            # Return 0.7 so it falls below the 0.85 threshold and goes to LLM classification
            mock_embedding.calculate_similarity = AsyncMock(return_value=0.7)
            mock_llm.classify_memory_operation = AsyncMock(
                return_value={
                    "operation": "UPDATE",
                    "related_memory_indices": [0],
                }
            )

            result = await memory_manager.process_conversation_turn(
                sample_turn, mock_db_session
            )

            assert result.operation_performed == "UPDATE"
            assert result.memory_id == 1

    @pytest.mark.asyncio
    async def test_retrieve_memories(self, memory_manager, mock_db_session):
        """Test memory retrieval"""

        # Mock user memories
        mock_memory = Mock()
        mock_memory.embedding = np.random.rand(1536).tolist()
        mock_memory.timestamp = datetime.now()
        mock_memory.importance_score = 0.5
        mock_memory.access_count = 5
        mock_memory.text = "Sample memory text"
        mock_memory.id = 1
        mock_memory.user_id = "test-user"
        mock_memory.conversation_id = "test-conversation"
        mock_memory.metadata = {}
        mock_memory.embedding_dimension = 1536

        with (
            patch.object(
                memory_manager, "_get_user_memories", return_value=[mock_memory]
            ),
            patch.object(memory_manager, "_update_access_count"),
            patch("services.memory_manager.embedding_service") as mock_embedding,
        ):
            mock_embedding.get_embedding = AsyncMock(return_value=np.random.rand(1536))
            mock_embedding.calculate_similarity = AsyncMock(return_value=0.8)

            query = MemoryQuery(
                query="What are my dietary preferences?", user_id="test-user", top_k=5
            )

            result = await memory_manager.retrieve_memories(query, mock_db_session)

            assert result.query == query.query
            assert result.total_found >= 0
            assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_retrieve_memories_empty_store(self, memory_manager, mock_db_session):
        """Test memory retrieval with empty memory store"""

        with (
            patch.object(memory_manager, "_get_user_memories", return_value=[]),
            patch("services.memory_manager.embedding_service") as mock_embedding,
        ):
            mock_embedding.get_embedding = AsyncMock(return_value=np.random.rand(1536))

            query = MemoryQuery(
                query="What are my dietary preferences?", user_id="test-user", top_k=5
            )

            result = await memory_manager.retrieve_memories(query, mock_db_session)

            assert result.query == query.query
            assert result.total_found == 0
            assert len(result.memories) == 0

    def test_similarity_threshold_configuration(self, memory_manager):
        """Test similarity threshold configuration"""
        assert memory_manager.similarity_threshold == 0.75

        # Test with custom threshold
        custom_manager = MemoryManager()
        custom_manager.similarity_threshold = 0.8
        assert custom_manager.similarity_threshold == 0.8

    def test_max_memories_per_user_configuration(self, memory_manager):
        """Test max memories per user configuration"""
        assert memory_manager.max_memories_per_user == 10000

        # Test with custom limit
        custom_manager = MemoryManager()
        custom_manager.max_memories_per_user = 5000
        assert custom_manager.max_memories_per_user == 5000

    @pytest.mark.asyncio
    async def test_classify_operation_update_at_075_threshold(
        self, memory_manager, mock_db_session, sample_turn
    ):
        """Similarity >= 0.75 should trigger UPDATE without LLM classification."""

        mock_memory = Mock()
        mock_memory.text = "I am vegetarian"
        mock_memory.id = 1
        mock_memory.embedding = np.random.rand(1536).tolist()
        mock_memory.metadata = "{}"

        with (
            patch("services.memory_manager.embedding_service") as mock_embedding,
            patch("services.memory_manager.llm_service") as mock_llm,
            patch.object(
                memory_manager, "_get_user_memories", return_value=[mock_memory]
            ),
            patch.object(memory_manager, "_update_memory"),
            patch.object(memory_manager, "_extract_and_store_entities"),
        ):
            mock_embedding.get_embedding = AsyncMock(return_value=np.random.rand(1536))
            # Similarity of 0.78 is above the new 0.75 threshold
            mock_embedding.calculate_similarity = AsyncMock(return_value=0.78)

            result = await memory_manager.process_conversation_turn(
                sample_turn, mock_db_session
            )

            assert result.operation_performed == "UPDATE"
            assert result.memory_id == 1
            # LLM classifier should NOT have been called
            mock_llm.classify_memory_operation.assert_not_called()

    @pytest.mark.asyncio
    async def test_acan_retrieval_returns_empty_below_cosine_floor(
        self, memory_manager
    ):
        """Memories below cosine floor (0.3) should not be returned."""

        mock_memory = Mock()
        mock_memory.embedding = np.random.rand(1536).tolist()
        mock_memory.timestamp = datetime.now()
        mock_memory.importance_score = 10.0  # High importance
        mock_memory.access_count = 100  # High access count

        with patch("services.memory_manager.embedding_service") as mock_embedding:
            # Very low cosine similarity — completely unrelated query
            mock_embedding.calculate_similarity = AsyncMock(return_value=0.1)

            result = await memory_manager._acan_retrieval(
                query_embedding=np.random.rand(1536),
                memories=[mock_memory],
                top_k=5,
                similarity_threshold=0.7,
                scoring_profile="semantic",
            )

            # With cosine floor of 0.3, no memories should be returned
            assert result == []
