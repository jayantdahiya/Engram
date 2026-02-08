"""Integration tests for memory endpoints."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


class TestMemoryRouter:
    """Integration test cases for memory endpoints."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session

    @pytest.fixture
    def mock_memory_manager(self):
        """Create mock memory manager."""
        with patch("api.routers.memory.memory_manager") as mock:
            yield mock

    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service."""
        with patch("api.routers.memory.embedding_service") as mock:
            mock.get_embedding = AsyncMock(
                return_value=np.random.rand(1536).astype(np.float32)
            )
            mock.process_embedding_request = AsyncMock()
            yield mock

    @pytest.fixture
    def sample_memory_response(self):
        """Sample memory response data."""
        return {
            "id": 1,
            "text": "I am vegetarian",
            "user_id": "test-user-id",
            "conversation_id": "test-conversation",
            "timestamp": datetime.utcnow(),
            "access_count": 0,
            "importance_score": 5.0,
            "embedding_dimension": 1536,
        }

    @pytest.mark.asyncio
    async def test_create_memory_success(
        self, mock_db_session, mock_embedding_service, test_memory_data
    ):
        """Test successful memory creation."""
        from api.routers.memory import create_memory
        from models.memory import MemoryEntryCreate

        # Mock database insert
        mock_result = MagicMock()
        # First scalar call: memory count (0)
        # Second scalar call (via _add_memory -> RETURNING id): memory_id (1)
        mock_result.scalar.side_effect = [0, 1]
        mock_memory = MagicMock(
            id=1,
            text=test_memory_data["text"],
            user_id="test-user-id",
            conversation_id=test_memory_data["conversation_id"],
            timestamp=datetime.utcnow(),
            importance_score=0.0,
            access_count=0,
            metadata={},
            embedding=[0.1] * 1536,
        )
        mock_result.fetchone.return_value = mock_memory
        mock_db_session.execute.return_value = mock_result

        memory_data = MemoryEntryCreate(**test_memory_data)

        with patch("api.routers.memory.logger"):
            result = await create_memory(
                memory_data=memory_data,
                current_user="test-user-id",
                db_session=mock_db_session,
            )

        assert result.text == test_memory_data["text"]
        assert result.user_id == test_memory_data["user_id"]

    @pytest.mark.asyncio
    async def test_get_memory_success(self, mock_db_session, sample_memory_response):
        """Test successful memory retrieval."""
        from api.routers.memory import get_memory

        # Mock memory lookup
        mock_row = MagicMock()
        for key, value in sample_memory_response.items():
            setattr(mock_row, key, value)
        mock_row.embedding = [0.1] * 1536
        mock_row.metadata = {}  # Explicitly set metadata to dict

        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_row
        mock_db_session.execute.return_value = mock_result

        result = await get_memory(
            memory_id=1,
            current_user="test-user-id",
            db_session=mock_db_session,
        )

        assert result.id == 1
        assert result.text == sample_memory_response["text"]

    @pytest.mark.asyncio
    async def test_get_memory_not_found(self, mock_db_session):
        """Test memory not found."""
        from fastapi import HTTPException
        from api.routers.memory import get_memory

        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_db_session.execute.return_value = mock_result

        with pytest.raises(HTTPException) as exc_info:
            await get_memory(
                memory_id=99999,
                current_user="test-user-id",
                db_session=mock_db_session,
            )

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_update_memory_success(
        self, mock_db_session, mock_embedding_service, sample_memory_response
    ):
        """Test successful memory update."""
        from api.routers.memory import update_memory
        from models.memory import MemoryEntryUpdate

        # Mock existing memory
        mock_row = MagicMock()
        for key, value in sample_memory_response.items():
            setattr(mock_row, key, value)
        mock_row.embedding = [0.1] * 1536
        mock_row.metadata = {}

        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_row
        mock_db_session.execute.return_value = mock_result

        update_data = MemoryEntryUpdate(text="I am now vegan", importance_score=8.0)

        with patch("api.routers.memory.logger"):
            result = await update_memory(
                memory_id=1,
                memory_update=update_data,
                current_user="test-user-id",
                db_session=mock_db_session,
            )

        # Should return updated memory
        assert result is not None

    @pytest.mark.asyncio
    async def test_update_memory_not_found(self, mock_db_session):
        """Test update memory not found."""
        from fastapi import HTTPException
        from api.routers.memory import update_memory
        from models.memory import MemoryEntryUpdate

        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_db_session.execute.return_value = mock_result

        update_data = MemoryEntryUpdate(text="Updated text")

        with pytest.raises(HTTPException) as exc_info:
            await update_memory(
                memory_id=99999,
                memory_update=update_data,
                current_user="test-user-id",
                db_session=mock_db_session,
            )

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_memory_success(self, mock_db_session, sample_memory_response):
        """Test successful memory deletion."""
        from api.routers.memory import delete_memory

        # Mock existing memory
        mock_row = MagicMock()
        mock_row.id = 1
        mock_row.user_id = "test-user-id"

        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_row
        mock_db_session.execute.return_value = mock_result

        with patch("api.routers.memory.logger"):
            result = await delete_memory(
                memory_id=1,
                current_user="test-user-id",
                db_session=mock_db_session,
            )

        assert "deleted" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_delete_memory_not_found(self, mock_db_session):
        """Test delete memory not found."""
        from fastapi import HTTPException
        from api.routers.memory import delete_memory

        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_result.rowcount = 0
        mock_db_session.execute.return_value = mock_result

        with pytest.raises(HTTPException) as exc_info:
            await delete_memory(
                memory_id=99999,
                current_user="test-user-id",
                db_session=mock_db_session,
            )

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_query_memories_success(self, mock_db_session, mock_memory_manager):
        """Test successful memory query."""
        from api.routers.memory import query_memories
        from models.memory import MemoryQuery

        # Mock memory manager response
        mock_memory_manager.retrieve_memories = AsyncMock(
            return_value=MagicMock(
                query="test query",
                memories=[],
                total_found=0,
                processing_time_ms=10.5,
            )
        )

        query = MemoryQuery(
            query="What are my preferences?",
            user_id="test-user-id",
            top_k=5,
        )

        result = await query_memories(
            query=query,
            current_user="test-user-id",
            db_session=mock_db_session,
        )

        mock_memory_manager.retrieve_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_conversation_turn_success(
        self, mock_db_session, mock_memory_manager
    ):
        """Test processing conversation turn."""
        from api.routers.memory import process_conversation_turn
        from models.memory import ConversationTurn

        # Mock memory manager response
        mock_memory_manager.process_conversation_turn = AsyncMock(
            return_value=MagicMock(
                turn_id="turn-123",
                operation_performed="ADD",
                memory_id=1,
                processing_time_ms=50.0,
                memories_affected=1,
            )
        )

        turn = ConversationTurn(
            user_message="I love pizza",
            user_id="test-user-id",
            conversation_id="test-conversation",
        )

        result = await process_conversation_turn(
            turn=turn,
            current_user="test-user-id",
            db_session=mock_db_session,
        )

        mock_memory_manager.process_conversation_turn.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_memories_success(self, mock_db_session, sample_memory_response):
        """Test listing memories with pagination."""
        from api.routers.memory import list_memories

        # Mock memory list
        mock_row = MagicMock()
        for key, value in sample_memory_response.items():
            setattr(mock_row, key, value)
        mock_row.embedding = [0.1] * 1536
        mock_row.metadata = {}

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        mock_db_session.execute.return_value = mock_result

        result = await list_memories(
            current_user="test-user-id",
            db_session=mock_db_session,
            limit=50,
            offset=0,
        )

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_list_memories_with_conversation_filter(
        self, mock_db_session, sample_memory_response
    ):
        """Test listing memories filtered by conversation."""
        from api.routers.memory import list_memories

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_db_session.execute.return_value = mock_result

        result = await list_memories(
            current_user="test-user-id",
            db_session=mock_db_session,
            limit=50,
            offset=0,
            conversation_id="specific-conversation",
        )

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_memory_stats_success(self, mock_db_session):
        """Test getting memory statistics."""
        from api.routers.memory import get_memory_stats

        # Mock stats query results
        mock_result = MagicMock()
        mock_result.fetchone.return_value = MagicMock(
            total_memories=100,
            average_importance=6.5,
        )
        mock_result.fetchall.return_value = []
        mock_db_session.execute.return_value = mock_result

        with patch("api.routers.memory.logger"):
            result = await get_memory_stats(
                current_user="test-user-id",
                db_session=mock_db_session,
            )

        assert result is not None

    @pytest.mark.asyncio
    async def test_generate_embedding_success(self, mock_embedding_service):
        """Test embedding generation endpoint."""
        from api.routers.memory import generate_embedding
        from models.memory import EmbeddingRequest, EmbeddingResponse

        # Mock embedding service response
        mock_embedding_service.process_embedding_request.return_value = (
            EmbeddingResponse(
                text="test text",
                embedding=[0.1] * 1536,
                model="nomic-embed-text",
                dimension=1536,
                tokens_used=5,
            )
        )

        request = EmbeddingRequest(text="test text")

        result = await generate_embedding(
            request=request,
            current_user="test-user-id",
        )

        assert result.text == "test text"
        assert len(result.embedding) == 1536
