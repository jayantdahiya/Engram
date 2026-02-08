"""Integration tests for conversation endpoints."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestConversationRouter:
    """Integration test cases for conversation endpoints."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session

    @pytest.fixture
    def sample_conversation_response(self):
        """Sample conversation response data."""
        return {
            "id": "conv-123-uuid",
            "title": "Test Conversation",
            "user_id": "test-user-id",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "turn_count": 0,
            "metadata": {},
        }

    @pytest.fixture
    def sample_turn_response(self):
        """Sample conversation turn response data."""
        return {
            "id": "turn-123-uuid",
            "conversation_id": "conv-123-uuid",
            "user_message": "Hello",
            "assistant_response": "Hi there!",
            "turn_number": 1,
            "timestamp": datetime.utcnow(),
        }

    @pytest.mark.asyncio
    async def test_create_conversation_success(
        self, mock_db_session, test_conversation_data
    ):
        """Test successful conversation creation."""
        from api.routers.conversation import create_conversation
        from models.conversation import ConversationCreate

        # Mock database insert
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None  # No existing conversation
        mock_db_session.execute.return_value = mock_result

        conv_data = ConversationCreate(**test_conversation_data)

        with (
            patch("api.routers.conversation.logger"),
            patch("api.routers.conversation.uuid") as mock_uuid,
        ):
            mock_uuid.uuid4.return_value = MagicMock(__str__=lambda x: "conv-123-uuid")
            result = await create_conversation(
                conversation_data=conv_data,
                current_user="test-user-id",
                db_session=mock_db_session,
            )

        assert result.title == test_conversation_data["title"]

    @pytest.mark.asyncio
    async def test_get_conversation_success(
        self, mock_db_session, sample_conversation_response
    ):
        """Test successful conversation retrieval."""
        from api.routers.conversation import get_conversation

        # Mock conversation lookup
        mock_conv = MagicMock()
        for key, value in sample_conversation_response.items():
            setattr(mock_conv, key, value)

        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_conv
        mock_result.fetchall.return_value = []
        mock_db_session.execute.return_value = mock_result

        result = await get_conversation(
            conversation_id="conv-123-uuid",
            current_user="test-user-id",
            db_session=mock_db_session,
        )

        assert result.id == "conv-123-uuid"

    @pytest.mark.asyncio
    async def test_get_conversation_not_found(self, mock_db_session):
        """Test conversation not found."""
        from fastapi import HTTPException
        from api.routers.conversation import get_conversation

        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_db_session.execute.return_value = mock_result

        with pytest.raises(HTTPException) as exc_info:
            await get_conversation(
                conversation_id="nonexistent-conv",
                current_user="test-user-id",
                db_session=mock_db_session,
            )

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_update_conversation_success(
        self, mock_db_session, sample_conversation_response
    ):
        """Test successful conversation update."""
        from api.routers.conversation import update_conversation
        from models.conversation import ConversationUpdate

        # Mock existing conversation
        mock_conv = MagicMock()
        for key, value in sample_conversation_response.items():
            setattr(mock_conv, key, value)

        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_conv
        mock_db_session.execute.return_value = mock_result

        update_data = ConversationUpdate(title="Updated Title")

        with patch("api.routers.conversation.logger"):
            result = await update_conversation(
                conversation_id="conv-123-uuid",
                conversation_update=update_data,
                current_user="test-user-id",
                db_session=mock_db_session,
            )

        assert result is not None

    @pytest.mark.asyncio
    async def test_update_conversation_not_found(self, mock_db_session):
        """Test update conversation not found."""
        from fastapi import HTTPException
        from api.routers.conversation import update_conversation
        from models.conversation import ConversationUpdate

        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_db_session.execute.return_value = mock_result

        update_data = ConversationUpdate(title="Updated Title")

        with pytest.raises(HTTPException) as exc_info:
            await update_conversation(
                conversation_id="nonexistent-conv",
                conversation_update=update_data,
                current_user="test-user-id",
                db_session=mock_db_session,
            )

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_conversation_success(
        self, mock_db_session, sample_conversation_response
    ):
        """Test successful conversation deletion."""
        from api.routers.conversation import delete_conversation

        # Mock existing conversation
        mock_conv = MagicMock()
        mock_conv.id = "conv-123-uuid"
        mock_conv.user_id = "test-user-id"

        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_conv
        mock_db_session.execute.return_value = mock_result

        with patch("api.routers.conversation.logger"):
            result = await delete_conversation(
                conversation_id="conv-123-uuid",
                current_user="test-user-id",
                db_session=mock_db_session,
            )

        assert "deleted" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_delete_conversation_not_found(self, mock_db_session):
        """Test delete conversation not found."""
        from fastapi import HTTPException
        from api.routers.conversation import delete_conversation

        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_db_session.execute.return_value = mock_result

        with pytest.raises(HTTPException) as exc_info:
            await delete_conversation(
                conversation_id="nonexistent-conv",
                current_user="test-user-id",
                db_session=mock_db_session,
            )

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_list_conversations_success(
        self, mock_db_session, sample_conversation_response
    ):
        """Test listing conversations with pagination."""
        from api.routers.conversation import list_conversations

        # Mock conversation list
        mock_conv = MagicMock()
        for key, value in sample_conversation_response.items():
            setattr(mock_conv, key, value)

        # Mock count query
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 1

        # Mock list query
        mock_list_result = MagicMock()
        mock_list_result.fetchall.return_value = [mock_conv]

        mock_db_session.execute.side_effect = [mock_count_result, mock_list_result]

        result = await list_conversations(
            current_user="test-user-id",
            db_session=mock_db_session,
            page=1,
            page_size=20,
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_list_conversations_with_search(self, mock_db_session):
        """Test listing conversations with search filter."""
        from api.routers.conversation import list_conversations

        # Mock empty results
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 0

        mock_list_result = MagicMock()
        mock_list_result.fetchall.return_value = []

        mock_db_session.execute.side_effect = [mock_count_result, mock_list_result]

        result = await list_conversations(
            current_user="test-user-id",
            db_session=mock_db_session,
            page=1,
            page_size=20,
            search="specific topic",
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_add_conversation_turn_success(
        self, mock_db_session, sample_conversation_response, sample_turn_response
    ):
        """Test adding a turn to a conversation."""
        from api.routers.conversation import add_conversation_turn
        from models.conversation import ConversationTurnCreate

        # Mock conversation lookup
        mock_conv = MagicMock()
        for key, value in sample_conversation_response.items():
            setattr(mock_conv, key, value)

        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_conv
        mock_db_session.execute.return_value = mock_result

        turn_data = ConversationTurnCreate(
            user_message="Hello",
            assistant_response="Hi there!",
            user_id="test-user-id",
            conversation_id="conv-123-uuid",
            turn_number=1,
        )

        with (
            patch("api.routers.conversation.logger"),
            patch("api.routers.conversation.uuid") as mock_uuid,
        ):
            mock_uuid.uuid4.return_value = MagicMock(__str__=lambda x: "turn-123-uuid")
            result = await add_conversation_turn(
                conversation_id="conv-123-uuid",
                turn_data=turn_data,
                current_user="test-user-id",
                db_session=mock_db_session,
            )

        assert result is not None

    @pytest.mark.asyncio
    async def test_add_conversation_turn_conversation_not_found(self, mock_db_session):
        """Test adding turn to non-existent conversation."""
        from fastapi import HTTPException
        from api.routers.conversation import add_conversation_turn
        from models.conversation import ConversationTurnCreate

        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_db_session.execute.return_value = mock_result

        turn_data = ConversationTurnCreate(
            user_message="Hello",
            user_id="test-user-id",
            conversation_id="nonexistent-conv",
            turn_number=1,
        )

        with pytest.raises(HTTPException) as exc_info:
            await add_conversation_turn(
                conversation_id="nonexistent-conv",
                turn_data=turn_data,
                current_user="test-user-id",
                db_session=mock_db_session,
            )

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_conversation_stats_success(self, mock_db_session):
        """Test getting conversation statistics."""
        from api.routers.conversation import get_conversation_stats

        # Mock stats query results
        mock_result = MagicMock()
        # First call: total conversations (scalar)
        # Second call: total turns (scalar)
        mock_result.scalar.side_effect = [10, 50]

        # Third call: recent conversations (fetchall)
        mock_result.fetchall.return_value = []

        mock_db_session.execute.return_value = mock_result
        # Let's verify if explicit property setting is needed.
        stats_mock = MagicMock()
        stats_mock.total_conversations = 10
        stats_mock.total_turns = 50
        stats_mock.average_turns_per_conversation = 5.0
        mock_result.fetchone.return_value = stats_mock
        mock_db_session.execute.return_value = mock_result

        with patch("api.routers.conversation.logger"):
            result = await get_conversation_stats(
                current_user="test-user-id",
                db_session=mock_db_session,
            )

        assert result is not None
