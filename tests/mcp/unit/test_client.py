"""Unit tests for EngramClient."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestEngramClient:
    """Test cases for EngramClient HTTP client."""

    @pytest.fixture
    def client_config(self):
        """Client configuration."""
        return {
            "api_url": "http://localhost:8000",
            "username": "testuser",
            "password": "testpass123",
        }

    @pytest.fixture
    def mock_http_client(self):
        """Create mock httpx AsyncClient."""
        client = AsyncMock()
        client.post = AsyncMock()
        client.get = AsyncMock()
        client.delete = AsyncMock()
        client.aclose = AsyncMock()
        return client

    @pytest.fixture
    def sample_login_response(self):
        """Sample login response."""
        return {
            "access_token": "test_token_123",
            "token_type": "bearer",
            "expires_in": 1800,
        }

    @pytest.fixture
    def sample_user_response(self):
        """Sample user response."""
        return {
            "id": "user-uuid-123",
            "username": "testuser",
            "email": "test@example.com",
        }

    @pytest.mark.asyncio
    async def test_client_start(
        self,
        client_config,
        mock_http_client,
        sample_login_response,
        sample_user_response,
    ):
        """Test client initialization and authentication."""
        from engram_mcp.client import EngramClient

        with patch(
            "engram_mcp.client.httpx.AsyncClient", return_value=mock_http_client
        ):
            # Mock login response
            mock_http_client.post.return_value = MagicMock(
                json=lambda: sample_login_response, raise_for_status=MagicMock()
            )
            # Mock user info response
            mock_http_client.get.return_value = MagicMock(
                json=lambda: sample_user_response, raise_for_status=MagicMock()
            )

            client = EngramClient(**client_config)
            await client.start()

            assert client._token == "test_token_123"
            assert client._user_id == "user-uuid-123"

    @pytest.mark.asyncio
    async def test_client_stop(self, client_config, mock_http_client):
        """Test client cleanup."""
        from engram_mcp.client import EngramClient

        client = EngramClient(**client_config)
        client._http = mock_http_client

        await client.stop()

        mock_http_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_login_success(
        self, client_config, mock_http_client, sample_login_response
    ):
        """Test successful login."""
        from engram_mcp.client import EngramClient

        mock_http_client.post.return_value = MagicMock(
            json=lambda: sample_login_response, raise_for_status=MagicMock()
        )

        client = EngramClient(**client_config)
        client._http = mock_http_client

        await client._login()

        assert client._token == "test_token_123"
        assert client._token_expires_at > time.time()

    @pytest.mark.asyncio
    async def test_register_success(self, client_config, mock_http_client):
        """Test successful registration."""
        from engram_mcp.client import EngramClient

        mock_http_client.post.return_value = MagicMock(
            status_code=200, raise_for_status=MagicMock()
        )

        client = EngramClient(**client_config)
        client._http = mock_http_client

        await client._register()

        mock_http_client.post.assert_called_with(
            "/auth/register",
            json={
                "username": "testuser",
                "email": "testuser@engram.local",
                "password": "testpass123",
            },
        )

    @pytest.mark.asyncio
    async def test_register_already_exists(self, client_config, mock_http_client):
        """Test registration when user already exists."""
        from engram_mcp.client import EngramClient

        mock_http_client.post.return_value = MagicMock(
            status_code=400,  # Already registered
            raise_for_status=MagicMock(),
        )

        client = EngramClient(**client_config)
        client._http = mock_http_client

        # Should not raise
        await client._register()

    @pytest.mark.asyncio
    async def test_ensure_token_valid(self, client_config, mock_http_client):
        """Test token check when token is still valid."""
        from engram_mcp.client import EngramClient

        client = EngramClient(**client_config)
        client._http = mock_http_client
        client._token = "valid_token"
        client._token_expires_at = time.time() + 3600  # Expires in 1 hour

        await client._ensure_token()

        # Should not make any API calls
        mock_http_client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_token_refresh(
        self, client_config, mock_http_client, sample_login_response
    ):
        """Test token refresh when token is about to expire."""
        from engram_mcp.client import EngramClient

        mock_http_client.post.return_value = MagicMock(
            json=lambda: sample_login_response, raise_for_status=MagicMock()
        )

        client = EngramClient(**client_config)
        client._http = mock_http_client
        client._token = "old_token"
        client._token_expires_at = (
            time.time() + 30
        )  # Expires in 30 seconds (within 60s buffer)

        await client._ensure_token()

        # Should have called refresh
        mock_http_client.post.assert_called()

    @pytest.mark.asyncio
    async def test_process_turn_success(self, client_config, mock_http_client):
        """Test processing a conversation turn."""
        from engram_mcp.client import EngramClient

        mock_response = {
            "turn_id": "turn-123",
            "operation_performed": "ADD",
            "memory_id": 1,
            "memories_affected": 1,
            "processing_time_ms": 50.0,
        }

        mock_http_client.post.return_value = MagicMock(
            json=lambda: mock_response, raise_for_status=MagicMock()
        )

        client = EngramClient(**client_config)
        client._http = mock_http_client
        client._token = "valid_token"
        client._token_expires_at = time.time() + 3600
        client._user_id = "user-123"

        result = await client.process_turn("I love pizza", "conv-123")

        assert result["operation_performed"] == "ADD"
        assert result["memory_id"] == 1

    @pytest.mark.asyncio
    async def test_query_memories_success(self, client_config, mock_http_client):
        """Test querying memories."""
        from engram_mcp.client import EngramClient

        mock_response = {
            "query": "What do I like?",
            "memories": [{"id": 1, "text": "I love pizza", "importance_score": 5.0}],
            "total_found": 1,
            "processing_time_ms": 25.0,
        }

        mock_http_client.post.return_value = MagicMock(
            json=lambda: mock_response, raise_for_status=MagicMock()
        )

        client = EngramClient(**client_config)
        client._http = mock_http_client
        client._token = "valid_token"
        client._token_expires_at = time.time() + 3600
        client._user_id = "user-123"

        result = await client.query_memories("What do I like?", top_k=5)

        assert result["total_found"] == 1
        assert len(result["memories"]) == 1

    @pytest.mark.asyncio
    async def test_create_memory_success(self, client_config, mock_http_client):
        """Test creating a memory directly."""
        from engram_mcp.client import EngramClient

        mock_response = {
            "id": 1,
            "text": "I am a software engineer",
            "user_id": "user-123",
            "importance_score": 7.0,
        }

        mock_http_client.post.return_value = MagicMock(
            json=lambda: mock_response, raise_for_status=MagicMock()
        )

        client = EngramClient(**client_config)
        client._http = mock_http_client
        client._token = "valid_token"
        client._token_expires_at = time.time() + 3600
        client._user_id = "user-123"

        result = await client.create_memory(
            text="I am a software engineer",
            importance_score=7.0,
        )

        assert result["id"] == 1
        assert result["text"] == "I am a software engineer"

    @pytest.mark.asyncio
    async def test_delete_memory_success(self, client_config, mock_http_client):
        """Test deleting a memory."""
        from engram_mcp.client import EngramClient

        mock_response = {"message": "Memory deleted successfully"}

        mock_http_client.delete.return_value = MagicMock(
            json=lambda: mock_response, raise_for_status=MagicMock()
        )

        client = EngramClient(**client_config)
        client._http = mock_http_client
        client._token = "valid_token"
        client._token_expires_at = time.time() + 3600
        client._user_id = "user-123"

        result = await client.delete_memory(1)

        assert "deleted" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_get_stats_success(self, client_config, mock_http_client):
        """Test getting memory statistics."""
        from engram_mcp.client import EngramClient

        mock_response = {
            "total_memories": 100,
            "average_importance": 6.5,
            "most_accessed_memories": [],
            "recent_memories": [],
        }

        mock_http_client.get.return_value = MagicMock(
            json=lambda: mock_response, raise_for_status=MagicMock()
        )

        client = EngramClient(**client_config)
        client._http = mock_http_client
        client._token = "valid_token"
        client._token_expires_at = time.time() + 3600
        client._user_id = "user-123"

        result = await client.get_stats()

        assert result["total_memories"] == 100

    @pytest.mark.asyncio
    async def test_health_check_success(self, client_config, mock_http_client):
        """Test health check endpoint."""
        from engram_mcp.client import EngramClient

        mock_response = {
            "status": "healthy",
            "version": "1.0.0",
        }

        mock_http_client.get.return_value = MagicMock(
            json=lambda: mock_response, raise_for_status=MagicMock()
        )

        client = EngramClient(**client_config)
        client._http = mock_http_client
        # No token needed for health check

        result = await client.health_check()

        assert result["status"] == "healthy"

    def test_auth_headers(self, client_config):
        """Test auth headers generation."""
        from engram_mcp.client import EngramClient

        client = EngramClient(**client_config)
        client._token = "test_token_123"

        headers = client._auth_headers()

        assert headers["Authorization"] == "Bearer test_token_123"
