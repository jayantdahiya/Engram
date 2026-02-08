"""Integration tests for MCP server."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestMCPServer:
    """Integration test cases for MCP server."""

    @pytest.fixture
    def mock_client(self):
        """Create mock EngramClient."""
        client = AsyncMock()
        client.start = AsyncMock()
        client.stop = AsyncMock()
        client.process_turn = AsyncMock()
        client.query_memories = AsyncMock()
        client.get_stats = AsyncMock()
        client.health_check = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_lifespan_success(self, mock_client):
        """Test server lifespan context manager success."""
        from fastmcp import FastMCP
        from engram_mcp.server import lifespan

        mcp = FastMCP(name="test")

        with (
            patch("engram_mcp.server.get_config") as mock_config,
            patch("engram_mcp.server.EngramClient", return_value=mock_client),
        ):
            mock_config.return_value = {
                "api_url": "http://localhost:8000",
                "username": "testuser",
                "password": "testpass",
            }

            async with lifespan(mcp) as context:
                assert context["client"] == mock_client
                mock_client.start.assert_called_once()

            mock_client.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_connection_failure(self):
        """Test server handles connection failure gracefully."""
        from fastmcp import FastMCP
        from engram_mcp.server import lifespan

        mcp = FastMCP(name="test")

        with (
            patch("engram_mcp.server.get_config") as mock_config,
            patch("engram_mcp.server.EngramClient") as mock_client_class,
        ):
            mock_config.return_value = {
                "api_url": "http://localhost:8000",
                "username": "testuser",
                "password": "testpass",
            }
            mock_client_instance = AsyncMock()
            mock_client_instance.start.side_effect = Exception("Connection refused")
            mock_client_class.return_value = mock_client_instance

            # Should not raise, yields None client
            async with lifespan(mcp) as context:
                assert context["client"] is None

    def test_mcp_instance_creation(self):
        """Test MCP instance is created correctly."""
        from engram_mcp.server import mcp

        assert mcp.name == "engram"

    def test_tools_registered(self):
        """Test all tools are registered with MCP."""
        from engram_mcp.server import mcp

        tool_names = [tool.name for tool in mcp._tool_manager._tools.values()]

        expected_tools = [
            "remember",
            "recall",
            "store_memory",
            "forget",
            "memory_stats",
            "check_health",
        ]

        for expected in expected_tools:
            assert expected in tool_names, f"Tool '{expected}' not registered"

    def test_instructions_defined(self):
        """Test MCP instructions are defined."""
        from engram_mcp.server import INSTRUCTIONS

        assert "remember" in INSTRUCTIONS.lower()
        assert "recall" in INSTRUCTIONS.lower()
        assert "memory" in INSTRUCTIONS.lower()

    @pytest.mark.asyncio
    async def test_end_to_end_remember_recall_flow(self, mock_client):
        """Test complete remember-recall flow."""
        from fastmcp import FastMCP
        from engram_mcp.tools import register_tools

        # Setup mock responses
        mock_client.process_turn.return_value = {
            "operation_performed": "ADD",
            "memory_id": 1,
            "memories_affected": 1,
            "processing_time_ms": 50.0,
        }
        mock_client.query_memories.return_value = {
            "memories": [{"id": 1, "text": "I love pizza", "importance_score": 5.0}],
            "total_found": 1,
        }

        mcp = FastMCP(name="test")
        register_tools(mcp)

        # Create mock context
        ctx = MagicMock()
        ctx.request_context = MagicMock()
        ctx.request_context.lifespan_context = {"client": mock_client}

        # Find tools
        remember_tool = None
        recall_tool = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "remember":
                remember_tool = tool
            elif tool.name == "recall":
                recall_tool = tool

        # Step 1: Remember something
        with patch("engram_mcp.tools._get_client", return_value=mock_client):
            remember_result = await remember_tool.fn(
                user_message="I love pizza",
                ctx=ctx,
            )
            assert "ADD" in remember_result

        # Step 2: Recall it
        with patch("engram_mcp.tools._get_client", return_value=mock_client):
            recall_result = await recall_tool.fn(
                query="What food do I like?",
                ctx=ctx,
            )
            assert "pizza" in recall_result

    @pytest.mark.asyncio
    async def test_end_to_end_store_forget_flow(self, mock_client):
        """Test complete store-forget flow."""
        from fastmcp import FastMCP
        from engram_mcp.tools import register_tools

        # Setup mock responses
        mock_client.create_memory.return_value = {"id": 10}
        mock_client.delete_memory.return_value = {"message": "Memory deleted"}

        mcp = FastMCP(name="test")
        register_tools(mcp)

        ctx = MagicMock()
        ctx.request_context = MagicMock()
        ctx.request_context.lifespan_context = {"client": mock_client}

        store_tool = None
        forget_tool = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "store_memory":
                store_tool = tool
            elif tool.name == "forget":
                forget_tool = tool

        # Step 1: Store a memory
        with patch("engram_mcp.tools._get_client", return_value=mock_client):
            store_result = await store_tool.fn(
                text="Temporary note",
                ctx=ctx,
                importance_score=3.0,
            )
            assert "10" in store_result

        # Step 2: Forget it
        with patch("engram_mcp.tools._get_client", return_value=mock_client):
            forget_result = await forget_tool.fn(
                memory_id=10,
                ctx=ctx,
            )
            assert "deleted" in forget_result.lower()

    @pytest.mark.asyncio
    async def test_tools_handle_client_errors(self, mock_client):
        """Test tools handle client errors gracefully."""
        from fastmcp import FastMCP
        from engram_mcp.tools import register_tools
        import httpx

        mock_client.process_turn.side_effect = httpx.HTTPStatusError(
            "Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )

        mcp = FastMCP(name="test")
        register_tools(mcp)

        ctx = MagicMock()
        ctx.request_context = MagicMock()
        ctx.request_context.lifespan_context = {"client": mock_client}

        remember_tool = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "remember":
                remember_tool = tool
                break

        with patch("engram_mcp.tools._get_client", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                await remember_tool.fn(
                    user_message="This should fail",
                    ctx=ctx,
                )
