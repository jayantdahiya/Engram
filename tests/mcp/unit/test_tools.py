"""Unit tests for MCP tool definitions."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestMCPTools:
    """Test cases for MCP tool functions."""

    @pytest.fixture
    def mock_client(self):
        """Create mock EngramClient."""
        client = AsyncMock()
        client.process_turn = AsyncMock()
        client.query_memories = AsyncMock()
        client.create_memory = AsyncMock()
        client.delete_memory = AsyncMock()
        client.get_stats = AsyncMock()
        client.health_check = AsyncMock()
        return client

    @pytest.fixture
    def mock_context(self, mock_client):
        """Create mock MCP context with client."""
        ctx = MagicMock()
        ctx.request_context = MagicMock()
        ctx.request_context.lifespan_context = {"client": mock_client}
        return ctx

    @pytest.fixture
    def mock_context_no_client(self):
        """Create mock MCP context without client."""
        ctx = MagicMock()
        ctx.request_context = MagicMock()
        ctx.request_context.lifespan_context = {"client": None}
        return ctx

    def test_get_client_success(self, mock_context, mock_client):
        """Test successful client retrieval from context."""
        from engram_mcp.tools import _get_client

        result = _get_client(mock_context)

        assert result == mock_client

    def test_get_client_no_context(self):
        """Test client retrieval with no context."""
        from engram_mcp.tools import _get_client

        ctx = MagicMock()
        ctx.request_context = None

        with pytest.raises(RuntimeError, match="no request context"):
            _get_client(ctx)

    def test_get_client_not_initialized(self, mock_context_no_client):
        """Test client retrieval when client is None."""
        from engram_mcp.tools import _get_client

        with pytest.raises(RuntimeError, match="not initialized"):
            _get_client(mock_context_no_client)

    @pytest.mark.asyncio
    async def test_remember_tool(self, mock_context, mock_client):
        """Test remember tool stores memory."""
        from fastmcp import FastMCP
        from engram_mcp.tools import register_tools

        mock_client.process_turn.return_value = {
            "operation_performed": "ADD",
            "memory_id": 1,
            "memories_affected": 1,
            "processing_time_ms": 50.0,
        }

        # Create MCP instance and register tools
        mcp = FastMCP(name="test")
        register_tools(mcp)

        # Get the remember tool
        remember_tool = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "remember":
                remember_tool = tool
                break

        assert remember_tool is not None

        # Call the tool function directly with mock context
        with patch("engram_mcp.tools._get_client", return_value=mock_client):
            result = await remember_tool.fn(
                user_message="I love hiking",
                ctx=mock_context,
            )

        assert "ADD" in result
        assert "Memory ID: 1" in result

    @pytest.mark.asyncio
    async def test_remember_tool_with_conversation_id(self, mock_context, mock_client):
        """Test remember tool with explicit conversation ID."""
        from fastmcp import FastMCP
        from engram_mcp.tools import register_tools

        mock_client.process_turn.return_value = {
            "operation_performed": "UPDATE",
            "memory_id": 2,
            "memories_affected": 1,
            "processing_time_ms": 30.0,
        }

        mcp = FastMCP(name="test")
        register_tools(mcp)

        remember_tool = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "remember":
                remember_tool = tool
                break

        with patch("engram_mcp.tools._get_client", return_value=mock_client):
            result = await remember_tool.fn(
                user_message="I now prefer vegan food",
                ctx=mock_context,
                conversation_id="conv-123",
            )

        assert "UPDATE" in result
        mock_client.process_turn.assert_called_with(
            "I now prefer vegan food", "conv-123"
        )

    @pytest.mark.asyncio
    async def test_recall_tool(self, mock_context, mock_client):
        """Test recall tool queries memories."""
        from fastmcp import FastMCP
        from engram_mcp.tools import register_tools

        mock_client.query_memories.return_value = {
            "memories": [
                {"id": 1, "text": "I love hiking", "importance_score": 7.0},
                {
                    "id": 2,
                    "text": "I enjoy outdoor activities",
                    "importance_score": 5.0,
                },
            ],
            "total_found": 2,
        }

        mcp = FastMCP(name="test")
        register_tools(mcp)

        recall_tool = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "recall":
                recall_tool = tool
                break

        with patch("engram_mcp.tools._get_client", return_value=mock_client):
            result = await recall_tool.fn(
                query="What outdoor activities do I like?",
                ctx=mock_context,
            )

        assert "Found 2 memories" in result
        assert "hiking" in result

    @pytest.mark.asyncio
    async def test_recall_tool_no_memories(self, mock_context, mock_client):
        """Test recall tool with no matching memories."""
        from fastmcp import FastMCP
        from engram_mcp.tools import register_tools

        mock_client.query_memories.return_value = {
            "memories": [],
            "total_found": 0,
        }

        mcp = FastMCP(name="test")
        register_tools(mcp)

        recall_tool = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "recall":
                recall_tool = tool
                break

        with patch("engram_mcp.tools._get_client", return_value=mock_client):
            result = await recall_tool.fn(
                query="Unknown topic",
                ctx=mock_context,
            )

        assert "No memories found" in result

    @pytest.mark.asyncio
    async def test_recall_tool_limits_top_k(self, mock_context, mock_client):
        """Test recall tool enforces max top_k of 20."""
        from fastmcp import FastMCP
        from engram_mcp.tools import register_tools

        mock_client.query_memories.return_value = {"memories": [], "total_found": 0}

        mcp = FastMCP(name="test")
        register_tools(mcp)

        recall_tool = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "recall":
                recall_tool = tool
                break

        with patch("engram_mcp.tools._get_client", return_value=mock_client):
            await recall_tool.fn(
                query="test",
                ctx=mock_context,
                top_k=100,  # Should be capped to 20
            )

        mock_client.query_memories.assert_called_with("test", 20)

    @pytest.mark.asyncio
    async def test_store_memory_tool(self, mock_context, mock_client):
        """Test store_memory tool creates memory directly."""
        from fastmcp import FastMCP
        from engram_mcp.tools import register_tools

        mock_client.create_memory.return_value = {
            "id": 5,
            "text": "Important fact",
        }

        mcp = FastMCP(name="test")
        register_tools(mcp)

        store_tool = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "store_memory":
                store_tool = tool
                break

        with patch("engram_mcp.tools._get_client", return_value=mock_client):
            result = await store_tool.fn(
                text="Important fact about user",
                ctx=mock_context,
                importance_score=8.0,
            )

        assert "Memory created with ID: 5" in result

    @pytest.mark.asyncio
    async def test_forget_tool(self, mock_context, mock_client):
        """Test forget tool deletes memory."""
        from fastmcp import FastMCP
        from engram_mcp.tools import register_tools

        mock_client.delete_memory.return_value = {
            "message": "Memory deleted successfully",
        }

        mcp = FastMCP(name="test")
        register_tools(mcp)

        forget_tool = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "forget":
                forget_tool = tool
                break

        with patch("engram_mcp.tools._get_client", return_value=mock_client):
            result = await forget_tool.fn(
                memory_id=1,
                ctx=mock_context,
            )

        assert "deleted" in result.lower()
        mock_client.delete_memory.assert_called_with(1)

    @pytest.mark.asyncio
    async def test_memory_stats_tool(self, mock_context, mock_client):
        """Test memory_stats tool retrieves statistics."""
        from fastmcp import FastMCP
        from engram_mcp.tools import register_tools

        mock_client.get_stats.return_value = {
            "total_memories": 42,
            "average_importance": 6.5,
            "most_accessed_memories": [
                {"id": 1, "text": "Frequent memory", "access_count": 10}
            ],
            "recent_memories": [{"id": 2, "text": "Recent memory"}],
        }

        mcp = FastMCP(name="test")
        register_tools(mcp)

        stats_tool = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "memory_stats":
                stats_tool = tool
                break

        with patch("engram_mcp.tools._get_client", return_value=mock_client):
            result = await stats_tool.fn(ctx=mock_context)

        assert "Total memories: 42" in result
        assert "Average importance: 6.5" in result

    @pytest.mark.asyncio
    async def test_check_health_tool(self, mock_context, mock_client):
        """Test check_health tool verifies backend."""
        from fastmcp import FastMCP
        from engram_mcp.tools import register_tools

        mock_client.health_check.return_value = {
            "status": "healthy",
            "version": "1.0.0",
        }

        mcp = FastMCP(name="test")
        register_tools(mcp)

        health_tool = None
        for tool in mcp._tool_manager._tools.values():
            if tool.name == "check_health":
                health_tool = tool
                break

        with patch("engram_mcp.tools._get_client", return_value=mock_client):
            result = await health_tool.fn(ctx=mock_context)

        assert "Status: healthy" in result
        assert "Version: 1.0.0" in result
