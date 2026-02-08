"""MCP tool definitions for Engram memory operations."""

from __future__ import annotations

from typing import Any

from fastmcp import Context, FastMCP

from engram_mcp.client import EngramClient


def _get_client(ctx: Context) -> EngramClient:
    req_ctx = ctx.request_context
    if req_ctx is None:
        raise RuntimeError("Engram client not initialized (no request context)")
    client = req_ctx.lifespan_context.get("client")
    if client is None:
        raise RuntimeError("Engram client not initialized")
    return client


def register_tools(mcp_instance: FastMCP) -> None:
    """Register all Engram tools with the MCP instance."""
    
    @mcp_instance.tool
    async def remember(
        user_message: str,
        ctx: Context,
        conversation_id: str | None = None,
    ) -> str:
        """Store a memory from conversation. The system auto-classifies it as ADD, UPDATE, or CONSOLIDATE."""
        # Pass None to create memories without a conversation context
        client = _get_client(ctx)
        result = await client.process_turn(user_message, conversation_id)
        op = result.get("operation_performed", "unknown")
        mid = result.get("memory_id")
        affected = result.get("memories_affected", 0)
        ms = result.get("processing_time_ms", 0)
        parts = [f"Operation: {op}"]
        if mid is not None:
            parts.append(f"Memory ID: {mid}")
        parts.append(f"Memories affected: {affected}")
        parts.append(f"Processing time: {ms:.0f}ms")
        return " | ".join(parts)
    
    @mcp_instance.tool
    async def recall(
        query: str,
        ctx: Context,
        top_k: int | None = None,
    ) -> str:
        """Search memories using semantic similarity (ACAN retrieval). Returns the most relevant memories."""
        if top_k is None:
            top_k = 5
        if top_k > 20:
            top_k = 20
        client = _get_client(ctx)
        result = await client.query_memories(query, top_k)
        memories = result.get("memories", [])
        total = result.get("total_found", 0)
        if not memories:
            return "No memories found."
        lines = [f"Found {total} memories (showing top {len(memories)}):"]
        for mem in memories:
            mid = mem.get("id")
            text = mem.get("text", "")
            score = mem.get("importance_score", 0)
            lines.append(f"  [{mid}] (importance: {score}) {text}")
        return "\n".join(lines)
    
    @mcp_instance.tool
    async def store_memory(
        text: str,
        ctx: Context,
        importance_score: float | None = None,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Directly create a memory with explicit text, importance (0-10), and optional metadata."""
        if importance_score is None:
            importance_score = 5.0
        client = _get_client(ctx)
        result = await client.create_memory(text, importance_score, conversation_id, metadata)
        mid = result.get("id")
        return f"Memory created with ID: {mid}"
    
    @mcp_instance.tool
    async def forget(memory_id: int, ctx: Context) -> str:
        """Delete a specific memory by its ID."""
        client = _get_client(ctx)
        result = await client.delete_memory(memory_id)
        return result.get("message", "Memory deleted.")
    
    @mcp_instance.tool
    async def memory_stats(ctx: Context) -> str:
        """Get an overview of stored memories: total count, most accessed, and recent memories."""
        client = _get_client(ctx)
        result = await client.get_stats()
        total = result.get("total_memories", 0)
        avg_imp = result.get("average_importance", 0)
        lines = [f"Total memories: {total}", f"Average importance: {avg_imp:.1f}"]
        most_accessed = result.get("most_accessed_memories", [])
        if most_accessed:
            lines.append("Most accessed:")
            for mem in most_accessed:
                lines.append(f"  [{mem.get('id')}] ({mem.get('access_count', 0)} accesses) {mem.get('text', '')}")
        recent = result.get("recent_memories", [])
        if recent:
            lines.append("Recent:")
            for mem in recent:
                lines.append(f"  [{mem.get('id')}] {mem.get('text', '')}")
        return "\n".join(lines)
    
    @mcp_instance.tool
    async def check_health(ctx: Context) -> str:
        """Check if the Engram backend is reachable and healthy."""
        client = _get_client(ctx)
        result = await client.health_check()
        status = result.get("status", "unknown")
        version = result.get("version", "?")
        return f"Status: {status} | Version: {version}"
