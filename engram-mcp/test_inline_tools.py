"""Test with tools defined inline."""

from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastmcp import FastMCP, Context

class MockClient:
    def __init__(self):
        self.name = "mock"

@asynccontextmanager
async def lifespan(mcp: FastMCP) -> AsyncIterator[dict]:
    client = MockClient()
    yield {"client": client}

mcp = FastMCP(name="test-inline", lifespan=lifespan)

@mcp.tool
async def test_tool_1(text: str, ctx: Context) -> str:
    """First test tool."""
    client = ctx.lifespan_context.get("client")
    return f"Tool1: {text}, client: {client.name}"

@mcp.tool
async def test_tool_2(value: int, ctx: Context) -> str:
    """Second test tool."""
    return f"Tool2: {value}"

if __name__ == "__main__":
    mcp.run(show_banner=False)
