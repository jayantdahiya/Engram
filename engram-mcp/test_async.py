"""Test async tools with Context and lifespan."""

from contextlib import asynccontextmanager
from typing import AsyncIterator
from fastmcp import FastMCP, Context

@asynccontextmanager
async def lifespan(mcp: FastMCP) -> AsyncIterator[dict]:
    print("Lifespan starting", flush=True)
    yield {"test_value": "hello"}
    print("Lifespan ending", flush=True)

mcp = FastMCP(name="test-async", lifespan=lifespan)

@mcp.tool
async def async_test(text: str, ctx: Context) -> str:
    """An async test tool with context."""
    value = ctx.lifespan_context.get("test_value", "no value")
    return f"Echo: {text}, Context: {value}"

if __name__ == "__main__":
    mcp.run(show_banner=False)
