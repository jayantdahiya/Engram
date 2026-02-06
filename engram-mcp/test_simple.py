"""Minimal FastMCP server to test tool registration."""

from fastmcp import FastMCP

mcp = FastMCP(name="test")

@mcp.tool
def simple_test(text: str) -> str:
    """A simple test tool."""
    return f"Echo: {text}"

if __name__ == "__main__":
    mcp.run(show_banner=False)
