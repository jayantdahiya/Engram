"""FastMCP server for Engram persistent memory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastmcp import FastMCP

from engram_mcp.client import EngramClient
from engram_mcp.config import get_config

INSTRUCTIONS = (
    "Engram gives you persistent long-term memory across conversations. "
    "Use 'remember' to store important facts from the current conversation. "
    "Use 'recall' to search your memories before answering questions. "
    "Use 'answer' to recall and extract a concise answer from memory context. "
    "Use 'store_memory' for direct memory creation with explicit importance. "
    "Use 'forget' to delete outdated or incorrect memories. "
    "Use 'memory_stats' to see an overview of stored memories. "
    "Use 'check_health' to verify the backend is reachable."
)


@asynccontextmanager
async def lifespan(mcp: FastMCP) -> AsyncIterator[dict]:
    try:
        cfg = get_config()
        client = EngramClient(cfg["api_url"], cfg["username"], cfg["password"])
        await client.start()
        try:
            yield {"client": client}
        finally:
            await client.stop()
    except Exception as e:
        # If backend connection fails, yield a None client
        # Tools will need to handle this gracefully
        import sys
        print(f"Warning: Failed to connect to Engram backend: {e}", file=sys.stderr)
        yield {"client": None}


mcp = FastMCP(name="engram", instructions=INSTRUCTIONS, lifespan=lifespan)

# Register tools with the mcp instance
from engram_mcp.tools import register_tools  # noqa: E402
register_tools(mcp)


def main() -> None:
    # Suppress verbose logging for STDIO compatibility
    import logging
    import os
    
    # Set minimal logging - only CRITICAL errors to stderr
    logging.basicConfig(level=logging.CRITICAL)
    logging.getLogger("httpcore").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    logging.getLogger("mcp").setLevel(logging.CRITICAL)
    logging.getLogger("docket").setLevel(logging.CRITICAL)
    logging.getLogger("fakeredis").setLevel(logging.CRITICAL)
    logging.getLogger("fastmcp").setLevel(logging.CRITICAL)
    
    # Disable rich console output
    os.environ["TERM"] = "dumb"
    os.environ["NO_COLOR"] = "1"
    
    # Run without banner (critical for STDIO protocol compatibility)
    mcp.run(show_banner=False)


if __name__ == "__main__":
    main()
