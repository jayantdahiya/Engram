"""Debug wrapper to test MCP server startup."""

import sys
import os
import logging

# Set up logging to stderr (not stdout, since stdout is for MCP protocol)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

logger.info("=== Engram MCP Server Debug Start ===")
logger.info(f"Python: {sys.version}")
logger.info(f"CWD: {os.getcwd()}")
logger.info(f"ENGRAM_API_URL: {os.environ.get('ENGRAM_API_URL', 'NOT SET')}")
logger.info(f"ENGRAM_USERNAME: {os.environ.get('ENGRAM_USERNAME', 'NOT SET')}")
logger.info(f"ENGRAM_PASSWORD: {'SET' if os.environ.get('ENGRAM_PASSWORD') else 'NOT SET'}")

try:
    logger.info("Importing engram_mcp...")
    from engram_mcp.server import mcp
    logger.info("Import successful!")
    
    logger.info("Starting MCP server...")
    mcp.run()
    logger.info("MCP server stopped normally")
except Exception as e:
    logger.exception(f"Failed to start server: {e}")
    sys.exit(1)
