#!/bin/bash
# Wrapper script to start the engram MCP server
# This avoids the uv runtime issues on macOS

cd "$(dirname "$0")"

# Debug: Log environment variables
echo "=== Engram MCP Starting ===" >> /tmp/engram-mcp-debug.log
echo "ENGRAM_API_URL=$ENGRAM_API_URL" >> /tmp/engram-mcp-debug.log
echo "ENGRAM_USERNAME=$ENGRAM_USERNAME" >> /tmp/engram-mcp-debug.log
echo "ENGRAM_PASSWORD set: ${ENGRAM_PASSWORD:+yes}" >> /tmp/engram-mcp-debug.log
date >> /tmp/engram-mcp-debug.log

# Use python3 directly from the venv (avoiding activation issues with paths containing spaces)
VENV_PYTHON="$(dirname "$0")/.venv/bin/python3"

# Suppress terminal features that might interfere with STDIO protocol
export TERM=dumb
export NO_COLOR=1

# Run the server (stderr goes to log file for debugging)
exec "$VENV_PYTHON" -m engram_mcp.server 2>>/tmp/engram-mcp-error.log
