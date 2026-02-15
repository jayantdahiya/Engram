"""Unit tests for LOCOMO MCP benchmark helpers."""

from unittest.mock import AsyncMock

import pytest

from engram_mcp.locomo_benchmark import (
    extract_memory_id,
    llm_extractive_answer,
    parse_recall_output,
    simple_extractive_answer,
)


class TestLocomoBenchmarkHelpers:
    """Validate helper parsing logic used by LOCOMO benchmark runner."""

    def test_extract_memory_id_from_remember(self):
        output = "Operation: ADD | Memory ID: 55 | Memories affected: 1 | Processing time: 20ms"
        assert extract_memory_id(output) == 55

    def test_extract_memory_id_from_store(self):
        output = "Memory created with ID: 99"
        assert extract_memory_id(output) == 99

    def test_parse_recall_output(self):
        output = (
            "Found 2 memories (showing top 2):\n"
            "  [11] (importance: 0.7) Caroline said, \"I love hiking.\"\n"
            "  [12] (importance: 0.3) Mel said, \"I like coffee.\""
        )
        hits = parse_recall_output(output)
        assert len(hits) == 2
        assert hits[0].memory_id == 11
        assert "hiking" in hits[0].text

    def test_simple_extractive_answer_prefers_overlap(self):
        output = (
            "Found 2 memories (showing top 2):\n"
            "  [11] (importance: 0.2) Caroline said, \"I love hiking.\"\n"
            "  [12] (importance: 0.9) Mel said, \"I like coffee.\""
        )
        hits = parse_recall_output(output)
        answer = simple_extractive_answer("What does Caroline love?", hits)
        assert answer == "I love hiking."

    @pytest.mark.asyncio
    async def test_llm_extractive_answer(self):
        output = (
            "Found 1 memories (showing top 1):\n"
            "  [11] (importance: 0.5) Caroline went on 7 May 2023."
        )
        hits = parse_recall_output(output)
        mock_client = AsyncMock()
        mock_client.generate_answer = AsyncMock(return_value="7 May 2023")

        answer = await llm_extractive_answer(
            "When did Caroline go?",
            hits,
            mock_client,
        )
        assert answer == "7 May 2023"
