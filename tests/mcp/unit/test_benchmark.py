"""Unit tests for MCP benchmark helpers."""

from pathlib import Path

from engram_mcp.benchmark import (
    PhaseMetrics,
    extract_memory_id,
    parse_total_memories,
    percentile,
    write_phase_csv,
)


class TestBenchmarkHelpers:
    """Test helper functions used by benchmark runner."""

    def test_extract_memory_id_remember(self):
        result = "Operation: ADD | Memory ID: 42 | Memories affected: 1 | Processing time: 5ms"
        assert extract_memory_id(result) == 42

    def test_extract_memory_id_store(self):
        result = "Memory created with ID: 108"
        assert extract_memory_id(result) == 108

    def test_extract_memory_id_not_found(self):
        assert extract_memory_id("No memory id here") is None

    def test_parse_total_memories(self):
        output = "Total memories: 91\nAverage importance: 0.5"
        assert parse_total_memories(output) == 91

    def test_percentile_linear_interpolation(self):
        values = [10.0, 20.0, 30.0, 40.0]
        assert percentile(values, 50) == 25.0
        assert percentile(values, 95) == 38.5

    def test_write_phase_csv(self, tmp_path: Path):
        csv_path = tmp_path / "phases.csv"
        phases = [
            PhaseMetrics(
                phase="remember",
                tool="remember",
                total_calls=30,
                success_calls=30,
                failed_calls=0,
                duration_ms=1000.0,
                throughput_ops_s=30.0,
                min_latency_ms=10.0,
                p50_latency_ms=20.0,
                p95_latency_ms=35.0,
                max_latency_ms=40.0,
            )
        ]

        write_phase_csv(csv_path, phases)

        assert csv_path.exists()
        content = csv_path.read_text(encoding="utf-8")
        assert "phase,tool,total_calls" in content
        assert "remember,remember,30,30,0" in content
