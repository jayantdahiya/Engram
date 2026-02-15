"""Repeatable MCP tool benchmark runner with CSV/JSON phase metrics output."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastmcp import FastMCP

from engram_mcp.client import EngramClient
from engram_mcp.config import get_config
from engram_mcp.tools import register_tools


_REMEMBER_ID_PATTERN = re.compile(r"Memory ID:\s*(\d+)")
_STORE_ID_PATTERN = re.compile(r"Memory created with ID:\s*(\d+)")
_TOTAL_MEMORIES_PATTERN = re.compile(r"Total memories:\s*(\d+)")


@dataclass
class ToolCallResult:
    """Result for one MCP tool call."""

    tool: str
    success: bool
    latency_ms: float
    memory_id: int | None
    output: str | None = None
    error: str | None = None


@dataclass
class PhaseMetrics:
    """Aggregated metrics for one benchmark phase."""

    phase: str
    tool: str
    total_calls: int
    success_calls: int
    failed_calls: int
    duration_ms: float
    throughput_ops_s: float
    min_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float


def extract_memory_id(result_text: str) -> int | None:
    """Extract created memory id from remember/store tool output strings."""
    for pattern in (_REMEMBER_ID_PATTERN, _STORE_ID_PATTERN):
        match = pattern.search(result_text)
        if match:
            return int(match.group(1))
    return None


def parse_total_memories(memory_stats_output: str) -> int | None:
    """Parse total memory count from memory_stats tool output."""
    match = _TOTAL_MEMORIES_PATTERN.search(memory_stats_output)
    if not match:
        return None
    return int(match.group(1))


def percentile(sorted_values: list[float], p: float) -> float:
    """Compute percentile p in [0, 100] with linear interpolation."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (len(sorted_values) - 1) * (p / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def write_phase_csv(path: Path, phase_metrics: list[PhaseMetrics]) -> None:
    """Write per-phase metrics to CSV."""
    fieldnames = [
        "phase",
        "tool",
        "total_calls",
        "success_calls",
        "failed_calls",
        "duration_ms",
        "throughput_ops_s",
        "min_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "max_latency_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for metrics in phase_metrics:
            writer.writerow(asdict(metrics))


class MCPBenchmarkRunner:
    """Run a phased benchmark through MCP tool calls."""

    def __init__(
        self,
        *,
        api_url: str,
        username: str,
        password: str,
        run_id: str,
        remember_ops: int,
        store_ops: int,
        recall_ops: int,
        concurrency: int,
        keep_data: bool,
    ) -> None:
        self.api_url = api_url
        self.username = username
        self.password = password
        self.run_id = run_id
        self.remember_ops = max(remember_ops, 0)
        self.store_ops = max(store_ops, 0)
        self.recall_ops = max(recall_ops, 0)
        self.concurrency = max(concurrency, 1)
        self.keep_data = keep_data

        self._client: EngramClient | None = None
        self._ctx: Any = None
        self._tools: dict[str, Any] = {}

    async def __aenter__(self) -> MCPBenchmarkRunner:
        client = EngramClient(
            api_url=self.api_url,
            username=self.username,
            password=self.password,
        )
        await client.start()
        self._client = client

        mcp = FastMCP(name="engram-benchmark")
        register_tools(mcp)
        self._tools = dict(mcp._tool_manager._tools)
        self._ctx = SimpleNamespace(
            request_context=SimpleNamespace(lifespan_context={"client": client})
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.stop()

    async def _call_tool(self, tool_name: str, payload: dict[str, Any]) -> ToolCallResult:
        tool = self._tools[tool_name]
        start = time.perf_counter()
        try:
            result = await tool.fn(ctx=self._ctx, **payload)
            latency_ms = (time.perf_counter() - start) * 1000.0
            output = result if isinstance(result, str) else str(result)
            memory_id = extract_memory_id(output)
            return ToolCallResult(
                tool=tool_name,
                success=True,
                latency_ms=latency_ms,
                memory_id=memory_id,
                output=output,
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return ToolCallResult(
                tool=tool_name,
                success=False,
                latency_ms=latency_ms,
                memory_id=None,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _run_phase(
        self,
        *,
        phase_name: str,
        tool_name: str,
        payloads: list[dict[str, Any]],
    ) -> tuple[PhaseMetrics, list[ToolCallResult]]:
        if not payloads:
            empty = PhaseMetrics(
                phase=phase_name,
                tool=tool_name,
                total_calls=0,
                success_calls=0,
                failed_calls=0,
                duration_ms=0.0,
                throughput_ops_s=0.0,
                min_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                max_latency_ms=0.0,
            )
            return empty, []

        semaphore = asyncio.Semaphore(self.concurrency)

        async def run_one(payload: dict[str, Any]) -> ToolCallResult:
            async with semaphore:
                return await self._call_tool(tool_name, payload)

        phase_start = time.perf_counter()
        results = await asyncio.gather(*(run_one(payload) for payload in payloads))
        duration_ms = (time.perf_counter() - phase_start) * 1000.0

        latencies = sorted(result.latency_ms for result in results)
        successes = sum(1 for result in results if result.success)
        failures = len(results) - successes
        throughput_ops_s = (
            (len(results) / (duration_ms / 1000.0))
            if duration_ms > 0
            else 0.0
        )
        metrics = PhaseMetrics(
            phase=phase_name,
            tool=tool_name,
            total_calls=len(results),
            success_calls=successes,
            failed_calls=failures,
            duration_ms=duration_ms,
            throughput_ops_s=throughput_ops_s,
            min_latency_ms=latencies[0],
            p50_latency_ms=percentile(latencies, 50),
            p95_latency_ms=percentile(latencies, 95),
            max_latency_ms=latencies[-1],
        )
        return metrics, results

    async def run(self) -> dict[str, Any]:
        """Execute benchmark phases and return structured results."""
        phase_metrics: list[PhaseMetrics] = []
        phase_errors: dict[str, list[str]] = {}
        created_memory_ids: set[int] = set()
        deleted_memory_ids: set[int] = set()

        started_at = datetime.now(UTC)
        started_ms = int(time.time() * 1000)

        health_result = await self._call_tool("check_health", {})
        baseline_stats_result = await self._call_tool("memory_stats", {})
        baseline_total = parse_total_memories(baseline_stats_result.output or "")

        conversation_id = self.run_id

        remember_payloads = [
            {
                "conversation_id": conversation_id,
                "user_message": (
                    f"{self.run_id} remember i{i:03d}: "
                    f"synthetic preference payload number {i}."
                ),
            }
            for i in range(self.remember_ops)
        ]
        remember_metrics, remember_results = await self._run_phase(
            phase_name="remember",
            tool_name="remember",
            payloads=remember_payloads,
        )
        phase_metrics.append(remember_metrics)
        phase_errors["remember"] = [
            result.error for result in remember_results if not result.success and result.error
        ]
        for result in remember_results:
            if result.success and result.memory_id is not None:
                created_memory_ids.add(result.memory_id)

        store_payloads = [
            {
                "conversation_id": conversation_id,
                "text": (
                    f"{self.run_id} store i{i:03d}: "
                    f"synthetic benchmark note number {i}."
                ),
                "importance_score": float(5 + (i % 5)),
                "metadata": {
                    "source": "mcp_benchmark",
                    "run_id": self.run_id,
                    "index": i,
                },
            }
            for i in range(self.store_ops)
        ]
        store_metrics, store_results = await self._run_phase(
            phase_name="store_memory",
            tool_name="store_memory",
            payloads=store_payloads,
        )
        phase_metrics.append(store_metrics)
        phase_errors["store_memory"] = [
            result.error for result in store_results if not result.success and result.error
        ]
        for result in store_results:
            if result.success and result.memory_id is not None:
                created_memory_ids.add(result.memory_id)

        recall_payloads = [
            {
                "query": f"{self.run_id} synthetic preference payload {i}",
                "top_k": 10,
            }
            for i in range(self.recall_ops)
        ]
        recall_metrics, recall_results = await self._run_phase(
            phase_name="recall",
            tool_name="recall",
            payloads=recall_payloads,
        )
        phase_metrics.append(recall_metrics)
        phase_errors["recall"] = [
            result.error for result in recall_results if not result.success and result.error
        ]

        if self.keep_data:
            cleanup_metrics = PhaseMetrics(
                phase="forget",
                tool="forget",
                total_calls=0,
                success_calls=0,
                failed_calls=0,
                duration_ms=0.0,
                throughput_ops_s=0.0,
                min_latency_ms=0.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                max_latency_ms=0.0,
            )
            cleanup_results: list[ToolCallResult] = []
        else:
            cleanup_payloads = [
                {"memory_id": memory_id}
                for memory_id in sorted(created_memory_ids)
            ]
            cleanup_metrics, cleanup_results = await self._run_phase(
                phase_name="forget",
                tool_name="forget",
                payloads=cleanup_payloads,
            )
            for payload, call in zip(cleanup_payloads, cleanup_results, strict=False):
                if call.success:
                    deleted_memory_ids.add(payload["memory_id"])
        phase_metrics.append(cleanup_metrics)
        phase_errors["forget"] = [
            result.error for result in cleanup_results if not result.success and result.error
        ]

        final_stats_output = await self._tools["memory_stats"].fn(ctx=self._ctx)
        final_total = parse_total_memories(final_stats_output)

        ended_at = datetime.now(UTC)
        ended_ms = int(time.time() * 1000)

        return {
            "run_id": self.run_id,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "started_ms": started_ms,
            "ended_ms": ended_ms,
            "total_duration_ms": ended_ms - started_ms,
            "config": {
                "api_url": self.api_url,
                "username": self.username,
                "remember_ops": self.remember_ops,
                "store_ops": self.store_ops,
                "recall_ops": self.recall_ops,
                "concurrency": self.concurrency,
                "keep_data": self.keep_data,
            },
            "health_check": {
                "success": health_result.success,
                "error": health_result.error,
                "latency_ms": health_result.latency_ms,
            },
            "baseline": {
                "total_memories": baseline_total,
            },
            "final": {
                "total_memories": final_total,
            },
            "phase_metrics": [asdict(metrics) for metrics in phase_metrics],
            "phase_errors": phase_errors,
            "created_memory_ids_count": len(created_memory_ids),
            "deleted_memory_ids_count": len(deleted_memory_ids),
            "created_memory_ids_sample": sorted(created_memory_ids)[:20],
        }


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run benchmark and return metrics payload."""
    cfg: dict[str, Any] = {}
    if not (args.api_url and args.username and args.password):
        cfg = get_config()

    api_url = args.api_url or cfg.get("api_url")
    username = args.username or cfg.get("username")
    password = args.password or cfg.get("password")

    if not api_url or not username or not password:
        raise RuntimeError(
            "Missing benchmark connection settings. Provide --api-url/--username/--password "
            "or set ENGRAM_API_URL/ENGRAM_USERNAME/ENGRAM_PASSWORD."
        )

    run_id = args.run_id or f"mcp-bench-{datetime.now(UTC).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"

    async with MCPBenchmarkRunner(
        api_url=api_url,
        username=username,
        password=password,
        run_id=run_id,
        remember_ops=args.remember_ops,
        store_ops=args.store_ops,
        recall_ops=args.recall_ops,
        concurrency=args.concurrency,
        keep_data=args.keep_data,
    ) as runner:
        return await runner.run()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Run repeatable MCP benchmark and write phase metrics as CSV/JSON.",
    )
    parser.add_argument("--api-url", default=None, help="Engram API base URL")
    parser.add_argument("--username", default=None, help="Engram username")
    parser.add_argument("--password", default=None, help="Engram password")
    parser.add_argument("--run-id", default=None, help="Explicit run id")
    parser.add_argument("--remember-ops", type=int, default=30, help="remember calls")
    parser.add_argument("--store-ops", type=int, default=30, help="store_memory calls")
    parser.add_argument("--recall-ops", type=int, default=15, help="recall calls")
    parser.add_argument("--concurrency", type=int, default=5, help="parallel calls per phase")
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for output files",
    )
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Skip forget cleanup phase",
    )
    return parser


def write_outputs(output_dir: Path, run_data: dict[str, Any]) -> tuple[Path, Path]:
    """Write JSON and CSV outputs and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_data["run_id"]
    json_path = output_dir / f"mcp_benchmark_{run_id}.json"
    csv_path = output_dir / f"mcp_benchmark_{run_id}.csv"

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(run_data, json_file, indent=2)

    phases = [PhaseMetrics(**phase) for phase in run_data["phase_metrics"]]
    write_phase_csv(csv_path, phases)
    return json_path, csv_path


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()
    run_data = asyncio.run(run_benchmark(args))
    json_path, csv_path = write_outputs(Path(args.output_dir), run_data)
    print(f"Benchmark run complete: {run_data['run_id']}")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
