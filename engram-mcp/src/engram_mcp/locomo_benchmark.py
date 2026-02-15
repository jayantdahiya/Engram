"""Run official LOCOMO QA evaluation using MCP tool calls for memory operations."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import re
import statistics
import sys
import time
import uuid
from collections import defaultdict
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
from typing import Any

from fastmcp import FastMCP

from engram_mcp.client import EngramClient
from engram_mcp.config import get_config
from engram_mcp.tools import register_tools

_REMEMBER_ID_PATTERN = re.compile(r"Memory ID:\s*(\d+)")
_STORE_ID_PATTERN = re.compile(r"Memory created with ID:\s*(\d+)")
_RECALL_LINE_PATTERN = re.compile(r"^\s*\[(\d+)\]\s+\(importance:\s*([^)]+)\)\s*(.*)$")


@dataclass
class RecallHit:
    """Single parsed item from recall tool output."""

    memory_id: int
    importance: float
    text: str


def extract_memory_id(tool_output: str) -> int | None:
    """Extract memory id from remember/store output."""
    for pattern in (_REMEMBER_ID_PATTERN, _STORE_ID_PATTERN):
        match = pattern.search(tool_output)
        if match:
            return int(match.group(1))
    return None


def parse_recall_output(tool_output: str) -> list[RecallHit]:
    """Parse recall response into structured hits."""
    hits: list[RecallHit] = []
    for line in tool_output.splitlines():
        match = _RECALL_LINE_PATTERN.match(line)
        if not match:
            continue
        memory_id = int(match.group(1))
        importance_raw = match.group(2).strip()
        text = match.group(3).strip()
        try:
            importance = float(importance_raw)
        except ValueError:
            importance = 0.0
        hits.append(RecallHit(memory_id=memory_id, importance=importance, text=text))
    return hits


def simple_extractive_answer(question: str, hits: list[RecallHit]) -> str:
    """Heuristic answer extraction from top recall hits."""
    if not hits:
        return "No information available."

    question_tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", question)
        if len(token) > 2
    }

    best_text = hits[0].text
    best_score = -1.0
    for hit in hits:
        text_tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9]+", hit.text)
            if len(token) > 2
        }
        overlap = len(question_tokens & text_tokens)
        score = overlap + (0.001 * max(hit.importance, 0.0))
        if score > best_score:
            best_score = score
            best_text = hit.text

    # Light cleanup: remove speaker prefix if present.
    # Example: "Caroline said, \"...\"" -> "..."
    quoted = re.findall(r"\"([^\"]+)\"", best_text)
    if quoted:
        return quoted[0].strip()
    return best_text.strip()


async def llm_extractive_answer(
    question: str,
    hits: list[RecallHit],
    client: EngramClient,
) -> str:
    """Use configured LLM to extract concise answer from recall hits."""
    if not hits:
        return "No information available."

    context_memories = [hit.text for hit in hits[:5] if hit.text]
    if not context_memories:
        return "No information available."

    try:
        answer = await client.generate_answer(question, context_memories)
    except Exception:
        # Keep benchmark resilient if local LLM is unavailable.
        return simple_extractive_answer(question, hits)

    answer = answer.strip()
    return answer if answer else "No information available."


def session_keys(conversation: dict[str, Any]) -> list[int]:
    """Return sorted session indices available in conversation dict."""
    values: list[int] = []
    for key, value in conversation.items():
        if key.startswith("session_") and isinstance(value, list):
            try:
                values.append(int(key.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return sorted(values)


class LocomoMCPBenchmark:
    """Benchmark runner that uses MCP tools and official LOCOMO evaluator."""

    def __init__(
        self,
        *,
        api_url: str,
        username: str,
        password: str,
        run_id: str,
        top_k: int,
        ingest_tool: str,
        answer_mode: str,
        scoring_profile: str,
        keep_data: bool,
    ) -> None:
        self.api_url = api_url
        self.username = username
        self.password = password
        self.run_id = run_id
        self.top_k = max(top_k, 1)
        self.ingest_tool = ingest_tool
        self.answer_mode = answer_mode
        self.scoring_profile = scoring_profile
        self.keep_data = keep_data

        self._client: EngramClient | None = None
        self._ctx: Any = None
        self._tools: dict[str, Any] = {}

    async def __aenter__(self) -> LocomoMCPBenchmark:
        client = EngramClient(
            api_url=self.api_url,
            username=self.username,
            password=self.password,
        )
        await client.start()
        self._client = client

        mcp = FastMCP(name="engram-locomo-benchmark")
        register_tools(mcp)
        self._tools = dict(mcp._tool_manager._tools)
        self._ctx = SimpleNamespace(
            request_context=SimpleNamespace(lifespan_context={"client": client})
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            await self._client.stop()

    async def _tool_call(self, name: str, **kwargs: Any) -> str:
        result = await self._tools[name].fn(ctx=self._ctx, **kwargs)
        return result if isinstance(result, str) else str(result)

    async def ingest_sample(
        self,
        sample: dict[str, Any],
    ) -> tuple[set[int], dict[int, str], int]:
        """Store conversation turns as memories via MCP tool calls."""
        created_memory_ids: set[int] = set()
        memory_id_to_dia_id: dict[int, str] = {}
        failed_ingest_calls = 0

        sample_id = sample["sample_id"]
        conv = sample["conversation"]
        conv_id = f"{self.run_id}:{sample_id}"

        for s_idx in session_keys(conv):
            date_time = conv.get(f"session_{s_idx}_date_time", "")
            dialogs = conv[f"session_{s_idx}"]
            for turn in dialogs:
                speaker = turn.get("speaker", "Speaker")
                dia_id = turn.get("dia_id", "")
                text = turn.get("text", "")
                memory_text = f"({date_time}) {speaker} said, \"{text}\""
                metadata = {
                    "source": "locomo_benchmark",
                    "run_id": self.run_id,
                    "sample_id": sample_id,
                    "session": s_idx,
                    "dia_id": dia_id,
                    "date_time": date_time,
                }

                if self.ingest_tool == "remember":
                    try:
                        output = await self._tool_call(
                            "remember",
                            conversation_id=conv_id,
                            user_message=memory_text,
                        )
                    except Exception:
                        failed_ingest_calls += 1
                        continue
                else:
                    try:
                        output = await self._tool_call(
                            "store_memory",
                            conversation_id=conv_id,
                            text=memory_text,
                            importance_score=5.0,
                            metadata=metadata,
                        )
                    except Exception:
                        failed_ingest_calls += 1
                        continue

                memory_id = extract_memory_id(output)
                if memory_id is not None:
                    created_memory_ids.add(memory_id)
                    memory_id_to_dia_id[memory_id] = dia_id

        return created_memory_ids, memory_id_to_dia_id, failed_ingest_calls

    async def evaluate_sample(
        self,
        sample: dict[str, Any],
        *,
        prediction_key: str,
        max_questions: int,
        memory_id_to_dia_id: dict[int, str],
    ) -> tuple[dict[str, Any], int]:
        """Run QA over sample using MCP recall results."""
        sample_id = sample["sample_id"]
        output_sample = {"sample_id": sample_id, "qa": []}
        failed_recall_calls = 0

        qa_items = sample["qa"][:max_questions] if max_questions > 0 else sample["qa"]
        for qa in qa_items:
            question = qa["question"]
            recall_kwargs: dict[str, Any] = {"query": question, "top_k": self.top_k}
            if self.scoring_profile:
                recall_kwargs["scoring_profile"] = self.scoring_profile
            try:
                recall_output = await self._tool_call("recall", **recall_kwargs)
            except Exception:
                failed_recall_calls += 1
                recall_output = "No memories found."
            hits = parse_recall_output(recall_output)

            if self.answer_mode == "oracle":
                prediction = str(qa["answer"])
            elif self.answer_mode == "llm":
                if self._client is None:
                    prediction = "No information available."
                else:
                    prediction = await llm_extractive_answer(question, hits, self._client)
            else:
                prediction = simple_extractive_answer(question, hits)

            pred_qa = dict(qa)
            pred_qa[prediction_key] = prediction
            pred_qa[prediction_key + "_context"] = [
                memory_id_to_dia_id[hit.memory_id]
                for hit in hits
                if hit.memory_id in memory_id_to_dia_id
            ]
            output_sample["qa"].append(pred_qa)

        return output_sample, failed_recall_calls

    async def cleanup(self, memory_ids: set[int]) -> int:
        """Delete created benchmark memories."""
        if self.keep_data:
            return 0

        deleted = 0
        for memory_id in sorted(memory_ids):
            try:
                await self._tool_call("forget", memory_id=memory_id)
                deleted += 1
            except Exception:
                continue
        return deleted


def _install_bert_score_stub_if_missing() -> None:
    """Install a lightweight bert_score stub to import official evaluator."""
    if importlib.util.find_spec("bert_score") is not None:
        return

    stub = ModuleType("bert_score")

    def _missing_score(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "bertscore is not installed in this environment. "
            "Install bert-score/torch only if you need bert_score metric paths."
        )

    setattr(stub, "score", _missing_score)
    sys.modules.setdefault("bert_score", stub)


def load_eval_function(locomo_root: Path) -> tuple[Any, str]:
    """Load official LOCOMO QA evaluator from local repo checkout."""
    try:
        _install_bert_score_stub_if_missing()
        sys.path.insert(0, str(locomo_root))
        from task_eval.evaluation import eval_question_answering  # type: ignore

        return eval_question_answering, "official"
    except Exception:
        return eval_question_answering_fallback, "fallback"


def normalize_answer_local(text: str) -> str:
    """Approximate normalize_answer from LOCOMO evaluation.py."""
    text = text.replace(",", "")
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\b(a|an|the|and)\b", " ", text)
    text = " ".join(text.split())
    return text


def f1_score_local(prediction: str, ground_truth: str) -> float:
    """Compute single-answer token F1 mirroring official LOCOMO logic."""
    prediction_tokens = normalize_answer_local(prediction).split()
    ground_truth_tokens = normalize_answer_local(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / max(len(prediction_tokens), 1)
    recall = num_same / max(len(ground_truth_tokens), 1)
    return (2.0 * precision * recall) / (precision + recall)


def f1_multi_local(prediction: str, ground_truth: str) -> float:
    """Compute multi-answer F1 approximation used by LOCOMO category 1."""
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]
    if not ground_truths:
        return 0.0
    per_gt = []
    for gt in ground_truths:
        per_gt.append(max((f1_score_local(pred, gt) for pred in predictions), default=0.0))
    return sum(per_gt) / len(per_gt)


def eval_question_answering_fallback(
    qas: list[dict[str, Any]],
    eval_key: str = "prediction",
    metric: str = "f1",
) -> tuple[list[float], float, list[float]]:
    """Fallback evaluator mirroring official LOCOMO QA evaluation flow."""
    del metric  # LOCOMO QA eval currently uses F1-style scores.
    scores: list[float] = []
    recalls: list[float] = []

    for line in qas:
        answer = line["answer"]
        if not isinstance(answer, list):
            answer = str(answer)
        output = str(line.get(eval_key, ""))
        category = int(line["category"])

        if category in {2, 3, 4}:
            if category == 3:
                answer = str(answer).split(";")[0].strip()
            score = f1_score_local(output, str(answer))
        elif category == 1:
            score = f1_multi_local(output, str(answer))
        elif category == 5:
            lower = output.lower()
            score = 1.0 if ("no information available" in lower or "not mentioned" in lower) else 0.0
        else:
            score = 0.0
        scores.append(score)

        ctx_key = eval_key + "_context"
        evidence = line.get("evidence", [])
        if ctx_key in line and evidence:
            predicted_ctx = line.get(ctx_key, [])
            recall_acc = sum(1 for ev in evidence if ev in predicted_ctx) / len(evidence)
            recalls.append(float(recall_acc))
        else:
            recalls.append(1.0)

    return scores, 0.0, recalls


def aggregate_scores(qas: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    """Aggregate scores overall and by QA category."""
    values = [float(q.get(score_key, 0.0)) for q in qas]
    by_category: dict[int, list[float]] = defaultdict(list)
    for q in qas:
        by_category[int(q["category"])].append(float(q.get(score_key, 0.0)))

    return {
        "count": len(values),
        "mean_f1": statistics.mean(values) if values else 0.0,
        "median_f1": statistics.median(values) if values else 0.0,
        "by_category_mean_f1": {
            str(cat): (statistics.mean(cat_vals) if cat_vals else 0.0)
            for cat, cat_vals in sorted(by_category.items())
        },
    }


async def run_locomo_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run LOCOMO benchmark and return structured results."""
    cfg: dict[str, Any] = {}
    if not (args.api_url and args.username and args.password):
        cfg = get_config()

    api_url = args.api_url or cfg.get("api_url")
    username = args.username or cfg.get("username")
    password = args.password or cfg.get("password")
    if not api_url or not username or not password:
        raise RuntimeError(
            "Missing MCP credentials. Provide --api-url/--username/--password "
            "or set ENGRAM_API_URL/ENGRAM_USERNAME/ENGRAM_PASSWORD."
        )

    locomo_root = Path(args.locomo_root).resolve()
    data_file = Path(args.data_file).resolve() if args.data_file else locomo_root / "data/locomo10.json"
    if not data_file.exists():
        raise FileNotFoundError(f"LOCOMO data file not found: {data_file}")

    run_id = args.run_id or f"locomo-mcp-{datetime.now(UTC).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}"
    prediction_key = f"{args.prediction_prefix}_prediction"
    score_key = f"{args.prediction_prefix}_f1"

    eval_question_answering, evaluator_source = load_eval_function(locomo_root)
    samples = json.load(data_file.open("r", encoding="utf-8"))
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    benchmark_outputs: list[dict[str, Any]] = []
    all_qas: list[dict[str, Any]] = []
    created_memory_ids: set[int] = set()
    deleted_memory_ids = 0
    failed_ingest_calls = 0
    failed_recall_calls = 0

    started_ms = int(time.time() * 1000)
    started_at = datetime.now(UTC).isoformat()

    async with LocomoMCPBenchmark(
        api_url=api_url,
        username=username,
        password=password,
        run_id=run_id,
        top_k=args.top_k,
        ingest_tool=args.ingest_tool,
        answer_mode=args.answer_mode,
        scoring_profile=args.scoring_profile,
        keep_data=args.keep_data,
    ) as runner:
        for sample in samples:
            sample_created_ids, memory_id_to_dia_id, sample_ingest_failures = (
                await runner.ingest_sample(sample)
            )
            failed_ingest_calls += sample_ingest_failures
            created_memory_ids.update(sample_created_ids)

            evaluated, sample_recall_failures = await runner.evaluate_sample(
                sample,
                prediction_key=prediction_key,
                max_questions=args.max_questions_per_sample,
                memory_id_to_dia_id=memory_id_to_dia_id,
            )
            failed_recall_calls += sample_recall_failures

            scores, _, recalls = eval_question_answering(evaluated["qa"], prediction_key)
            for idx, qa in enumerate(evaluated["qa"]):
                qa[score_key] = round(float(scores[idx]), 6)
                qa[prediction_key + "_recall"] = round(float(recalls[idx]), 6)
                all_qas.append(qa)

            benchmark_outputs.append(evaluated)

        deleted_memory_ids = await runner.cleanup(created_memory_ids)

    ended_ms = int(time.time() * 1000)
    ended_at = datetime.now(UTC).isoformat()

    summary = aggregate_scores(all_qas, score_key)
    return {
        "run_id": run_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "started_ms": started_ms,
        "ended_ms": ended_ms,
        "duration_ms": ended_ms - started_ms,
        "locomo_root": str(locomo_root),
        "data_file": str(data_file),
        "config": {
            "api_url": api_url,
            "username": username,
            "top_k": args.top_k,
            "ingest_tool": args.ingest_tool,
            "answer_mode": args.answer_mode,
            "scoring_profile": args.scoring_profile,
            "max_samples": args.max_samples,
            "max_questions_per_sample": args.max_questions_per_sample,
            "keep_data": args.keep_data,
        },
        "prediction_key": prediction_key,
        "score_key": score_key,
        "evaluator_source": evaluator_source,
        "summary": summary,
        "created_memory_ids_count": len(created_memory_ids),
        "deleted_memory_ids_count": deleted_memory_ids,
        "failed_ingest_calls": failed_ingest_calls,
        "failed_recall_calls": failed_recall_calls,
        "results": benchmark_outputs,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate MCP memory pipeline on official LOCOMO QA benchmark.",
    )
    parser.add_argument("--locomo-root", required=True, help="Path to cloned LOCOMO repository")
    parser.add_argument("--data-file", default=None, help="Optional path to locomo*.json")
    parser.add_argument("--api-url", default=None, help="Engram backend URL")
    parser.add_argument("--username", default=None, help="Engram username")
    parser.add_argument("--password", default=None, help="Engram password")
    parser.add_argument("--run-id", default=None, help="Optional explicit run id")
    parser.add_argument("--top-k", type=int, default=5, help="Recall top_k")
    parser.add_argument(
        "--ingest-tool",
        choices=["store_memory", "remember"],
        default="store_memory",
        help="Tool used to ingest LOCOMO conversation turns",
    )
    parser.add_argument(
        "--answer-mode",
        choices=["extractive", "llm", "oracle"],
        default="extractive",
        help="Answer generation strategy after recall",
    )
    parser.add_argument(
        "--scoring-profile",
        choices=["balanced", "semantic"],
        default="balanced",
        help="Retrieval scoring profile to use for recall queries",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    parser.add_argument(
        "--max-questions-per-sample",
        type=int,
        default=0,
        help="0 means all questions per sample",
    )
    parser.add_argument(
        "--prediction-prefix",
        default="mcp_locomo",
        help="Prefix used for prediction/scoring keys in output",
    )
    parser.add_argument("--output-dir", default=".", help="Directory for output JSON")
    parser.add_argument("--keep-data", action="store_true", help="Skip memory cleanup")
    return parser


def write_output(output_dir: Path, result: dict[str, Any]) -> Path:
    """Write benchmark JSON output."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"locomo_mcp_benchmark_{result['run_id']}.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return out_file


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()
    result = asyncio.run(run_locomo_benchmark(args))
    out_file = write_output(Path(args.output_dir), result)
    print(f"LOCOMO MCP benchmark completed: {result['run_id']}")
    print(f"Output: {out_file}")
    print(f"Mean F1: {result['summary']['mean_f1']:.4f}")


if __name__ == "__main__":
    main()
