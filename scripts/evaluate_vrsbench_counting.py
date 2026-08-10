#!/usr/bin/env python3
"""Evaluate VRSBench-counting through the real CountingAgent.

使用真实 CountingAgent 对 VRSBench-counting 做全流程计数评测：只经公共
Runtime.create() 组装真实 Qwen + 完整后端注册表 + 回退，逐样本调用
CountingAgent.run()；答案来源取自 trace（final_backend/kind、primary、
fallback、target），指标复用生产 count_deterministic_metrics /
aggregate_counting，车辆 hint 复用生产 count_target_hint。输出流式
results.jsonl + 原子 summary.json / unsupported_or_error.json；字段
JSON-safe，错误只记稳定类型名/错误码，不写密钥、绝对路径或原始异常。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Standalone-script bootstrap: make the repo root importable when executed
# directly. 独立脚本引导：直接执行时把仓库根加入 sys.path。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from application.runtime import Runtime
from application.settings import load_settings
from agents.base import AgentContext
from data.adapters.vrsbench.ontology import count_target_hint
from data.schema import GroundTruth, ImageRef, UnifiedSample, stable_sample_id
from evaluation.metrics.counting import aggregate_counting, count_deterministic_metrics
from evaluation.records import EvaluationRecord

DEFAULT_INPUT = Path("data/VRSBench-counting/VRSBench_train_counting.jsonl")
DEFAULT_DATA_ROOT = Path("data/VRSBench-full")
DEFAULT_OUTPUT_DIR = Path("outputs/vrsbench_counting_eval")
DEFAULT_LIMIT = 0  # 0 means all rows. / 0 表示全量。
DEFAULT_CONCURRENCY = 1
PROGRESS_EVERY = 200

# CountingResult status -> public bucket; unknown fails closed as error.
# CountingResult 状态 -> 公共桶；未知状态以 error 关闭。
_COMPLETED_STATUSES = frozenset({"completed", "completed_with_warnings"})
_UNSUPPORTED_STATUSES = frozenset({"partial", "failed"})
_SOURCE_BUCKETS = (
    "yolo_obb",
    "qwen_point",
    "quantity_proposal",
    "semantic_segmentation",
)


class _RowValidationError(ValueError):
    """Stable row-validation code carried on the exception. / 稳定的行校验错误码。"""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI. / 构建评测 CLI。"""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the full VRSBench-counting pipeline through the real "
            "CountingAgent and annotate every final answer source."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--limit", type=_non_negative_int, default=DEFAULT_LIMIT,
        help="Maximum rows to evaluate; 0 means all.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help=(
            "Optional settings YAML; must point models.qwen.model at the "
            "local checkpoint and set models.qwen.cache_model_id for paths."
        ),
    )
    parser.add_argument(
        "--concurrency", type=_positive_int, default=DEFAULT_CONCURRENCY,
        help=(
            "Sample-level concurrency. Default 1 because YOLO count() is a "
            "synchronous blocking call and ONNX session thread safety is "
            "unverified."
        ),
    )
    return parser


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


async def run_evaluation(args: argparse.Namespace) -> int:
    """Assemble the real runtime once, then stream one row per sample.
    组装一次真实运行时，然后逐样本流式执行。"""
    project_root = Path(__file__).resolve().parents[1]
    settings = load_settings(args.config, environ=os.environ)
    runtime = Runtime.create(
        settings=settings,
        project_root=project_root,
        api_key=None,  # No DeepSeek judge; deterministic-only. / 纯确定性。
    )
    rows = _read_rows(args.input, args.limit)
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "results.jsonl"
    jsonl_path.write_text("", encoding="utf-8")
    semaphore = asyncio.Semaphore(args.concurrency)

    async def run_one(
        index: int,
        row: dict[str, Any],
    ) -> tuple[dict[str, Any], EvaluationRecord | None]:
        async with semaphore:
            return await _run_one(index, row, runtime, args)

    results: list[tuple[dict[str, Any], EvaluationRecord | None]] = []
    with jsonl_path.open("a", encoding="utf-8") as jsonl_handle:
        tasks = [asyncio.create_task(run_one(index, row)) for index, row in enumerate(rows)]
        for done in asyncio.as_completed(tasks):
            row_result, evaluation = await done
            _append_jsonl(jsonl_handle, row_result)
            results.append((row_result, evaluation))
            if len(results) % PROGRESS_EVERY == 0:
                print(f"processed {len(results)}/{len(rows)}", file=sys.stderr, flush=True)

    ordered = sorted((row_result for row_result, _ in results), key=lambda item: item["index"])
    records = [evaluation for _, evaluation in results if evaluation is not None]
    summary = _build_summary(ordered, records, args, settings)
    unsupported = [
        row_result for row_result in ordered
        if row_result["status"] in {"error", "unsupported", "gold_parse_error"}
    ]
    _atomic_write_json(output_dir / "summary.json", summary)
    _atomic_write_json(output_dir / "unsupported_or_error.json", unsupported)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    partial_failures = sum(
        1 for row_result in ordered if row_result["status"] in {"error", "unsupported"}
    )
    return 2 if partial_failures else 0


async def _run_one(
    index: int,
    row: dict[str, Any],
    runtime: Runtime,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], EvaluationRecord | None]:
    """Run one row through CountingAgent.run and annotate the answer source.
    通过 CountingAgent.run 执行一行并标注答案来源。"""
    record = _base_row(index, row)
    try:
        sample, gold, hint_used = _build_sample(index, row)
        record["sample_id"] = sample.sample_id
        record["gold"] = gold
        record["hint_used"] = hint_used
        # Per-sample dirs isolate backend artifacts, even at concurrency > 1.
        # 逐样本目录隔离后端产物，并发 > 1 也不冲突。
        artifact_dir = args.output_dir / "agent_artifacts" / f"sample_{index:06d}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        context = AgentContext(
            artifact_dir=artifact_dir,
            qwen_client=runtime.components.qwen_client,
            call_budget=runtime.components.call_budget_factory.create_for_sample(
                "counting"
            ),
            data_root=args.data_root,
            judge_client=None,
        )
        agent = runtime.components.agent_registry.get("counting_agent")
        execution = await agent.run(sample, context)
        payload = execution.payload
        trace = execution.trace if isinstance(execution.trace, dict) else {}
        record.update({
            "target": trace.get("target"),
            "predicted": getattr(payload, "final_count", None),
            "answer_source": trace.get("final_backend"),
            "answer_source_kind": trace.get("final_backend_kind"),
            "primary_backend": trace.get("primary_backend"),
            "fallback_triggered": bool(trace.get("fallback_triggered")),
            "fallback_kind": trace.get("fallback_kind"),
            "fallback_reason_code": trace.get("fallback_reason_code"),
            "target_classes": list(trace.get("target_classes") or []),
        })
        public_status = _public_status(str(getattr(payload, "status", "completed")))
        # Agent still runs for non-integer answers; only GT comparison is lost.
        # 非整数标答仍执行 Agent，只丢失 GT 对比。
        record["status"] = "gold_parse_error" if gold is None else public_status
        if public_status == "error":
            record["error_type"] = "UNKNOWN_COUNTING_STATUS"
        metrics = None
        predicted = record["predicted"]
        if isinstance(predicted, int) and gold is not None:
            metrics = count_deterministic_metrics(predicted, gold)
            record["exact_match"] = metrics.exact_match
            record["absolute_error"] = metrics.absolute_error
        evaluation = EvaluationRecord(
            sample_id=sample.sample_id,
            task="counting",
            deterministic_metrics=metrics,
            judge_status="not_requested",
        )
        return record, evaluation
    except Exception as error:
        # Keep the batch alive; stable code/type name only, never raw text.
        # 保持整批继续；只记稳定错误码/类型名，绝不记录原始文本。
        record["status"] = "error"
        record["error_type"] = getattr(error, "code", None) or type(error).__name__
        return record, None


def _read_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    """Read up to ``limit`` JSONL objects (0 means all). / 读取最多 limit 条。"""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if limit > 0 and len(rows) >= limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL at line {line_number}: {type(error).__name__}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            rows.append(row)
    return rows


def _parse_gold(value: Any) -> int | None:
    """Return the integer ground truth, or None (gold_parse_error).
    返回整数标答；非整数返回 None。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _build_sample(index: int, row: dict[str, Any]) -> tuple[UnifiedSample, int | None, bool]:
    """Build one UnifiedSample; vehicle hints come from the production data
    layer, other targets fall through to the agent's Qwen parser.
    构造一条 UnifiedSample；车辆 hint 来自生产数据层，其余走 Agent Qwen 解析。"""
    question = row.get("question")
    image = row.get("image")
    if not isinstance(question, str) or not question.strip():
        raise _RowValidationError("INVALID_ROW_MISSING_QUESTION")
    if not isinstance(image, str) or not image.strip():
        raise _RowValidationError("INVALID_ROW_MISSING_IMAGE")
    gold = _parse_gold(row.get("answer"))
    hint = count_target_hint(question)
    metadata: dict[str, Any] = {}
    if hint is not None:
        metadata["count_target_hint"] = hint
    sample = UnifiedSample(
        sample_id=stable_sample_id(
            dataset="VRSBench",
            split="train",
            source_id=row.get("id"),
            relative_image_paths=[image],
            question=question,
            source_index=index,
        ),
        dataset="VRSBench",
        split="train",
        task="counting",
        images=[ImageRef(image_id="image-0", path=Path(image), role="image")],
        question=question,
        ground_truth=GroundTruth(count=gold) if gold is not None else None,
        metadata=metadata,
    )
    return sample, gold, hint is not None


def _base_row(index: int, row: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe row skeleton before execution. / 执行前的 JSON 安全行骨架。"""
    return {
        "index": index,
        "sample_id": str(row.get("id") or f"vrsbench/row/{index}"),
        "image": row.get("image"),
        "question": row.get("question"),
        "target": None,
        "gold": None,
        "predicted": None,
        "exact_match": None,
        "absolute_error": None,
        "answer_source": None,
        "answer_source_kind": None,
        "primary_backend": None,
        "fallback_triggered": None,
        "fallback_kind": None,
        "fallback_reason_code": None,
        "target_classes": [],
        "status": "error",
        "error_type": None,
        "hint_used": None,
    }


def _public_status(payload_status: str) -> str:
    """Map a CountingResult status to the public bucket.
    将 CountingResult 状态映射到公共桶。"""
    if payload_status in _COMPLETED_STATUSES:
        return "succeeded"
    if payload_status in _UNSUPPORTED_STATUSES:
        return "unsupported"
    return "error"


def _build_summary(
    rows: list[dict[str, Any]],
    records: list[EvaluationRecord],
    args: argparse.Namespace,
    settings: Any,
) -> dict[str, Any]:
    """Aggregate metrics, answer sources, fallback, statuses, targets.
    汇总指标、答案来源、回退、状态与目标分布。"""
    total = len(rows)
    status_counts = {
        "total": total,
        "succeeded": 0,
        "unsupported": 0,
        "error": 0,
        "gold_parse_error": 0,
    }
    for row_result in rows:
        status_counts[row_result["status"]] += 1
    return {
        "input_file": str(args.input),
        "data_root": str(args.data_root),
        "output_dir": str(args.output_dir),
        "limit": args.limit,
        "concurrency": args.concurrency,
        "model_id": settings.models.qwen.effective_cache_model_id,
        "default_backend": settings.agents.counting.default_backend,
        "total": total,
        "status_counts": status_counts,
        "metrics": aggregate_counting(records),
        "answer_source_buckets": _source_buckets(rows),
        "fallback": _fallback_stats(rows),
        "target_distribution": dict(
            Counter(
                row_result["target"]
                for row_result in rows
                if row_result.get("target") is not None
            )
        ),
    }


def _source_buckets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per answer-source count, evaluated count, accuracy, and MAE.
    按答案来源统计数量、可评测数、accuracy 与 MAE。"""
    buckets: dict[str, dict[str, Any]] = {
        key: {"count": 0, "evaluated": 0}
        for key in (*_SOURCE_BUCKETS, "error", "unknown")
    }
    for row_result in rows:
        key = (
            "error"
            if row_result["status"] == "error"
            else row_result.get("answer_source_kind") or "unknown"
        )
        bucket = buckets.setdefault(key, {"count": 0, "evaluated": 0})
        bucket["count"] += 1
        if (
            row_result.get("exact_match") is not None
            and row_result.get("absolute_error") is not None
        ):
            bucket["evaluated"] += 1
            bucket["exact_sum"] = bucket.get("exact_sum", 0) + int(row_result["exact_match"])
            bucket["mae_sum"] = bucket.get("mae_sum", 0) + float(row_result["absolute_error"])
    for bucket in buckets.values():
        evaluated = bucket["evaluated"]
        bucket["accuracy"] = bucket.get("exact_sum", 0) / evaluated if evaluated else None
        bucket["mae"] = bucket.get("mae_sum", 0) / evaluated if evaluated else None
    return buckets


def _fallback_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count fallback triggers and kinds among executed samples.
    统计已执行样本中的回退触发次数与种类。"""
    executed = [
        row_result for row_result in rows
        if row_result.get("fallback_triggered") is not None
    ]
    triggered = [row_result for row_result in executed if row_result["fallback_triggered"]]
    return {
        "executed_samples": len(executed),
        "fallback_triggered_count": len(triggered),
        "fallback_triggered_ratio": (
            round(len(triggered) / len(executed), 6) if executed else None
        ),
        "by_fallback_kind": dict(
            Counter(
                row_result["fallback_kind"]
                for row_result in triggered
                if row_result.get("fallback_kind") is not None
            )
        ),
    }


def _append_jsonl(handle: Any, record: dict[str, Any]) -> None:
    """Append one streamed line and flush so a crash keeps prior rows.
    追加一行流式结果并 flush，使中途崩溃保留已完成行。"""
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def _atomic_write_json(path: Path, value: Any) -> None:
    """Atomic JSON write via temp file + replace (stdlib only).
    经临时文件与 replace 的原子 JSON 写入（仅 stdlib）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    """Parse args, run the async loop, return a stable exit code.
    解析参数、运行异步主流程并返回稳定退出码。"""
    args = build_parser().parse_args()
    try:
        return asyncio.run(run_evaluation(args))
    except KeyboardInterrupt:
        print("vrsbench counting evaluation interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        # Public failure output never carries raw exception text or secrets.
        # 公共失败输出绝不携带原始异常文本或密钥。
        print(f"vrsbench counting evaluation failed: {type(error).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
