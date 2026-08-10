#!/usr/bin/env python3
"""Classify VRSBench test VQA tasks through the TaskResolver model path.

VRSBench 测试 VQA 集任务分类脚本：直接读取派生 JSONL（不构造
UnifiedSample、不触碰数据契约），为每条样本构造 TaskResolutionRequest 并
调用 TaskResolver 的模型解析路径（explicit_task=None 且 question 非空必然
走模型路径，不送图）。Qwen 客户端只创建一次、全样本共享；JsonResponseCache
按请求哈希去重，重跑可命中缓存。产物只含稳定 code 与 JSON-safe 字段：不写
secret、不写原始异常文本。脚本复用架构既有入口（models.entry.create_model /
VisionLanguageClient.complete_json），不是第二条模型调用链。
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

# Standalone-script bootstrap: make the repo root importable when this file is
# executed directly (python scripts/classify_tasks.py). This only adjusts the
# module search path; the import DAG and package boundaries are unchanged and
# the script never reaches into legacy packages. 独立脚本引导：直接执行本文件
# 时把仓库根加入 sys.path。它只调整模块查找路径，不改变 import DAG 或包
# 边界，也绝不触碰旧包。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from application.prompts import PromptCatalog
from application.settings import load_settings
from models.cache import JsonResponseCache
from models.entry import create_model
from routing.schema import TaskResolution, TaskResolutionRequest
from workflows.task_resolver import TaskResolver, TaskResolutionError

DEFAULT_INPUT = Path("datasets/vrsbench/VRSBench_test_vqa.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/task_classification")
DEFAULT_LIMIT = 1000
DEFAULT_CONCURRENCY = 8
QUESTION_TRUNCATE = 120

# Interpretive mapping from VRSBench source task names to the closed internal
# TaskName set. This is explanatory only -- never a ground-truth mapping; the
# original source_task and the predicted_task are always shown verbatim.
# VRSBench 源 task 到内部封闭 TaskName 集合的解释性映射；仅供展示，绝非
# Ground Truth。原始 source_task 与 predicted_task 始终原样展示。
_SOURCE_TASK_MAPPING = {
    "object_existence": "general_vqa",
    "object_classification": "general_vqa",
    "scene_classification": "scene_classification",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the classification CLI. / 构建分类 CLI。"""

    parser = argparse.ArgumentParser(
        description=(
            "Classify VRSBench test VQA tasks through the TaskResolver "
            "model path (auto mode)."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--limit", type=_positive_int, default=DEFAULT_LIMIT)
    parser.add_argument("--concurrency", type=_positive_int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional settings YAML; must point models.qwen.model at the "
            "local 8B checkpoint and set models.qwen.cache_model_id."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prompts-root", type=Path, default=None)
    return parser


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


async def run_classification(args: argparse.Namespace) -> int:
    """Assemble the minimal classifier and run --limit rows concurrently.
    组装最小分类器并并发分类前 --limit 行。"""

    project_root = Path(__file__).resolve().parents[1]
    settings = load_settings(args.config, environ=os.environ)
    catalog = PromptCatalog(args.prompts_root or project_root / "prompts")
    # Same composition pieces as application/bootstrap.py: service cache plus
    # one shared Qwen client; no segformer/judge/agent registry here.
    # 与 application/bootstrap.py 相同的组装构件：service 缓存 + 单个共享
    # Qwen 客户端；此处不碰 segformer/judge/agent registry。
    service_cache = JsonResponseCache(settings.runs.root / "service" / "cache")
    qwen_client = create_model(
        "qwen_transformers",
        settings=settings.models.qwen,
        repair_prompt=catalog["json_repair"],
        cache=service_cache,
    )
    resolver = TaskResolver(
        qwen_client,
        system_prompt=catalog["task_resolver"],
        confidence_threshold=settings.router.confidence_threshold,
    )
    rows = _read_rows(args.input, args.limit)
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "results.jsonl"
    jsonl_path.write_text("", encoding="utf-8")
    semaphore = asyncio.Semaphore(args.concurrency)

    async def classify_one(index: int, row: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            record = _base_record(index, row)
            if record["error"] is None:
                try:
                    resolution = await resolver.resolve(
                        TaskResolutionRequest(
                            explicit_task=None,
                            question=row["question"],
                            image_count=1,
                        ),
                        sample_id=record["sample_id"],
                        artifact_dir=output_dir / "artifacts" / str(index),
                        budget=None,
                    )
                    _apply_resolution(record, resolution)
                except TaskResolutionError as exc:
                    record["error"] = f"TASK_RESOLUTION_FAILED:{exc.code}"
                except Exception as exc:
                    # Keep the batch alive; only the stable type name is
                    # recorded, never raw exception text. 保持整批继续；只记录
                    # 稳定类型名，绝不记录原始异常文本。
                    record["error"] = f"UNEXPECTED:{type(exc).__name__}"
            _append_jsonl(jsonl_handle, record)
            return record

    with jsonl_path.open("a", encoding="utf-8") as jsonl_handle:
        records = await asyncio.gather(
            *(classify_one(index, row) for index, row in enumerate(rows))
        )
    ordered = sorted(records, key=lambda record: record["index"])
    summary = _build_summary(ordered, args, settings, catalog)
    _atomic_write_json(
        output_dir / "results.json",
        {"summary": summary, "records": ordered},
    )
    _atomic_write_text(output_dir / "results.html", _render_html(ordered, summary))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _read_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    """Read at most ``limit`` JSONL objects without interpreting the schema.
    读取最多 ``limit`` 条 JSONL 对象，不做 schema 解读。"""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if len(rows) >= limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at line {line_number}: {type(exc).__name__}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            rows.append(row)
    return rows


def _mapped_expected_task(source_task: str | None) -> str | None:
    """Interpretive source -> internal task mapping (explanatory, not GT).
    解释性源任务到内部任务映射（仅供展示，非 Ground Truth）。"""

    if source_task is None:
        return None
    return _SOURCE_TASK_MAPPING.get(source_task)


def _base_record(index: int, row: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-safe record; malformed rows get a stable INVALID_ROW code
    and never reach the model. 构造 JSON 安全记录；畸形行获得稳定
    INVALID_ROW code 且绝不调用模型。"""

    source_task = row.get("task")
    if not isinstance(source_task, str) or not source_task.strip():
        error = "INVALID_ROW_MISSING_TASK"
        source_task = None
    else:
        error = None
    sample_id = row.get("id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        error = error or "INVALID_ROW_MISSING_ID"
        sample_id = f"vrsbench/row/{index}"
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        error = error or "INVALID_ROW_MISSING_QUESTION"
        question = question if isinstance(question, str) else ""
    return {
        "index": index,
        "sample_id": sample_id,
        "source_task": source_task,
        "mapped_expected_task": _mapped_expected_task(source_task),
        "question": question,
        "predicted_task": None,
        "confidence": None,
        "resolution_source": None,
        "candidate_tasks": [],
        "needs_candidate_fallback": False,
        "reason_codes": [],
        "match": False,
        "error": error,
    }


def _apply_resolution(
    record: dict[str, Any],
    resolution: TaskResolution,
) -> None:
    """Copy one structured resolution into the record and compute the
    explanatory match column. 把一条结构化解析结果写入记录，并计算解释性
    match 列。"""

    record["predicted_task"] = resolution.task
    record["confidence"] = resolution.confidence
    record["resolution_source"] = resolution.source
    record["candidate_tasks"] = list(resolution.candidate_tasks)
    record["needs_candidate_fallback"] = resolution.needs_candidate_fallback
    record["reason_codes"] = list(resolution.reason_codes)
    record["match"] = (
        record["mapped_expected_task"] is not None
        and resolution.task == record["mapped_expected_task"]
    )


def _append_jsonl(handle: Any, record: dict[str, Any]) -> None:
    """Append one streamed line and flush so a crash keeps prior rows.
    追加一行流式结果并 flush，使中途崩溃保留已完成行。"""

    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def _build_summary(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    settings: Any,
    catalog: PromptCatalog,
) -> dict[str, Any]:
    """Aggregate match rates by source task and predicted-task distribution.
    按源任务聚合匹配率与预测任务分布。"""

    total = len(records)
    matched = sum(1 for record in records if record["match"])
    errored = sum(1 for record in records if record.get("error") is not None)
    source_tasks: dict[str, dict[str, Any]] = {}
    for record in records:
        key = record["source_task"] or "unknown"
        entry = source_tasks.setdefault(key, {"count": 0, "match_count": 0})
        entry["count"] += 1
        if record["match"]:
            entry["match_count"] += 1
    for entry in source_tasks.values():
        entry["match_rate"] = (
            round(entry["match_count"] / entry["count"], 6)
            if entry["count"]
            else None
        )
    predicted_tasks: dict[str, int] = {}
    for record in records:
        key = record["predicted_task"] or "error"
        predicted_tasks[key] = predicted_tasks.get(key, 0) + 1
    sources: dict[str, int] = {}
    for record in records:
        key = record["resolution_source"] or "error"
        sources[key] = sources.get(key, 0) + 1
    return {
        "input_file": str(args.input),
        "limit": args.limit,
        "concurrency": args.concurrency,
        "resolver_confidence_threshold": settings.router.confidence_threshold,
        "model_id": settings.models.qwen.effective_cache_model_id,
        "prompt_version": catalog.version("task_resolver"),
        "total": total,
        "matched": matched,
        "match_rate": round(matched / total, 6) if total else None,
        "errored": errored,
        "resolution_sources": sources,
        "source_tasks": source_tasks,
        "predicted_tasks": predicted_tasks,
    }


def _render_html(
    records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    """Render a self-contained HTML page; every dynamic value is escaped.
    渲染自包含 HTML 页面；所有动态值均转义。"""

    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="zh"><head><meta charset="utf-8">',
            "<title>VRSBench Test VQA Task Classification</title>",
            "<style>",
            _CSS,
            "</style></head><body>",
            "<h1>VRSBench Test VQA — Task Classification</h1>",
            _summary_html(summary),
            _records_html(records),
            "</body></html>",
        ]
    )


def _summary_html(summary: dict[str, Any]) -> str:
    source_cells = "".join(
        "<tr>"
        f"<td>{_esc(str(task))}</td>"
        f"<td>{entry['count']}</td>"
        f"<td>{entry['match_count']}</td>"
        f"<td>{_esc(_format_rate(entry['match_rate']))}</td>"
        "</tr>"
        for task, entry in sorted(summary["source_tasks"].items())
    )
    source_block = (
        "<h3>Source tasks</h3>"
        "<table><tr><th>source_task</th><th>count</th>"
        "<th>match_count</th><th>match_rate</th></tr>"
        + source_cells
        + "</table>"
    )
    predicted_cells = "".join(
        "<tr>"
        f"<td>{_esc(str(task))}</td><td>{count}</td></tr>"
        for task, count in sorted(summary["predicted_tasks"].items())
    )
    predicted_block = (
        "<h3>Predicted task distribution</h3>"
        "<table><tr><th>predicted_task</th><th>count</th></tr>"
        + predicted_cells
        + "</table>"
    )
    return (
        "<h2>Summary</h2>"
        "<table><tr><th>total</th><th>matched</th><th>match_rate</th>"
        "<th>errored</th></tr>"
        f"<tr><td>{summary['total']}</td><td>{summary['matched']}</td>"
        f"<td>{_esc(_format_rate(summary['match_rate']))}</td>"
        f"<td>{summary['errored']}</td></tr></table>"
        + source_block
        + predicted_block
    )


def _records_html(records: list[dict[str, Any]]) -> str:
    header = (
        "<tr><th>#</th><th>sample_id</th><th>source_task</th>"
        "<th>mapped_expected</th><th>predicted_task</th><th>match</th>"
        "<th>confidence</th><th>source</th><th>reason_codes</th>"
        "<th>question</th></tr>"
    )
    body = "".join(_record_row(record) for record in records)
    return f"<h2>Records</h2><table>{header}{body}</table>"


def _record_row(record: dict[str, Any]) -> str:
    match = "yes" if record["match"] else "no"
    match_class = "match-yes" if record["match"] else "match-no"
    error = record.get("error")
    error_text = f" ({_esc(error)})" if error else ""
    return (
        "<tr>"
        f"<td>{record['index']}</td>"
        f"<td>{_esc(record['sample_id'])}</td>"
        f"<td>{_esc(str(record['source_task'] or ''))}</td>"
        f"<td>{_esc(str(record['mapped_expected_task'] or ''))}</td>"
        f"<td>{_esc(str(record['predicted_task'] or ''))}{error_text}</td>"
        f'<td class="{match_class}">{match}</td>'
        f"<td>{_esc(_format_confidence(record['confidence']))}</td>"
        f"<td>{_esc(record['resolution_source'] or '—')}</td>"
        f"<td>{_esc(', '.join(record['reason_codes']))}</td>"
        f"<td>{_esc(_truncate(record['question'], QUESTION_TRUNCATE))}</td>"
        "</tr>"
    )


def _format_confidence(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.2f}"
    return "—"


def _format_rate(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.4f}"
    return "—"


def _truncate(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[: max(0, max_length - 3)] + "..."


def _esc(value: str) -> str:
    """HTML-escape dynamic text; never trust raw input or model output.
    HTML 转义动态文本；绝不信任原始输入或模型输出。"""

    return html.escape(str(value), quote=True)


def _atomic_write_json(path: Path, value: Any) -> None:
    """Atomic JSON write via a temporary file and replace (stdlib only).
    经临时文件与 replace 的原子 JSON 写入（仅 stdlib）。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    """Atomic text write via a temporary file and replace.
    经临时文件与 replace 的原子文本写入。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return asyncio.run(run_classification(args))
    except KeyboardInterrupt:
        print("task classification interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        # Public failure output never carries raw exception text or secrets.
        # 公共失败输出绝不携带原始异常文本或密钥。
        print(
            f"task classification failed: {type(error).__name__}",
            file=sys.stderr,
        )
        return 1


_CSS = """
body { font-family: system-ui, sans-serif; margin: 1.5rem; color: #1a1a1a; }
table { border-collapse: collapse; margin-bottom: 1.5rem; }
th, td { border: 1px solid #ccc; padding: 0.3rem 0.6rem; text-align: left; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; } h3 { font-size: 0.95rem; }
td.match-yes { color: #1a7f37; }
td.match-no { color: #b35900; }
"""


if __name__ == "__main__":
    raise SystemExit(main())
