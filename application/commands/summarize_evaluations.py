"""Public `summarize-evaluations` CLI command: deterministic evaluation
aggregation for one run.

公开 `summarize-evaluations` CLI 命令：单个 run 的确定性评估聚合。解析
run 内全部当前 EvaluationRecord（含 legacy VQAEvaluationRecord 兼容形状）；
任何损坏记录稳定失败；按 task 同构分组后经 evaluation.metrics.aggregate
确定性聚合；纯本地计算，无模型调用。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from application.settings import load_settings
from evaluation.metrics.aggregate import aggregate
from evaluation.records import EvaluationRecord, VQAEvaluationRecord

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_INTERRUPTED = 130

# Evaluation artifact names produced by the shared deterministic dispatch.
# 共享确定性分派产出的评估产物名。
_KNOWN_EVALUATION_FILENAMES = (
    "vqa_evaluation.json",
    "counting_evaluation.json",
    "grounding_evaluation.json",
    "caption_evaluation.json",
)


def run_summarize_evaluations(args: argparse.Namespace) -> int:
    """Aggregate every evaluation record of one run and print the result.
    聚合一个 run 的全部评估记录并输出结果。"""

    try:
        settings = load_settings(
            Path(args.config) if getattr(args, "config", None) else None,
        )
        run_dir = settings.runs.root / args.run_id
        if not run_dir.is_dir():
            raise ValueError("run does not exist")
        records = _collect_evaluation_records(run_dir)
        if not records:
            raise ValueError("no evaluation records found")
        groups: dict[str, list[Any]] = {}
        for record in records:
            task = (
                record.task
                if isinstance(record, EvaluationRecord)
                else "general_vqa"
            )
            groups.setdefault(task, []).append(record)
        aggregates = {
            task: aggregate(group)
            for task, group in sorted(groups.items())
        }
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    except Exception as error:
        print(
            json.dumps({"status": "failed", "error": f"{type(error).__name__}"}),
            file=sys.stderr,
        )
        return EXIT_RUNTIME
    print(
        json.dumps(
            {
                "status": "ok",
                "run_id": args.run_id,
                "record_count": len(records),
                "aggregates": aggregates,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return EXIT_OK


def _collect_evaluation_records(run_dir: Path) -> list[Any]:
    """Parse every current evaluation artifact under tasks/*/samples/*/; a
    malformed file fails stably instead of being skipped.
    解析 tasks/*/samples/*/ 下全部当前评估产物；损坏文件稳定失败而非跳过。"""

    records: list[Any] = []
    tasks_dir = run_dir / "tasks"
    if not tasks_dir.is_dir():
        return records
    for task_dir in sorted(tasks_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        samples_root = task_dir / "samples"
        if not samples_root.is_dir():
            continue
        for sample_dir in sorted(samples_root.iterdir()):
            if not sample_dir.is_dir():
                continue
            for name in _KNOWN_EVALUATION_FILENAMES:
                path = sample_dir / name
                if not path.is_file():
                    continue
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"evaluation record is malformed: {name}") from exc
                record = _parse_record(raw)
                if record is not None:
                    records.append(record)
    return records


def _parse_record(raw: Any) -> Any | None:
    """Validate one evaluation artifact as the unified EvaluationRecord or
    the legacy VQA wrapper; anything else fails stably.
    将一份评估产物校验为统一 EvaluationRecord 或 legacy VQA 包装；其他
    形状稳定失败。"""

    try:
        return EvaluationRecord.model_validate(raw)
    except ValueError:
        pass
    try:
        return VQAEvaluationRecord.model_validate(raw)
    except ValueError as exc:
        raise ValueError("evaluation record is invalid") from exc
