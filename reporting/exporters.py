"""Deterministic JSON and CSV exporters for the report model.

报告模型的确定性 JSON 与 CSV 导出。JSON 为稳定布局（sort_keys、indent）；
CSV 使用 utf-8-sig（Windows Excel 兼容）且只含稳定字段与 run 相对路径。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from reporting.schema import Report

_CSV_COLUMNS = (
    "run_task",
    "sample_id",
    "task",
    "state",
    "error_code",
    "fallback_used",
    "judge_status",
    "execution_agent",
    "inference_seconds",
    "prediction",
    "result_path",
    "updated_at",
)


def write_json(report: Report, path: Path) -> Path:
    """Write the report as stable-layout JSON. 以稳定布局写出 JSON 报告。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def write_csv(report: Report, path: Path) -> Path:
    """Write the per-sample CSV with utf-8-sig encoding; only stable fields
    and run-relative paths. 以 utf-8-sig 编码写出逐样本 CSV；只含稳定字段与
    run 相对路径。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(_CSV_COLUMNS)
        for sample in report.samples:
            writer.writerow(
                [
                    sample.run_task,
                    sample.sample_id,
                    sample.task,
                    sample.state,
                    sample.error_code or "",
                    "yes" if sample.fallback_used else "no",
                    sample.judge_status,
                    sample.execution_agent or "",
                    sample.inference_seconds if sample.inference_seconds is not None else "",
                    sample.prediction or "",
                    sample.result_path or "",
                    sample.updated_at or "",
                ]
            )
    return path
