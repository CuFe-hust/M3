"""Deterministic JSON and CSV exporters for the report model.

报告模型的确定性 JSON 与 CSV 导出。JSON 为稳定布局（sort_keys、indent）；
CSV 使用 utf-8-sig（Windows Excel 兼容）且只含稳定字段与 run 相对路径。

本模块同时承载基准/审计导出能力：逐样本 samples.jsonl、DeepSeek 审计
JSONL（只含稳定 hash/request/status/parsed 元数据，绝无 auth/原始 secret）、
run 元数据 JSON（无主机绝对路径）、外部标准指标（external_standard 命名
空间，绝不并入确定性指标名）与 MME-RealWorld 官方提交导出（源记录只读，
未关联字段原样保留）。新统一 Reporting 始终权威；不重建旧 HTML builder。
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from reporting.schema import Report, ReportSample

# Stable schema version for the metadata export. / 元数据导出的稳定 schema 版本。
REPORT_SCHEMA_VERSION = "report-v1"

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

_MME_CONTAINER_KEYS = ("samples", "data", "annotations", "items", "images")
_MME_QUESTION_ID_KEYS = ("Question_id", "question_id")

# Counting task family for judge artifact directory selection.
# 用于 judge 产物目录选择的计数任务族。
_COUNTING_TASKS = frozenset({"counting", "fine_grained_counting"})


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


# ── benchmark / audit exports / 基准与审计导出 ─────────────────────────────


def write_samples_jsonl(report: Report, path: Path) -> Path:
    """Write one deterministic JSON row per current sample (sorted keys,
    UTF-8). 为每个当前样本写一行确定性 JSON（排序键、UTF-8）。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for sample in report.samples:
            handle.write(
                json.dumps(
                    sample.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
                )
                + "\n"
            )
    return path


def write_deepseek_audit(
    report: Report,
    path: Path,
    *,
    run_dir: Path | None = None,
) -> Path:
    """Write one auditable DeepSeek record per judged sample. Rows carry only
    stable metadata — request id/hash/prompt version from the actual persisted
    RequestMeta (never synthesized from the verdict), judge status, stable
    error, and the validated structured judge_parsed output. Never auth keys,
    never raw responses or secrets; missing RequestMeta yields null identity
    fields instead of fabricated values.
    为每个被 judge 的样本写一条可审计 DeepSeek 记录。行只携带稳定元数据——
    来自实际持久化 RequestMeta 的 request id/hash/prompt version（绝不从
    判决合成）、judge 状态、稳定错误与校验后的结构化 judge_parsed。绝不
    输出 auth 键、原始响应或密钥；缺失 RequestMeta 时身份字段输出 null
    而非伪造值。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for sample in report.samples:
            row = _deepseek_audit_row(sample, run_dir=run_dir)
            if row is not None:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def _deepseek_audit_row(
    sample: ReportSample,
    *,
    run_dir: Path | None,
) -> dict[str, Any] | None:
    """Build one stable audit row; samples without a judge pass yield None.
    Request identity comes from the persisted RequestMeta of the matching
    judge artifact directory; without it the identity fields stay null.
    构建一条稳定审计行；未参与 judge 的样本返回 None。请求身份来自匹配
    judge 产物目录中的持久化 RequestMeta；缺失时身份字段保持 null。"""

    evaluation = sample.evaluation
    if evaluation is None or evaluation.judge_status == "not_requested":
        return None
    request_meta = _load_request_meta(sample, run_dir) if run_dir is not None else None
    return {
        "sample_id": sample.sample_id,
        "task": sample.task,
        "request_id": (
            request_meta.get("request_id") if isinstance(request_meta, dict) else None
        ),
        "request_hash": (
            request_meta.get("request_hash")
            if isinstance(request_meta, dict)
            else None
        ),
        "prompt_version": (
            request_meta.get("prompt_version")
            if isinstance(request_meta, dict)
            else None
        ),
        "judge_status": evaluation.judge_status,
        "judge_error": evaluation.judge_error,
        "judge_parsed": _json_value(evaluation.judge_parsed),
    }


def _load_request_meta(
    sample: ReportSample,
    run_dir: Path,
) -> dict[str, Any] | None:
    """Read the actual persisted RequestMeta for the sample's judge artifact
    directory. The sample directory is derived from the frozen storage
    identity inside the run; arbitrary paths from model/user output are never
    trusted. 读取样本 judge 产物目录中的实际持久化 RequestMeta。样本目录
    由 run 内冻结存储身份推导；绝不信任来自模型/用户输出的任意路径。"""

    from reporting.adapters import sample_dir_for_row

    sample_dir = sample_dir_for_row(
        run_dir,
        {"run_task": sample.run_task, "sample_id": sample.sample_id},
    )
    if sample_dir is None:
        return None
    task = sample.task
    if task in _COUNTING_TASKS:
        meta_path = sample_dir / "deepseek" / "request_meta.json"
    else:
        meta_path = sample_dir / "deepseek_vqa_judge" / "request_meta.json"
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def write_metadata_json(
    report: Report,
    path: Path,
    *,
    run_dir: Path | None = None,
) -> Path:
    """Write run metadata JSON: run id, dataset, split, model ids, counts,
    already-persisted manifest timestamp, and the report schema version.
    Never host absolute paths. When run_dir is given, split/model ids and the
    manifest created_at are recovered read-only from the persisted manifest.
    写出 run 元数据 JSON：run id、dataset、split、model ids、计数、已持久化
    的 manifest 时间戳与报告 schema 版本。绝不包含主机绝对路径。提供
    run_dir 时从持久化 manifest 只读恢复 split/model ids 与 created_at。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _read_run_manifest(run_dir) if run_dir is not None else None
    metadata: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": report.run_id,
        "dataset": report.dataset,
        "split": (manifest or {}).get("split"),
        "model_ids": (manifest or {}).get("model_ids"),
        "counts": {
            "total": report.total,
            "succeeded": report.succeeded,
            "partial": report.partial,
            "failed": report.failed,
            "skipped": report.skipped,
        },
        "created_at": (manifest or {}).get("created_at"),
        "sample_count": len(report.samples),
    }
    path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _read_run_manifest(run_dir: Path) -> dict[str, Any] | None:
    """Read the persisted run manifest read-only; corrupt or missing files
    yield None (never fatal). 只读已持久化的 run manifest；损坏或缺失返回
    None（绝不致命）。"""

    manifest_path = run_dir / "manifest.json"
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def write_external_standard_report(
    standard_report: Mapping[str, Any],
    path: Path,
) -> Path:
    """Persist the external standard-evaluator report under the
    ``external_standard`` namespace — never merged into deterministic metric
    names. 将外部标准评估器报告持久化在 ``external_standard`` 命名空间下
    ——绝不并入确定性指标名。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "external_standard": dict(standard_report),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def persist_report_bundle(
    run_dir: Path,
    report: Report,
    *,
    external_standard: Mapping[str, Any] | None = None,
) -> Path:
    """Persist the unified current-generation report bundle under
    ``runs/<run_id>/report/``: report.html, report.json, samples.csv,
    samples.jsonl, metadata.json, deepseek_audit.jsonl (when judge records
    exist) and external_standard.json (when provided). The report builder
    stays read-only with respect to execution artifacts; only the report
    output directory is written. 将统一当前代报告 bundle 持久化到
    ``runs/<run_id>/report/``：report.html、report.json、samples.csv、
    samples.jsonl、metadata.json、deepseek_audit.jsonl（存在 judge 记录时）
    与 external_standard.json（提供时）。报告构建器对执行产物保持只读；
    只写报告输出目录。"""

    from reporting.html import build_html

    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report, report_dir / "report.json")
    (report_dir / "report.html").write_text(
        build_html(report) + "\n", encoding="utf-8"
    )
    write_csv(report, report_dir / "samples.csv")
    write_samples_jsonl(report, report_dir / "samples.jsonl")
    write_metadata_json(report, report_dir / "metadata.json", run_dir=run_dir)
    write_deepseek_audit(
        report, report_dir / "deepseek_audit.jsonl", run_dir=run_dir
    )
    if external_standard is not None:
        write_external_standard_report(
            external_standard, report_dir / "external_standard.json"
        )
    return report_dir


def write_mme_official_export(
    source_path: Path,
    predictions: Mapping[str, str],
    output_path: Path,
) -> Path:
    """Build the MME-RealWorld official submission: read the original records
    read-only, map represented predictions (keyed by question id) into the
    official ``Output`` field, and preserve every unrelated field exactly.
    The source file is never mutated; a missing source fails stably.
    构建 MME-RealWorld 官方提交：只读原始记录，将已有预测（按 question id
    键）映射到官方 ``Output`` 字段，未关联字段原样保留。源文件绝不修改；
    源缺失稳定失败。"""

    if not source_path.is_file():
        raise FileNotFoundError("MME source file does not exist")
    rows = _read_mme_rows(source_path)
    exported: list[dict[str, Any]] = []
    for row in rows:
        exported_row = dict(row)  # unrelated fields preserved exactly
        question_id = _mme_question_id(row)
        exported_row["Output"] = predictions.get(question_id, "") if question_id else ""
        exported.append(exported_row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(exported, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def _read_mme_rows(path: Path) -> list[dict[str, Any]]:
    """Parse the MME annotation file read-only: JSONL, a top-level list, or a
    single supported record container. Unsupported shapes fail stably.
    只读解析 MME 标注文件：JSONL、顶层 list 或单一受支持容器。不支持形状
    稳定失败。"""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("MME source file is invalid JSON") from exc
    if isinstance(raw, list):
        rows = raw
    elif isinstance(raw, dict):
        container = [
            raw[key] for key in _MME_CONTAINER_KEYS if isinstance(raw.get(key), list)
        ]
        if len(container) != 1:
            raise ValueError("MME source must be a JSON list or one record container")
        rows = container[0]
    else:
        raise ValueError("MME source must be a JSON list or object")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("MME source records must be JSON objects")
    return rows


def _mme_question_id(row: dict[str, Any]) -> str | None:
    """Return the record question id under any supported spelling.
    返回任意受支持写法下的记录 question id。"""

    for key in _MME_QUESTION_ID_KEYS:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _json_value(value: Any) -> Any:
    """Normalize a pydantic model or plain value into JSON-safe data.
    将 pydantic 模型或普通值规范化为 JSON 安全数据。"""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value
