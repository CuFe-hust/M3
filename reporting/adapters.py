"""Run-artifact adapters: read the persisted execution index and sample-level
artifacts into report inputs. Read-only and best-effort: corrupt or missing
optional artifacts degrade to None, never raise. The reporting layer never
calls a model and never recomputes model results.

运行产物适配器：把持久化执行索引与样本级产物读入报告输入。只读且尽力而为：
损坏或缺失的可选产物降级为 None，绝不抛出。报告层绝不调用模型、绝不重新
计算模型结果。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from agents.counting.schema import CountingResult
from agents.schema import AgentResult
from data.schema import UnifiedSample
from evaluation.records import (
    EvaluationRecord,
    evaluation_filename_for_runtime_task,
    evaluation_task_for_runtime_task,
)
from reporting.schema import RunMetadata
from workflows.schema import RunRequest, SampleRunStatus


def load_run_manifest(run_dir: Path) -> RunMetadata | None:
    """Load the typed, allowlisted reproducibility manifest only."""

    raw = read_json(run_dir / "manifest.json")
    if not isinstance(raw, dict):
        return None
    try:
        return RunMetadata.model_validate(raw)
    except ValueError:
        return None


def load_run_request(run_dir: Path) -> RunRequest | None:
    """Load the private materialization context.

    The returned dataset root is intentionally never copied into a report
    model; it may only be consumed internally by visualization materializers.
    """

    raw = read_json(run_dir / "run_request.json")
    if not isinstance(raw, dict):
        return None
    try:
        return RunRequest.model_validate(raw)
    except ValueError:
        return None


def read_json(path: Path) -> Any | None:
    """Read one JSON artifact; unreadable or unparseable files return None and
    never raise raw errors into the report. 读取一份 JSON 产物；不可读或无法
    解析的文件返回 None，绝不向报告抛原始错误。"""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def iter_current_predictions(run_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield the current-state execution-index rows: append-only history is
    collapsed to the last row per (run_task, sample_id), preserving the first
    appearance order. 产出当前状态执行索引行：append-only 历史按 (run_task,
    sample_id) 收敛到最后一行，保留首次出现顺序。"""

    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.is_file():
        return
    current: dict[tuple[str, str], dict[str, Any]] = {}
    try:
        lines = predictions_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # corrupt index line is skipped, never fatal
        if not isinstance(row, dict):
            continue
        sample_id = row.get("sample_id")
        run_task = row.get("run_task")
        if not isinstance(sample_id, str) or not isinstance(run_task, str):
            continue
        current[(run_task, sample_id)] = row
    for row in current.values():
        yield row


def load_status(sample_dir: Path) -> SampleRunStatus | None:
    """Load the persisted sample status; corrupt or schema-invalid files
    return None (the execution-index row remains authoritative).
    读取持久化样本状态；损坏或 schema 非法文件返回 None（执行索引行仍为
    权威来源）。"""

    raw = read_json(sample_dir / "status.json")
    if not isinstance(raw, dict):
        return None
    try:
        return SampleRunStatus.model_validate(raw)
    except ValueError:
        return None


def load_sample(sample_dir: Path) -> UnifiedSample | None:
    """Load the canonical persisted sample; corrupt files return None.
    读取 canonical 持久化样本；损坏文件返回 None。"""

    raw = read_json(sample_dir / "sample.json")
    if not isinstance(raw, dict):
        return None
    try:
        return UnifiedSample.model_validate(raw)
    except ValueError:
        return None


def load_trace(sample_dir: Path) -> dict[str, Any] | None:
    """Load the agent trace; corrupt files return None. 读取 agent trace；
    损坏文件返回 None。"""

    raw = read_json(sample_dir / "agent_trace.json")
    return raw if isinstance(raw, dict) else None


def load_evaluation(sample_dir: Path, task: str) -> EvaluationRecord | None:
    """Load the sample-level deterministic evaluation for the execution task;
    missing or corrupt files return None. 按执行任务读取样本级确定性评估；
    缺失或损坏文件返回 None。"""

    filename = evaluation_filename_for_runtime_task(task)
    if filename is None:
        return None
    raw = read_json(sample_dir / filename)
    if not isinstance(raw, dict):
        return None
    try:
        return EvaluationRecord.model_validate(raw)
    except ValueError:
        return None


def load_payload(sample_dir: Path, task: str) -> object | None:
    """Load the persisted execution payload (counting tasks read
    counting_result.json, other evaluated tasks read agent_result.json);
    missing or corrupt files return None. 读取持久化执行载荷（计数任务读
    counting_result.json，其余已评估任务读 agent_result.json）；缺失或损坏
    返回 None。"""

    family = evaluation_task_for_runtime_task(task)
    if family == "counting":
        path = sample_dir / "counting_result.json"
        model = CountingResult
    elif family is not None:
        path = sample_dir / "agent_result.json"
        model = AgentResult
    else:
        return None
    raw = read_json(path)
    if not isinstance(raw, dict):
        return None
    try:
        return model.model_validate(raw)
    except ValueError:
        return None


def prediction_text(payload: object | None) -> str | None:
    """A short human-readable prediction: the counting final count or the
    Agent answer. 简短可读预测：计数最终数量或 Agent 答案。"""

    if payload is None:
        return None
    if isinstance(payload, CountingResult):
        return str(payload.final_count)
    return str(getattr(payload, "answer", None) or "").strip() or None


def sample_dir_for_row(
    run_dir: Path,
    row: Mapping[str, Any],
) -> Path | None:
    """Derive the sample directory from the frozen storage identity
    (run_task, sample_id) — never from result_path, which is display-only and
    may be absent or corrupt. 从冻结存储身份（run_task, sample_id）推导样本
    目录——绝不使用仅用于展示且可能缺失/损坏的 result_path。"""

    run_task = row.get("run_task")
    sample_id = row.get("sample_id")
    if not isinstance(run_task, str) or not _safe_run_task(run_task):
        return None
    if not isinstance(sample_id, str) or not sample_id:
        return None
    key = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]
    return run_dir / "tasks" / run_task / "samples" / key


def _safe_run_task(run_task: str) -> bool:
    """A run-task namespace must be a plain directory name: separators, dot
    segments, drive prefixes, UNC, and control characters are rejected.
    run-task 命名空间必须是纯目录名：分隔符、dot 段、drive 前缀、UNC 与
    控制字符一律拒绝。"""

    if not run_task or run_task in {".", ".."}:
        return False
    if "/" in run_task or "\\" in run_task:
        return False
    if any(ord(character) < 32 for character in run_task):
        return False
    if len(run_task) >= 2 and run_task[0].isalpha() and run_task[1] == ":":
        return False
    return True


def safe_result_path(run_dir: Path, value: Any) -> str | None:
    """Fail-closed display path: keep only run-relative values whose canonical
    resolution stays inside the run directory; corrupt index entries degrade
    to None without failing the report. 展示路径 fail-closed：只保留 run
    相对且 canonical 解析后仍在 run 目录内的值；损坏索引条目降级为 None，
    不使报告失败。"""

    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("\\", "/")
    if normalized.startswith("/"):
        return None
    if len(normalized) >= 2 and normalized[0].isalpha() and normalized[1] == ":":
        return None
    if any(segment in ("", ".", "..") for segment in normalized.split("/")):
        return None
    candidate = run_dir / value
    try:
        if not candidate.resolve().is_relative_to(run_dir.resolve()):
            return None
    except OSError:
        return None
    return normalized
