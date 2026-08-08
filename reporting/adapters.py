"""Run-artifact adapters: read the persisted execution index and sample-level
artifacts into report inputs. Read-only and best-effort: corrupt or missing
optional artifacts degrade to None, never raise. The reporting layer never
calls a model and never recomputes model results.

运行产物适配器：把持久化执行索引与样本级产物读入报告输入。只读且尽力而为：
损坏或缺失的可选产物降级为 None，绝不抛出。报告层绝不调用模型、绝不重新
计算模型结果。
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from agents.counting.schema import CountingResult
from agents.schema import AgentResult
from data.schema import UnifiedSample
from evaluation.records import EvaluationRecord
from workflows.schema import SampleRunStatus

# Mirrors the runtime contract in workflows.sample_runner; the architecture
# rule forbids reporting from importing that module, so the mapping is
# maintained locally. 镜像 workflows.sample_runner 中的运行时契约；架构规则
# 禁止 reporting 导入该模块，因此映射在本模块本地维护。
_VQA_EVALUATION_FILENAME = "vqa_evaluation.json"
_COUNTING_EVALUATION_FILENAME = "counting_evaluation.json"
_GROUNDING_EVALUATION_FILENAME = "grounding_evaluation.json"
_CAPTION_EVALUATION_FILENAME = "caption_evaluation.json"

_VQA_TASKS = frozenset({"general_vqa", "multiple_choice_vqa", "scene_classification"})
_COUNTING_TASKS = frozenset({"counting", "fine_grained_counting"})


def evaluation_filename_for_task(task: str) -> str | None:
    """Sample-level deterministic evaluation artifact for a task; None when
    the task has no wired sample-level metric. 任务的样本级确定性评估产物名；
    无已接线样本级指标时返回 None。"""

    if task in _VQA_TASKS:
        return _VQA_EVALUATION_FILENAME
    if task in _COUNTING_TASKS:
        return _COUNTING_EVALUATION_FILENAME
    if task == "grounding":
        return _GROUNDING_EVALUATION_FILENAME
    if task == "caption":
        return _CAPTION_EVALUATION_FILENAME
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

    filename = evaluation_filename_for_task(task)
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

    if task in _COUNTING_TASKS:
        path = sample_dir / "counting_result.json"
        model = CountingResult
    else:
        path = sample_dir / "agent_result.json"
        model = AgentResult
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


def sample_dir_for_row(run_dir: Path, row: Mapping[str, Any]) -> Path | None:
    """Derive the sample directory from the run-relative result path; rows
    without a result path cannot be artifact-enriched. 从 run 相对结果路径
    推导样本目录；无结果路径的行无法做产物增强。"""

    result_path = row.get("result_path")
    if not isinstance(result_path, str) or not result_path:
        return None
    candidate = (run_dir / result_path).parent
    try:
        return candidate if candidate.is_relative_to(run_dir) else None
    except ValueError:
        return None
