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
import math
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from agents.counting.schema import CountingExecutionAudit, CountingResult
from agents.schema import AgentResult
from data.schema import UnifiedSample
from evaluation.records import (
    EvaluationRecord,
    evaluation_filename_for_runtime_task,
    evaluation_task_for_runtime_task,
)
from reporting.schema import ModelCallAuditView, RunMetadata, StructuredArtifactView
from routing.schema import RoutingDecision
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


def load_counting_attempts(sample_dir: Path) -> CountingExecutionAudit | None:
    """Load an optional ordered counting audit without failing old reports."""
    raw = read_json(sample_dir / "counting_attempts.json")
    if not isinstance(raw, dict):
        return None
    try:
        return CountingExecutionAudit.model_validate(raw)
    except ValueError:
        return None


def load_routing_decision(sample_dir: Path) -> dict[str, Any] | None:
    """Read the typed routing artifact without exposing arbitrary JSON."""
    raw = read_json(sample_dir / "routing_decision.json")
    if not isinstance(raw, dict):
        return None
    try:
        validated = RoutingDecision.model_validate(raw).model_dump(mode="json")
        return {
            str(key): value for key, value in validated.items()
            if _routing_value_is_safe(value)
        }
    except ValueError:
        allowed = {
            "source_task", "resolved_task", "task", "router_used", "confidence",
            "reason", "evidence", "reason_codes", "primary_agent",
            "fallback_agents", "execution_mode", "requires_tiling",
        }
        projected = {
            str(key): value
            for key, value in raw.items()
            if str(key) in allowed and _routing_value_is_safe(value)
        }
        return projected or None


def _routing_value_is_safe(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return not isinstance(value, str) or _safe_meta_text(value) is not None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    if isinstance(value, list) and len(value) <= 20:
        return all(_routing_value_is_safe(item) and not isinstance(item, list) for item in value)
    return False


_MODEL_CALL_RAW_LIMIT = 8000
_UNSAFE_ARTIFACT_TEXT_RE = re.compile(
    r"(?i)(?:data:image/[^;]+;base64,|\b(?:bearer\s+|sk-[a-z0-9_-]{6,})|"
    r"[\"']?(?:api[_-]?key|authorization|access_token)[\"']?\s*:|"
    r"(?:^|[^a-z0-9])(?:[a-z]:[\\/]|/(?:home|tmp|users|private|var)/))"
)
_UNSAFE_ARTIFACT_KEYS = {
    "api_key", "apikey", "authorization", "auth_header", "access_token",
    "refresh_token", "password", "secret", "artifact_dir", "dataset_root",
    "checkpoint", "checkpoint_path", "weights", "weights_path",
}


def _inside(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root)
    except OSError:
        return False


def _read_json_inside(path: Path, root: Path) -> Any | None:
    return read_json(path) if _inside(path, root) else None


def _read_text_inside(path: Path, root: Path) -> str | None:
    if not _inside(path, root):
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _safe_meta_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value or _unsafe_artifact_text(value):
        return None
    return value


def _safe_artifact_text(value: str | None) -> str | None:
    if value is None or _unsafe_artifact_text(value):
        return None
    return value


def _unsafe_artifact_text(value: str) -> bool:
    if _UNSAFE_ARTIFACT_TEXT_RE.search(value):
        return True
    for match in re.finditer(r"[A-Za-z0-9+/]{200,}={0,2}", value):
        if re.search(r"[0-9+/]", match.group(0)):
            return True
    return False


def _safe_json(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return None
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else None
    if isinstance(value, str):
        return _safe_artifact_text(value)
    if isinstance(value, list):
        return [_safe_json(item, depth=depth + 1) for item in value[:100]]
    if isinstance(value, dict):
        return {
            str(key): _safe_json(item, depth=depth + 1)
            for key, item in value.items()
            if str(key).casefold() not in _UNSAFE_ARTIFACT_KEYS
        }
    return None


def _format_safe_json(value: Any) -> str | None:
    safe = _safe_json(value)
    if safe is None:
        return None
    try:
        return json.dumps(safe, ensure_ascii=False, indent=2, sort_keys=True)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _token_usage(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    result = {
        str(key): int(item)
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, int)
        and not isinstance(item, bool) and item >= 0
    }
    return result or None


def load_model_calls(sample_dir: Path) -> list[ModelCallAuditView]:
    """Discover sanitized model-call artifacts below ``sample_dir`` only."""
    try:
        root = sample_dir.resolve()
        candidates = sorted(
            (path for path in root.rglob("request_meta.json") if _inside(path, root)),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    except (OSError, ValueError):
        return []
    result: list[ModelCallAuditView] = []
    for meta_path in candidates:
        raw_meta = read_json(meta_path)
        if not isinstance(raw_meta, dict):
            continue
        request_id = _safe_meta_text(raw_meta.get("request_id"))
        prompt_version = _safe_meta_text(raw_meta.get("prompt_version"))
        if not request_id or not prompt_version:
            continue
        directory = meta_path.parent
        request = _read_json_inside(directory / "request.json", root)
        parsed = _read_json_inside(directory / "parsed.json", root)
        validation = _read_json_inside(directory / "validation.json", root)
        raw_response = _read_text_inside(directory / "raw_response.txt", root)
        raw_view = _safe_artifact_text(raw_response)
        raw_truncated = bool(raw_response is not None and len(raw_response) > _MODEL_CALL_RAW_LIMIT)
        if raw_view is not None and raw_truncated:
            marker = "\n[truncated]"
            raw_view = raw_view[:_MODEL_CALL_RAW_LIMIT - len(marker)] + marker
        parsed_view = _format_safe_json(parsed)
        request_view = _format_safe_json(request)
        metadata = validation.get("response_metadata") if isinstance(validation, dict) else None
        if not isinstance(metadata, dict):
            metadata = validation if isinstance(validation, dict) else {}
        result.append(ModelCallAuditView(
            request_id=request_id,
            prompt_version=prompt_version,
            request_hash=_safe_meta_text(raw_meta.get("request_hash")),
            sample_id=_safe_meta_text(raw_meta.get("sample_id")),
            tile_id=_safe_meta_text(raw_meta.get("tile_id")),
            image_sha256=_safe_meta_text(raw_meta.get("image_sha256")),
            cache_hit=_bool_or_none(validation.get("cache_hit")) if isinstance(validation, dict) else None,
            valid=_bool_or_none(validation.get("valid")) if isinstance(validation, dict) else None,
            repair_used=_bool_or_none(metadata.get("repair_used")),
            latency_seconds=_finite_float(metadata.get("latency_seconds")),
            token_usage=_token_usage(metadata.get("token_usage")),
            raw_response=raw_view,
            raw_response_truncated=raw_truncated,
            parsed_response=parsed_view,
            request_summary=request_view,
        ))
    return result


load_model_call_artifacts = load_model_calls


_STRUCTURED_ARTIFACT_FILENAMES = (
    "vqa_evidence.json",
    "grounding_evidence.json",
    "visual_plan.json",
    "joint_visual_plan.json",
)


def load_structured_artifacts(sample_dir: Path) -> list[StructuredArtifactView]:
    """Load allowlisted structured submodel artifacts for the HTML audit.
    为 HTML 审计加载允许列表内的结构化子模型产物。

    The payload is passed through the same depth/secret/path sanitizer as
    model-call views. This exposes detector/evidence state without granting
    reporting arbitrary file access or persisting credentials.
    载荷复用模型调用视图的深度/敏感信息/路径清洗；既展示检测器/证据状态，
    又不向 reporting 开放任意文件读取或持久化凭据。
    """

    result: list[StructuredArtifactView] = []
    for filename in _STRUCTURED_ARTIFACT_FILENAMES:
        raw = read_json(sample_dir / filename)
        if raw is None:
            continue
        safe = _safe_json(raw)
        if safe is None:
            continue
        try:
            result.append(StructuredArtifactView(filename=filename, payload=safe))
        except ValueError:
            continue
    return result


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
