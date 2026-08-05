"""Bridge persisted counting artifacts into the shared HTML audit report.
将持久化计数产物桥接到共享 HTML 审计报告。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.schema import CanonicalPrediction, CanonicalSample
from eval.audit_report import AuditReportWriter, build_audit_report
from spacers_agent.schemas import CountingResult, UnifiedSample
from spacers_agent.settings import QwenSettings


def build_multiagent_counting_report(
    run_dir: Path,
    *,
    qwen: QwenSettings,
    model_load_seconds: float = 0.0,
    max_samples: int = 200,
) -> Path | None:
    """Build a counting-only audit report without changing evaluation metrics.
    构建仅计数的审计报告，而不改变评估指标。
    """
    sample_dirs, state_counts = _counting_sample_dirs(run_dir)
    if not sample_dirs:
        return None
    result_path = run_dir / "counting.jsonl"
    lines: list[str] = []
    with AuditReportWriter(result_path, max_samples=max_samples) as writer:
        for sample_dir in sample_dirs:
            sample = UnifiedSample.model_validate_json((sample_dir / "sample.json").read_text(encoding="utf-8"))
            result = CountingResult.model_validate_json((sample_dir / "counting_result.json").read_text(encoding="utf-8"))
            trace = _read_json(sample_dir / "agent_trace.json")
            references = _references(sample)
            canonical_sample = CanonicalSample(
                id=sample.sample_id, task_type="counting", images=[image.path for image in sample.images],
                prompt=sample.question, answers=references,
                meta={"source": sample.dataset, "question": sample.question, "original_task": sample.task},
            )
            prediction = CanonicalPrediction(
                id=sample.sample_id, task_type="counting", text=str(result.final_count), answer=str(result.final_count),
                meta={"raw_text": str(result.final_count), "counting_status": result.status, "target": result.target,
                      "accepted_count": result.final_count, "warning_codes": [warning.code for warning in result.warnings]},
            )
            lines.append(json.dumps({"sample": canonical_sample.serializable(), "prediction": prediction.serializable()}, ensure_ascii=False))
            evidence = [
                {"label": point.target, "point": [point.global_x_norm, point.global_y_norm], "confidence": point.confidence,
                 "image_id": sample.images[0].image_id, "coordinate_frame": "normalized_0_999_top_left"}
                for point in result.global_points if point.accepted
            ]
            writer.capture(canonical_sample, prediction, float(trace.get("inference_seconds", 0.0) or 0.0), agent_trace=trace, visual_evidence=evidence)
    result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    metadata = {
        "dataset": "counting", "completed_samples": len(sample_dirs), "partial_samples": state_counts.get("partial", 0),
        "failed_samples": state_counts.get("failed", 0),
        "model": {"id": qwen.model, "backend": "transformers", "dtype": qwen.dtype, "max_new_tokens": qwen.max_tokens,
                  "local_files_only": qwen.local_files_only, "yolo_enabled": True},
        "model_load_seconds": model_load_seconds,
        "pipeline": ["DatasetAdapter", "SampleRunner", "CountingAgent", "BackendSelector", "YoloOBBCountingBackend | QwenPointCountingBackend", "AuditReportWriter"],
    }
    result_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return build_audit_report(result_path)


def _counting_sample_dirs(run_dir: Path) -> tuple[list[Path], dict[str, int]]:
    """Return visible successful or partial counting samples and every state count.
    返回可见的成功或部分计数样本以及所有状态计数。
    """
    directories: list[Path] = []
    states: dict[str, int] = {}
    for status_path in (run_dir / "samples").glob("*/status.json"):
        status = _read_json(status_path)
        state = str(status.get("state", "unknown"))
        states[state] = states.get(state, 0) + 1
        directory = status_path.parent
        sample_path = directory / "sample.json"
        if state in {"succeeded", "partial"} and sample_path.is_file() and (directory / "counting_result.json").is_file():
            sample = _read_json(sample_path)
            if sample.get("task") in {"counting", "fine_grained_counting"}:
                directories.append(directory)
    return sorted(directories, key=lambda path: (0, int(path.name)) if path.name.isdigit() else (1, path.name)), states


def _references(sample: UnifiedSample) -> list[str]:
    if sample.ground_truth is None:
        return []
    if sample.ground_truth.count is not None:
        return [str(sample.ground_truth.count)]
    return sample.ground_truth.answers


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}
