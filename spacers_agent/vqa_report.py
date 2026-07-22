"""Bridge persisted multi-Agent VQA artifacts into the default HTML audit report.
将持久化的多 Agent VQA 产物接入默认 HTML 审计报告。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.schema import CanonicalPrediction, CanonicalSample
from eval.audit_report import AuditReportWriter, build_audit_report, write_deepseek_audit
from spacers_agent.schemas import ExpertResult, UnifiedSample
from spacers_agent.settings import QwenSettings


def build_multiagent_vqa_report(
    run_dir: Path,
    *,
    qwen: QwenSettings,
    model_load_seconds: float = 0.0,
    max_samples: int = 200,
) -> Path | None:
    """Create canonical result, metrics, DeepSeek audit, CSV, and HTML artifacts.
    生成统一结果、指标、DeepSeek 审计、CSV 与 HTML 产物。
    """

    result_path = run_dir / "vrsbench_vqa.jsonl"
    metrics_path = run_dir / "vrsbench_vqa.metrics.json"
    audit_path = run_dir / "vrsbench_vqa.deepseek_audit.jsonl"
    sample_dirs = _successful_vqa_sample_dirs(run_dir)
    if not sample_dirs:
        return None
    exact_correct = 0
    inference_seconds = 0.0
    deepseek_audit: list[dict[str, Any]] = []
    result_lines: list[str] = []
    with AuditReportWriter(result_path, max_samples=max_samples) as writer:
        for sample_dir in sample_dirs:
            sample = UnifiedSample.model_validate_json((sample_dir / "sample.json").read_text(encoding="utf-8"))
            result = ExpertResult.model_validate_json((sample_dir / "expert_result.json").read_text(encoding="utf-8"))
            trace = _read_json(sample_dir / "agent_trace.json")
            evaluation = _read_json(sample_dir / "vqa_evaluation.json")
            references = sample.ground_truth.answers if sample.ground_truth is not None else []
            canonical_sample = CanonicalSample(
                id=sample.sample_id,
                task_type="vqa",
                images=[ref.path for ref in sample.images],
                prompt=sample.question,
                answers=references,
                meta={"source": sample.dataset, "question": sample.question},
            )
            raw_text = _read_raw_qwen(sample_dir / "general_vqa_expert" / "raw_response.txt", result.answer)
            canonical_prediction = CanonicalPrediction(
                id=sample.sample_id,
                task_type="vqa",
                text=result.answer,
                answer=result.answer,
                boxes=result.boxes,
                meta={"model_id": qwen.model, "raw_text": raw_text},
            )
            record = {
                "sample": canonical_sample.serializable(),
                "prediction": canonical_prediction.serializable(),
            }
            result_lines.append(json.dumps(record, ensure_ascii=False))
            exact_correct += int(bool(evaluation.get("exact_match")))
            seconds = float(trace.get("inference_seconds", 0.0) or 0.0)
            inference_seconds += seconds
            writer.capture(canonical_sample, canonical_prediction, seconds, agent_trace=trace)
            deepseek_audit.append(_deepseek_audit_record(sample_dir, sample, result, evaluation))
    result_path.write_text("\n".join(result_lines) + "\n", encoding="utf-8")
    evaluated = [record for record in deepseek_audit if record.get("score") is not None]
    metrics = {
        "metric": "exact_match_accuracy",
        "correct": exact_correct,
        "total": len(sample_dirs),
        "score": exact_correct / len(sample_dirs),
        "deepseek_proxy": {
            "metric": "deepseek_semantic_match_proxy",
            "score": sum(float(record["score"]) for record in evaluated) / len(evaluated) if evaluated else 0.0,
            "evaluated": len(evaluated),
            "failed": [
                {"id": record["sample_id"], "error": record.get("error")}
                for record in deepseek_audit
                if record.get("error")
            ],
            "comparability_note": "Text-only DeepSeek validation; not the official VRSBench GPT metric.",
        },
    }
    _write_json(metrics_path, metrics)
    write_deepseek_audit(audit_path, deepseek_audit)
    _write_json(
        result_path.with_suffix(".metadata.json"),
        {
            "dataset": "vrsbench_vqa",
            "completed_samples": len(sample_dirs),
            "model": {
                "id": qwen.model,
                "backend": qwen.backend,
                "dtype": qwen.dtype,
                "max_new_tokens": qwen.max_tokens,
                "local_files_only": qwen.local_files_only,
            },
            "model_load_seconds": model_load_seconds,
            "inference_seconds": round(inference_seconds, 6),
            "pipeline": [
                "VRSBenchVQAAdapter",
                "TaskRouter",
                "GeneralVQAExpert",
                "QwenTransformersClient",
                "DeepSeekJudgeClient",
                "AuditReportWriter",
            ],
        },
    )
    return build_audit_report(result_path, metrics_path, audit_path)


def _successful_vqa_sample_dirs(run_dir: Path) -> list[Path]:
    directories = []
    for status_path in (run_dir / "samples").glob("*/status.json"):
        status = _read_json(status_path)
        sample_path = status_path.parent / "sample.json"
        if status.get("state") != "succeeded" or not sample_path.is_file():
            continue
        sample = _read_json(sample_path)
        if sample.get("task") == "general_vqa" and (status_path.parent / "expert_result.json").is_file():
            directories.append(status_path.parent)
    return sorted(directories, key=lambda path: _sample_sort_key(path.name))


def _sample_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _deepseek_audit_record(
    sample_dir: Path,
    sample: UnifiedSample,
    result: ExpertResult,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    judge_dir = sample_dir / "deepseek_vqa_judge"
    validation = _read_json(judge_dir / "validation.json")
    raw_content = (judge_dir / "raw_response.txt").read_text(encoding="utf-8") if (judge_dir / "raw_response.txt").is_file() else ""
    metadata = validation.get("response_metadata") or {}
    return {
        "sample_id": sample.sample_id,
        "question": sample.question,
        "reference_answers": sample.ground_truth.answers if sample.ground_truth is not None else [],
        "candidate_answer": result.answer,
        "score": evaluation.get("judge_score"),
        "raw_content": raw_content,
        "parsed_result": evaluation.get("judge_parsed"),
        "duration_seconds": metadata.get("latency_seconds", ""),
        "attempts": 1 + len(validation.get("attempt_errors", [])) if validation else 0,
        "token_usage": metadata.get("token_usage"),
        "error": evaluation.get("judge_error"),
    }


def _read_raw_qwen(path: Path, fallback: str) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else fallback


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
