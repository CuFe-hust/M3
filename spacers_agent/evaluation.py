"""Deterministic counting metrics and text-only DeepSeek judge integration records.
确定性计数指标与仅文本 DeepSeek 评估器集成记录。
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from spacers_agent.schemas import CountTargetSpec, CountingResult, GroundTruth


class CountDeterministicMetrics(BaseModel):
    """Deterministic metrics for a counting sample with a known count.
    对具有已知计数真值样本的确定性指标。
    """

    model_config = ConfigDict(extra="forbid")

    predicted_count: int = Field(ge=0)
    gold_count: int = Field(ge=0)
    exact_match: int = Field(ge=0, le=1)
    absolute_error: int = Field(ge=0)
    relative_error: float = Field(ge=0.0)
    smooth_error_score: float = Field(ge=0.0, le=1.0)


class DeepSeekJudgeResult(BaseModel):
    """Structured judge output that explicitly excludes visual truth verification.
    明确排除视觉真相核验的结构化评估器输出。
    """

    model_config = ConfigDict(extra="forbid")

    judge_scope: Literal["text_and_structured_evidence_only"]
    can_verify_visual_truth: Literal[False]
    semantic_correctness: float = Field(ge=0.0, le=1.0)
    answer_evidence_consistency: float = Field(ge=0.0, le=1.0)
    constraint_following: float = Field(ge=0.0, le=1.0)
    clarity: float = Field(ge=0.0, le=1.0)
    verdict: Literal["correct", "mostly_correct", "incorrect", "not_judgeable"]
    issues: list[str] = Field(default_factory=list)
    concise_rationale: str = Field(max_length=500)


class CountJudgeResult(BaseModel):
    """Counting-specific Judge contract that cannot override metrics. / 不能覆盖指标的计数专用 Judge 契约。"""
    model_config = ConfigDict(extra="forbid")
    verdict: Literal["correct", "incorrect", "uncertain", "not_visually_verifiable"]
    pipeline_consistency: Literal["pass", "fail"]
    completeness_claim_valid: bool
    error_codes: list[Literal["COUNT_POINT_MISMATCH", "PARTIAL_CLAIMED_COMPLETE", "UNRESOLVED_CONFLICT", "LOW_CONFIDENCE_RISK", "GROUND_TRUTH_MISMATCH", "INSUFFICIENT_INFORMATION"]] = Field(default_factory=list)
    score: float = Field(ge=0.0, le=1.0)
    short_reason: str = Field(max_length=500)


def sample_count_evidence(result: CountingResult, *, seed: int = 0, limit: int = 64) -> list[dict[str, Any]]:
    """Deterministically stratify sampled point evidence by tile. / 按 tile 确定性分层抽样点证据。"""
    import random
    buckets: dict[str, list[Any]] = {}
    for point in result.global_points:
        buckets.setdefault(point.source_tile_id, []).append(point)
    rng = random.Random(seed); chosen = []
    for tile_id in sorted(buckets):
        point = rng.choice(sorted(buckets[tile_id], key=lambda item: item.global_id))
        chosen.append({"point_id": point.global_id, "tile_id": tile_id, "accepted": point.accepted, "confidence": point.confidence, "reason": point.rejection_reason})
    remaining = [point for tile in buckets.values() for point in tile if point.global_id not in {item["point_id"] for item in chosen}]
    rng.shuffle(remaining)
    chosen.extend({"point_id": point.global_id, "tile_id": point.source_tile_id, "accepted": point.accepted, "confidence": point.confidence, "reason": point.rejection_reason} for point in remaining[:max(0, limit-len(chosen))])
    return chosen[:limit]


class EvaluationRecord(BaseModel):
    """One merged deterministic and optional text-only judge evaluation record.
    一条合并确定性指标与可选仅文本评估器结果的评估记录。
    """

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    task: str
    deterministic_metrics: CountDeterministicMetrics | None = None
    judge_status: Literal["not_requested", "succeeded", "failed"]
    judge_raw: str | None = None
    judge_parsed: DeepSeekJudgeResult | None = None
    judge_inconsistency: bool = False
    judge_error: str | None = None


def count_deterministic_metrics(predicted_count: int, gold_count: int) -> CountDeterministicMetrics:
    """Calculate exact, absolute, relative, and smooth counting error metrics.
    计算精确匹配、绝对误差、相对误差和平滑误差分数。
    """

    if predicted_count < 0 or gold_count < 0:
        raise ValueError("counts must not be negative")
    absolute_error = abs(predicted_count - gold_count)
    denominator = abs(gold_count) + 1
    return CountDeterministicMetrics(
        predicted_count=predicted_count,
        gold_count=gold_count,
        exact_match=int(predicted_count == gold_count),
        absolute_error=absolute_error,
        relative_error=absolute_error / denominator,
        smooth_error_score=math.exp(-3 * absolute_error / denominator),
    )


def build_count_judge_payload(
    *,
    question: str,
    target: CountTargetSpec,
    display_answer: str,
    counting: CountingResult,
    ground_truth: GroundTruth | None,
    min_confidence: float,
) -> dict[str, Any]:
    """Build a compact text-and-evidence payload that never includes image data.
    构建绝不包含图像数据的紧凑文本与证据载荷。
    """

    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")
    metrics = (
        count_deterministic_metrics(counting.final_count, ground_truth.count)
        if ground_truth is not None and ground_truth.count is not None
        else None
    )
    accepted_points = [point for point in counting.global_points if point.accepted]
    return {
        "task": "counting",
        "question": question,
        "target_spec": {
            "canonical_label": target.canonical_label,
            "inclusion_rule": target.inclusion_rule,
            "exclusion_rule": target.exclusion_rule,
        },
        "prediction": {
            "display_answer": display_answer,
            "final_count": counting.final_count,
            "point_count": len(accepted_points),
            "failed_tiles": counting.failed_tiles,
            "unresolved_conflicts": counting.unresolved_conflicts,
        },
        "ground_truth": (
            {"count": ground_truth.count, "answers": ground_truth.answers} if ground_truth is not None else None
        ),
        "deterministic_metrics": metrics.model_dump(mode="json") if metrics is not None else None,
        "evidence_summary": {
            "tile_count": counting.tile_count,
            "succeeded_tiles": len(counting.succeeded_tiles),
            "low_confidence_points": sum(point.confidence < min_confidence for point in accepted_points),
            "seam_merges": len(counting.merged_groups),
        },
    }


def build_judge_request_hash(
    *,
    model: str,
    prompt_text: str,
    sample_id: str,
    payload: dict[str, Any],
) -> str:
    """Hash the exact judge inputs without adding image payloads to the cache key.
    对精确评估器输入计算哈希且不向缓存键添加图像载荷。
    """

    prediction = payload.get("prediction")
    ground_truth = payload.get("ground_truth")
    metrics = payload.get("deterministic_metrics")
    stable = {
        "model": model,
        "prompt_sha256": _stable_hash(prompt_text),
        "sample_id": sample_id,
        "prediction_sha256": _stable_hash(prediction),
        "ground_truth_sha256": _stable_hash(ground_truth),
        "deterministic_metrics_sha256": _stable_hash(metrics),
    }
    return _stable_hash(stable)


def merge_count_evaluation(
    *,
    sample_id: str,
    counting: CountingResult,
    ground_truth: GroundTruth | None,
    judge_raw: str | None = None,
    judge_parsed: DeepSeekJudgeResult | None = None,
    judge_error: str | None = None,
) -> EvaluationRecord:
    """Preserve judge output and visibly flag conflict with deterministic truth.
    保留评估器输出，并显式标记与确定性真值的冲突。
    """

    metrics = (
        count_deterministic_metrics(counting.final_count, ground_truth.count)
        if ground_truth is not None and ground_truth.count is not None
        else None
    )
    if judge_error is not None:
        return EvaluationRecord(
            sample_id=sample_id,
            task="counting",
            deterministic_metrics=metrics,
            judge_status="failed",
            judge_raw=judge_raw,
            judge_error=judge_error,
        )
    if judge_parsed is None:
        return EvaluationRecord(
            sample_id=sample_id,
            task="counting",
            deterministic_metrics=metrics,
            judge_status="not_requested",
        )
    inconsistency = metrics is not None and metrics.exact_match == 0 and judge_parsed.verdict == "correct"
    return EvaluationRecord(
        sample_id=sample_id,
        task="counting",
        deterministic_metrics=metrics,
        judge_status="succeeded",
        judge_raw=judge_raw,
        judge_parsed=judge_parsed,
        judge_inconsistency=inconsistency,
    )


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
