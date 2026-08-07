"""Deterministic counting metrics and judge merging.

确定性计数指标与 judge 合并。纯本地计算，无网络副作用；judge 结果只能
旁路记录，绝不能覆盖 deterministic_metrics。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from agents.counting.schema import CountingResult
from data.schema import GroundTruth
from evaluation.records import CountDeterministicMetrics, EvaluationRecord


def count_deterministic_metrics(
    predicted_count: int, gold_count: int
) -> CountDeterministicMetrics:
    """Calculate exact, absolute, relative, and smooth counting error metrics.
    计算精确匹配、绝对误差、相对误差和平滑误差分数。"""

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


def merge_count_evaluation(
    *,
    sample_id: str,
    counting: CountingResult,
    ground_truth: GroundTruth | None,
    judge_raw: str | None = None,
    judge_parsed: Any | None = None,
    judge_error: str | None = None,
) -> EvaluationRecord:
    """Preserve judge output and visibly flag conflict with deterministic
    truth; judge verdicts never overwrite deterministic metrics.
    保留 judge 输出并显式标记与确定性真值的冲突；judge 结论绝不覆盖
    确定性指标。"""

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
    inconsistency = (
        metrics is not None
        and metrics.exact_match == 0
        and getattr(judge_parsed, "verdict", None) == "correct"
    )
    return EvaluationRecord(
        sample_id=sample_id,
        task="counting",
        deterministic_metrics=metrics,
        judge_status="succeeded",
        judge_raw=judge_raw,
        judge_parsed=judge_parsed,
        judge_inconsistency=inconsistency,
    )


def aggregate_counting(records: Sequence[EvaluationRecord]) -> dict[str, Any]:
    """Aggregate deterministic counting metrics across records.
    跨记录汇总确定性计数指标。"""

    metrics = [
        record.deterministic_metrics
        for record in records
        if record.deterministic_metrics is not None
    ]
    total = len(records)
    if not metrics:
        return {
            "metric": "counting_deterministic",
            "total": total,
            "samples_with_ground_truth": 0,
        }
    return {
        "metric": "counting_deterministic",
        "total": total,
        "samples_with_ground_truth": len(metrics),
        "exact_match_accuracy": sum(item.exact_match for item in metrics) / len(metrics),
        "mean_absolute_error": sum(item.absolute_error for item in metrics) / len(metrics),
        "mean_relative_error": sum(item.relative_error for item in metrics) / len(metrics),
        "mean_smooth_error_score": sum(item.smooth_error_score for item in metrics) / len(metrics),
    }
