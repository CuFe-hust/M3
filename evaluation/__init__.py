"""Deterministic evaluation records and metrics exports.
确定性评估记录与指标导出。"""

from evaluation.metrics.aggregate import aggregate
from evaluation.metrics.caption import evaluate_caption
from evaluation.metrics.counting import (
    aggregate_counting,
    count_deterministic_metrics,
    merge_count_evaluation,
)
from evaluation.metrics.grounding import aggregate_grounding, box_iou
from evaluation.metrics.vqa import (
    aggregate_vqa,
    exact_match,
    merge_vqa_evaluation,
    normalize_answer,
)
from evaluation.records import (
    CountDeterministicMetrics,
    EvaluationRecord,
    VQAEvaluationRecord,
)

__all__ = [
    "CountDeterministicMetrics",
    "EvaluationRecord",
    "VQAEvaluationRecord",
    "aggregate",
    "aggregate_counting",
    "aggregate_grounding",
    "aggregate_vqa",
    "box_iou",
    "count_deterministic_metrics",
    "evaluate_caption",
    "exact_match",
    "merge_count_evaluation",
    "merge_vqa_evaluation",
    "normalize_answer",
]
