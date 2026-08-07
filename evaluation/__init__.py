"""Deterministic evaluation records and metrics exports.
确定性评估记录与指标导出。"""

from evaluation.metrics.aggregate import aggregate
from evaluation.metrics.caption import aggregate_caption, evaluate_caption
from evaluation.metrics.counting import (
    aggregate_counting,
    count_deterministic_metrics,
    merge_count_evaluation,
)
from evaluation.metrics.grounding import (
    aggregate_grounding,
    box_iou,
    grounding_deterministic_metrics,
)
from evaluation.metrics.vqa import (
    aggregate_vqa,
    exact_match,
    merge_vqa_evaluation,
    normalize_answer,
    vqa_deterministic_metrics,
)
from evaluation.records import (
    CaptionDeterministicMetrics,
    CountDeterministicMetrics,
    DeterministicMetrics,
    EvaluationRecord,
    EvaluationTask,
    GroundingDeterministicMetrics,
    VQADeterministicMetrics,
    VQAEvaluationRecord,
)

__all__ = [
    "CaptionDeterministicMetrics",
    "CountDeterministicMetrics",
    "DeterministicMetrics",
    "EvaluationRecord",
    "EvaluationTask",
    "GroundingDeterministicMetrics",
    "VQADeterministicMetrics",
    "VQAEvaluationRecord",
    "aggregate",
    "aggregate_caption",
    "aggregate_counting",
    "aggregate_grounding",
    "aggregate_vqa",
    "box_iou",
    "count_deterministic_metrics",
    "evaluate_caption",
    "exact_match",
    "grounding_deterministic_metrics",
    "merge_count_evaluation",
    "merge_vqa_evaluation",
    "normalize_answer",
    "vqa_deterministic_metrics",
]
