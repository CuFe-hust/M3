"""VRSBench adapter-layer exports (ontology and task normalization).
VRSBench 适配器层导出（类别体系与任务规范化）。

This package only re-exports stable symbols.
本包只重导出稳定符号。
"""

from data.adapters.vrsbench.adapter import VRSBenchAdapter
from data.adapters.vrsbench.ontology import (
    LARGE_VEHICLE_ALIASES,
    LARGE_VEHICLE_CLASS,
    SMALL_VEHICLE_ALIASES,
    SMALL_VEHICLE_CLASS,
    canonical_vehicle_class,
    count_target_hint,
)
from data.adapters.vrsbench.task_normalizer import (
    NORMALIZER_NAME,
    NORMALIZER_VERSION,
    QuestionSubtype,
    classify_question_subtype,
    normalize_task,
)

__all__ = [
    "LARGE_VEHICLE_ALIASES",
    "LARGE_VEHICLE_CLASS",
    "NORMALIZER_NAME",
    "NORMALIZER_VERSION",
    "QuestionSubtype",
    "SMALL_VEHICLE_ALIASES",
    "SMALL_VEHICLE_CLASS",
    "VRSBenchAdapter",
    "canonical_vehicle_class",
    "classify_question_subtype",
    "count_target_hint",
    "normalize_task",
]
