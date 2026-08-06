"""VRSBench vehicle ontology and count-target hints (data layer).

VRSBench 车辆类别体系与计数目标提示（数据层）。只包含数据事实，
不依赖模型、Router、Agent 或任何后端运行时。
"""

from __future__ import annotations

import re
from typing import Any

SMALL_VEHICLE_ALIASES = (
    "small vehicle",
    "small-vehicle",
    "car",
    "automobile",
    "passenger car",
    "motorcycle",
)
LARGE_VEHICLE_ALIASES = (
    "large vehicle",
    "large-vehicle",
    "truck",
    "bus",
    "trailer",
    "semi-truck",
)
# Official VRSBench class labels. / VRSBench 官方类别标签。
SMALL_VEHICLE_CLASS = "small-vehicle"
LARGE_VEHICLE_CLASS = "large-vehicle"
GENERIC_VEHICLE_CLASS = "vehicle"


def _lowered(value: str) -> str:
    return " ".join(value.casefold().replace("-", " ").split())


def canonical_vehicle_class(label: str) -> str | None:
    """Normalize a vehicle label to an official class, or None.
    将车辆标签规范化为官方类别，无法识别时返回 None。"""
    lowered = _lowered(label)
    if any(alias in lowered for alias in SMALL_VEHICLE_ALIASES):
        return SMALL_VEHICLE_CLASS
    if any(alias in lowered for alias in LARGE_VEHICLE_ALIASES):
        return LARGE_VEHICLE_CLASS
    return None


def count_target_hint(question: str) -> dict[str, Any] | None:
    """Return the audited vehicle count-target hint without any model call.
    不调用模型，返回经审计的车辆计数目标提示。"""
    lowered = _lowered(question)
    if "small vehicle" in lowered:
        return {
            "canonical_label": SMALL_VEHICLE_CLASS,
            "aliases": list(SMALL_VEHICLE_ALIASES),
        }
    if "large vehicle" in lowered:
        return {
            "canonical_label": LARGE_VEHICLE_CLASS,
            "aliases": list(LARGE_VEHICLE_ALIASES),
        }
    if re.search(r"\bvehicles?\b", lowered):
        return {
            "canonical_label": GENERIC_VEHICLE_CLASS,
            "aliases": list(SMALL_VEHICLE_ALIASES) + list(LARGE_VEHICLE_ALIASES),
        }
    return None
