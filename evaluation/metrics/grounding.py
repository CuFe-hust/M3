"""Deterministic axis-aligned grounding IoU metrics.

确定性轴对齐接地 IoU 指标。纯几何本地计算，无网络副作用；official
oriented-box 指标由上游评测器负责，本模块只做轴对齐近似。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Axis-aligned IoU of two xyxy boxes. / 两个 xyxy 框的轴对齐 IoU。"""

    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    denominator = first_area + second_area - intersection
    return intersection / denominator if denominator else 0.0


def aggregate_grounding(
    pairs: Sequence[tuple[Sequence[float], Sequence[float]]],
) -> dict[str, Any]:
    """Aggregate mean IoU and IoU@0.5 accuracy over box pairs.
    对框对汇总平均 IoU 与 IoU@0.5 准确率。"""

    total = len(pairs)
    if not total:
        return {
            "metric": "axis_aligned_iou_at_0_5",
            "total": 0,
            "mean_iou": 0.0,
            "accuracy": 0.0,
        }
    ious = [box_iou(predicted, expected) for predicted, expected in pairs]
    success = sum(iou >= 0.5 for iou in ious)
    return {
        "metric": "axis_aligned_iou_at_0_5",
        "total": total,
        "mean_iou": sum(ious) / total,
        "accuracy": success / total,
        "official_note": (
            "Use the upstream VRSBench or XLRS-Bench evaluator for official "
            "oriented-box metrics."
        ),
    }
