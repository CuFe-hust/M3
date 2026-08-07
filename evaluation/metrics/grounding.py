"""Deterministic axis-aligned grounding IoU metrics.

确定性轴对齐接地 IoU 指标。纯几何本地计算，无网络副作用；official
oriented-box 指标由上游评测器负责，本模块只做轴对齐近似。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from evaluation.records import EvaluationRecord, GroundingDeterministicMetrics


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


def grounding_deterministic_metrics(iou: float) -> GroundingDeterministicMetrics:
    """Wrap one box-pair IoU into the unified typed record. The raw IoU is
    stored unrounded and the 0.5 threshold is decided exactly once here; the
    aggregate must reuse the stored flag.
    将单对框 IoU 包装进统一类型化记录。原始 IoU 不做舍入存储，0.5 阈值
    在此唯一判定一次；聚合必须复用存储的标志位。"""

    value = float(iou)
    return GroundingDeterministicMetrics(iou=value, iou_at_0_5=value >= 0.5)


def aggregate_grounding(records: Sequence[EvaluationRecord]) -> dict[str, Any]:
    """Aggregate mean IoU and IoU@0.5 accuracy over unified grounding
    records; the accuracy uses each record's stored threshold flag, never a
    second comparison. 对统一 grounding 记录汇总平均 IoU 与 IoU@0.5 准确
    率；准确率使用每条记录存储的阈值标志，绝不二次比较。"""

    total = len(records)
    if not total:
        return {
            "metric": "axis_aligned_iou_at_0_5",
            "total": 0,
            "mean_iou": 0.0,
            "accuracy": 0.0,
        }
    metrics_list: list[GroundingDeterministicMetrics] = []
    for record in records:
        metrics = record.deterministic_metrics
        if not isinstance(metrics, GroundingDeterministicMetrics):
            raise ValueError(
                f"grounding record {record.sample_id!r} lacks GroundingDeterministicMetrics"
            )
        metrics_list.append(metrics)
    success = sum(int(metrics.iou_at_0_5) for metrics in metrics_list)
    return {
        "metric": "axis_aligned_iou_at_0_5",
        "total": total,
        "mean_iou": sum(metrics.iou for metrics in metrics_list) / total,
        "accuracy": success / total,
        "official_note": (
            "Use the upstream VRSBench or XLRS-Bench evaluator for official "
            "oriented-box metrics."
        ),
    }
