"""Contract tests for deterministic grounding IoU metrics.

确定性接地 IoU 指标契约测试：重叠/包含/不相交/退化框的 IoU 数值、
IoU@0.5 准确率与平均 IoU 汇总、空输入、无网络副作用。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.metrics.grounding import (
    aggregate_grounding,
    box_iou,
    grounding_deterministic_metrics,
)
from evaluation.records import EvaluationRecord

REPO_ROOT = Path(__file__).resolve().parents[2]


def _grounding_record(iou: float) -> EvaluationRecord:
    return EvaluationRecord(
        sample_id=f"s{iou}",
        task="grounding",
        deterministic_metrics=grounding_deterministic_metrics(iou),
        judge_status="not_requested",
    )


def test_box_iou_perfect_overlap() -> None:
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_box_iou_no_overlap() -> None:
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_box_iou_partial_overlap() -> None:
    # Intersection is 5x5=25; union is 100+100-25=175. / 交集 5x5=25；并集 175。
    assert box_iou([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25 / 175)


def test_box_iou_containment() -> None:
    # Inner box fully inside the outer one. / 内框完全包含于外框。
    assert box_iou([0, 0, 10, 10], [2, 2, 8, 8]) == pytest.approx(36 / 100)


def test_box_iou_degenerate_boxes() -> None:
    assert box_iou([0, 0, 0, 0], [0, 0, 10, 10]) == 0.0
    assert box_iou([0, 0, 10, 10], [5, 5, 5, 5]) == 0.0


def test_aggregate_grounding() -> None:
    records = [
        _grounding_record(1.0),  # iou 1.0 -> success
        _grounding_record(25 / 175),  # iou ~0.143 -> fail
        _grounding_record(0.0),  # iou 0.0 -> fail
    ]
    summary = aggregate_grounding(records)
    assert summary["metric"] == "axis_aligned_iou_at_0_5"
    assert summary["total"] == 3
    assert summary["accuracy"] == pytest.approx(1 / 3)
    assert summary["mean_iou"] == pytest.approx((1.0 + 25 / 175 + 0.0) / 3)
    assert "official" in summary["official_note"]
    assert "oriented" not in summary["metric"]
    assert "benchmark" not in summary


def test_aggregate_grounding_empty() -> None:
    summary = aggregate_grounding([])
    assert summary["total"] == 0
    assert summary["mean_iou"] == 0.0
    assert summary["accuracy"] == 0.0


def test_grounding_typed_record_serializes() -> None:
    record = _grounding_record(0.75)
    payload = record.model_dump(mode="json")
    assert payload["task"] == "grounding"
    assert payload["deterministic_metrics"] == {"iou": 0.75, "iou_at_0_5": True}
    assert payload["judge_status"] == "not_requested"


def test_grounding_metrics_have_no_network_side_effects() -> None:
    source = (REPO_ROOT / "evaluation/metrics/grounding.py").read_text(encoding="utf-8")
    for token in ("urlopen", "requests", "socket", "httpx", "api.deepseek", "http://", "https://"):
        assert token not in source, token


# ── 阈值一致性 / threshold consistency (33.6) ─────────────────────────────


@pytest.mark.parametrize(
    ("iou", "expected_flag"),
    [
        (0.4999994, False),
        (0.4999996, False),
        (0.5, True),
        (0.5000004, True),
    ],
)
def test_iou_threshold_decided_once_per_record(iou: float, expected_flag: bool) -> None:
    """The stored flag is the single threshold authority; the raw value is
    stored unrounded. 存储标志是唯一阈值权威；原始值不做舍入存储。"""
    metrics = grounding_deterministic_metrics(iou)
    assert metrics.iou == pytest.approx(iou)
    assert metrics.iou_at_0_5 is expected_flag


def test_aggregate_uses_stored_threshold_flag() -> None:
    """Aggregate accuracy must come from the stored flag, never a second
    comparison against the stored (possibly rounded) value.
    聚合准确率必须来自存储标志，绝不再次比较（可能已舍入的）存储值。"""
    records = [
        _grounding_record(0.4999996),
        _grounding_record(0.5),
        _grounding_record(0.5000004),
    ]
    summary = aggregate_grounding(records)
    assert summary["accuracy"] == pytest.approx(2 / 3)
    assert summary["mean_iou"] == pytest.approx((0.4999996 + 0.5 + 0.5000004) / 3)
