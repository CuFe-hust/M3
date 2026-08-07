"""Contract tests for deterministic grounding IoU metrics.

确定性接地 IoU 指标契约测试：重叠/包含/不相交/退化框的 IoU 数值、
IoU@0.5 准确率与平均 IoU 汇总、空输入、无网络副作用。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.metrics.grounding import aggregate_grounding, box_iou

REPO_ROOT = Path(__file__).resolve().parents[2]


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
    pairs = [
        ([0, 0, 10, 10], [0, 0, 10, 10]),  # iou 1.0 -> success
        ([0, 0, 10, 10], [5, 5, 15, 15]),  # iou ~0.143 -> fail
        ([0, 0, 10, 10], [20, 20, 30, 30]),  # iou 0.0 -> fail
    ]
    summary = aggregate_grounding(pairs)
    assert summary["metric"] == "axis_aligned_iou_at_0_5"
    assert summary["total"] == 3
    assert summary["accuracy"] == pytest.approx(1 / 3)
    assert summary["mean_iou"] == pytest.approx((1.0 + 25 / 175 + 0.0) / 3)
    assert "official" in summary["official_note"]


def test_aggregate_grounding_empty() -> None:
    summary = aggregate_grounding([])
    assert summary["total"] == 0
    assert summary["mean_iou"] == 0.0
    assert summary["accuracy"] == 0.0


def test_grounding_metrics_have_no_network_side_effects() -> None:
    source = (REPO_ROOT / "evaluation/metrics/grounding.py").read_text(encoding="utf-8")
    for token in ("urlopen", "requests", "socket", "httpx", "api.deepseek", "http://", "https://"):
        assert token not in source, token
