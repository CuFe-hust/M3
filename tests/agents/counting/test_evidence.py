"""Contract tests for counting evidence rules.

计数证据契约测试：数量解析、box→点转换、去重（IoU/点距/混合）、边界残片
丢弃、merge group 与 warning 计数逻辑的纯确定性。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.counting.evidence import (
    accepted_count_evidence,
    box_evidence,
    global_count_point,
    is_tiny_border_fragment,
    merge_count_evidence,
    parse_count_answer,
    recover_count_proposal_header,
)
from agents.schema import VisualEvidence

REPO_ROOT = Path(__file__).resolve().parents[3]


def _box(label: str, box: list[int]) -> VisualEvidence:
    return VisualEvidence(label=label, box=box)


def _point(label: str, point: list[int]) -> VisualEvidence:
    return VisualEvidence(label=label, point=point)


# ── 数量解析 / count parsing ──────────────────────────────────────────────


def test_parse_count_answer() -> None:
    assert parse_count_answer(" 42 ") == 42
    assert parse_count_answer("0") == 0
    with pytest.raises(ValueError, match="non-negative integer"):
        parse_count_answer("twelve")
    with pytest.raises(ValueError, match="non-negative integer"):
        parse_count_answer("-1")


def test_recover_count_proposal_header() -> None:
    assert recover_count_proposal_header('{"answer": "7"') == 7
    assert recover_count_proposal_header('garbage "answer": "3"') == 3
    assert recover_count_proposal_header('{"label": "x"}') is None


# ── box 证据 / box evidence ────────────────────────────────────────────────


def test_box_evidence_normalizes_and_drops_invalid() -> None:
    evidence = box_evidence(
        [[100, 100, 200, 200], [300, 300, 200, 200], [50, 50, 50, 60], [1, 2, 3], [9999, 0, 0, 9999]],
        target="car",
        image_id="i1",
    )
    assert len(evidence) == 3
    assert evidence[0].box == [100, 100, 200, 200]
    assert evidence[1].box == [200, 200, 300, 300]  # corners reordered / 角点重排
    assert evidence[2].box == [0, 0, 999, 999]  # clamped to 0..999 / 截断到 0..999


def test_is_tiny_border_fragment() -> None:
    assert is_tiny_border_fragment([0, 100, 40, 140]) is True   # left border / 左缘
    assert is_tiny_border_fragment([959, 0, 999, 40]) is True   # top border / 上缘
    assert is_tiny_border_fragment([960, 100, 999, 140]) is True  # right border / 右缘
    assert is_tiny_border_fragment([100, 960, 140, 999]) is True  # bottom border / 下缘
    assert is_tiny_border_fragment([100, 100, 200, 200]) is False


# ── 去重与合并 / dedup and merge ───────────────────────────────────────────


def test_merge_count_evidence_merges_by_iou() -> None:
    items = [
        _box("car", [100, 100, 200, 200]),
        _box("car", [102, 102, 198, 198]),
        _box("car", [300, 300, 400, 400]),
    ]
    merged = merge_count_evidence(items)
    assert len(merged) == 2
    assert merged[0].box == [100, 100, 200, 200]


def test_merge_count_evidence_keeps_adjacent_distinct() -> None:
    """Strict dedup: adjacent distinct targets stay separate.
    严格去重：相邻的不同目标保持独立。"""
    items = [
        _box("car", [100, 100, 200, 200]),
        _box("car", [220, 100, 320, 200]),
    ]
    assert len(merge_count_evidence(items)) == 2


def test_merge_count_evidence_point_distance() -> None:
    items = [
        _point("car", [100, 100]),
        _point("car", [108, 100]),
        _point("car", [200, 200]),
    ]
    merged = merge_count_evidence(items)
    assert len(merged) == 2


def test_merge_count_evidence_box_point_mix() -> None:
    items = [
        _box("car", [100, 100, 200, 200]),
        _point("car", [150, 150]),  # centre within 12px / 中心在 12px 内
        _point("truck", [150, 150]),  # different label / 不同标签
    ]
    merged = merge_count_evidence(items)
    assert len(merged) == 2


def test_merge_count_evidence_label_normalization() -> None:
    """Generic label normalization — no dataset-specific class mapping.
    通用标签归一化——无数据集特定类别映射。"""
    items = [_box(" Car ", [100, 100, 200, 200]), _box("car", [102, 102, 198, 198])]
    assert len(merge_count_evidence(items)) == 1


def test_accepted_count_evidence_drops_fragments_and_dedups() -> None:
    items = [
        _box("car", [0, 100, 40, 140]),          # tiny border fragment / 边界残片
        _box("car", [100, 100, 200, 200]),
        _box("car", [102, 102, 198, 198]),        # duplicate of previous / 重复
        _point("car", [300, 300]),
    ]
    points, boxes, dropped = accepted_count_evidence(items, target="car", image_id="i1")
    assert len(points) == 2
    assert len(boxes) == 1
    assert boxes[0] == [100, 100, 200, 200]
    # duplicate box + border fragment = 2 dropped
    # 重复框 + 边界残片 = 丢弃 2
    assert dropped == 2
    assert all(point.point is not None for point in points)


def test_global_count_point_conversion() -> None:
    point = global_count_point(
        "s1", "car", _point("car", [100, 200]), index=3, width=1000, height=1000
    )
    assert point.global_id == "s1:whole_image_overview:p003"
    assert point.local_id == "p003"
    assert point.global_x_norm == 100
    assert point.global_y_norm == 200
    assert point.global_x_px == round(100 * 999 / 999)
    assert point.accepted is True
    assert point.ownership_valid is True


def test_global_count_point_requires_point() -> None:
    with pytest.raises(ValueError, match="requires a point"):
        global_count_point("s1", "car", _box("car", [100, 100, 200, 200]), 0, 1000, 1000)


# ── 边界 / boundaries ──────────────────────────────────────────────────────


def test_evidence_has_no_dataset_branch() -> None:
    source = (REPO_ROOT / "agents" / "counting" / "evidence.py").read_text(encoding="utf-8")
    assert "VRSBench" not in source
    assert "vrsbench" not in source
    assert "spacers_agent" not in source
    assert "eval" not in source
