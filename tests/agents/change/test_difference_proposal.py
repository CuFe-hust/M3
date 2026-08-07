"""Contract tests for change difference proposals.

差异候选契约测试：加权差异图、连通域过滤、候选数量/面积/score/box 稳定、
overlay 不修改源数组。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from agents.change.difference_proposal import propose_changes, render_overlay
from agents.change.settings import ChangeProposalSettings

REPO_ROOT = Path(__file__).resolve().parents[3]


def _rgb(seed: int = 0, size: tuple[int, int] = (128, 128)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(*size, 3), dtype=np.uint8)


def _pair_with_patch(seed: int = 0, *, size: tuple[int, int] = (128, 128)) -> tuple[np.ndarray, np.ndarray]:
    """A pair with one clearly different square patch covering >10% of pixels.

    一个含清晰差异方块区域的图对；补丁面积需超过默认 threshold_quantile
    的缺失比例 (1 - 0.9)，否则分位数阈值恰好为 0（与基线行为一致）。
    """
    t1 = _rgb(seed, size)
    t2 = t1.copy()
    t2[40:88, 40:88] = np.clip(t2[40:88, 40:88].astype(np.int16) + 120, 0, 255).astype(np.uint8)
    return t1, t2


def _settings(**overrides) -> ChangeProposalSettings:
    values = dict(
        min_component_area_ratio=0.001,
        max_component_area_ratio=0.30,
        max_proposals=4,
    )
    values.update(overrides)
    return ChangeProposalSettings(**values)


def test_propose_changes_finds_the_patch() -> None:
    t1, t2 = _pair_with_patch(1)
    score, proposals = propose_changes(t1, t2, _settings())
    assert score.shape == t1.shape[:2]
    assert score.dtype == np.float32
    assert 0.0 <= float(score.min()) and float(score.max()) <= 1.0
    assert proposals, "expected at least one proposal for a clear patch"
    # The patch is (40..88, 40..88). / 方块位于 (40..88, 40..88)。
    best = proposals[0]
    x1, y1, x2, y2 = best.pixel_box
    assert x1 <= 88 and 40 <= x2 and y1 <= 88 and 40 <= y2


def test_identical_images_yield_no_proposals() -> None:
    image = _rgb(2)
    score, proposals = propose_changes(image, image.copy(), _settings())
    assert proposals == []
    assert float(score.max()) <= 1e-8


def test_proposal_fields_are_stable_and_bounded() -> None:
    t1, t2 = _pair_with_patch(3)
    score, proposals = propose_changes(t1, t2, _settings(max_proposals=3))
    assert len(proposals) <= 3
    for index, proposal in enumerate(proposals):
        assert proposal.proposal_id == f"change_{index:03d}"
        assert 0.0 <= proposal.score <= 1.0
        assert 0.0 <= proposal.area_ratio <= 1.0
        assert len(proposal.box) == 4
        assert all(0 <= value <= 999 for value in proposal.box)
        assert proposal.box[0] < proposal.box[2]
        assert proposal.box[1] < proposal.box[3]
        assert proposal.source == "difference_map_v1"


def test_proposals_are_deterministic() -> None:
    t1, t2 = _pair_with_patch(4)
    settings = _settings()
    score1, proposals1 = propose_changes(t1, t2, settings)
    score2, proposals2 = propose_changes(t1, t2, settings)
    assert np.array_equal(score1, score2)
    assert [item.model_dump() for item in proposals1] == [
        item.model_dump() for item in proposals2
    ]


def test_max_proposals_is_respected() -> None:
    # Several distinct patches / 多个独立差异方块。
    t1 = _rgb(5)
    t2 = t1.copy()
    for x, y in ((10, 10), (60, 10), (10, 70), (60, 70), (30, 40)):
        t2[y : y + 15, x : x + 15] = np.clip(
            t2[y : y + 15, x : x + 15].astype(np.int16) + 150, 0, 255
        ).astype(np.uint8)
    _, proposals = propose_changes(t1, t2, _settings(max_proposals=3))
    assert len(proposals) == 3


def test_render_overlay_does_not_modify_source() -> None:
    t1, t2 = _pair_with_patch(6)
    _, proposals = propose_changes(t1, t2, _settings(max_proposals=2))
    before = t2.copy()
    overlay = render_overlay(t2, proposals)
    assert np.array_equal(t2, before)
    assert overlay.shape == t2.shape
    assert overlay.dtype == t2.dtype


def test_settings_require_positive_weight_sum() -> None:
    with pytest.raises(Exception, match="positive sum"):
        ChangeProposalSettings(rgb_weight=0.0, edge_weight=0.0, structure_weight=0.0)


def test_difference_proposal_has_no_dataset_or_model_branch() -> None:
    source = (REPO_ROOT / "agents" / "change" / "difference_proposal.py").read_text(
        encoding="utf-8"
    )
    assert "VRSBench" not in source
    assert "dataset" not in source
    assert "spacers_agent" not in source
    assert "qwen" not in source.casefold()
