from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import agents.change.proposal_fusion as proposal_fusion_module
from agents.change.proposal_fusion import (
    PROPOSAL_FUSION_VERSION,
    fuse_change_proposals,
)
from agents.change.settings import ChangeProposalSettings


def _settings(**overrides: object) -> ChangeProposalSettings:
    values: dict[str, object] = {
        "min_component_area_ratio": 0.001,
        "max_component_area_ratio": 0.50,
        "max_proposals": 6,
        "mask_close_kernel": 1,
        "proposal_padding_ratio": 0.10,
    }
    values.update(overrides)
    return ChangeProposalSettings(**values)


def _maps(size: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.zeros((size, size), dtype=np.float32) for _ in range(3))


def _pif_without_patch(
    size: int = 64,
    patch: tuple[slice, slice] = (slice(20, 36), slice(22, 38)),
) -> np.ndarray:
    pif = np.ones((size, size), dtype=np.uint8)
    pif[patch] = 0
    return pif


def test_low_no_change_noise_does_not_create_proposals_or_inflate_scores() -> None:
    rng = np.random.default_rng(71)
    low = rng.uniform(0.0, 0.04, size=(64, 64)).astype(np.float32)
    feature = rng.uniform(0.0, 0.02, size=(16, 16)).astype(np.float32)
    semantic = rng.uniform(0.0, 0.01, size=(32, 32)).astype(np.float32)

    result = fuse_change_proposals(
        low,
        feature,
        semantic,
        np.ones((128, 128), dtype=np.uint8),
        _settings(),
    )

    assert result.proposals == []
    assert not bool(np.any(result.binary_change_mask))
    assert float(np.max(result.fused_score_map)) < 0.04
    assert result.diagnostics["reason_code"] == "NO_SCORE_ABOVE_FLOOR"
    assert result.diagnostics["threshold_mode"] == "no_change_floor"


def test_localized_three_source_agreement_produces_mask_and_v2_proposal() -> None:
    low, feature, semantic = _maps()
    patch = np.s_[20:36, 22:38]
    low[patch] = 0.80
    feature[patch] = 0.90
    semantic[patch] = 0.70

    result = fuse_change_proposals(
        low,
        feature,
        semantic,
        _pif_without_patch(),
        _settings(),
    )

    assert len(result.proposals) == 1
    proposal = result.proposals[0]
    assert proposal.source == "fused_change_v2"
    assert proposal.proposal_id == "change_000"
    assert proposal.mask_filename == "change_000_mask.png"
    assert proposal.score == pytest.approx(0.825)
    assert proposal.component_scores == pytest.approx(
        {
            "low_level": 0.80,
            "feature": 0.90,
            "semantic": 0.70,
            "fused": 0.825,
        }
    )
    component_mask = result.component_masks[proposal.mask_filename]
    assert component_mask.dtype == np.uint8
    assert set(np.unique(component_mask)) <= {0, 255}
    assert int(np.count_nonzero(component_mask)) == 16 * 16
    assert result.diagnostics["threshold_mode"] == "pif_robust"
    assert result.diagnostics["version"] == PROPOSAL_FUSION_VERSION


def test_low_level_only_brightness_change_is_downweighted() -> None:
    low, feature, semantic = _maps()
    patch = np.s_[16:40, 16:40]
    low[patch] = 1.0

    result = fuse_change_proposals(
        low,
        feature,
        semantic,
        _pif_without_patch(patch=(slice(16, 40), slice(16, 40))),
        _settings(),
    )

    assert float(np.median(result.fused_score_map[patch])) == pytest.approx(0.25)
    assert result.proposals[0].component_scores["fused"] == pytest.approx(0.25)
    assert result.proposals[0].component_scores["feature"] == 0.0
    assert result.proposals[0].component_scores["semantic"] == 0.0


def test_feature_and_semantic_agreement_remains_high_without_low_level_change() -> None:
    low, feature, semantic = _maps()
    patch = np.s_[18:42, 19:43]
    feature[patch] = 1.0
    semantic[patch] = 1.0

    result = fuse_change_proposals(
        low,
        feature,
        semantic,
        _pif_without_patch(patch=(slice(18, 42), slice(19, 43))),
        _settings(),
    )

    assert float(np.median(result.fused_score_map[patch])) == pytest.approx(0.75)
    assert len(result.proposals) == 1
    assert result.proposals[0].score == pytest.approx(0.75)


def test_insufficient_pif_uses_explicit_quantile_fallback() -> None:
    low, feature, semantic = _maps()
    low[12:36, 12:36] = 0.8
    feature[12:36, 12:36] = 0.8
    semantic[12:36, 12:36] = 0.8
    pif = np.zeros((64, 64), dtype=np.uint8)
    pif[0, 0] = 1

    result = fuse_change_proposals(
        low,
        feature,
        semantic,
        pif,
        _settings(pif_fallback_quantile=0.80),
        min_pif_pixels=32,
    )

    assert result.diagnostics["threshold_mode"] == "quantile_fallback"
    assert result.diagnostics["pif_threshold_fallback_used"] is True
    assert result.diagnostics["pif_pixels"] == 1


def test_feature_and_semantic_maps_are_bilinearly_resized_to_canonical_grid() -> None:
    low = np.zeros((64, 64), dtype=np.float32)
    feature = np.zeros((16, 16), dtype=np.float32)
    semantic = np.zeros((32, 32), dtype=np.float32)
    feature[4:8, 4:8] = 1.0
    semantic[8:16, 8:16] = 1.0

    result = fuse_change_proposals(
        low,
        feature,
        semantic,
        _pif_without_patch(patch=(slice(16, 32), slice(16, 32))),
        _settings(),
    )

    assert result.fused_score_map.shape == (64, 64)
    assert result.binary_change_mask.shape == (64, 64)
    assert float(result.fused_score_map[24, 24]) > 0.70
    assert float(result.fused_score_map[0, 0]) == 0.0


def test_equal_components_use_stable_coordinate_ordering() -> None:
    low, feature, semantic = _maps()
    for row_slice, column_slice in (
        (slice(8, 16), slice(40, 48)),
        (slice(8, 16), slice(8, 16)),
        (slice(40, 48), slice(8, 16)),
    ):
        low[row_slice, column_slice] = 0.8
        feature[row_slice, column_slice] = 0.8
        semantic[row_slice, column_slice] = 0.8
    pif = (low == 0.0).astype(np.uint8)

    first = fuse_change_proposals(low, feature, semantic, pif, _settings())
    second = fuse_change_proposals(low, feature, semantic, pif, _settings())

    assert [proposal.pixel_box for proposal in first.proposals] == [
        [7, 7, 17, 17],
        [39, 7, 49, 17],
        [7, 39, 17, 49],
    ]
    assert [proposal.model_dump() for proposal in first.proposals] == [
        proposal.model_dump() for proposal in second.proposals
    ]
    assert all(
        np.array_equal(first.component_masks[name], second.component_masks[name])
        for name in first.component_masks
    )


def test_missing_component_weights_are_renormalized_over_available_maps() -> None:
    low = np.full((32, 32), 0.2, dtype=np.float32)
    feature = np.full((8, 8), 0.8, dtype=np.float32)

    result = fuse_change_proposals(
        low,
        feature,
        None,
        np.ones((32, 32), dtype=np.uint8),
        _settings(),
        fallback_reason="SEMANTIC_UNAVAILABLE",
    )

    assert float(np.median(result.fused_score_map)) == pytest.approx(0.6)
    assert result.diagnostics["available_components"] == ["low_level", "feature"]
    assert result.diagnostics["missing_components"] == ["semantic"]
    assert result.diagnostics["effective_weights"] == pytest.approx(
        {"low_level": 1 / 3, "feature": 2 / 3}
    )
    assert result.diagnostics["fallback_reason"] == "SEMANTIC_UNAVAILABLE"


def test_edge_component_padding_is_clipped_and_mask_is_crop_local() -> None:
    low, feature, semantic = _maps()
    low[0:8, 0:8] = 0.9
    feature[0:8, 0:8] = 0.9
    semantic[0:8, 0:8] = 0.9
    pif = np.ones((64, 64), dtype=np.uint8)
    pif[0:8, 0:8] = 0

    result = fuse_change_proposals(
        low,
        feature,
        semantic,
        pif,
        _settings(proposal_padding_ratio=0.25),
    )

    proposal = result.proposals[0]
    assert proposal.pixel_box == [0, 0, 10, 10]
    assert all(0 <= coordinate <= 64 for coordinate in proposal.pixel_box)
    mask = result.component_masks[proposal.mask_filename]
    assert mask.shape == (10, 10)
    assert bool(np.all(mask[:8, :8] == 255))
    assert bool(np.all(mask[8:, :] == 0))
    assert bool(np.all(mask[:, 8:] == 0))


def test_threshold_comparison_is_strict_greater_than() -> None:
    low = np.full((32, 32), 0.1, dtype=np.float32)
    feature = low.copy()
    semantic = low.copy()

    result = fuse_change_proposals(
        low,
        feature,
        semantic,
        np.ones((32, 32), dtype=np.uint8),
        _settings(threshold_floor=0.1),
    )

    assert result.diagnostics["threshold"] == pytest.approx(0.1)
    assert result.diagnostics["threshold_comparison"] == ">"
    assert not bool(np.any(result.binary_change_mask))
    assert result.proposals == []


def test_module_is_model_free_and_does_not_write_masks() -> None:
    source_path = Path(proposal_fusion_module.__file__ or "")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots = {
        node.names[0].name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
    }
    imported_roots.update(
        (node.module or "").split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )

    assert "models" not in imported_roots
    assert "pathlib" not in imported_roots


def test_nonfinite_map_is_rejected() -> None:
    low, feature, semantic = _maps()
    feature[0, 0] = np.nan

    with pytest.raises(ValueError, match="PROPOSAL_FUSION_FEATURE_NONFINITE"):
        fuse_change_proposals(
            low,
            feature,
            semantic,
            np.ones((64, 64), dtype=np.uint8),
            _settings(),
        )
