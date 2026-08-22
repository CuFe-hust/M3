from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import agents.change.proposal_fusion as proposal_fusion_module
from agents.change.proposal_fusion import (
    PROPOSAL_FUSION_VERSION,
    compute_reliabilities,
    compute_temporal_semantic_stability,
    fuse_change_proposals,
    fuse_feature_evidence,
    fuse_semantic_evidence,
)
from agents.change.schema import (
    HarmonizationDecision,
    RegistrationDecision,
    RegistrationMetrics,
    RegistrationReport,
)
from agents.change.settings import ChangeProposalSettings, ChangeReliabilitySettings


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


def test_semantic_evidence_fuses_maps_without_cross_taxonomy_shapes() -> None:
    isaid = np.zeros((16, 16), dtype=np.float32)
    oem = np.zeros((8, 8), dtype=np.float32)
    isaid[4:8, 4:8] = 0.8
    oem[2:4, 2:4] = 0.9

    fused, diagnostics = fuse_semantic_evidence(
        [isaid, oem], [0.8, 0.9]
    )

    assert fused.shape == (16, 16)
    assert diagnostics["method"] == "semantic_consensus_union"
    assert diagnostics["expert_count"] == 2
    assert float(fused[6, 6]) > 0.0


def test_feature_evidence_allows_partial_success() -> None:
    first = np.full((8, 8), 0.2, dtype=np.float32)
    second = np.full((4, 4), 0.8, dtype=np.float32)

    fused, diagnostics = fuse_feature_evidence([first, second], [0.7, 0.3])

    assert fused is not None
    assert fused.shape == first.shape
    assert diagnostics["method"] == "reliability_weighted_mean"


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


def test_shadow_learned_map_cannot_change_deterministic_result() -> None:
    low, feature, semantic = _maps()
    patch = np.s_[20:36, 22:38]
    low[patch] = 0.80
    feature[patch] = 0.90
    semantic[patch] = 0.70
    baseline = fuse_change_proposals(
        low, feature, semantic, _pif_without_patch(), _settings()
    )
    for learned_map in (np.ones_like(low), np.random.default_rng(7).random(low.shape)):
        shadow = fuse_change_proposals(
            low,
            feature,
            semantic,
            _pif_without_patch(),
            _settings(),
            learned_map=learned_map,
            learned_weight=1.0,
            learned_requested=True,
            learned_mode="shadow",
            learned_rescue_threshold=0.5,
            learned_rescue_min_reliability=0.0,
        )
        assert np.array_equal(shadow.fused_score_map, baseline.fused_score_map)
        assert [item.model_dump() for item in shadow.proposals] == [
            item.model_dump() for item in baseline.proposals
        ]


def test_learned_rescue_top_k_and_provenance_are_recorded() -> None:
    low, feature, semantic = _maps()
    learned = np.zeros_like(low)
    learned[4:10, 4:10] = 0.95
    learned[20:28, 20:28] = 0.90
    learned[40:48, 40:48] = 0.85
    result = fuse_change_proposals(
        low,
        feature,
        semantic,
        np.ones_like(low, dtype=np.uint8),
        _settings(),
        learned_map=learned,
        learned_weight=0.0,
        learned_requested=True,
        learned_mode="assist",
        learned_rescue_threshold=0.8,
        learned_rescue_min_reliability=0.0,
        learned_rescue_min_component_area_ratio=0.001,
        learned_rescue_max_proposals=2,
    )
    assert result.diagnostics["learned_rescue_component_count"] == 2
    assert result.diagnostics["learned_rescue_max_proposals"] == 2
    rescued = [item for item in result.diagnostics["components"] if item["learned_rescue"]]
    assert len(rescued) == 2
    assert all(item["source"] == "learned_rescue" for item in rescued)
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


def test_reliability_modulates_base_weights_without_changing_default_weights() -> None:
    low = np.full((32, 32), 0.2, dtype=np.float32)
    feature = np.full((32, 32), 0.8, dtype=np.float32)
    semantic = np.full((32, 32), 0.8, dtype=np.float32)

    result = fuse_change_proposals(
        low,
        feature,
        semantic,
        np.ones((32, 32), dtype=np.uint8),
        _settings(),
        reliability={
            "low_level": 0.1,
            "feature": 1.0,
            "semantic": 1.0,
            "registration": 0.25,
        },
    )

    assert result.diagnostics["reliability"]["registration"] == pytest.approx(0.25)
    assert result.diagnostics["effective_weights"]["low_level"] < 0.25
    assert sum(result.diagnostics["effective_weights"].values()) == pytest.approx(1.0)


def test_low_quality_registration_reduces_registration_reliability() -> None:
    report = RegistrationReport(
        decision=RegistrationDecision(
            version="global_registration_v1",
            status="applied",
            model="affine",
            reason_codes=["REGISTRATION_APPLIED"],
            used_for_comparison=True,
        ),
        metrics=RegistrationMetrics(
            match_count=20,
            inlier_count=4,
            inlier_ratio=0.20,
            median_reprojection_error=8.0,
            overlap_ratio=0.50,
        ),
        transform_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    reliability, diagnostics = compute_reliabilities(
        registration_report=report,
        feature_diagnostics={"pif_feature_cells": 64, "median_score_pif": 0.0},
        semantic_diagnostics={"mean_confidence": 0.9},
        harmonization_decision=None,
        settings=ChangeReliabilitySettings(),
    )

    assert reliability["registration"] < reliability["semantic"]
    assert diagnostics["raw"]["registration"] < 0.5


def _semantic_probabilities(first_label: int, second_label: int, size: int = 8) -> tuple[np.ndarray, np.ndarray]:
    first = np.zeros((2, size, size), dtype=np.float32)
    second = np.zeros_like(first)
    first[first_label] = 1.0
    second[second_label] = 1.0
    return first, second


def test_temporal_stability_reports_stable_pif_without_penalty() -> None:
    first, second = _semantic_probabilities(0, 0)
    diagnostics = compute_temporal_semantic_stability(
        first,
        second,
        pif_mask=np.ones((8, 8), dtype=np.uint8),
        class_names=("background", "building"),
        neutral_labels=("background",),
    )
    assert diagnostics["temporal_stability_status"] == "available"
    assert diagnostics["pif_pixel_count"] == 64
    assert diagnostics["pif_label_flip_rate_all"] == pytest.approx(0.0)
    assert diagnostics["temporal_stability_multiplier"] == pytest.approx(1.0)


def test_temporal_stability_penalty_is_smooth_and_optional() -> None:
    first, second = _semantic_probabilities(0, 1)
    enabled = compute_temporal_semantic_stability(
        first,
        second,
        pif_mask=np.ones((8, 8), dtype=np.uint8),
        class_names=("background", "building"),
        neutral_labels=("background",),
        enabled=True,
        soft_flip_rate=0.10,
        hard_flip_rate=0.90,
        floor=0.25,
    )
    disabled = compute_temporal_semantic_stability(
        first,
        second,
        pif_mask=np.ones((8, 8), dtype=np.uint8),
        class_names=("background", "building"),
        neutral_labels=("background",),
        enabled=False,
        soft_flip_rate=0.10,
        hard_flip_rate=0.90,
        floor=0.25,
    )
    assert enabled["pif_label_flip_rate_non_neutral"] == pytest.approx(1.0)
    assert enabled["temporal_stability_multiplier"] == pytest.approx(0.25)
    assert disabled["temporal_stability_multiplier"] == pytest.approx(1.0)


def test_temporal_stability_is_neutral_when_pif_is_unavailable() -> None:
    first, second = _semantic_probabilities(0, 1)
    diagnostics = compute_temporal_semantic_stability(
        first,
        second,
        pif_mask=np.zeros((8, 8), dtype=np.uint8),
        enabled=True,
    )
    assert diagnostics["temporal_stability_status"] == "unavailable"
    assert diagnostics["temporal_stability_multiplier"] == pytest.approx(1.0)


@pytest.mark.parametrize("reason_codes", [["REGISTRATION_NOT_NEEDED", "METADATA_ALIGNMENT_USED"], ["REGISTRATION_NOT_NEEDED", "IDENTICAL_INPUTS"]])
def test_trusted_identity_registration_has_full_reliability(
    reason_codes: list[str],
) -> None:
    report = RegistrationReport(
        decision=RegistrationDecision(
            version="global_registration_v1",
            status="skipped",
            model="identity",
            reason_codes=reason_codes,
            used_for_comparison=True,
        ),
        metrics=RegistrationMetrics(),
    )

    reliability, _ = compute_reliabilities(
        registration_report=report,
        feature_diagnostics=None,
        semantic_diagnostics=None,
        harmonization_decision=None,
        settings=ChangeReliabilitySettings(),
    )

    assert reliability["registration"] == pytest.approx(1.0)


def test_disabled_registration_has_legacy_reliability() -> None:
    report = RegistrationReport(
        decision=RegistrationDecision(
            version="global_registration_v1",
            status="skipped",
            model="identity",
            reason_codes=["REGISTRATION_DISABLED"],
            used_for_comparison=False,
        ),
    )
    reliability, _ = compute_reliabilities(
        registration_report=report,
        feature_diagnostics=None,
        semantic_diagnostics=None,
        harmonization_decision=None,
        settings=ChangeReliabilitySettings(),
    )
    assert reliability["registration"] == pytest.approx(1.0)


def test_invalid_overlap_is_a_hard_mask_for_threshold_and_components() -> None:
    low, feature, semantic = _maps(32)
    low[:, :] = 1.0
    feature[:, :] = 1.0
    semantic[:, :] = 1.0
    valid = np.zeros((32, 32), dtype=bool)
    valid[8:24, 8:24] = True

    result = fuse_change_proposals(
        low,
        feature,
        semantic,
        np.ones((32, 32), dtype=np.uint8),
        _settings(),
        valid_overlap_mask=valid,
    )

    assert not bool(np.any(result.binary_change_mask[~valid]))
    assert result.diagnostics["valid_overlap_ratio"] == pytest.approx(0.25)
    assert all(
        valid[proposal.pixel_box[1] : proposal.pixel_box[3], proposal.pixel_box[0] : proposal.pixel_box[2]].any()
        for proposal in result.proposals
    )


def test_invalid_overlap_remains_masked_after_morphology_close() -> None:
    low = np.zeros((32, 32), dtype=np.float32)
    feature = np.zeros((32, 32), dtype=np.float32)
    semantic = np.zeros((32, 32), dtype=np.float32)
    valid = np.zeros((32, 32), dtype=bool)
    valid[8:24, 8:16] = True
    valid[8:24, 18:26] = True
    low[valid] = 1.0
    feature[valid] = 1.0
    semantic[valid] = 1.0

    result = fuse_change_proposals(
        low,
        feature,
        semantic,
        np.zeros((32, 32), dtype=np.uint8),
        _settings(mask_close_kernel=5, threshold_floor=0.1),
        min_pif_pixels=1,
        valid_overlap_mask=valid,
    )

    assert not bool(np.any(result.binary_change_mask[~valid]))


@pytest.mark.parametrize(
    ("weights", "expected_score"),
    [
        ((0.50, 0.00, 0.50), 0.40),  # low + semantic
        ((1 / 3, 2 / 3, 0.00), 0.60),  # low + feature
        ((0.25, 0.50, 0.25), 0.60),  # three-source
    ],
)
def test_ablation_zero_weights_are_respected_without_runtime_switches(
    weights: tuple[float, float, float],
    expected_score: float,
) -> None:
    low = np.full((16, 16), 0.2, dtype=np.float32)
    feature = np.full((16, 16), 0.8, dtype=np.float32)
    semantic = np.full((16, 16), 0.6, dtype=np.float32)
    settings = _settings(
        fusion_low_level_weight=weights[0],
        fusion_feature_weight=weights[1],
        fusion_semantic_weight=weights[2],
    )

    result = fuse_change_proposals(
        low,
        feature,
        semantic,
        np.ones((16, 16), dtype=np.uint8),
        settings,
    )

    assert float(np.median(result.fused_score_map)) == pytest.approx(expected_score)
    assert result.diagnostics["effective_weights"] == pytest.approx(
        {
            "low_level": weights[0],
            "feature": weights[1],
            "semantic": weights[2],
        }
    )


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
