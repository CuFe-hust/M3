from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from models.base import (
    LEARNED_CHANGE_INPUT_CONTRACT_VERSION,
    LEARNED_CHANGE_OUTPUT_CONTRACT_VERSION,
)
from models.change_head import checkpoint as checkpoint_module
from models.change_head.calibration import ChangeHeadCalibration
from scripts.calibrate_change_head import fit_temperature, search_rescue_threshold
from training.change_head.evaluator import evaluate_probability_maps
from training.change_head.release_gate import evaluate_release_gates


def _gate_config() -> dict[str, object]:
    return {
        "critical_no_change": {
            "max_new_false_positive_samples": 0,
            "max_new_false_positive_components": 0,
            "max_scene_fp_rate_increase": 0.0,
        },
        "normal_changed": {
            "max_proposal_recall_drop": 0.0,
            "max_proposal_f1_drop": 0.0,
        },
        "residual_hard_cases": {"require_net_improvement": True, "min_net_improvement": 1},
        "building_edge": {"max_proposal_recall_drop": 0.0, "allow_missing_subset": False},
        "broad_validation": {"max_proposal_f1_drop": 0.01, "max_pixel_f1_drop": 0.01},
    }


def _metrics(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "scene_nochange_fp_rate": 0.0,
        "normal_proposal_recall": 1.0,
        "normal_proposal_f1": 1.0,
        "proposal_f1": 1.0,
        "pixel_f1": 1.0,
        "building_edge_proposal_recall": 1.0,
        "critical_new_fp_samples": 0,
        "critical_new_fp_components": 0,
    }
    value.update(overrides)
    return value


def test_scene_nochange_fp_rate_is_per_scene() -> None:
    result = evaluate_probability_maps(
        [np.array([[0.9]]), np.array([[0.1]]), np.array([[0.9]])],
        [np.array([[0]]), np.array([[0]]), np.array([[1]])],
        [np.ones((1, 1), dtype=bool)] * 3,
    )
    assert result["nochange_scene_count"] == 2
    assert result["nochange_scene_fp_count"] == 1
    assert result["scene_nochange_fp_rate"] == pytest.approx(0.5)


def test_hard_case_rescued_is_not_hardcoded() -> None:
    result = evaluate_release_gates(
        shadow_parity=True,
        baseline=_metrics(),
        assist=_metrics(),
        residual_hard_cases_rescued=2,
        residual_hard_cases_regressed=0,
        config=_gate_config(),
    )
    assert result["gates"]["residual_hard_cases"] is True
    assert result["gate_details"]["hard_cases"]["rescued"] == 2


def test_hard_case_regression_reduces_net_gain() -> None:
    result = evaluate_release_gates(
        shadow_parity=True,
        baseline=_metrics(),
        assist=_metrics(),
        residual_hard_cases_rescued=2,
        residual_hard_cases_regressed=2,
        config=_gate_config(),
    )
    assert result["gate_details"]["hard_cases"]["net_improvement"] == 0
    assert result["gates"]["residual_hard_cases"] is False


def test_building_edge_gate_executes_and_missing_subset_is_not_pass() -> None:
    result = evaluate_release_gates(
        shadow_parity=True,
        baseline=_metrics(),
        assist=_metrics(building_edge_proposal_recall=0.9),
        residual_hard_cases_rescued=1,
        config=_gate_config(),
    )
    assert result["gates"]["building_edge"] is False
    assert result["gate_details"]["building_edge"]["status"] == "available"

    baseline_missing = _metrics()
    assist_missing = _metrics()
    baseline_missing.pop("building_edge_proposal_recall")
    assist_missing.pop("building_edge_proposal_recall")
    missing = evaluate_release_gates(
        shadow_parity=True,
        baseline=baseline_missing,
        assist=assist_missing,
        residual_hard_cases_rescued=1,
        config=_gate_config(),
    )
    assert missing["gate_details"]["building_edge"]["status"] == "skipped"
    assert missing["gates"]["building_edge"] is False


def test_pixel_f1_gate_executes() -> None:
    result = evaluate_release_gates(
        shadow_parity=True,
        baseline=_metrics(pixel_f1=1.0),
        assist=_metrics(pixel_f1=0.8),
        residual_hard_cases_rescued=1,
        config=_gate_config(),
    )
    assert result["gates"]["broad_pixel_f1"] is False
    assert result["gates"]["broad_validation"] is False


def test_critical_nochange_new_fp_blocks_release() -> None:
    result = evaluate_release_gates(
        shadow_parity=True,
        baseline=_metrics(),
        assist=_metrics(critical_new_fp_samples=1),
        residual_hard_cases_rescued=1,
        config=_gate_config(),
    )
    assert result["gates"]["critical_no_change"] is False
    assert result["passed"] is False


def test_temperature_fit_ignores_invalid_pixels() -> None:
    logits = np.array([[[-3.0, 3.0], [100.0, -100.0]]])
    targets = np.array([[[0, 1], [1, 0]]])
    valid = np.array([[[1, 1], [0, 0]]], dtype=bool)
    assert fit_temperature(logits, targets, valid) == fit_temperature(
        logits[:, :1, :], targets[:, :1, :]
    )


def test_threshold_search_prefers_safe_hardcase_improvement() -> None:
    threshold, metrics = search_rescue_threshold(
        [np.array([[0.1]]), np.array([[0.7, 0.4]])],
        [np.array([[0]]), np.array([[1, 0]])],
        [np.ones((1, 1), dtype=bool), np.ones((1, 2), dtype=bool)],
        tags=[["no_change"], ["hard_case"]],
        candidates=[0.5, 0.8],
    )
    assert threshold == 0.5
    assert metrics["scene_nochange_fp_rate"] == 0.0


def test_no_safe_threshold_marks_calibration_failed() -> None:
    with pytest.raises(ValueError, match="CALIBRATION_NO_SAFE_RESCUE_THRESHOLD"):
        search_rescue_threshold(
            [np.array([[0.8]])],
            [np.array([[0]])],
            [np.ones((1, 1), dtype=bool)],
            candidates=[0.5, 0.7],
        )


def _manifest_payload(weight_sha: str) -> dict[str, object]:
    return {
        "input_contract_version": LEARNED_CHANGE_INPUT_CONTRACT_VERSION,
        "output_contract_version": LEARNED_CHANGE_OUTPUT_CONTRACT_VERSION,
        "architecture": {
            "name": "multi_expert_siamese_change_head_v1",
            "hidden_dim": 4,
            "semantic_dim": 4,
            "decoder_dim": 4,
            "optional_expert_dropout_supported": True,
            "use_pif_mask": True,
            "use_rgb_pair": False,
        },
        "experts": [{
            "expert_id": "expert_1",
            "logical_model_id": "SegFormer-MiT-B2:iSAID:local",
            "weights_sha256": "a" * 64,
            "class_names_sha256": "b" * 64,
            "feature_stages": [1],
            "feature_channels_by_stage": {"1": 2},
            "required": True,
            "use_semantic_probabilities": False,
            "missing_policy": "error",
        }],
        "pipeline_fingerprint": "c" * 64,
        "model_weights_sha256": weight_sha,
        "created_from_git_commit": "test",
        "training_manifest_sha256": "e" * 64,
    }


def test_calibration_checkpoint_sha_mismatch_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "checkpoint"
    root.mkdir()
    weight_sha = "d" * 64
    (root / "manifest.json").write_text(json.dumps(_manifest_payload(weight_sha)), encoding="utf-8")
    (root / "calibration.json").write_text(json.dumps({
        "temperature": 1.0,
        "rescue_probability_threshold": 0.8,
        "rescue_min_component_area_ratio": 0.01,
        "validation_reliability": 0.9,
        "created_from_checkpoint_sha256": "f" * 64,
    }), encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(checkpoint_module, "validate_local_model_asset", lambda *args, **kwargs: None)
    with pytest.raises(checkpoint_module.ChangeHeadCheckpointError, match="LEARNED_CHANGE_CALIBRATION_INVALID"):
        checkpoint_module.load_change_head_checkpoint(root)


def test_calibration_model_persists_validation_identity() -> None:
    calibration = ChangeHeadCalibration(
        temperature=1.0,
        rescue_probability_threshold=0.8,
        rescue_min_component_area_ratio=0.01,
        validation_reliability=0.9,
        validation_fingerprint="v" * 64,
        created_from_checkpoint_sha256="w" * 64,
        metrics={"post_ece": 0.1},
    )
    assert calibration.validation_fingerprint.startswith("v")
    assert calibration.metrics["post_ece"] == 0.1
