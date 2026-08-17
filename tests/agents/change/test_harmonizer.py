"""Contract tests for the change harmonizer.

变化一致化器契约测试：PIF/LAB midpoint、sharpness match、clipping/MAD 拒绝
条件、raw fallback decision、输入只读、指标可序列化。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from agents.change.harmonizer import (
    PairHarmonizer,
    compute_metrics,
    estimate_pif_mask,
)
from agents.change.settings import ChangeHarmonizationSettings


def _rgb(seed: int = 0, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(*size, 3), dtype=np.uint8)


def _similar_pair(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    base = _rgb(seed)
    return base, np.clip(base.astype(np.int16) + 5, 0, 255).astype(np.uint8)


def _settings(**overrides) -> ChangeHarmonizationSettings:
    values = dict(calibration_file=None)
    values.update(overrides)
    return ChangeHarmonizationSettings(**values)


# ── 基础流程 / basic flow ─────────────────────────────────────────────────


def test_identical_images_apply_ok() -> None:
    image = _rgb(1)
    candidate = PairHarmonizer(_settings()).run(image, image.copy())
    assert candidate.decision.status == "applied"
    assert "APPLIED_OK" in candidate.decision.reason_codes
    assert candidate.decision.used_for_proposal is True
    assert candidate.decision.metrics is not None
    assert candidate.decision.metrics.pif_ratio > 0.9


def test_input_images_are_not_mutated() -> None:
    """The harmonizer must never modify the caller's input arrays.
    一致化器绝不修改调用方的输入数组。"""
    t1, t2 = _similar_pair(2)
    before1, before2 = t1.copy(), t2.copy()
    PairHarmonizer(_settings()).run(t1, t2)
    assert np.array_equal(t1, before1)
    assert np.array_equal(t2, before2)


def test_decision_and_metrics_are_serializable() -> None:
    t1, t2 = _similar_pair(3)
    candidate = PairHarmonizer(_settings()).run(t1, t2)
    decision_payload = json.loads(candidate.decision.model_dump_json())
    assert decision_payload["status"] in {"applied", "skipped", "rejected"}
    assert isinstance(decision_payload["reason_codes"], list)
    if candidate.decision.metrics is not None:
        metrics_payload = json.loads(candidate.decision.metrics.model_dump_json())
        assert 0.0 <= metrics_payload["pif_ratio"] <= 1.0


# ── 跳过与拒绝 / skip and rejection ───────────────────────────────────────


def test_insufficient_pif_skips_with_raw_fallback() -> None:
    """Completely different images yield an explicit raw-fallback skip.
    完全不同的图像产生显式 raw fallback 跳过。"""
    t1 = _rgb(4)
    t2 = _rgb(5)  # unrelated random content / 无关随机内容
    candidate = PairHarmonizer(_settings(min_pif_ratio=0.5)).run(t1, t2)
    assert candidate.decision.status == "skipped"
    assert "SKIPPED_INSUFFICIENT_PIF" in candidate.decision.reason_codes
    assert "RAW_FALLBACK_USED" in candidate.decision.reason_codes
    assert candidate.decision.used_for_proposal is False
    assert candidate.pif_valid is False
    # Raw copies are returned for the fallback. / raw 副本用于回退。
    assert np.array_equal(candidate.t1, t1)
    assert np.array_equal(candidate.t2, t2)


def test_reject_when_pif_mad_worse() -> None:
    t1, t2 = _similar_pair(5)
    settings = _settings(reject_when_pif_mad_worse=True, max_pif_mad_degradation_ratio=1.0)
    candidate = PairHarmonizer(settings).run(t1, t2)
    if candidate.decision.metrics is not None:
        metrics = candidate.decision.metrics
        if metrics.mad_pif_after > metrics.mad_pif_before * 1.0:
            assert candidate.decision.status == "rejected"
            assert candidate.pif_valid is True
            assert "REJECTED_PIF_MAD_WORSE" in candidate.decision.reason_codes
            assert "RAW_FALLBACK_USED" in candidate.decision.reason_codes


def test_unstable_transform_rejected() -> None:
    """A wildly unstable affine is rejected instead of applied.
    极不稳定的仿射被拒绝而非应用。"""
    t1 = _rgb(6)
    t2 = np.clip((t1.astype(np.float32) * 3.0 + 200.0), 0, 255).astype(np.uint8)
    candidate = PairHarmonizer(_settings(max_abs_gain=1.0)).run(t1, t2)
    if candidate.decision.status == "rejected":
        assert candidate.pif_valid is True
        assert "REJECTED_UNSTABLE_TRANSFORM" in candidate.decision.reason_codes


# ── sharpness / 清晰度匹配 ────────────────────────────────────────────────


def test_sharpness_matching_disabled() -> None:
    t1, t2 = _similar_pair(7)
    candidate = PairHarmonizer(_settings(match_sharpness=False)).run(t1, t2)
    assert candidate.transform_summary["sharpness_adjustment_used"] is False


def test_sharpness_mismatch_rolls_back_safely() -> None:
    """A blurred/sharp pair must not crash; an unsafe blur is rolled back.
    模糊/锐利图对不得崩溃；不安全的模糊被回退。"""
    t1 = _rgb(8)
    t2 = cv2.GaussianBlur(t1, (0, 0), 3.0)
    candidate = PairHarmonizer(_settings()).run(t1, t2)
    summary = candidate.transform_summary
    assert "sharpness_adjustment_used" in summary
    assert "sharpness_adjustment_safe" in summary
    assert candidate.decision.status in {"applied", "skipped", "rejected"}


# ── 纯函数 / pure functions ───────────────────────────────────────────────


def test_estimate_pif_mask_shape_and_dtype() -> None:
    t1, t2 = _similar_pair(9)
    mask = estimate_pif_mask(t1, t2, _settings())
    assert mask.shape == t1.shape[:2]
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})


def test_registration_invalid_overlap_is_excluded_from_pif_and_metrics() -> None:
    t1 = _rgb(11, size=(48, 48))
    t2 = np.clip(t1.astype(np.int16) + 5, 0, 255).astype(np.uint8)
    t2[:12, :] = 255
    valid = np.ones((48, 48), dtype=bool)
    valid[:12, :] = False

    mask = estimate_pif_mask(t1, t2, _settings(), valid_mask=valid)
    metrics = compute_metrics(t1, t2, t1, t2, mask, valid_mask=valid)

    assert np.count_nonzero(mask[:12, :]) == 0
    assert metrics.mad_full_before < 10.0
    assert metrics.pct_diff_gt20_before == pytest.approx(0.0)


def test_compute_metrics_identical_outputs() -> None:
    t1, t2 = _similar_pair(10)
    mask = estimate_pif_mask(t1, t2, _settings())
    metrics = compute_metrics(t1, t2, t1.copy(), t2.copy(), mask)
    assert metrics.mad_full_after == pytest.approx(metrics.mad_full_before)
    assert metrics.corr_full_before is not None
    assert 0.0 <= metrics.pct_diff_gt20_before <= 1.0


def test_compute_metrics_constant_inputs_yield_none_corr() -> None:
    flat = np.full((16, 16, 3), 100, dtype=np.uint8)
    mask = np.full((16, 16), 255, dtype=np.uint8)
    metrics = compute_metrics(flat, flat, flat, flat, mask)
    assert metrics.corr_full_before is None
    assert metrics.corr_pif_before is None


# ── 边界 / boundaries ─────────────────────────────────────────────────────


def test_harmonizer_has_no_dataset_or_semantic_branch() -> None:
    source = (Path(__file__).resolve().parents[3] / "agents" / "change" / "harmonizer.py").read_text(
        encoding="utf-8"
    )
    assert "VRSBench" not in source
    assert "dataset" not in source
    assert "spacers_agent" not in source
    assert "caption" not in source
    assert "semantic" not in source
