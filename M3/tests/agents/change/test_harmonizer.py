"""Deterministic harmonizer and gate tests. / 确定性一致化与门控测试。"""

import cv2
import numpy as np

from spacers_agent.agents.change.harmonizer import PairHarmonizer, _lapvar, _match_sharpness
from spacers_agent.agents.change.schemas import HarmonizationMetrics
from spacers_agent.settings import ChangeHarmonizationSettings


def _texture(size: int = 64) -> np.ndarray:
    y, x = np.mgrid[:size, :size]
    base = ((x * 5 + y * 3) % 180 + 30).astype(np.uint8)
    return np.stack([base, np.roll(base, 3, axis=0), np.roll(base, 4, axis=1)], axis=2)


def test_identical_pair_preserves_shape_dtype_and_input() -> None:
    image = _texture()
    original = image.copy()
    result = PairHarmonizer(ChangeHarmonizationSettings()).run(image, image)
    assert result.decision.status == "applied"
    assert result.t1.shape == image.shape and result.t1.dtype == np.uint8
    assert np.array_equal(image, original)
    assert result.decision.metrics is not None
    assert result.decision.metrics.corr_full_before == 1.0


def test_brightness_shift_reduces_pif_mad_deterministically() -> None:
    first = _texture()
    second = np.clip(first.astype(np.int16) + 35, 0, 255).astype(np.uint8)
    harmonizer = PairHarmonizer(ChangeHarmonizationSettings(match_sharpness=False))
    one, two = harmonizer.run(first, second), harmonizer.run(first, second)
    assert one.decision.status == "applied"
    assert one.decision.metrics is not None
    assert one.decision.metrics.mad_pif_after < one.decision.metrics.mad_pif_before
    assert np.array_equal(one.t1, two.t1) and np.array_equal(one.pif_mask, two.pif_mask)


def test_constant_pair_never_serializes_nan_correlation() -> None:
    first = np.full((32, 32, 3), 100, dtype=np.uint8)
    second = np.full((32, 32, 3), 120, dtype=np.uint8)
    result = PairHarmonizer(ChangeHarmonizationSettings(match_sharpness=False)).run(first, second)
    assert result.decision.metrics is not None
    assert result.decision.metrics.corr_full_before is None
    assert "NaN" not in result.decision.model_dump_json()


def test_small_pif_is_skipped_with_raw_fallback() -> None:
    black = np.zeros((16, 16, 3), dtype=np.uint8)
    result = PairHarmonizer(ChangeHarmonizationSettings(min_pif_pixels=10)).run(black, black)
    assert result.decision.status == "skipped"
    assert "RAW_FALLBACK_USED" in result.decision.reason_codes
    assert not result.decision.used_for_proposal


def test_pif_mad_degradation_is_rejected(monkeypatch) -> None:
    degraded = HarmonizationMetrics(
        pif_ratio=0.5, mad_full_before=10, mad_full_after=11,
        mad_pif_before=10, mad_pif_after=11,
        pct_diff_gt20_before=0.1, pct_diff_gt20_after=0.2,
        lapvar_t1_before=100, lapvar_t2_before=100,
        lapvar_t1_after=90, lapvar_t2_after=90,
    )
    monkeypatch.setattr("spacers_agent.agents.change.harmonizer.compute_metrics", lambda *args: degraded)
    result = PairHarmonizer(ChangeHarmonizationSettings(match_sharpness=False)).run(_texture(), np.roll(_texture(), 1, axis=0))
    assert result.decision.status == "rejected"
    assert "REJECTED_PIF_MAD_WORSE" in result.decision.reason_codes
    assert "RAW_FALLBACK_USED" in result.decision.reason_codes


def test_sharpness_match_never_sharpens_blurry_side() -> None:
    sharp = _texture()
    blurry = cv2.GaussianBlur(sharp, (0, 0), 0.5)
    out_sharp, out_blurry, used, _, _, _ = _match_sharpness(
        sharp, blurry, ChangeHarmonizationSettings(min_retained_lapvar_ratio=0.1)
    )
    assert np.array_equal(out_blurry, blurry)
    assert _lapvar(out_sharp) <= _lapvar(sharp)
    assert used
