from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import agents.change.feature_residual as feature_residual_module
from agents.change.feature_residual import (
    FEATURE_RESIDUAL_VERSION,
    compute_feature_residual,
    compute_multiscale_feature_residual,
)


def _features(seed: int = 7, shape: tuple[int, int, int] = (8, 24, 24)) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=shape).astype(np.float32)


def _full_pif(height: int = 24, width: int = 24) -> np.ndarray:
    return np.ones((height, width), dtype=np.uint8)


def test_identical_features_have_near_zero_residual() -> None:
    first = _features()

    result = compute_feature_residual(first, first.copy(), _full_pif())

    assert result.score_map.dtype == np.float32
    assert result.valid_mask.dtype == np.bool_
    assert bool(np.all(result.valid_mask))
    assert float(np.max(result.score_map)) < 1e-6
    assert result.diagnostics["alignment_status"] == "aligned"
    assert result.diagnostics["version"] == FEATURE_RESIDUAL_VERSION


def test_multiscale_residual_is_deterministic_and_reports_stage_diagnostics() -> None:
    rng = np.random.default_rng(121)
    first = {
        1: rng.normal(size=(8, 16, 16)).astype(np.float32),
        2: rng.normal(size=(12, 8, 8)).astype(np.float32),
        3: rng.normal(size=(16, 4, 4)).astype(np.float32),
    }
    second = {stage: value.copy() for stage, value in first.items()}
    pif = _full_pif(64, 64)
    kwargs = {
        "feature_stages": (1, 2, 3),
        "feature_stage_weights": {1: 0.4, 2: 0.35, 3: 0.25},
        "image_size": (64, 64),
        "min_pif_feature_cells": 4,
        "local_match_radius": 1,
    }
    first_result = compute_multiscale_feature_residual(first, second, pif, **kwargs)
    second_result = compute_multiscale_feature_residual(first, second, pif, **kwargs)

    np.testing.assert_array_equal(first_result.score_map, second_result.score_map)
    assert first_result.diagnostics["effective_stages"] == [1, 2, 3]
    assert first_result.diagnostics["missing_stages"] == []
    assert [item["stage"] for item in first_result.diagnostics["per_stage"]] == [1, 2, 3]


def test_multiscale_residual_reports_missing_stage_without_fabricating_features() -> None:
    first = {1: _features(shape=(4, 8, 8))}
    second = {1: first[1].copy()}
    result = compute_multiscale_feature_residual(
        first,
        second,
        _full_pif(32, 32),
        feature_stages=(1, 2),
        image_size=(32, 32),
        min_pif_feature_cells=4,
    )
    assert result.diagnostics["effective_stages"] == [1]
    assert result.diagnostics["missing_stages"] == [2]


def test_per_channel_shift_and_scale_are_removed_by_robust_normalization() -> None:
    first = _features(seed=11)
    scale = np.linspace(0.5, 2.2, first.shape[0], dtype=np.float32)[:, None, None]
    shift = np.linspace(-4.0, 3.0, first.shape[0], dtype=np.float32)[:, None, None]
    second = (first * scale + shift).astype(np.float32)

    result = compute_feature_residual(
        first,
        second,
        _full_pif(),
        local_match_radius=0,
    )

    first_unit = first / np.maximum(np.linalg.norm(first, axis=0), 1e-6)
    second_unit = second / np.maximum(np.linalg.norm(second, axis=0), 1e-6)
    naive_residual = np.clip(
        (1.0 - np.sum(first_unit * second_unit, axis=0)) / 2.0,
        0.0,
        1.0,
    )

    assert float(np.median(result.score_map)) < 1e-6
    assert float(np.median(naive_residual)) > 0.05


def test_local_direction_change_scores_higher_than_unchanged_region() -> None:
    first = _features(seed=19)
    second = first.copy()
    changed = np.zeros(first.shape[1:], dtype=bool)
    changed[8:16, 9:17] = True
    second[:, changed] *= np.float32(-1.0)

    result = compute_feature_residual(
        first,
        second,
        _full_pif(),
        local_match_radius=0,
    )

    changed_median = float(np.median(result.score_map[changed]))
    unchanged_median = float(np.median(result.score_map[~changed]))
    assert changed_median > 0.75
    assert changed_median > unchanged_median + 0.5


def test_one_cell_translation_is_tolerated_by_radius_one() -> None:
    first = _features(seed=23, shape=(16, 20, 20))
    second = np.empty_like(first)
    second[:, :, 1:] = first[:, :, :-1]
    second[:, :, 0] = _features(seed=29, shape=(16, 20, 1))[:, :, 0]

    exact = compute_feature_residual(
        first,
        second,
        _full_pif(20, 20),
        local_match_radius=0,
    )
    tolerant = compute_feature_residual(
        first,
        second,
        _full_pif(20, 20),
        local_match_radius=1,
    )

    interior = np.s_[:, :-1]
    exact_median = float(np.median(exact.score_map[interior]))
    tolerant_median = float(np.median(tolerant.score_map[interior]))
    assert exact_median > 0.25
    assert tolerant_median < 0.01
    assert tolerant_median < exact_median * 0.1


def test_ten_percent_pif_outliers_do_not_destroy_alignment() -> None:
    first = _features(seed=31, shape=(12, 30, 30))
    scale = np.linspace(0.7, 1.8, first.shape[0], dtype=np.float32)[:, None, None]
    shift = np.linspace(-2.0, 2.0, first.shape[0], dtype=np.float32)[:, None, None]
    second = (first * scale + shift).astype(np.float32)
    rng = np.random.default_rng(37)
    contaminated = np.zeros(first.shape[1:], dtype=bool)
    indices = rng.choice(contaminated.size, size=contaminated.size // 10, replace=False)
    contaminated.flat[indices] = True
    second[:, contaminated] = rng.normal(
        loc=100.0,
        scale=20.0,
        size=(first.shape[0], int(np.count_nonzero(contaminated))),
    ).astype(np.float32)

    result = compute_feature_residual(
        first,
        second,
        _full_pif(30, 30),
        local_match_radius=0,
    )

    assert float(np.median(result.score_map[~contaminated])) < 0.01
    assert float(np.median(result.score_map)) < 0.02


def test_insufficient_pif_is_explicit_and_does_not_estimate_from_full_grid() -> None:
    first = _features(seed=41, shape=(4, 8, 8))
    second = _features(seed=43, shape=(4, 8, 8))
    pif = np.zeros((8, 8), dtype=np.uint8)
    pif[0, 0] = 1

    result = compute_feature_residual(
        first,
        second,
        pif,
        min_pif_feature_cells=2,
    )

    assert result.diagnostics["alignment_status"] == "insufficient_pif"
    assert result.diagnostics["pif_feature_cells"] == 1
    assert not bool(np.any(result.valid_mask))
    assert bool(np.all(result.score_map == 0.0))
    assert bool(np.all(np.isfinite(result.score_map)))


def test_constant_channel_is_protected_by_epsilon() -> None:
    first = _features(seed=47)
    first[0] = np.float32(5.0)
    scale = np.linspace(0.8, 1.6, first.shape[0], dtype=np.float32)[:, None, None]
    shift = np.linspace(-1.0, 1.0, first.shape[0], dtype=np.float32)[:, None, None]
    second = (first * scale + shift).astype(np.float32)

    result = compute_feature_residual(
        first,
        second,
        _full_pif(),
        local_match_radius=0,
    )

    assert bool(np.all(np.isfinite(result.score_map)))
    assert float(np.median(result.score_map)) < 1e-6


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_features_are_rejected(bad_value: float) -> None:
    first = _features()
    second = first.copy()
    second[0, 0, 0] = bad_value

    with pytest.raises(ValueError, match="FEATURE_RESIDUAL_NONFINITE"):
        compute_feature_residual(first, second, _full_pif())


def test_pif_resize_explicitly_uses_nearest_neighbor(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def resize(image: np.ndarray, size: tuple[int, int], *, interpolation: int) -> np.ndarray:
        calls.append(interpolation)
        return cv2.resize(image, size, interpolation=interpolation)

    monkeypatch.setattr(
        feature_residual_module,
        "_require_cv2",
        lambda: SimpleNamespace(INTER_NEAREST=cv2.INTER_NEAREST, resize=resize),
    )
    first = _features(seed=53, shape=(4, 4, 4))

    result = compute_feature_residual(
        first,
        first.copy(),
        np.ones((16, 16), dtype=np.uint8),
        min_pif_feature_cells=1,
    )

    assert calls == [cv2.INTER_NEAREST]
    assert result.diagnostics["pif_feature_cells"] == 16


def test_score_range_and_diagnostics_contract() -> None:
    first = _features(seed=59)
    second = _features(seed=61)

    result = compute_feature_residual(first, second, _full_pif())

    assert float(np.min(result.score_map)) >= 0.0
    assert float(np.max(result.score_map)) <= 1.0
    assert {
        "pif_feature_cells",
        "pif_feature_ratio",
        "local_match_radius",
        "channels",
        "feature_height",
        "feature_width",
        "median_score_pif",
        "median_score_full",
        "p95_score_full",
        "alignment_status",
        "version",
    } <= result.diagnostics.keys()


def test_multiscale_residual_keeps_change_outside_pif() -> None:
    pif = np.ones((64, 64), dtype=np.uint8)
    change = np.s_[24:40, 24:40]
    pif[change] = 0
    features_t1_by_stage: dict[int, np.ndarray] = {}
    features_t2_by_stage: dict[int, np.ndarray] = {}
    for stage, size in ((1, 8), (2, 16), (3, 32)):
        first = _features(seed=stage + 100, shape=(8, size, size))
        second = first.copy()
        y1, y2 = int(24 * size / 64), int(40 * size / 64)
        second[:, y1:y2, y1:y2] *= np.float32(-1.0)
        features_t1_by_stage[stage] = first
        features_t2_by_stage[stage] = second

    valid_mask = np.ones((64, 64), dtype=bool)
    valid_mask[:8, :8] = False
    result = compute_multiscale_feature_residual(
        features_t1_by_stage,
        features_t2_by_stage,
        pif,
        feature_stages=(1, 2, 3),
        image_size=(64, 64),
        valid_mask=valid_mask,
        local_match_radius=0,
        min_pif_feature_cells=8,
    )

    assert float(np.median(result.score_map[change])) > 0.10
    assert bool(np.any(result.score_map[pif == 0] > 0.0))
    assert not bool(np.any(result.score_map[:8, :8]))
    assert result.diagnostics["pif_feature_cells"] < 64 * 64


def test_module_has_no_model_dependency() -> None:
    source_path = Path(feature_residual_module.__file__ or "")
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


def test_large_feature_grid_smoke_uses_vectorized_spatial_math() -> None:
    first = _features(seed=67, shape=(320, 64, 64))
    second = first.copy()
    second[:, 24:40, 24:40] *= np.float32(-1.0)

    result = compute_feature_residual(
        first,
        second,
        _full_pif(64, 64),
        local_match_radius=1,
    )

    assert result.score_map.shape == (64, 64)
    assert bool(np.all(np.isfinite(result.score_map)))
    source = Path(feature_residual_module.__file__ or "").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_ranges = {
        argument.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
        for argument in node.args
        if isinstance(argument, ast.Name)
        and argument.id in {"height", "width", "channels"}
    }
    assert forbidden_ranges == set()


@pytest.mark.parametrize(
    ("first", "second", "pif", "error_code"),
    [
        (
            np.zeros((2, 3, 4), dtype=np.float32),
            np.zeros((2, 3, 5), dtype=np.float32),
            np.ones((3, 4), dtype=np.uint8),
            "FEATURE_RESIDUAL_SHAPE_MISMATCH",
        ),
        (
            np.zeros((2, 3, 4), dtype=np.float64),
            np.zeros((2, 3, 4), dtype=np.float32),
            np.ones((3, 4), dtype=np.uint8),
            "FEATURE_RESIDUAL_DTYPE_INVALID",
        ),
        (
            np.zeros((2, 3, 4), dtype=np.float32),
            np.zeros((2, 3, 4), dtype=np.float32),
            np.ones((3, 4), dtype=np.float32),
            "FEATURE_RESIDUAL_PIF_DTYPE_INVALID",
        ),
    ],
)
def test_invalid_contract_inputs_are_rejected(
    first: np.ndarray,
    second: np.ndarray,
    pif: np.ndarray,
    error_code: str,
) -> None:
    with pytest.raises(ValueError, match=error_code):
        compute_feature_residual(first, second, pif)
