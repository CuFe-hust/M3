"""PIF-guided robust feature normalization and local cosine residual.

This module is deliberately limited to deterministic array mathematics.  It
does not load models, publish artifacts, or make network/model calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.errors import OptionalDependencyMissingError


FEATURE_RESIDUAL_VERSION = "pif_robust_local_cosine_v1"


def _require_numpy():
    try:
        import numpy as np
    except ImportError as error:
        raise OptionalDependencyMissingError("change", dependency="numpy") from error
    return np


def _require_cv2():
    try:
        import cv2
    except ImportError as error:
        raise OptionalDependencyMissingError(
            "change", dependency="opencv-python-headless"
        ) from error
    return cv2


@dataclass(frozen=True)
class FeatureResidualResult:
    """Feature-grid residual and its compact, serializable diagnostics."""

    score_map: Any
    valid_mask: Any
    diagnostics: dict[str, object]


def compute_feature_residual(
    features_t1: Any,
    features_t2: Any,
    pif_mask: Any,
    *,
    local_match_radius: int = 1,
    min_pif_feature_cells: int = 32,
    feature_scale_epsilon: float = 1e-3,
) -> FeatureResidualResult:
    """Compute a PIF-normalized, translation-tolerant cosine residual.

    ``features_t1`` and ``features_t2`` must be float32 arrays in ``[D,H,W]``
    order. ``pif_mask`` must be a two-dimensional uint8 or boolean array at
    image resolution; it is resized to the feature grid with nearest-neighbor
    interpolation.
    """

    np = _require_numpy()
    cv2 = _require_cv2()
    feature_1, feature_2 = _validate_inputs(
        features_t1,
        features_t2,
        pif_mask,
        local_match_radius=local_match_radius,
        min_pif_feature_cells=min_pif_feature_cells,
        feature_scale_epsilon=feature_scale_epsilon,
        np=np,
    )
    pif = np.asarray(pif_mask)
    channels, height, width = feature_1.shape
    pif_grid = cv2.resize(
        (pif != 0).astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool, copy=False)
    pif_cells = int(np.count_nonzero(pif_grid))

    common_diagnostics: dict[str, object] = {
        "pif_feature_cells": pif_cells,
        "pif_feature_ratio": float(pif_cells / (height * width)),
        "local_match_radius": local_match_radius,
        "channels": channels,
        "feature_height": height,
        "feature_width": width,
        "version": FEATURE_RESIDUAL_VERSION,
    }
    if pif_cells < min_pif_feature_cells:
        score_map = np.zeros((height, width), dtype=np.float32)
        valid_mask = np.zeros((height, width), dtype=bool)
        return FeatureResidualResult(
            score_map=score_map,
            valid_mask=valid_mask,
            diagnostics={
                **common_diagnostics,
                "median_score_pif": 0.0,
                "median_score_full": 0.0,
                "p95_score_full": 0.0,
                "alignment_status": "insufficient_pif",
            },
        )

    normalized_1, nonzero_1 = _robust_normalize(
        feature_1,
        pif_grid,
        epsilon=feature_scale_epsilon,
        np=np,
    )
    normalized_2, nonzero_2 = _robust_normalize(
        feature_2,
        pif_grid,
        epsilon=feature_scale_epsilon,
        np=np,
    )
    score_map, valid_mask = _local_cosine_residual(
        normalized_1,
        normalized_2,
        nonzero_1,
        nonzero_2,
        radius=local_match_radius,
        np=np,
    )

    full_scores = score_map[valid_mask]
    pif_scores = score_map[pif_grid & valid_mask]
    diagnostics = {
        **common_diagnostics,
        "median_score_pif": float(np.median(pif_scores)),
        "median_score_full": float(np.median(full_scores)),
        "p95_score_full": float(np.percentile(full_scores, 95)),
        "alignment_status": "aligned",
    }
    return FeatureResidualResult(
        score_map=score_map,
        valid_mask=valid_mask,
        diagnostics=diagnostics,
    )


def _validate_inputs(
    features_t1: Any,
    features_t2: Any,
    pif_mask: Any,
    *,
    local_match_radius: int,
    min_pif_feature_cells: int,
    feature_scale_epsilon: float,
    np: Any,
) -> tuple[Any, Any]:
    feature_1 = np.asarray(features_t1)
    feature_2 = np.asarray(features_t2)
    if feature_1.ndim != 3 or any(dimension <= 0 for dimension in feature_1.shape):
        raise ValueError("FEATURE_RESIDUAL_SHAPE_INVALID: expected non-empty [D,H,W]")
    if feature_2.shape != feature_1.shape:
        raise ValueError("FEATURE_RESIDUAL_SHAPE_MISMATCH")
    if feature_1.dtype != np.float32 or feature_2.dtype != np.float32:
        raise ValueError("FEATURE_RESIDUAL_DTYPE_INVALID: features must be float32")
    if not bool(np.all(np.isfinite(feature_1))) or not bool(
        np.all(np.isfinite(feature_2))
    ):
        raise ValueError("FEATURE_RESIDUAL_NONFINITE")

    pif = np.asarray(pif_mask)
    if pif.ndim != 2 or any(dimension <= 0 for dimension in pif.shape):
        raise ValueError("FEATURE_RESIDUAL_PIF_SHAPE_INVALID")
    if pif.dtype != np.bool_ and pif.dtype != np.uint8:
        raise ValueError("FEATURE_RESIDUAL_PIF_DTYPE_INVALID: expected bool or uint8")
    if isinstance(local_match_radius, bool) or not isinstance(local_match_radius, int):
        raise ValueError("FEATURE_RESIDUAL_RADIUS_INVALID")
    if local_match_radius < 0:
        raise ValueError("FEATURE_RESIDUAL_RADIUS_INVALID")
    if isinstance(min_pif_feature_cells, bool) or not isinstance(
        min_pif_feature_cells, int
    ):
        raise ValueError("FEATURE_RESIDUAL_MIN_PIF_INVALID")
    if min_pif_feature_cells < 1:
        raise ValueError("FEATURE_RESIDUAL_MIN_PIF_INVALID")
    if (
        isinstance(feature_scale_epsilon, bool)
        or not isinstance(feature_scale_epsilon, (int, float))
        or not bool(np.isfinite(feature_scale_epsilon))
        or feature_scale_epsilon <= 0
    ):
        raise ValueError("FEATURE_RESIDUAL_EPSILON_INVALID")
    return feature_1, feature_2


def _robust_normalize(
    features: Any,
    pif_grid: Any,
    *,
    epsilon: float,
    np: Any,
) -> tuple[Any, Any]:
    """Normalize channels from PIF statistics, then spatial feature vectors."""

    channels, height, width = features.shape
    # Float64 working values keep subtraction and norms finite even when valid
    # float32 inputs lie near opposite representable extremes.
    working = features.astype(np.float64, copy=False)
    pif_values = working[:, pif_grid]
    medians = np.median(pif_values, axis=1, keepdims=True)
    deviations = np.median(np.abs(pif_values - medians), axis=1, keepdims=True)
    scales = np.maximum(1.4826 * deviations, float(epsilon))
    standardized = (working.reshape(channels, -1) - medians) / scales
    standardized = standardized.reshape(channels, height, width)
    norms = np.linalg.norm(standardized, axis=0)
    nonzero = norms > float(epsilon)
    normalized = np.divide(
        standardized,
        norms[None, :, :],
        out=np.zeros_like(standardized, dtype=np.float32),
        where=nonzero[None, :, :],
    )
    return normalized.astype(np.float32, copy=False), nonzero


def _local_cosine_residual(
    normalized_1: Any,
    normalized_2: Any,
    nonzero_1: Any,
    nonzero_2: Any,
    *,
    radius: int,
    np: Any,
) -> tuple[Any, Any]:
    """Take the minimum valid cosine distance across local integer offsets."""

    _, height, width = normalized_1.shape
    best_distance = np.full((height, width), np.inf, dtype=np.float32)
    valid_mask = np.zeros((height, width), dtype=bool)

    for delta_y in range(-radius, radius + 1):
        y1_start = max(0, -delta_y)
        y1_end = min(height, height - delta_y)
        y2_start = y1_start + delta_y
        y2_end = y1_end + delta_y
        for delta_x in range(-radius, radius + 1):
            x1_start = max(0, -delta_x)
            x1_end = min(width, width - delta_x)
            x2_start = x1_start + delta_x
            x2_end = x1_end + delta_x
            if y1_start >= y1_end or x1_start >= x1_end:
                continue

            first = normalized_1[:, y1_start:y1_end, x1_start:x1_end]
            second = normalized_2[:, y2_start:y2_end, x2_start:x2_end]
            cosine = np.sum(first * second, axis=0, dtype=np.float32)
            first_nonzero = nonzero_1[y1_start:y1_end, x1_start:x1_end]
            second_nonzero = nonzero_2[y2_start:y2_end, x2_start:x2_end]
            both_zero = ~first_nonzero & ~second_nonzero
            one_zero = first_nonzero ^ second_nonzero
            cosine = np.where(both_zero, np.float32(1.0), cosine)
            cosine = np.where(one_zero, np.float32(0.0), cosine)
            distance = np.float32(1.0) - np.clip(
                cosine, np.float32(-1.0), np.float32(1.0)
            )

            destination = best_distance[y1_start:y1_end, x1_start:x1_end]
            np.minimum(destination, distance, out=destination)
            valid_mask[y1_start:y1_end, x1_start:x1_end] = True

    score_map = np.zeros((height, width), dtype=np.float32)
    score_map[valid_mask] = np.clip(
        best_distance[valid_mask] / np.float32(2.0),
        np.float32(0.0),
        np.float32(1.0),
    )
    return score_map, valid_mask


__all__ = [
    "FEATURE_RESIDUAL_VERSION",
    "FeatureResidualResult",
    "compute_feature_residual",
]
