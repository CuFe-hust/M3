"""PIF-guided robust feature normalization and local cosine residual.

This module is deliberately limited to deterministic array mathematics.  It
does not load models, publish artifacts, or make network/model calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
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


def compute_multiscale_feature_residual(
    features_t1_by_stage: Mapping[int, Any],
    features_t2_by_stage: Mapping[int, Any],
    pif_mask: Any,
    *,
    feature_stages: tuple[int, ...],
    feature_stage_weights: Mapping[int, float] | None = None,
    feature_strides_by_stage: Mapping[int, tuple[float, float]] | None = None,
    image_size: tuple[int, int] | None = None,
    valid_mask: Any | None = None,
    local_match_radius: int = 1,
    min_pif_feature_cells: int = 32,
    feature_scale_epsilon: float = 1e-3,
) -> FeatureResidualResult:
    """Fuse deterministic local-cosine residuals from several feature grids.

    Each stage is normalized from the same PIF mask independently.  Scores are
    then resized to one canonical grid before weighted fusion, so a coarse
    stage cannot silently change the meaning of a pixel merely because its
    native stride differs.  PIF is used for normalization and diagnostics, not
    as the final change-validity mask.  Missing requested stages are reported
    explicitly; no stage is fabricated by resizing another feature tensor.
    """

    np = _require_numpy()
    cv2 = _require_cv2()
    stages = tuple(dict.fromkeys(feature_stages))
    if not stages:
        raise ValueError("FEATURE_RESIDUAL_STAGES_EMPTY")
    missing_stages = [
        stage
        for stage in stages
        if stage not in features_t1_by_stage or stage not in features_t2_by_stage
    ]
    effective_stages = [stage for stage in stages if stage not in missing_stages]
    if image_size is None:
        if not effective_stages:
            raise ValueError("FEATURE_RESIDUAL_STAGE_MISSING")
        first_stage = effective_stages[0]
        first = np.asarray(features_t1_by_stage.get(first_stage))
        if first.ndim != 3:
            raise ValueError("FEATURE_RESIDUAL_CANONICAL_SIZE_MISSING")
        canonical_height, canonical_width = int(first.shape[1]), int(first.shape[2])
    else:
        if len(image_size) != 2 or image_size[0] <= 0 or image_size[1] <= 0:
            raise ValueError("FEATURE_RESIDUAL_CANONICAL_SIZE_INVALID")
        canonical_width, canonical_height = int(image_size[0]), int(image_size[1])

    weights = {
        int(stage): float((feature_stage_weights or {}).get(stage, 1.0))
        for stage in stages
    }
    if any(value <= 0.0 or not np.isfinite(value) for value in weights.values()):
        raise ValueError("FEATURE_RESIDUAL_STAGE_WEIGHT_INVALID")

    if not effective_stages:
        raise ValueError("FEATURE_RESIDUAL_STAGE_MISSING")

    score_accumulator = np.zeros((canonical_height, canonical_width), dtype=np.float32)
    valid_accumulator = np.zeros((canonical_height, canonical_width), dtype=bool)
    total_weight = sum(weights[stage] for stage in effective_stages)
    per_stage: list[dict[str, object]] = []
    stage_statuses: list[object] = []
    for stage in effective_stages:
        result = compute_feature_residual(
            features_t1_by_stage[stage],
            features_t2_by_stage[stage],
            pif_mask,
            local_match_radius=local_match_radius,
            min_pif_feature_cells=min_pif_feature_cells,
            feature_scale_epsilon=feature_scale_epsilon,
        )
        score = np.asarray(result.score_map, dtype=np.float32)
        valid = np.asarray(result.valid_mask, dtype=bool)
        resized_score = cv2.resize(
            score,
            (canonical_width, canonical_height),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32, copy=False)
        resized_valid = cv2.resize(
            valid.astype(np.uint8),
            (canonical_width, canonical_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool, copy=False)
        normalized_weight = float(weights[stage] / total_weight)
        score_accumulator += resized_score * np.float32(normalized_weight)
        valid_accumulator |= resized_valid
        stride = None
        if feature_strides_by_stage is not None:
            stride = feature_strides_by_stage.get(stage)
        per_stage.append(
            {
                "stage": int(stage),
                "shape": [int(value) for value in np.asarray(features_t1_by_stage[stage]).shape],
                "stride": (
                    [float(stride[0]), float(stride[1])]
                    if stride is not None and len(stride) == 2
                    else None
                ),
                "pif_feature_cells": int(result.diagnostics["pif_feature_cells"]),
                "median_score_pif": float(result.diagnostics["median_score_pif"]),
                "median_score_full": float(result.diagnostics["median_score_full"]),
                "p95": float(result.diagnostics["p95_score_full"]),
                "alignment_status": result.diagnostics["alignment_status"],
                "weight": normalized_weight,
            }
        )
        stage_statuses.append(result.diagnostics["alignment_status"])

    canonical_pif = cv2.resize(
        (np.asarray(pif_mask) != 0).astype(np.uint8),
        (canonical_width, canonical_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool, copy=False)
    registration_valid = None
    if valid_mask is not None:
        registration_valid = np.asarray(valid_mask)
        if registration_valid.ndim != 2 or any(
            dimension <= 0 for dimension in registration_valid.shape
        ):
            raise ValueError("FEATURE_RESIDUAL_VALID_MASK_SHAPE_INVALID")
        registration_valid = cv2.resize(
            (registration_valid != 0).astype(np.uint8),
            (canonical_width, canonical_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool, copy=False)
        valid_accumulator &= registration_valid
    score_accumulator[~valid_accumulator] = 0.0
    pif_valid = canonical_pif & valid_accumulator
    return FeatureResidualResult(
        score_map=np.ascontiguousarray(score_accumulator, dtype=np.float32),
        valid_mask=np.ascontiguousarray(valid_accumulator, dtype=bool),
        diagnostics={
            "version": FEATURE_RESIDUAL_VERSION,
            "multiscale": True,
            "per_stage": per_stage,
            "effective_stages": [int(stage) for stage in effective_stages],
            "missing_stages": [int(stage) for stage in missing_stages],
            "canonical_size": [canonical_width, canonical_height],
            "local_match_radius": local_match_radius,
            "alignment_status": (
                "aligned"
                if all(status == "aligned" for status in stage_statuses)
                else "insufficient_pif"
            ),
            "pif_feature_cells": int(np.count_nonzero(canonical_pif)),
            "median_score_pif": float(np.median(score_accumulator[pif_valid]))
            if np.any(pif_valid)
            else 0.0,
            "median_score_full": float(np.median(score_accumulator[valid_accumulator]))
            if np.any(valid_accumulator)
            else 0.0,
            "p95_score_full": float(np.percentile(score_accumulator[valid_accumulator], 95))
            if np.any(valid_accumulator)
            else 0.0,
        },
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
    "compute_multiscale_feature_residual",
]
