"""Confidence-weighted semantic divergence for dense class probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.errors import OptionalDependencyMissingError


SEMANTIC_DIFFERENCE_VERSION = "confidence_weighted_js_v1"
_PROBABILITY_SUM_TOLERANCE = 1e-2


def _require_numpy():
    try:
        import numpy as np
    except ImportError as error:
        raise OptionalDependencyMissingError("change", dependency="numpy") from error
    return np


@dataclass(frozen=True)
class SemanticDifferenceResult:
    """Semantic score grid and compact JSON-safe summary statistics."""

    score_map: Any
    diagnostics: dict[str, object]


def compute_semantic_difference(
    probabilities_t1: Any,
    probabilities_t2: Any,
    *,
    confidence_floor: float = 0.45,
    epsilon: float = 1e-6,
) -> SemanticDifferenceResult:
    """Compute normalized Jensen-Shannon divergence weighted by confidence."""

    np = _require_numpy()
    first, second = _validate_probabilities(
        probabilities_t1,
        probabilities_t2,
        confidence_floor=confidence_floor,
        epsilon=epsilon,
        np=np,
    )
    first = _renormalize(first, epsilon=epsilon, np=np)
    second = _renormalize(second, epsilon=epsilon, np=np)

    midpoint = (first + second) * 0.5
    kl_first = np.sum(
        first * np.log((first + epsilon) / (midpoint + epsilon)),
        axis=0,
    )
    kl_second = np.sum(
        second * np.log((second + epsilon) / (midpoint + epsilon)),
        axis=0,
    )
    js_divergence = np.clip(
        (kl_first + kl_second) * 0.5 / np.log(2.0),
        0.0,
        1.0,
    )

    confidence_first = np.max(first, axis=0)
    confidence_second = np.max(second, axis=0)
    confidence = np.minimum(confidence_first, confidence_second)
    if confidence_floor == 1.0:
        effective_confidence = (confidence >= 1.0).astype(np.float64)
    else:
        effective_confidence = np.clip(
            (confidence - confidence_floor) / (1.0 - confidence_floor),
            0.0,
            1.0,
        )
    score_map = np.clip(js_divergence * effective_confidence, 0.0, 1.0).astype(
        np.float32
    )

    changed_top_class_fraction = float(
        np.mean(np.argmax(first, axis=0) != np.argmax(second, axis=0))
    )
    diagnostics: dict[str, object] = {
        "changed_top_class_fraction": changed_top_class_fraction,
        "mean_confidence": float(np.mean(confidence)),
        "median_js": float(np.median(js_divergence)),
        "p95_js": float(np.percentile(js_divergence, 95)),
        "version": SEMANTIC_DIFFERENCE_VERSION,
    }
    return SemanticDifferenceResult(score_map=score_map, diagnostics=diagnostics)


def _validate_probabilities(
    probabilities_t1: Any,
    probabilities_t2: Any,
    *,
    confidence_floor: float,
    epsilon: float,
    np: Any,
) -> tuple[Any, Any]:
    first = np.asarray(probabilities_t1)
    second = np.asarray(probabilities_t2)
    if first.ndim != 3 or any(dimension <= 0 for dimension in first.shape):
        raise ValueError(
            "SEMANTIC_DIFFERENCE_SHAPE_INVALID: expected non-empty [C,H,W]"
        )
    if first.shape[0] < 2:
        raise ValueError("SEMANTIC_DIFFERENCE_CLASS_COUNT_INVALID")
    if second.shape != first.shape:
        raise ValueError("SEMANTIC_DIFFERENCE_SHAPE_MISMATCH")
    if first.dtype != np.float32 or second.dtype != np.float32:
        raise ValueError(
            "SEMANTIC_DIFFERENCE_DTYPE_INVALID: probabilities must be float32"
        )
    if not bool(np.all(np.isfinite(first))) or not bool(np.all(np.isfinite(second))):
        raise ValueError("SEMANTIC_DIFFERENCE_NONFINITE")
    if bool(np.any(first < 0.0)) or bool(np.any(second < 0.0)):
        raise ValueError("SEMANTIC_DIFFERENCE_NEGATIVE_PROBABILITY")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not bool(np.isfinite(epsilon))
        or epsilon <= 0.0
    ):
        raise ValueError("SEMANTIC_DIFFERENCE_EPSILON_INVALID")
    if (
        isinstance(confidence_floor, bool)
        or not isinstance(confidence_floor, (int, float))
        or not bool(np.isfinite(confidence_floor))
        or confidence_floor < 0.0
        or confidence_floor > 1.0
    ):
        raise ValueError("SEMANTIC_DIFFERENCE_CONFIDENCE_FLOOR_INVALID")

    first_sum = np.sum(first.astype(np.float64), axis=0)
    second_sum = np.sum(second.astype(np.float64), axis=0)
    if bool(np.any(first_sum <= epsilon)) or bool(np.any(second_sum <= epsilon)):
        raise ValueError("SEMANTIC_DIFFERENCE_ZERO_SUM")
    if bool(np.any(np.abs(first_sum - 1.0) > _PROBABILITY_SUM_TOLERANCE)) or bool(
        np.any(np.abs(second_sum - 1.0) > _PROBABILITY_SUM_TOLERANCE)
    ):
        raise ValueError("SEMANTIC_DIFFERENCE_SUM_INVALID")
    return first, second


def _renormalize(probabilities: Any, *, epsilon: float, np: Any) -> Any:
    working = np.clip(probabilities.astype(np.float64), 0.0, None)
    sums = np.sum(working, axis=0, keepdims=True)
    return working / np.maximum(sums, epsilon)


__all__ = [
    "SEMANTIC_DIFFERENCE_VERSION",
    "SemanticDifferenceResult",
    "compute_semantic_difference",
]
