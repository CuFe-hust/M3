"""Deterministic proposal-level semantic transition candidates."""

from __future__ import annotations

from typing import Any, Sequence

from agents.change.schema import SemanticTransition
from agents.errors import OptionalDependencyMissingError


SEMANTIC_TRANSITION_VERSION = "proposal_semantic_transition_v1"


def _require_numpy():
    try:
        import numpy as np
    except ImportError as error:
        raise OptionalDependencyMissingError("change", dependency="numpy") from error
    return np


def infer_semantic_transition(
    probabilities_t1: Any,
    probabilities_t2: Any,
    proposal_mask: Any,
    class_names: Sequence[str] | None = None,
    *,
    confidence_floor: float = 0.45,
    support_floor: float = 0.50,
    valid_mask: Any | None = None,
) -> SemanticTransition:
    """Aggregate class probabilities over a proposal mask.

    The function never makes a pixel-level argmax the sole decision.  It
    aggregates per-class probabilities, measures top-class support, and emits
    ``unknown`` whenever the evidence is too weak to name a transition.
    """

    np = _require_numpy()
    first, second = _validate_probabilities(probabilities_t1, probabilities_t2, np=np)
    mask = np.asarray(proposal_mask)
    if mask.ndim != 2 or any(int(value) <= 0 for value in mask.shape):
        raise ValueError("SEMANTIC_TRANSITION_MASK_INVALID")
    if valid_mask is not None:
        overlap = np.asarray(valid_mask)
        if overlap.ndim != 2 or any(int(value) <= 0 for value in overlap.shape):
            raise ValueError("SEMANTIC_TRANSITION_VALID_MASK_INVALID")
        overlap = _resize_nearest(overlap.astype(bool), mask.shape, np=np)
        mask = (mask != 0) & overlap
    else:
        mask = mask != 0
    selected = np.flatnonzero(mask.reshape(-1))
    if selected.size == 0:
        return SemanticTransition(
            from_class="unknown",
            from_confidence=0.0,
            to_class="unknown",
            to_confidence=0.0,
            changed_class="unknown",
            support_ratio=0.0,
            transition_confidence=0.0,
        )

    first_grid = _resize_probabilities(first, mask.shape, np=np)
    second_grid = _resize_probabilities(second, mask.shape, np=np)
    first_values = first_grid.reshape(first_grid.shape[0], -1)[:, selected]
    second_values = second_grid.reshape(second_grid.shape[0], -1)[:, selected]
    first_aggregate = _robust_mean(first_values, np=np)
    second_aggregate = _robust_mean(second_values, np=np)
    first_class = int(np.argmax(first_aggregate))
    second_class = int(np.argmax(second_aggregate))
    first_confidence = float(first_aggregate[first_class])
    second_confidence = float(second_aggregate[second_class])
    first_labels = np.argmax(first_values, axis=0)
    second_labels = np.argmax(second_values, axis=0)
    support_ratio = float(
        min(
            np.mean(first_labels == first_class),
            np.mean(second_labels == second_class),
        )
    )
    transition_confidence = float(
        np.clip(min(first_confidence, second_confidence) * support_ratio, 0.0, 1.0)
    )
    from_label = _class_label(first_class, class_names)
    to_label = _class_label(second_class, class_names)
    reliable = (
        first_confidence >= confidence_floor
        and second_confidence >= confidence_floor
        and support_ratio >= support_floor
    )
    if not reliable:
        from_label = to_label = "unknown"
        changed_class: str | None = "unknown"
    elif first_class == second_class:
        changed_class = None
    else:
        changed_class = to_label
    return SemanticTransition(
        from_class=from_label,
        from_confidence=first_confidence,
        to_class=to_label,
        to_confidence=second_confidence,
        changed_class=changed_class,
        support_ratio=support_ratio,
        transition_confidence=transition_confidence,
    )


def _validate_probabilities(first_value: Any, second_value: Any, *, np: Any) -> tuple[Any, Any]:
    first = np.asarray(first_value, dtype=np.float32)
    second = np.asarray(second_value, dtype=np.float32)
    if (
        first.ndim != 3
        or second.shape != first.shape
        or first.shape[0] < 2
        or any(int(value) <= 0 for value in first.shape)
    ):
        raise ValueError("SEMANTIC_TRANSITION_PROBABILITY_SHAPE_INVALID")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("SEMANTIC_TRANSITION_PROBABILITY_NONFINITE")
    if np.any(first < 0.0) or np.any(second < 0.0):
        raise ValueError("SEMANTIC_TRANSITION_PROBABILITY_NEGATIVE")
    first_sum = first.sum(axis=0, keepdims=True)
    second_sum = second.sum(axis=0, keepdims=True)
    if np.any(first_sum <= 0.0) or np.any(second_sum <= 0.0):
        raise ValueError("SEMANTIC_TRANSITION_PROBABILITY_ZERO_SUM")
    return first / first_sum, second / second_sum


def _resize_probabilities(value: Any, shape: tuple[int, int], *, np: Any) -> Any:
    if value.shape[1:] == shape:
        return value
    rows = np.minimum(np.arange(shape[0]) * value.shape[1] // shape[0], value.shape[1] - 1)
    columns = np.minimum(
        np.arange(shape[1]) * value.shape[2] // shape[1], value.shape[2] - 1
    )
    return value[:, rows[:, None], columns[None, :]]


def _resize_nearest(value: Any, shape: tuple[int, int], *, np: Any) -> Any:
    if value.shape == shape:
        return value
    rows = np.minimum(np.arange(shape[0]) * value.shape[0] // shape[0], value.shape[0] - 1)
    columns = np.minimum(
        np.arange(shape[1]) * value.shape[1] // shape[1], value.shape[1] - 1
    )
    return value[rows[:, None], columns[None, :]]


def _robust_mean(values: Any, *, np: Any) -> Any:
    if values.shape[1] < 10:
        return values.mean(axis=1)
    ordered = np.sort(values, axis=1)
    trim = max(1, int(values.shape[1] * 0.10))
    if trim * 2 >= values.shape[1]:
        return values.mean(axis=1)
    return ordered[:, trim:-trim].mean(axis=1)


def _class_label(index: int, class_names: Sequence[str] | None) -> str:
    if class_names is not None and index < len(class_names):
        label = str(class_names[index]).strip()
        if label:
            return label
    return f"class_{index}"


__all__ = ["SEMANTIC_TRANSITION_VERSION", "infer_semantic_transition"]
