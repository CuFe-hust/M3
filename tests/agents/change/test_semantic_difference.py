from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import agents.change.semantic_difference as semantic_difference_module
from agents.change.semantic_difference import (
    SEMANTIC_DIFFERENCE_VERSION,
    compute_semantic_difference,
)


def _probability_grid(values: list[float], *, height: int = 3, width: int = 4) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)[:, None, None]
    return np.broadcast_to(vector, (len(values), height, width)).copy()


def test_identical_distributions_have_zero_difference() -> None:
    probabilities = _probability_grid([0.7, 0.2, 0.1])

    result = compute_semantic_difference(probabilities, probabilities.copy())

    assert result.score_map.dtype == np.float32
    assert bool(np.all(result.score_map == 0.0))
    assert result.diagnostics == {
        "changed_top_class_fraction": 0.0,
        "mean_confidence": pytest.approx(0.7),
        "median_js": 0.0,
        "p95_js": 0.0,
        "version": SEMANTIC_DIFFERENCE_VERSION,
    }


def test_confident_class_switch_has_high_difference() -> None:
    first = _probability_grid([0.95, 0.05])
    second = _probability_grid([0.05, 0.95])

    result = compute_semantic_difference(first, second)

    assert float(np.min(result.score_map)) > 0.6
    assert result.diagnostics["changed_top_class_fraction"] == 1.0
    assert float(result.diagnostics["median_js"]) > 0.7


def test_uncertain_argmax_switch_is_strongly_suppressed() -> None:
    first = _probability_grid([0.51, 0.49])
    second = _probability_grid([0.49, 0.51])

    result = compute_semantic_difference(first, second)

    assert result.diagnostics["changed_top_class_fraction"] == 1.0
    assert float(np.max(result.score_map)) < 1e-3


def test_multiclass_divergence_is_symmetric() -> None:
    first = np.asarray(
        [
            [[0.80, 0.10], [0.25, 0.20]],
            [[0.15, 0.25], [0.50, 0.30]],
            [[0.05, 0.65], [0.25, 0.50]],
        ],
        dtype=np.float32,
    )
    second = np.asarray(
        [
            [[0.10, 0.70], [0.20, 0.25]],
            [[0.20, 0.20], [0.25, 0.50]],
            [[0.70, 0.10], [0.55, 0.25]],
        ],
        dtype=np.float32,
    )

    forward = compute_semantic_difference(first, second, confidence_floor=0.2)
    reverse = compute_semantic_difference(second, first, confidence_floor=0.2)

    np.testing.assert_allclose(forward.score_map, reverse.score_map, atol=1e-7)
    assert forward.diagnostics == reverse.diagnostics


def test_small_stitching_sum_error_is_renormalized() -> None:
    base = _probability_grid([0.75, 0.20, 0.05])
    first = (base * np.float32(0.999)).astype(np.float32)
    second = (base * np.float32(1.001)).astype(np.float32)

    result = compute_semantic_difference(first, second)

    assert float(np.max(result.score_map)) < 1e-7
    assert float(result.diagnostics["median_js"]) < 1e-12


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_probabilities_fail_stably(bad_value: float) -> None:
    first = _probability_grid([0.8, 0.2])
    second = first.copy()
    second[0, 0, 0] = bad_value

    with pytest.raises(ValueError, match="SEMANTIC_DIFFERENCE_NONFINITE"):
        compute_semantic_difference(first, second)


def test_negative_probability_fails_stably() -> None:
    first = _probability_grid([0.8, 0.2])
    second = first.copy()
    second[:, 0, 0] = np.asarray([1.01, -0.01], dtype=np.float32)

    with pytest.raises(
        ValueError,
        match="SEMANTIC_DIFFERENCE_NEGATIVE_PROBABILITY",
    ):
        compute_semantic_difference(first, second)


def test_shape_mismatch_fails_stably() -> None:
    first = _probability_grid([0.8, 0.2])
    second = _probability_grid([0.8, 0.2], width=5)

    with pytest.raises(ValueError, match="SEMANTIC_DIFFERENCE_SHAPE_MISMATCH"):
        compute_semantic_difference(first, second)


def test_zero_sum_cell_fails_instead_of_becoming_uniform() -> None:
    first = _probability_grid([0.8, 0.2])
    second = first.copy()
    second[:, 1, 1] = 0.0

    with pytest.raises(ValueError, match="SEMANTIC_DIFFERENCE_ZERO_SUM"):
        compute_semantic_difference(first, second)


def test_non_probability_sum_fails_instead_of_arbitrary_renormalization() -> None:
    first = _probability_grid([0.8, 0.2])
    second = first * np.float32(2.0)

    with pytest.raises(ValueError, match="SEMANTIC_DIFFERENCE_SUM_INVALID"):
        compute_semantic_difference(first, second)


def test_output_is_finite_and_bounded_for_random_multiclass_inputs() -> None:
    rng = np.random.default_rng(67)
    first = rng.random((7, 11, 13), dtype=np.float32)
    second = rng.random((7, 11, 13), dtype=np.float32)
    first /= np.sum(first, axis=0, keepdims=True)
    second /= np.sum(second, axis=0, keepdims=True)

    result = compute_semantic_difference(first, second, confidence_floor=0.0)

    assert bool(np.all(np.isfinite(result.score_map)))
    assert float(np.min(result.score_map)) >= 0.0
    assert float(np.max(result.score_map)) <= 1.0
    assert set(result.diagnostics) == {
        "changed_top_class_fraction",
        "mean_confidence",
        "median_js",
        "p95_js",
        "version",
    }
    assert all(
        isinstance(value, (float, str)) for value in result.diagnostics.values()
    )


def test_confidence_floor_one_is_supported_without_division_by_zero() -> None:
    first = _probability_grid([1.0, 0.0])
    second = _probability_grid([0.0, 1.0])

    result = compute_semantic_difference(first, second, confidence_floor=1.0)

    assert bool(np.all(np.isfinite(result.score_map)))
    assert float(np.min(result.score_map)) > 0.99


def test_module_has_no_model_dependency() -> None:
    source_path = Path(semantic_difference_module.__file__ or "")
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
