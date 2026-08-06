from __future__ import annotations

import math
import random
from statistics import mean

import pytest

from m3rs_eval.statistics import bootstrap_interval, wilson_interval


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(82, 100)

    assert low < 0.82 < high
    assert (low, high) == pytest.approx((0.7333, 0.8830), abs=1e-4)


@pytest.mark.parametrize(
    ("successes", "total", "confidence"),
    [(-1, 1, 0.95), (2, 1, 0.95), (0, 0, 0.95), (True, 1, 0.95), (1, 2, 1.0)],
)
def test_wilson_interval_rejects_invalid_inputs(successes: object, total: object, confidence: object):
    with pytest.raises(ValueError):
        wilson_interval(successes, total, confidence)


def test_bootstrap_is_reproducible():
    values = [0.0, 1.0, 1.0, 1.0]

    assert bootstrap_interval(values, mean, iterations=200, seed=7) == bootstrap_interval(
        values, mean, iterations=200, seed=7
    )


def test_bootstrap_does_not_depend_on_global_random_state():
    values = [0.0, 1.0, 1.0, 1.0]
    random.seed(1)
    first = bootstrap_interval(values, mean, iterations=200, seed=7)
    random.seed(999)
    second = bootstrap_interval(values, mean, iterations=200, seed=7)

    assert first == second


@pytest.mark.parametrize(
    ("values", "statistic", "iterations"),
    [
        ([], mean, 10),
        ([1.0, math.inf], mean, 10),
        ([1.0], mean, 0),
        ([1.0], None, 10),
        ([1.0], lambda _: math.nan, 10),
    ],
)
def test_bootstrap_rejects_invalid_inputs(values: object, statistic: object, iterations: object):
    with pytest.raises(ValueError):
        bootstrap_interval(values, statistic, iterations=iterations)
