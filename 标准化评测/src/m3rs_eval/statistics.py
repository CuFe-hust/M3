"""Deterministic confidence-interval helpers for evaluation metrics."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable
from numbers import Real
from statistics import NormalDist


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if not _nonnegative_int(successes):
        raise ValueError("successes must be a nonnegative integer")
    if not _positive_int(total):
        raise ValueError("total must be a positive integer")
    if successes > total:
        raise ValueError("successes must not exceed total")
    if not _finite_number(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be a finite number between 0 and 1")

    rate = successes / total
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / denominator
    return center - margin, center + margin


def bootstrap_interval(
    values: Iterable[float],
    statistic: Callable[[list[float]], float],
    iterations: int = 1000,
    seed: int = 20260804,
) -> tuple[float, float]:
    """Return a 95% percentile bootstrap interval using a local seeded RNG."""
    if not callable(statistic):
        raise ValueError("statistic must be callable")
    if not _positive_int(iterations):
        raise ValueError("iterations must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    observations = [_validated_value(value) for value in values]
    if not observations:
        raise ValueError("values must be nonempty")

    generator = random.Random(seed)
    size = len(observations)
    estimates: list[float] = []
    for _ in range(iterations):
        sample = [observations[generator.randrange(size)] for _ in range(size)]
        estimates.append(_validated_statistic(statistic, sample))
    estimates.sort()
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _validated_value(value: object) -> float:
    if not _finite_number(value):
        raise ValueError("values must contain only finite numbers")
    return float(value)


def _validated_statistic(statistic: Callable[[list[float]], float], sample: list[float]) -> float:
    try:
        value = statistic(sample)
    except Exception as error:
        raise ValueError("statistic failed for a bootstrap sample") from error
    if not _finite_number(value):
        raise ValueError("statistic must return a finite number")
    return float(value)


def _finite_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_int(value: object) -> bool:
    return _nonnegative_int(value) and value > 0
