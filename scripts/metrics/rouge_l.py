"""ROUGE-L (Lin, 2004) self-implemented: LCS-based F-measure.

ROUGE-L uses the longest common subsequence between candidate and
reference, with beta = 1.2 weighting recall over precision.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from .tokenize import tokenize

_BETA = 1.2
_BETA_SQUARED = _BETA * _BETA


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0] * (len(right) + 1)
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current[index] = previous[index - 1] + 1
            else:
                current[index] = max(previous[index], current[index - 1])
        previous = current
    return previous[-1]


def rouge_l_fmeasure(candidate: str, reference: str) -> float:
    """ROUGE-L F-measure for one candidate/reference pair, 0-1 scale."""
    candidate_tokens = tokenize(candidate)
    reference_tokens = tokenize(reference)
    if not reference_tokens:
        return 0.0
    if not candidate_tokens:
        return 0.0
    lcs = _lcs_length(candidate_tokens, reference_tokens)
    recall = lcs / len(reference_tokens)
    precision = lcs / len(candidate_tokens)
    if recall == 0.0 and precision == 0.0:
        return 0.0
    denominator = recall + _BETA_SQUARED * precision
    if denominator == 0.0:
        return 0.0
    fmeasure = (1.0 + _BETA_SQUARED) * recall * precision / denominator
    return min(1.0, max(0.0, fmeasure))


def corpus_rouge_l(candidates: Iterable[str], references_batch: Iterable[Sequence[str]]) -> float:
    """Mean ROUGE-L over all candidate/reference pairs (best reference per sample)."""
    total = 0.0
    samples = 0
    for candidate, references in zip(candidates, references_batch):
        best = max(
            (rouge_l_fmeasure(candidate, reference) for reference in references),
            default=0.0,
        )
        total += best
        samples += 1
    return total / samples if samples else 0.0


def rouge_l_stats(candidate: str, reference: str) -> tuple[int, int, int]:
    """Return (lcs, reference_length, candidate_length) for one pair (unit-testable)."""
    candidate_tokens = tokenize(candidate)
    reference_tokens = tokenize(reference)
    return (
        _lcs_length(candidate_tokens, reference_tokens),
        len(reference_tokens),
        len(candidate_tokens),
    )
