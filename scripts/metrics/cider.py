"""CIDEr / CIDEr-D (Vedantam et al., 2015) self-implemented.

TF-IDF weighted n-gram (n = 1..4) cosine similarity between candidate and
reference caption sets, with the sigma = 6 Gaussian kernel weighting used
by CIDEr-D.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

from .tokenize import tokenize

_MAX_N = 4
_SIGMA = 6.0


def _counts(tokens: Sequence[str], n: int) -> Counter:
    return Counter(
        tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)
    )


def _idf_weights(all_references: Sequence[Sequence[str]]) -> dict[int, dict[tuple, float]]:
    """IDF per n-gram computed from the whole reference corpus (as in the paper)."""
    document_frequency: dict[int, Counter] = {
        n: Counter() for n in range(1, _MAX_N + 1)
    }
    total_documents = len(all_references)
    for reference in all_references:
        for n in range(1, _MAX_N + 1):
            seen: set[tuple] = set()
            for gram in _counts(reference, n):
                if gram not in seen:
                    seen.add(gram)
                    document_frequency[n][gram] += 1
    weights: dict[int, dict[tuple, float]] = {}
    for n in range(1, _MAX_N + 1):
        weights[n] = {
            gram: math.log((total_documents + 1) / (count + 1)) + 1.0
            for gram, count in document_frequency[n].items()
        }
    return weights


def _tf_idf_vector(tokens: Sequence[str], n: int, weights: dict[tuple, float]) -> dict[tuple, float]:
    counts = _counts(tokens, n)
    total = max(1, sum(counts.values()))
    return {gram: (count / total) * weights.get(gram, 1.0) for gram, count in counts.items()}


def _cosine(left: dict[tuple, float], right: dict[tuple, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(count * right.get(gram, 0.0) for gram, count in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _cider_n(
    candidate_tokens: Sequence[str],
    reference_batch: Sequence[Sequence[str]],
    n: int,
    weights: dict[int, dict[tuple, float]],
) -> float:
    candidate_vector = _tf_idf_vector(candidate_tokens, n, weights[n])
    if not candidate_vector:
        return 0.0
    similarities = [
        _cosine(candidate_vector, _tf_idf_vector(reference, n, weights[n]))
        for reference in reference_batch
    ]
    if not similarities:
        return 0.0
    return sum(similarities) / len(similarities)


def corpus_cider(
    candidates: Iterable[str],
    references_batch: Iterable[Sequence[str]],
    use_cider_d: bool = True,
) -> float:
    """CIDEr (plain) or CIDEr-D (Gaussian sigma=6 weighted) over the corpus."""
    candidates = list(candidates)
    references_batch = list(references_batch)
    candidate_tokens = [tokenize(candidate) for candidate in candidates]
    reference_tokens = [
        [tokenize(reference) for reference in references] for references in references_batch
    ]
    all_references = [tokens for batch in reference_tokens for tokens in batch]
    weights = _idf_weights(all_references)

    total = 0.0
    samples = 0
    for candidate, references in zip(candidate_tokens, reference_tokens):
        per_n: dict[int, float] = {}
        for n in range(1, _MAX_N + 1):
            per_n[n] = _cider_n(candidate, references, n, weights)
        if use_cider_d:
            per_n = {
                n: value * math.exp(-((n - 1) ** 2) / (2.0 * _SIGMA * _SIGMA))
                for n, value in per_n.items()
            }
        total += sum(per_n.values()) / _MAX_N
        samples += 1
    return total / samples if samples else 0.0
