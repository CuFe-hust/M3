"""BLEU (Papineni et al., 2002) self-implemented from the paper formula.

Modified n-gram precision with a brevity penalty, no external libraries.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

from .tokenize import tokenize


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(
        tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)
    )


def _modified_precision(
    candidate: Sequence[str],
    references: Sequence[Sequence[str]],
    n: int,
) -> tuple[int, int]:
    candidate_counts = _ngrams(candidate, n)
    reference_counts: Counter = Counter()
    for reference in references:
        counts = _ngrams(reference, n)
        for gram, count in counts.items():
            if count > reference_counts[gram]:
                reference_counts[gram] = count
    clipped = sum(
        min(count, reference_counts.get(gram, 0))
        for gram, count in candidate_counts.items()
    )
    total = max(1, sum(candidate_counts.values()))
    return clipped, total


def _brevity_penalty(candidate_length: int, reference_lengths: Sequence[int]) -> float:
    if not reference_lengths:
        return 1.0
    best_reference = min(reference_lengths, key=lambda length: abs(length - candidate_length))
    if candidate_length > best_reference:
        return 1.0
    if candidate_length == 0:
        return 0.0
    return math.exp(1.0 - best_reference / candidate_length)


def corpus_bleu(
    candidates: Iterable[str],
    references_batch: Iterable[Sequence[str]],
    max_n: int = 4,
) -> dict[int, float]:
    """Corpus-level BLEU_n for n = 1..max_n on the 0-1 scale."""
    clipped_by_n: dict[int, int] = {n: 0 for n in range(1, max_n + 1)}
    total_by_n: dict[int, int] = {n: 0 for n in range(1, max_n + 1)}
    total_candidate_length = 0
    total_reference_length = 0
    samples = 0
    for candidate, references in zip(candidates, references_batch):
        candidate_tokens = tokenize(candidate)
        reference_tokens = [tokenize(reference) for reference in references]
        total_candidate_length += len(candidate_tokens)
        total_reference_length += min(
            (len(reference) for reference in reference_tokens), default=0
        )
        for n in range(1, max_n + 1):
            clipped, total = _modified_precision(candidate_tokens, reference_tokens, n)
            clipped_by_n[n] += clipped
            total_by_n[n] += total
        samples += 1
    if samples == 0:
        return {n: 0.0 for n in range(1, max_n + 1)}
    brevity = _brevity_penalty(total_candidate_length, [total_reference_length])
    scores: dict[int, float] = {}
    for n in range(1, max_n + 1):
        precision = clipped_by_n[n] / total_by_n[n]
        if precision == 0.0:
            scores[n] = 0.0
        else:
            scores[n] = brevity * math.exp(math.log(precision))
    return scores
