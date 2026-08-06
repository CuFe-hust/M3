"""METEOR (Banerjee & Lavie, 2005) self-implemented.

Basic version: greedy unigram alignment with exact matching, harmonic
F-mean (alpha=9, beta=10) and fragmentation penalty (gamma=0.5).
WordNet synonym/stem stages are out of scope and documented as such.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from .tokenize import tokenize


def _align(candidate: Sequence[str], reference: Sequence[str]) -> int:
    """Greedy one-to-one unigram alignment count (exact matches)."""
    matched = 0
    used: set[int] = set()
    for candidate_token in candidate:
        for index, reference_token in enumerate(reference):
            if index in used:
                continue
            if candidate_token == reference_token:
                matched += 1
                used.add(index)
                break
    return matched


def _chunks(candidate: Sequence[str], reference: Sequence[str]) -> int:
    """Count contiguous aligned-match chunks (adjacency in both strings)."""
    reference_positions: dict[str, list[int]] = {}
    for index, token in enumerate(reference):
        reference_positions.setdefault(token, []).append(index)
    used: set[int] = set()
    previous_position = -2
    chunks = 0
    for token in candidate:
        for position in reference_positions.get(token, []):
            if position not in used:
                used.add(position)
                if position != previous_position + 1:
                    chunks += 1
                previous_position = position
                break
    return chunks


def meteor_score(candidate: str, reference: str) -> float:
    """METEOR score for one candidate/reference pair, 0-1 scale."""
    candidate_tokens = tokenize(candidate)
    reference_tokens = tokenize(reference)
    if not reference_tokens or not candidate_tokens:
        return 0.0
    matches = _align(candidate_tokens, reference_tokens)
    if matches == 0:
        return 0.0
    precision = matches / len(candidate_tokens)
    recall = matches / len(reference_tokens)
    denominator = recall + 9.0 * precision
    if denominator == 0.0:
        return 0.0
    f_mean = 10.0 * precision * recall / denominator
    chunks = _chunks(candidate_tokens, reference_tokens)
    penalty = 0.5 * (chunks / matches) ** 3
    return max(0.0, min(1.0, f_mean * (1.0 - penalty)))


def corpus_meteor(candidates: Iterable[str], references_batch: Iterable[Sequence[str]]) -> float:
    """Mean METEOR over all samples (best reference per sample)."""
    total = 0.0
    samples = 0
    for candidate, references in zip(candidates, references_batch):
        best = max((meteor_score(candidate, reference) for reference in references), default=0.0)
        total += best
        samples += 1
    return total / samples if samples else 0.0
