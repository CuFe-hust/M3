"""Deterministic exact-match / choice accuracy self-implemented.

- Choice VQA: prediction must equal the reference answer after
  normalization (strip whitespace/case/punctuation, optional letter-only).
- Free-form short answers: normalized exact match (no synonyms; synonym
  handling is documented out of scope).
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Sequence

_STRIP_RE = re.compile(r"[\s\.,;:!?\"'()\[\]{}]+")


def _normalize(text: str) -> str:
    return _STRIP_RE.sub("", (text or "").strip().lower())


def exact_match(prediction: str, reference: str) -> bool:
    return _normalize(prediction) == _normalize(reference)


def choice_match(prediction: str, reference: str) -> bool:
    """Accept 'B' or 'B. airport' style answers against 'B'/'A' references."""
    normalized = _normalize(prediction)
    reference_letter = _normalize(reference)[:1]
    if normalized in {"", "none", "n/a"}:
        return False
    if reference_letter in {"a", "b", "c", "d", "e"}:
        if normalized[:1] == reference_letter:
            return True
        if normalized == _normalize(reference):
            return True
    return normalized == _normalize(reference)


def accuracy(
    pairs: Iterable[tuple[str, str]],
    *,
    choice: bool = False,
) -> float:
    """Accuracy over (prediction, reference) pairs, 0-1 scale."""
    correct = 0
    total = 0
    for prediction, reference in pairs:
        matched = choice_match(prediction, reference) if choice else exact_match(prediction, reference)
        correct += 1 if matched else 0
        total += 1
    return correct / total if total else 0.0


def grouped_accuracy(
    pairs: Iterable[tuple[str, str, str]],
    *,
    choice: bool = False,
) -> dict[str, float]:
    """Accuracy grouped by a category label: {category: accuracy}."""
    counts: dict[str, list[int]] = {}
    for prediction, reference, category in pairs:
        counts.setdefault(category, [0, 0])
        matched = choice_match(prediction, reference) if choice else exact_match(prediction, reference)
        counts[category][0] += 1 if matched else 0
        counts[category][1] += 1
    return {
        category: (correct / total if total else 0.0)
        for category, (correct, total) in sorted(counts.items())
    }
