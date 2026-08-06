"""LEVIR-CC change discrimination metrics self-implemented.

Classifies each model output as change vs no-change via five fixed
no-change templates (five-template compatibility), then reports accuracy
for all / change / no-change slices.
"""

from __future__ import annotations

import re
from typing import Iterable

# Five canonical no-change templates; matched case-insensitively.
_NO_CHANGE_TEMPLATES = (
    re.compile(r"no change", re.IGNORECASE),
    re.compile(r"nothing (?:has )?changed", re.IGNORECASE),
    re.compile(r"unchanged", re.IGNORECASE),
    re.compile(r"same(?: scene)?(?: as before)?", re.IGNORECASE),
    re.compile(r"没有变化|无变化|未变化", ),
)


def classify_no_change(output: str) -> bool:
    """True when the output matches any fixed no-change template."""
    text = (output or "").strip()
    return any(pattern.search(text) for pattern in _NO_CHANGE_TEMPLATES)


def discrimination_metrics(
    rows: Iterable[tuple[str, str]],
) -> dict[str, float]:
    """Return {all, change, no_change} accuracy over (output, is_no_change_ref) rows.

    rows: (prediction_output, reference_is_no_change: bool)
    """
    buckets: dict[str, list[int]] = {"all": [0, 0], "change": [0, 0], "no_change": [0, 0]}
    for output, reference_is_no_change in rows:
        predicted_no_change = classify_no_change(output)
        correct = predicted_no_change == bool(reference_is_no_change)
        buckets["all"][0] += 1 if correct else 0
        buckets["all"][1] += 1
        slice_name = "no_change" if reference_is_no_change else "change"
        buckets[slice_name][0] += 1 if correct else 0
        buckets[slice_name][1] += 1
    return {
        scope: (correct / total if total else 0.0)
        for scope, (correct, total) in buckets.items()
    }
