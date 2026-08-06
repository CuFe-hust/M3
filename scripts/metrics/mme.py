"""MME-RealWorld Remote Sensing metrics self-implemented (official protocol).

Parses model outputs into answers A-E; anything else counts as an error
and is separately counted as an invalid parse. Reports per-task accuracy
(Color/Count/Position), sample-weighted Avg, equally-weighted Avg-C,
invalid parse rate and choice-E rate.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, Sequence

_LETTER_RE = re.compile(r"\b([a-eA-E])\b|^([a-eA-E])[.\s]")
_FULL_CHOICE_RE = re.compile(r"(?:^|\s)([A-Ea-e])\s*[.)、]")


def parse_choice(output: str) -> str | None:
    """Extract the A-E letter from a model output, or None when unparseable."""
    text = (output or "").strip()
    if not text:
        return None
    if text in {"a", "b", "c", "d", "e", "A", "B", "C", "D", "E"}:
        return text.upper()
    match = _FULL_CHOICE_RE.search(text)
    if match:
        return match.group(1).upper()
    match = _LETTER_RE.search(text)
    if match:
        return (match.group(1) or match.group(2)).upper()
    return None


def task_metrics(
    rows: Iterable[tuple[str, str, str]],
) -> dict[str, dict[str, float]]:
    """Return {task: {accuracy, n, invalid_parse, choice_e}} per task.

    rows: (task, prediction, reference_answer); reference is the expected
    letter for the sample.
    """
    per_task: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"correct": [0, 0], "invalid": [0, 0], "choice_e": [0, 0]}
    )
    for task, prediction, reference in rows:
        parsed = parse_choice(prediction)
        counts = per_task[task]
        if parsed is None:
            counts["invalid"][0] += 1
        else:
            counts["correct"][0] += 1 if parsed == _letter(reference) else 0
            if parsed == "E":
                counts["choice_e"][0] += 1
        for bucket in (counts["correct"], counts["invalid"], counts["choice_e"]):
            bucket[1] += 1
    results: dict[str, dict[str, float]] = {}
    for task, counts in per_task.items():
        correct, total = counts["correct"]
        invalid, invalid_total = counts["invalid"]
        choice_e, _ = counts["choice_e"]
        results[task] = {
            "accuracy": correct / total if total else 0.0,
            "n": float(total),
            "invalid_parse": invalid / invalid_total if invalid_total else 0.0,
            "choice_e": choice_e / invalid_total if invalid_total else 0.0,
        }
    return dict(sorted(results.items()))


def aggregate(task_results: dict[str, dict[str, float]]) -> dict[str, float]:
    """Sample-weighted avg and equally-weighted avg-c over task results."""
    tasks = list(task_results)
    if not tasks:
        return {"avg": 0.0, "avg_c": 0.0}
    total_correct = sum(
        task_results[task]["accuracy"] * task_results[task]["n"] for task in tasks
    )
    total_samples = sum(task_results[task]["n"] for task in tasks)
    avg = total_correct / total_samples if total_samples else 0.0
    avg_c = sum(task_results[task]["accuracy"] for task in tasks) / len(tasks)
    return {"avg": avg, "avg_c": avg_c}


def _letter(reference: str) -> str:
    value = (reference or "").strip().upper()
    if value in {"A", "B", "C", "D", "E"}:
        return value
    match = re.search(r"[A-E]", value)
    return match.group(0) if match else ""
