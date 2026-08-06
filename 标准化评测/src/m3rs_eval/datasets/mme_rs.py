"""MME-RealWorld-RS materialization limited to the Remote_Sensing domain."""

from __future__ import annotations

import re
from collections.abc import Mapping
from statistics import fmean
from typing import Any

from m3rs_eval.evaluation import AlignmentResult, MetricContext, make_metric_record
from m3rs_eval.registry import MetricRegistry

from .base import BaseDatasetAdapter, DatasetError, MaterializedExample


_MME_TASKS = ("color", "count", "position")


class MMERSAdapter(BaseDatasetAdapter):
    dataset = "mme_rs"
    expected_files = ("questions.jsonl",)

    def _parse_examples(self) -> list[MaterializedExample]:
        annotation_path, rows = self._annotation_jsonl("questions.jsonl")
        examples = []
        for index, row in enumerate(rows, start=1):
            domain = _text(row, "domain", index)
            if domain != "Remote_Sensing":
                raise DatasetError(
                    f"{self.dataset}: formal materialization accepts only Remote_Sensing, got '{domain}'"
                )
            task = _text(row, "task", index)
            if task not in {"color", "count", "position"}:
                raise DatasetError(f"{self.dataset}: Remote_Sensing task must be Color, Count, or Position")
            sample_id = f"mme_rs:{task}:{_text(row, 'id', index)}"
            examples.append(
                MaterializedExample(
                    request={
                        "sample_id": sample_id,
                        "dataset": self.dataset,
                        "benchmark_version": "mme-realworld-rs:Remote_Sensing",
                        "split": domain,
                        "task": task,
                        "images": [(self.config.root / _text(row, "image", index)).resolve().as_posix()],
                        "prompt": _text(row, "question", index),
                        "choices": _choices(row, index),
                        "expected_output": "choice",
                    },
                    reference={"sample_id": sample_id, "answer": _text(row, "answer", index)},
                    annotation_path=annotation_path,
                    source_label=f"questions.jsonl row {index}",
                )
            )
        return examples

    def _coverage(self, selected, total, task_counts):
        coverage = super()._coverage(selected, total, task_counts)
        coverage.update({"domain": "Remote_Sensing", "separate_tasks": ["color", "count", "position"]})
        return coverage


def _text(row: dict[str, Any], field: str, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"mme_rs: questions.jsonl row {index} requires nonempty '{field}'")
    return value.strip()


def _choices(row: dict[str, Any], index: int) -> list[str]:
    choices = row.get("choices")
    if not isinstance(choices, list) or not choices or not all(isinstance(choice, str) and choice.strip() for choice in choices):
        raise DatasetError(f"mme_rs: questions.jsonl row {index} requires nonempty string choices")
    return [choice.strip() for choice in choices]


def extract_mme_answer(value: str | None, choices: tuple[str, ...] = ()) -> str:
    """Reproduce MME-RealWorld eval_your_results.py answer extraction."""
    if not isinstance(value, str) or not value.strip():
        return ""
    response = value.strip()
    # These are intentionally the effective Python literals in the pinned upstream source.
    prefixes = (
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option isThe correct option is",
        "Best answer:Best option:",
    )
    for prefix in prefixes:
        response = response.replace(prefix, "")
    if len(response.split()) > 10 and re.search("[ABCDE]", response) is None:
        return ""
    match = re.search(r"[ABCDE]", response)
    if match is not None:
        return match[0]
    for choice in choices:
        if response.lower() in choice.lower():
            if len(choice) < 2:
                # Defensive contract boundary for malformed nonempty choices upstream cannot index.
                return ""
            # Upstream indexes the second character, so "A. text" deliberately returns ".".
            return choice[1]
    return ""


def evaluate_mme_alignment(
    alignment: AlignmentResult,
    references: Mapping[str, Mapping[str, Any]],
    registry: MetricRegistry,
    context: MetricContext,
):
    """Compute registry-defined MME-RS task rates, Avg, Avg-C, and parser diagnostics."""
    observations: dict[str, list[tuple[bool, bool, bool]]] = {task: [] for task in _MME_TASKS}
    for row in alignment.rows:
        if row.request.dataset != "mme_rs" or row.request.task not in observations:
            continue
        reference = references.get(row.request.sample_id)
        if reference is None:
            raise DatasetError(f"mme_rs: missing reference for {row.request.sample_id}")
        expected = extract_mme_answer(str(reference.get("answer", "")))
        if not expected:
            raise DatasetError(f"mme_rs: invalid reference answer for {row.request.sample_id}")
        predicted = extract_mme_answer(
            row.prediction.prediction if row.prediction is not None else None,
            row.request.choices or (),
        )
        valid_parse = predicted in {"A", "B", "C", "D", "E"}
        observations[row.request.task].append(
            (valid_parse and predicted == expected, not valid_parse, predicted == "E")
        )

    records = []
    task_rates: list[float] = []
    all_rows = [item for task in _MME_TASKS for item in observations[task]]
    for task in _MME_TASKS:
        values = observations[task]
        metric_id = f"mme_rs.acc.{task}"
        if not values:
            records.append(
                make_metric_record(
                    registry,
                    context,
                    metric_id,
                    None,
                    n_samples=0,
                    n_failures=0,
                    provenance="supplemental",
                    availability="missing",
                    notes="locked MME-RS task absent from aligned materialization",
                )
            )
            continue
        successes = sum(correct for correct, _, _ in values)
        failures = sum(invalid for _, invalid, _ in values)
        rate = successes / len(values)
        task_rates.append(rate)
        records.append(
            make_metric_record(
                registry,
                context,
                metric_id,
                rate,
                n_samples=len(values),
                n_failures=failures,
                provenance="supplemental",
                binomial_successes=successes,
                notes=(
                    "official MME-RealWorld eval_your_results.py compatibility parser; "
                    "intentionally accepts the first uppercase A-E anywhere in short responses"
                ),
            )
        )

    if all_rows:
        total = len(all_rows)
        correct = sum(item[0] for item in all_rows)
        invalid = sum(item[1] for item in all_rows)
        choice_e = sum(item[2] for item in all_rows)
        records.extend(
            [
                make_metric_record(
                    registry,
                    context,
                    "mme_rs.avg",
                    correct / total,
                    n_samples=total,
                    n_failures=invalid,
                    provenance="supplemental",
                    binomial_successes=correct,
                    notes="sample-weighted total_correct/total_N",
                ),
                make_metric_record(
                    registry,
                    context,
                    "mme_rs.avg_c",
                    fmean(task_rates) if len(task_rates) == len(_MME_TASKS) else None,
                    n_samples=total,
                    n_failures=invalid,
                    provenance="supplemental",
                    availability="available" if len(task_rates) == len(_MME_TASKS) else "missing",
                    notes="equal-task macro average; aggregate derived score has no synthetic CI",
                ),
                make_metric_record(
                    registry,
                    context,
                    "mme_rs.invalid_parse_rate",
                    invalid / total,
                    n_samples=total,
                    n_failures=invalid,
                    provenance="supplemental",
                    binomial_successes=invalid,
                ),
                make_metric_record(
                    registry,
                    context,
                    "mme_rs.choice_e_rate",
                    choice_e / total,
                    n_samples=total,
                    n_failures=invalid,
                    provenance="supplemental",
                    binomial_successes=choice_e,
                ),
            ]
        )
    return records
