"""LEVIR-CC materialization with locked official test A/B ordering."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from m3rs_eval.evaluation import (
    AlignmentResult,
    MetricContext,
    OfficialMetricScore,
    make_metric_record,
    merge_metric_records,
    official_metric_record,
)
from m3rs_eval.registry import MetricRegistry

from .base import BaseDatasetAdapter, DatasetError, MaterializedExample


_NO_CHANGE_TEMPLATES = frozenset(
    {
        "there is no difference",
        "the two scenes seem identical",
        "the scene is the same as before",
        "no change has occurred",
        "almost nothing has changed",
    }
)
_CAPTION_METRICS = ("bleu_1", "bleu_2", "bleu_3", "bleu_4", "meteor", "rouge_l", "cider_d")


class LevirCCAdapter(BaseDatasetAdapter):
    dataset = "levir_cc"
    expected_files = ("annotations.json",)

    def _parse_examples(self) -> list[MaterializedExample]:
        annotation_path, rows = self._annotation_json("annotations.json")
        if not isinstance(rows, list):
            raise DatasetError(f"{self.dataset}: annotations.json must be a JSON array")

        examples = []
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                raise DatasetError(f"{self.dataset}: annotations.json row {index} must be an object")
            split = _text(row, "split", index)
            if split != "test":
                raise DatasetError(
                    f"{self.dataset}: formal materialization requires split exactly 'test', got '{split}'"
                )
            sample = _text(row, "id", index)
            image_a = (self.config.root / _text(row, "image_a", index)).resolve().as_posix()
            image_b = (self.config.root / _text(row, "image_b", index)).resolve().as_posix()
            caption = _text(row, "caption", index)
            change = _text(row, "change", index)
            sample_id = f"levir_cc:{sample}"
            examples.append(
                MaterializedExample(
                    request={
                        "sample_id": sample_id,
                        "dataset": self.dataset,
                        "benchmark_version": "levir-cc:test",
                        "split": split,
                        "task": "caption",
                        "images": [image_a, image_b],
                        "prompt": "Describe the change from image A to image B.",
                        "expected_output": "caption",
                    },
                    reference={"sample_id": sample_id, "answer": caption, "change": change},
                    annotation_path=annotation_path,
                    source_label=f"annotations.json row {index}",
                )
            )
        return examples

    def _scope_dimensions(self, example):
        return {"split": str(example.request["split"]), "task": str(example.request["task"])}

    def _coverage(self, selected, total, task_counts):
        coverage = super()._coverage(selected, total, task_counts)
        selected_slices = _slice_counts(selected)
        total_slices = _slice_counts(total)
        coverage.update(
            {
                "split": "test",
                "ordered_images": ["A", "B"],
                "slices": {
                    slice_name: {"selected": selected_slices.get(slice_name, 0), "total": total_slices[slice_name]}
                    for slice_name in sorted(total_slices)
                },
            }
        )
        return coverage


def _text(row: dict[str, Any], field: str, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"levir_cc: annotations.json row {index} requires nonempty '{field}'")
    return value.strip()


def _slice_counts(examples):
    counts = {}
    for example in examples:
        change = str(example.reference["change"])
        counts[change] = counts.get(change, 0) + 1
    return counts


def classify_no_change(caption: str | None) -> bool:
    """Match the five fixed official LEVIR no-change sentence templates."""
    if not isinstance(caption, str):
        return False
    normalized = re.sub(r"[^a-z0-9]+", " ", caption.casefold()).strip()
    return normalized in _NO_CHANGE_TEMPLATES


def evaluate_levir_alignment(
    alignment: AlignmentResult,
    references: Mapping[str, Mapping[str, Any]],
    registry: MetricRegistry,
    context: MetricContext,
    official_scores: Iterable[OfficialMetricScore] = (),
):
    """Ingest official Caption scores and compute five-template discrimination rates."""
    metric_context = replace(context, benchmark_version="levir-cc:test")
    scores = tuple(official_scores)
    score_by_id = {score.metric_id: score for score in scores}
    if len(score_by_id) != len(scores):
        raise DatasetError("levir_cc: official scorer returned duplicate metric IDs")
    foreign = [score.metric_id for score in scores if not score.metric_id.startswith("levir.caption.")]
    if foreign:
        raise DatasetError(
            f"levir_cc: official scorer returned foreign metric IDs: {', '.join(foreign)}"
        )
    official = []
    slice_counts = {"all": 0, "change": 0, "no_change": 0}
    slice_failures = {"all": 0, "change": 0, "no_change": 0}
    discrimination = {"all": [], "change": [], "no_change": []}

    for row in alignment.rows:
        if row.request.dataset != "levir_cc":
            continue
        reference = references.get(row.request.sample_id)
        if reference is None:
            raise DatasetError(f"levir_cc: missing reference for {row.request.sample_id}")
        change = str(reference.get("change", ""))
        if change not in {"change", "no-change"}:
            raise DatasetError(f"levir_cc: invalid change slice for {row.request.sample_id}")
        scope = "change" if change == "change" else "no_change"
        predicted_no_change = classify_no_change(
            row.prediction.prediction if row.prediction is not None else None
        )
        correct = predicted_no_change == (scope == "no_change") and row.failure is None
        for selected in ("all", scope):
            slice_counts[selected] += 1
            slice_failures[selected] += int(row.failure is not None)
            discrimination[selected].append(correct)

    for scope in ("all", "change", "no_change"):
        for metric_name in _CAPTION_METRICS:
            metric_id = f"levir.caption.{scope}.{metric_name}"
            if metric_id == "levir.caption.no_change.cider_d":
                official.append(
                    make_metric_record(
                        registry,
                        metric_context,
                        metric_id,
                        None,
                        n_samples=slice_counts[scope],
                        n_failures=slice_failures[scope],
                        provenance="official",
                        availability="not_applicable",
                        notes="registry marks no-change CIDEr-D not applicable for primary reporting",
                    )
                )
            elif metric_id in score_by_id:
                official.append(official_metric_record(registry, metric_context, score_by_id[metric_id]))
            else:
                official.append(
                    make_metric_record(
                        registry,
                        metric_context,
                        metric_id,
                        None,
                        n_samples=slice_counts[scope],
                        n_failures=slice_failures[scope],
                        provenance="official",
                        availability="missing",
                        notes="official caption scorer output did not provide this slice metric",
                    )
                )

    supplemental = []
    for scope in ("change", "no_change", "all"):
        values = discrimination[scope]
        if not values:
            continue
        successes = sum(values)
        supplemental.append(
            make_metric_record(
                registry,
                metric_context,
                f"levir.discrimination.{scope}.accuracy",
                successes / len(values),
                n_samples=len(values),
                n_failures=slice_failures[scope],
                provenance="supplemental",
                binomial_successes=successes,
                notes="five fixed official no-change templates; deterministic compatibility metric",
            )
        )
    return merge_metric_records(official, supplemental)
