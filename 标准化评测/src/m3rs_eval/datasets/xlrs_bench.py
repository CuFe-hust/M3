"""XLRS-Bench materialization that never aliases Full/Lite or en/zh scopes."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import replace
from statistics import fmean
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


XLRS_L2_TASKS = (
    "counting",
    "scene_classification",
    "object_spatial_relationship",
    "object_properties",
    "complex_reasoning",
    "planning",
    "spatiotemporal_reasoning",
    "anomaly_reasoning",
)
XLRS_L3_TASKS = (
    "overall_counting",
    "regional_counting",
    "overall_land_use_classification",
    "regional_land_use_classification",
    "object_spatial_relationship",
    "object_classification",
    "object_color",
    "object_motion_state",
    "environmental_condition_reasoning",
    "counting_with_complex_reasoning",
    "anomaly_detection_and_interpretation",
    "route_planning",
    "regional_counting_with_change_detection",
)


class XLRSBenchAdapter(BaseDatasetAdapter):
    dataset = "xlrs_bench"
    expected_files = ("tasks.jsonl",)

    def _parse_examples(self) -> list[MaterializedExample]:
        annotation_path, rows = self._annotation_jsonl("tasks.jsonl")
        examples = []
        for index, row in enumerate(rows, start=1):
            split = _text(row, "split", index)
            if split != "public_test":
                raise DatasetError(
                    f"{self.dataset}: formal materialization requires split exactly 'public_test', got '{split}'"
                )
            variant = _text(row, "variant", index)
            language = _text(row, "language", index)
            task = _text(row, "task", index)
            if variant not in {"full", "lite"} or language not in {"en", "zh"}:
                raise DatasetError(f"{self.dataset}: each sample must explicitly declare Full/Lite and en/zh")
            if task not in {"vqa", "caption", "grounding"}:
                raise DatasetError(f"{self.dataset}: unsupported task '{task}'")
            sample_id = f"xlrs_bench:{variant}:{language}:{task}:{_text(row, 'id', index)}"
            request: dict[str, Any] = {
                "sample_id": sample_id,
                "dataset": self.dataset,
                "benchmark_version": f"xlrs-bench:{variant}:{language}",
                "split": split,
                "task": task,
                "images": [(self.config.root / _text(row, "image", index)).resolve().as_posix()],
                "prompt": _text(row, "prompt", index),
                "expected_output": "boxes" if task == "grounding" else ("choice" if task == "vqa" else "caption"),
                "language": language,
                "variant": variant,
            }
            reference: dict[str, Any] = {"sample_id": sample_id}
            if task == "grounding":
                reference["boxes"] = _boxes(row, index)
            else:
                if task == "vqa":
                    request["choices"] = _choices(row, index)
                reference["answer"] = _text(row, "answer", index)
                for dimension in ("l2", "l3"):
                    value = row.get(dimension)
                    if value is not None:
                        if not isinstance(value, str) or not value.strip():
                            raise DatasetError(
                                f"{self.dataset}: tasks.jsonl row {index} has invalid '{dimension}'"
                            )
                        reference[dimension] = value.strip()
            examples.append(MaterializedExample(request, reference, annotation_path, f"tasks.jsonl row {index}"))
        return examples

    def _scope_dimensions(self, example):
        return {
            "variant": str(example.request["variant"]),
            "language": str(example.request["language"]),
            "task": str(example.request["task"]),
        }

    def _coverage(self, selected, total, task_counts):
        coverage = super()._coverage(selected, total, task_counts)
        present = {self._scope_label(example) for example in total}
        unavailable = [
            scope
            for scope in (
                "variant=full|language=en|task=vqa",
                "variant=full|language=zh|task=vqa",
                "variant=full|language=en|task=caption",
                "variant=full|language=en|task=grounding",
                "variant=full|language=zh|task=caption",
                "variant=full|language=zh|task=grounding",
                "variant=lite|language=en|task=vqa",
                "variant=lite|language=en|task=caption",
                "variant=lite|language=en|task=grounding",
                "variant=lite|language=zh|task=vqa",
                "variant=lite|language=zh|task=caption",
                "variant=lite|language=zh|task=grounding",
            )
            if scope not in present
        ]
        coverage.update(
            {
                "split": "public_test",
                "variants": ["full", "lite"],
                "languages": ["en", "zh"],
                "unavailable_scopes": unavailable,
            }
        )
        return coverage


def _text(row: dict[str, Any], field: str, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"xlrs_bench: tasks.jsonl row {index} requires nonempty '{field}'")
    return value.strip()


def _choices(row: dict[str, Any], index: int) -> list[str]:
    choices = row.get("choices")
    if not isinstance(choices, list) or not choices or not all(isinstance(choice, str) and choice.strip() for choice in choices):
        raise DatasetError(f"xlrs_bench: tasks.jsonl row {index} requires nonempty string choices")
    return [choice.strip() for choice in choices]


def _boxes(row: dict[str, Any], index: int) -> list[list[float]]:
    boxes = row.get("boxes")
    if not isinstance(boxes, list) or not boxes:
        raise DatasetError(f"xlrs_bench: tasks.jsonl row {index} requires grounding boxes")
    return boxes


def extract_choice_set(value: str | None) -> frozenset[str]:
    """Extract the unordered A-E option set required by XLRS set-EM tasks."""
    if not isinstance(value, str):
        return frozenset()
    return frozenset(
        match.upper() for match in re.findall(r"(?i)(?<![A-Za-z])[A-E](?![A-Za-z])", value)
    )


def evaluate_xlrs_alignment(
    alignment: AlignmentResult,
    references: Mapping[str, Mapping[str, Any]],
    registry: MetricRegistry,
    context: MetricContext,
    official_scores: Iterable[OfficialMetricScore] = (),
):
    """Compute Full paper L2/L3 diagnostics and separate Lite-English aggregates."""
    official = []
    for score in official_scores:
        if not (score.metric_id.startswith("xlrs.caption.") or score.metric_id.startswith("xlrs.grounding.")):
            raise DatasetError(
                f"xlrs_bench: official scorer returned foreign metric ID: {score.metric_id}"
            )
        language = score.metric_id.split(".")[2]
        official.append(
            official_metric_record(
                registry,
                replace(context, benchmark_version=f"xlrs-bench:full:{language}"),
                score,
            )
        )

    full_l2: dict[str, dict[str, list[tuple[bool, bool]]]] = {
        language: {task: [] for task in XLRS_L2_TASKS} for language in ("en", "zh")
    }
    full_l3: dict[str, dict[str, list[tuple[bool, bool]]]] = {
        language: {task: [] for task in XLRS_L3_TASKS} for language in ("en", "zh")
    }
    lite_en: dict[str, list[tuple[bool, bool]]] = {task: [] for task in XLRS_L3_TASKS}
    for row in alignment.rows:
        request = row.request
        if request.dataset != "xlrs_bench" or request.task != "vqa":
            continue
        reference = references.get(request.sample_id)
        if reference is None:
            raise DatasetError(f"xlrs_bench: missing reference for {request.sample_id}")
        language = request.language or ""
        variant = request.variant or ""
        if language not in {"en", "zh"} or variant not in {"full", "lite"}:
            raise DatasetError("xlrs_bench: aligned VQA rows require explicit variant and language")
        predicted = row.prediction.prediction if row.prediction else None
        correct = _xlrs_exact(predicted, str(reference.get("answer", "")))
        failed = row.failure is not None or not extract_choice_set(predicted)
        if variant == "full":
            l2 = reference.get("l2")
            l3 = reference.get("l3")
            if l2 is None and l3 is None:
                raise DatasetError("xlrs_bench: Full VQA row requires l2 and/or l3")
            if l2 is not None:
                if l2 not in XLRS_L2_TASKS:
                    raise DatasetError(f"xlrs_bench: unknown Full L2 task '{l2}'")
                full_l2[language][str(l2)].append((correct, failed))
            if l3 is not None:
                if l3 not in XLRS_L3_TASKS:
                    raise DatasetError(f"xlrs_bench: unknown Full L3 task '{l3}'")
                full_l3[language][str(l3)].append((correct, failed))
        elif language == "en":
            l3 = str(reference.get("l3", ""))
            if l3 not in XLRS_L3_TASKS:
                raise DatasetError(f"xlrs_bench: unknown Lite-English L3 task '{l3}'")
            lite_en[l3].append((correct, failed))
        # Lite Chinese has no registered metric lane and is intentionally not persisted.

    supplemental = []
    for language in ("en", "zh"):
        full_context = replace(context, benchmark_version=f"xlrs-bench:full:{language}")
        l2_rates = []
        l2_total = 0
        l2_failures = 0
        for task in XLRS_L2_TASKS:
            values = full_l2[language][task]
            if not values:
                supplemental.append(
                    make_metric_record(
                        registry,
                        full_context,
                        f"xlrs.vqa.{language}.l2.{task}.acc",
                        None,
                        n_samples=0,
                        n_failures=0,
                        provenance="supplemental",
                        availability="missing",
                        notes="supported Full L2 dimension absent from this materialization",
                    )
                )
                continue
            successes = sum(correct for correct, _ in values)
            failures = sum(failed for _, failed in values)
            l2_total += len(values)
            l2_failures += failures
            rate = successes / len(values)
            l2_rates.append(rate)
            supplemental.append(
                make_metric_record(
                    registry,
                    full_context,
                    f"xlrs.vqa.{language}.l2.{task}.acc",
                    rate,
                    n_samples=len(values),
                    n_failures=failures,
                    provenance="supplemental",
                    binomial_successes=successes,
                    notes="deterministic exact/choice-set matching",
                )
            )
        supplemental.append(
            make_metric_record(
                registry,
                full_context,
                f"xlrs.vqa.{language}.paper_avg_l2",
                fmean(l2_rates) if len(l2_rates) == len(XLRS_L2_TASKS) else None,
                n_samples=l2_total,
                n_failures=l2_failures,
                provenance="supplemental",
                availability="available" if len(l2_rates) == len(XLRS_L2_TASKS) else "missing",
                notes="paper 8-L2 equal-weight macro; never aliased to Lite L3 macro",
            )
        )

        for task in XLRS_L3_TASKS:
            values = full_l3[language][task]
            metric_id = f"xlrs.vqa.{language}.l3.{task}.acc"
            if not values:
                supplemental.append(
                    make_metric_record(
                        registry,
                        full_context,
                        metric_id,
                        None,
                        n_samples=0,
                        n_failures=0,
                        provenance="supplemental",
                        availability="missing",
                        notes="supported Full paper L3 dimension absent from this materialization",
                    )
                )
                continue
            successes = sum(correct for correct, _ in values)
            failures = sum(failed for _, failed in values)
            supplemental.append(
                make_metric_record(
                    registry,
                    full_context,
                    metric_id,
                    successes / len(values),
                    n_samples=len(values),
                    n_failures=failures,
                    provenance="supplemental",
                    binomial_successes=successes,
                    notes="Full paper L3 deterministic exact/choice-set matching",
                )
            )

    # Lite data feeds only the two explicit English Lite aggregate IDs.
    lite_context = replace(context, benchmark_version="xlrs-bench:lite:en")
    flat = [item for task in XLRS_L3_TASKS for item in lite_en[task]]
    l3_rates = []
    for task in XLRS_L3_TASKS:
        values = lite_en[task]
        if not values:
            continue
        successes = sum(correct for correct, _ in values)
        l3_rates.append(successes / len(values))
    if flat:
        successes = sum(correct for correct, _ in flat)
        failures = sum(failed for _, failed in flat)
        supplemental.extend(
            [
                make_metric_record(
                    registry,
                    lite_context,
                    "xlrs.vqa.en.lite.micro_acc",
                    successes / len(flat),
                    n_samples=len(flat),
                    n_failures=failures,
                    provenance="supplemental",
                    binomial_successes=successes,
                    notes="Lite sample micro accuracy",
                ),
                make_metric_record(
                    registry,
                    lite_context,
                    "xlrs.vqa.en.lite.macro_l3_acc",
                    fmean(l3_rates) if len(l3_rates) == len(XLRS_L3_TASKS) else None,
                    n_samples=len(flat),
                    n_failures=failures,
                    provenance="supplemental",
                    availability="available" if len(l3_rates) == len(XLRS_L3_TASKS) else "missing",
                    notes="Lite 13-L3 equal-weight macro; never aliased to paper L2",
                ),
            ]
        )
    else:
        supplemental.extend(
            make_metric_record(
                registry,
                lite_context,
                metric_id,
                None,
                n_samples=0,
                n_failures=0,
                provenance="supplemental",
                availability="missing",
                notes="Lite-English aggregate absent; Lite-zh is registry-unsupported for this lane",
            )
            for metric_id in (
                "xlrs.vqa.en.lite.micro_acc",
                "xlrs.vqa.en.lite.macro_l3_acc",
            )
        )
    return merge_metric_records(official, supplemental)


def _xlrs_exact(predicted: str | None, expected: str) -> bool:
    expected_choices = extract_choice_set(expected)
    predicted_choices = extract_choice_set(predicted)
    if expected_choices:
        return predicted_choices == expected_choices
    return _normalize_exact(predicted) == _normalize_exact(expected)


def _normalize_exact(value: str | None) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())
