"""VRSBench materialization with Caption, Grounding, and VQA kept separate."""

from __future__ import annotations

import re
import math
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


_VQA_CATEGORIES = (
    "category", "presence", "quantity", "color", "shape", "size",
    "position", "direction", "scene", "reasoning",
)


class VRSBenchAdapter(BaseDatasetAdapter):
    dataset = "vrsbench"
    expected_files = ("caption.jsonl", "grounding.jsonl", "vqa.jsonl")

    def _parse_examples(self) -> list[MaterializedExample]:
        examples: list[MaterializedExample] = []
        for task, filename, expected_output in (
            ("caption", "caption.jsonl", "caption"),
            ("grounding", "grounding.jsonl", "boxes"),
            ("vqa", "vqa.jsonl", "text"),
        ):
            annotation_path, rows = self._annotation_jsonl(filename)
            for index, row in enumerate(rows, start=1):
                split = _text(row, "split", filename, index)
                if split != "test":
                    raise DatasetError(
                        f"{self.dataset}: formal materialization accepts only 'test' records, got '{split}'"
                    )
                sample_id = f"vrsbench:{task}:{_text(row, 'id', filename, index)}"
                request: dict[str, Any] = {
                    "sample_id": sample_id,
                    "dataset": self.dataset,
                    "benchmark_version": "vrsbench:test",
                    "split": split,
                    "task": task,
                    "images": [(self.config.root / _text(row, "image", filename, index)).resolve().as_posix()],
                    "prompt": _prompt(row, task, filename, index),
                    "expected_output": expected_output,
                }
                reference: dict[str, Any] = {"sample_id": sample_id}
                if task == "caption":
                    reference["answer"] = _text(row, "caption", filename, index)
                elif task == "grounding":
                    reference["boxes"] = _boxes(row, filename, index)
                    reference["grounding_slice"] = _optional_enum(
                        row, "grounding_slice", {"unique", "non_unique"}, filename, index
                    )
                else:
                    # VRSBench official VQA is open-ended: choices are
                    # optional; with choices the task is choice-EM, without
                    # them it is free-form text matching.
                    choices = _optional_choices(row, filename, index)
                    if choices is not None:
                        request["choices"] = choices
                        request["expected_output"] = "choice"
                    reference["answer"] = _text(row, "answer", filename, index)
                    reference["vqa_category"] = _optional_enum(
                        row, "vqa_category", set(_VQA_CATEGORIES), filename, index
                    )
                examples.append(
                    MaterializedExample(request, reference, annotation_path, f"{filename} row {index}")
                )
        return examples

    def _coverage(self, selected, total, task_counts):
        coverage = super()._coverage(selected, total, task_counts)
        coverage.update({"split": "test", "separate_tasks": ["caption", "grounding", "vqa"]})
        return coverage


def _text(row: dict[str, Any], field: str, filename: str, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"vrsbench: {filename} row {index} requires nonempty '{field}'")
    return value.strip()


def _prompt(row: dict[str, Any], task: str, filename: str, index: int) -> str:
    if task == "caption":
        return "Describe the remote-sensing image."
    return _text(row, "question", filename, index)


def _choices(row: dict[str, Any], filename: str, index: int) -> list[str]:
    choices = row.get("choices")
    if not isinstance(choices, list) or not choices or not all(isinstance(choice, str) and choice.strip() for choice in choices):
        raise DatasetError(f"vrsbench: {filename} row {index} requires nonempty string choices")
    return [choice.strip() for choice in choices]


def _optional_choices(row: dict[str, Any], filename: str, index: int) -> list[str] | None:
    choices = row.get("choices")
    if choices is None:
        return None
    if not isinstance(choices, list) or not choices or not all(isinstance(choice, str) and choice.strip() for choice in choices):
        raise DatasetError(f"vrsbench: {filename} row {index} choices must be a nonempty string list or null")
    return [choice.strip() for choice in choices]


def _boxes(row: dict[str, Any], filename: str, index: int) -> list[list[float]]:
    boxes = row.get("boxes")
    if not isinstance(boxes, list) or not boxes:
        raise DatasetError(f"vrsbench: {filename} row {index} requires grounding boxes")
    return boxes


def _optional_enum(
    row: dict[str, Any], field: str, allowed: set[str], filename: str, index: int
) -> str:
    value = row.get(field)
    if not isinstance(value, str) or value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise DatasetError(
            f"vrsbench: {filename} row {index} requires '{field}' in: {choices}"
        )
    return value


def inclusive_hbb_iou(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> float:
    """Compute official +1 HBB IoU in the normalized inclusive 0-100 domain."""
    ax1, ay1, ax2, ay2 = _validated_hbb(first)
    bx1, by1, bx2, by2 = _validated_hbb(second)
    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1) + 1.0)
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1) + 1.0)
    intersection = intersection_width * intersection_height
    first_area = (ax2 - ax1 + 1.0) * (ay2 - ay1 + 1.0)
    second_area = (bx2 - bx1 + 1.0) * (by2 - by1 + 1.0)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def _validated_hbb(box: object) -> tuple[float, float, float, float]:
    if not isinstance(box, (tuple, list)) or len(box) != 4:
        raise DatasetError("vrsbench: HBB must contain four coordinates")
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in box
    ):
        raise DatasetError("vrsbench: HBB coordinates must be finite numbers")
    x1, y1, x2, y2 = (float(value) for value in box)
    if not all(0.0 <= value <= 100.0 for value in (x1, y1, x2, y2)):
        raise DatasetError("vrsbench: HBB coordinates must use the normalized 0-100 domain")
    if x2 < x1 or y2 < y1:
        raise DatasetError("vrsbench: HBB coordinates must satisfy x2>=x1 and y2>=y1")
    return x1, y1, x2, y2


def normalize_vqa_answer(value: str | None) -> str:
    """Apply deterministic Unicode, case, punctuation, article, and whitespace normalization."""
    if not isinstance(value, str):
        return ""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    tokens = [token for token in normalized.split() if token not in {"a", "an", "the"}]
    return " ".join(tokens)


def evaluate_vrs_alignment(
    alignment: AlignmentResult,
    references: Mapping[str, Mapping[str, Any]],
    registry: MetricRegistry,
    context: MetricContext,
    official_scores: Iterable[OfficialMetricScore] = (),
):
    """Compute VRS HBB/VQA diagnostics and ingest official Caption aggregates."""
    scores = tuple(official_scores)
    foreign = [score.metric_id for score in scores if not score.metric_id.startswith("vrs.caption.")]
    if foreign:
        raise DatasetError(f"vrsbench: official scorer returned foreign metric IDs: {', '.join(foreign)}")
    official = [
        official_metric_record(
            registry,
            replace(context, benchmark_version="vrsbench:test"),
            score,
        )
        for score in scores
    ]
    supplemental = []
    grounding: dict[str, list[float]] = {"unique": [], "non_unique": [], "all": []}
    grounding_failures: dict[str, int] = {"unique": 0, "non_unique": 0, "all": 0}
    vqa: dict[str, list[bool]] = {category: [] for category in _VQA_CATEGORIES}
    vqa["all"] = []
    vqa_failures: dict[str, int] = {category: 0 for category in vqa}

    for row in alignment.rows:
        if row.request.dataset != "vrsbench":
            continue
        reference = references.get(row.request.sample_id)
        if reference is None:
            raise DatasetError(f"vrsbench: missing reference for {row.request.sample_id}")
        if row.request.task == "grounding":
            score = _grounding_iou(row.prediction.boxes if row.prediction else None, reference.get("boxes"))
            slice_name = str(reference.get("grounding_slice", ""))
            scopes = ["all"] + ([slice_name] if slice_name in {"unique", "non_unique"} else [])
            for scope in scopes:
                grounding[scope].append(score)
                grounding_failures[scope] += int(row.failure is not None)
        elif row.request.task == "vqa":
            prediction = row.prediction.prediction if row.prediction else None
            correct = _vqa_equal(prediction, str(reference.get("answer", "")))
            category = str(reference.get("vqa_category", ""))
            scopes = ["all"] + ([category] if category in _VQA_CATEGORIES else [])
            for scope in scopes:
                vqa[scope].append(correct)
                vqa_failures[scope] += int(row.failure is not None)

    metric_context = replace(context, benchmark_version="vrsbench:test")
    for scope in ("unique", "non_unique", "all"):
        values = grounding[scope]
        if not values:
            continue
        for threshold, suffix in ((0.5, "acc_0_5"), (0.7, "acc_0_7")):
            successes = sum(value >= threshold for value in values)
            supplemental.append(
                make_metric_record(
                    registry,
                    metric_context,
                    f"vrs.grounding.hbb.{scope}.{suffix}",
                    successes / len(values),
                    n_samples=len(values),
                    n_failures=grounding_failures[scope],
                    provenance="supplemental",
                    binomial_successes=successes,
                    notes="official +1 inclusive-pixel HBB IoU convention",
                )
            )
        supplemental.append(
            make_metric_record(
                registry,
                metric_context,
                f"vrs.grounding.hbb.{scope}.mean_iou",
                fmean(values),
                n_samples=len(values),
                n_failures=grounding_failures[scope],
                provenance="supplemental",
                bootstrap_observations=values,
                notes="official +1 inclusive-pixel HBB IoU convention",
            )
        )

    for category, values in vqa.items():
        if not values:
            continue
        successes = sum(values)
        supplemental.append(
            make_metric_record(
                registry,
                metric_context,
                f"vrs.vqa.acc.{category}",
                successes / len(values),
                n_samples=len(values),
                n_failures=vqa_failures[category],
                provenance="supplemental",
                binomial_successes=successes,
                notes="deterministic normalization; not the GPT-4 official semantic judge",
            )
        )
    return merge_metric_records(official, supplemental)


def _grounding_iou(predicted: object, expected: object) -> float:
    if not isinstance(predicted, tuple) or not predicted:
        return 0.0
    if not isinstance(expected, list) or not expected:
        raise DatasetError("vrsbench: grounding reference requires boxes")
    expected_boxes = []
    for box in expected:
        if not isinstance(box, list) or len(box) != 4:
            raise DatasetError("vrsbench: malformed grounding reference box")
        expected_boxes.append(tuple(float(value) for value in box))
    return max(inclusive_hbb_iou(candidate, target) for candidate in predicted for target in expected_boxes)


def _vqa_equal(predicted: str | None, expected: str) -> bool:
    expected_choices = _choice_letters(expected)
    predicted_choices = _choice_letters(predicted or "")
    if expected_choices:
        return len(expected_choices) == 1 and predicted_choices == expected_choices
    return normalize_vqa_answer(predicted) == normalize_vqa_answer(expected)


def _choice_letters(value: str) -> frozenset[str]:
    return frozenset(
        match.upper() for match in re.findall(r"(?i)(?<![A-Za-z])[A-E](?![A-Za-z])", value)
    )
