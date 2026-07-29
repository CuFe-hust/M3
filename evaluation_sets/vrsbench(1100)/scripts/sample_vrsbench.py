#!/usr/bin/env python3
"""Build the fixed VRSBench 1,100-record comparison subset.

This script is intentionally self-contained and conservative:
- source data is read-only;
- existing staging/final directories are never overwritten;
- all selections are made before the staging directory is created;
- records written to the three official JSON files are unmodified source dicts;
- validation is delegated to validate_vrsbench.py before atomic publication.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SEED = 42
CAP_FILE = "VRSBench_EVAL_Cap.json"
REF_FILE = "VRSBench_EVAL_referring.json"
VQA_FILE = "VRSBench_EVAL_vqa.json"

CORE_VQA_TYPES = [
    "object category",
    "object existence",
    "object quantity",
    "object color",
    "object shape",
    "object size",
    "object position",
    "object direction",
    "scene type",
    "reasoning",
]
SUPPLEMENTAL_VQA_TYPES = ["rural or urban", "image"]
LENGTH_ORDER = ["short", "medium", "long"]
RELATION_PHRASES = [
    "relative to",
    "between",
    "closest",
    "farthest",
    "nearest",
    "left-most",
    "right-most",
    "top-most",
    "bottom-most",
    "larger",
    "smaller",
    "largest",
    "smallest",
    "compared",
    "adjacent",
    "surrounding",
    "above",
    "below",
    "next to",
]
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
RELATION_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    + "|".join(re.escape(x) for x in sorted(RELATION_PHRASES, key=len, reverse=True))
    + r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)

ONES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


class BuildError(RuntimeError):
    """A clear, user-facing build failure."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/user/下载/datasets/vrsbench"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/user/silverdew/VRSBench_ModelCompare_1100"),
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path("/home/user/silverdew/.VRSBench_ModelCompare_1100_staging"),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Select in memory and print feasibility statistics; write nothing.",
    )
    return parser.parse_args()


def load_json_array(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise BuildError(f"Missing source file: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(data, list):
        raise BuildError(f"Source JSON is not an array: {path}")
    if not all(isinstance(item, dict) for item in data):
        raise BuildError(f"Source JSON contains a non-object record: {path}")
    return data


def dump_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def require_text(record: dict[str, Any], field: str, label: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise BuildError(f"{label}: field {field!r} must be a non-empty string")
    return value


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def assign_rank_groups(
    candidates: list[dict[str, Any]],
    metric_name: str,
    low_name: str,
    high_name: str,
) -> dict[str, Any]:
    ordered = sorted(candidates, key=lambda c: (c[metric_name], c["source_index"]))
    edge = math.floor(0.30 * len(ordered))
    for position, candidate in enumerate(ordered):
        if position < edge:
            candidate["group"] = low_name
        elif position >= len(ordered) - edge:
            candidate["group"] = high_name
        else:
            candidate["group"] = "medium"
    summary: dict[str, Any] = {"candidate_count": len(ordered), "edge_count": edge}
    for group in [low_name, "medium", high_name]:
        values = [c[metric_name] for c in ordered if c["group"] == group]
        summary[group] = {
            "count": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return summary


def shuffled_ranks(candidates: Iterable[dict[str, Any]], rng: random.Random) -> None:
    ordered = sorted(candidates, key=lambda c: c["source_index"])
    shuffled = list(ordered)
    rng.shuffle(shuffled)
    for rank, candidate in enumerate(shuffled):
        candidate["random_rank"] = rank


def choose_prefer_new_images(
    candidates: list[dict[str, Any]],
    count: int,
    used_images: set[str],
) -> list[dict[str, Any]]:
    if len(candidates) < count:
        raise BuildError(f"Need {count} candidates but only {len(candidates)} are available")
    remaining = list(candidates)
    chosen: list[dict[str, Any]] = []
    while len(chosen) < count:
        remaining.sort(
            key=lambda c: (
                c["image_id"] in used_images,
                c["random_rank"],
                c["source_index"],
            )
        )
        candidate = remaining.pop(0)
        chosen.append(candidate)
        used_images.add(candidate["image_id"])
    return chosen


def flatten_numbers(value: Any) -> list[float]:
    if isinstance(value, (list, tuple)):
        result: list[float] = []
        for child in value:
            result.extend(flatten_numbers(child))
        return result
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("coordinate is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("coordinate is not finite")
    return [number]


def referring_area(obj_corner: Any) -> float:
    """Return the area of the axis-aligned box enclosing four polygon corners."""
    flat = flatten_numbers(obj_corner)
    if len(flat) != 8:
        raise ValueError(f"expected 8 coordinate values, got {len(flat)}")
    xs = flat[0::2]
    ys = flat[1::2]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    if width < 0 or height < 0:
        raise ValueError("negative bounding-box side")
    return width * height


def parse_nonnegative_integer(answer: str) -> int | None:
    normalized = answer.strip().casefold().replace("‑", "-").replace("–", "-")
    if re.fullmatch(r"\d+", normalized):
        return int(normalized)
    tokens = normalized.replace("-", " ").split()
    if len(tokens) == 1 and tokens[0] in ONES:
        return ONES[tokens[0]]
    if len(tokens) == 1 and tokens[0] in TENS:
        return TENS[tokens[0]]
    if len(tokens) == 2 and tokens[0] in TENS and tokens[1] in ONES:
        tail = ONES[tokens[1]]
        if 1 <= tail <= 9:
            return TENS[tokens[0]] + tail
    return None


def quantity_group(value: int) -> str:
    if value <= 2:
        return "low"
    if value <= 5:
        return "medium"
    return "high"


def base_candidate(
    record: dict[str, Any],
    source_file: str,
    source_index: int,
    task: str,
    subset: str | None,
) -> dict[str, Any]:
    image_id = require_text(record, "image_id", f"{source_file}[{source_index}]")
    return {
        "record": record,
        "source_file": source_file,
        "source_index": source_index,
        "task": task,
        "subset": subset,
        "image_id": image_id,
    }


def caption_candidates(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        label = f"{CAP_FILE}[{index}]"
        text = require_text(record, "ground_truth", label)
        candidate = base_candidate(record, CAP_FILE, index, "caption", None)
        candidate["word_count"] = word_count(text)
        candidates.append(candidate)
    boundaries = assign_rank_groups(candidates, "word_count", "short", "long")
    return candidates, boundaries


def referring_candidates(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        label = f"{REF_FILE}[{index}]"
        unique = record.get("unique")
        if unique is not True and unique is not False:
            raise BuildError(f"{label}: unique must be a JSON boolean")
        obj_cls = require_text(record, "obj_cls", label)
        try:
            area = referring_area(record.get("obj_corner"))
        except ValueError as exc:
            raise BuildError(f"{label}: invalid obj_corner: {exc}") from exc
        candidate = base_candidate(record, REF_FILE, index, "referring", None)
        candidate.update({"unique": unique, "obj_cls": obj_cls, "area": area})
        candidates.append(candidate)
    boundaries = assign_rank_groups(candidates, "area", "small", "large")
    return candidates, boundaries


def vqa_candidates(
    records: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], Counter[str]]:
    by_type: dict[str, list[dict[str, Any]]] = {
        item: [] for item in CORE_VQA_TYPES + SUPPLEMENTAL_VQA_TYPES
    }
    filtered: Counter[str] = Counter()
    for index, record in enumerate(records):
        label = f"{VQA_FILE}[{index}]"
        question_type = require_text(record, "type", label)
        if question_type not in by_type:
            continue
        question = require_text(record, "question", label)
        answer = require_text(record, "ground_truth", label)
        subset = "core" if question_type in CORE_VQA_TYPES else "supplemental"
        candidate = base_candidate(record, VQA_FILE, index, "vqa", subset)
        candidate.update(
            {
                "question_type": question_type,
                "question_word_count": word_count(question),
                "answer_word_count": word_count(answer),
                "answer": answer,
            }
        )
        if question_type == "object existence":
            normalized = answer.strip().casefold()
            if normalized not in {"yes", "no"}:
                filtered["object existence: answer is not yes/no"] += 1
                continue
            candidate["answer_group"] = normalized
        elif question_type == "object quantity":
            parsed = parse_nonnegative_integer(answer)
            if parsed is None:
                filtered["object quantity: answer is not one clear nonnegative integer"] += 1
                continue
            candidate["quantity_value"] = parsed
            candidate["answer_group"] = quantity_group(parsed)
        by_type[question_type].append(candidate)

    boundaries: dict[str, Any] = {}
    for question_type, candidates in by_type.items():
        boundaries[question_type] = assign_rank_groups(
            candidates, "question_word_count", "short", "long"
        )
    return by_type, boundaries, filtered


def select_caption(
    candidates: list[dict[str, Any]],
    used_images: set[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    quotas = {"short": 75, "medium": 100, "long": 75}
    selected: list[dict[str, Any]] = []
    for group in LENGTH_ORDER:
        pool = [c for c in candidates if c["group"] == group]
        shuffled_ranks(pool, rng)
        selected.extend(choose_prefer_new_images(pool, quotas[group], used_images))
    return selected


def select_referring(
    candidates: list[dict[str, Any]],
    used_images: set[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    quotas = {
        (True, "small"): 37,
        (True, "medium"): 50,
        (True, "large"): 38,
        (False, "small"): 38,
        (False, "medium"): 50,
        (False, "large"): 37,
    }
    for candidate in candidates:
        candidate["cell"] = (candidate["unique"], candidate["group"])
    shuffled_ranks(candidates, rng)

    cell_order = list(quotas)
    class_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for cell in cell_order:
        pool = [c for c in candidates if c["cell"] == cell and c not in selected]
        needed = quotas[cell]
        while needed:
            eligible = [c for c in pool if class_counts[c["obj_cls"]] < 37]
            if not eligible:
                raise BuildError(
                    "Referring quotas cannot be completed without exceeding "
                    f"the 37-record obj_cls cap; failed at cell {cell}"
                )
            eligible.sort(
                key=lambda c: (
                    c["image_id"] in used_images,
                    class_counts[c["obj_cls"]],
                    c["obj_cls"],
                    c["random_rank"],
                    c["source_index"],
                )
            )
            candidate = eligible[0]
            pool.remove(candidate)
            selected.append(candidate)
            used_images.add(candidate["image_id"])
            class_counts[candidate["obj_cls"]] += 1
            needed -= 1
    return selected


def bounded_compositions(total: int, limits: list[int]) -> Iterable[tuple[int, ...]]:
    if len(limits) == 1:
        if 0 <= total <= limits[0]:
            yield (total,)
        return
    for first in range(min(total, limits[0]) + 1):
        for rest in bounded_compositions(total - first, limits[1:]):
            yield (first,) + rest


def feasible_tables(
    row_totals: list[int],
    col_totals: list[int],
    capacities: list[list[int]],
) -> Iterable[tuple[tuple[int, ...], ...]]:
    first_limits = [min(capacities[0][j], col_totals[j]) for j in range(len(col_totals))]
    for first in bounded_compositions(row_totals[0], first_limits):
        remaining_after_first = [col_totals[j] - first[j] for j in range(len(col_totals))]
        second_limits = [
            min(capacities[1][j], remaining_after_first[j]) for j in range(len(col_totals))
        ]
        for second in bounded_compositions(row_totals[1], second_limits):
            third = tuple(
                remaining_after_first[j] - second[j] for j in range(len(col_totals))
            )
            if sum(third) != row_totals[2]:
                continue
            if any(third[j] > capacities[2][j] for j in range(len(col_totals))):
                continue
            yield (first, second, third)


def choose_for_table(
    table: tuple[tuple[int, ...], ...],
    cells: list[list[list[dict[str, Any]]]],
    used_images: set[str],
) -> tuple[list[dict[str, Any]], int, int]:
    temp_used = set(used_images)
    selected: list[dict[str, Any]] = []
    rank_sum = 0
    for row in range(len(table)):
        for col in range(len(table[row])):
            pool = list(cells[row][col])
            for _ in range(table[row][col]):
                pool.sort(
                    key=lambda c: (
                        c["image_id"] in temp_used,
                        c["random_rank"],
                        c["source_index"],
                    )
                )
                candidate = pool.pop(0)
                selected.append(candidate)
                temp_used.add(candidate["image_id"])
                rank_sum += candidate["random_rank"]
    new_image_count = len(temp_used - used_images)
    return selected, new_image_count, rank_sum


def select_joint_margins(
    candidates: list[dict[str, Any]],
    answer_groups: list[str],
    answer_totals: list[int],
    used_images: set[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    shuffled_ranks(candidates, rng)
    cells: list[list[list[dict[str, Any]]]] = []
    for length_group in LENGTH_ORDER:
        row: list[list[dict[str, Any]]] = []
        for answer_group in answer_groups:
            row.append(
                [
                    c
                    for c in candidates
                    if c["group"] == length_group and c["answer_group"] == answer_group
                ]
            )
        cells.append(row)
    capacities = [[len(cell) for cell in row] for row in cells]
    tables = feasible_tables([15, 20, 15], answer_totals, capacities)

    best_key: tuple[Any, ...] | None = None
    best_selected: list[dict[str, Any]] | None = None
    for table in tables:
        selected, new_images, rank_sum = choose_for_table(table, cells, used_images)
        flat_table = tuple(value for row in table for value in row)
        key = (-new_images, rank_sum, flat_table)
        if best_key is None or key < best_key:
            best_key = key
            best_selected = selected
    if best_selected is None:
        raise BuildError(
            "No joint solution for length quotas and answer quotas; "
            f"capacities={capacities}, answer_totals={answer_totals}"
        )
    used_images.update(c["image_id"] for c in best_selected)
    return best_selected


def select_vqa(
    by_type: dict[str, list[dict[str, Any]]],
    used_images: set[str],
    rng: random.Random,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for question_type in CORE_VQA_TYPES + SUPPLEMENTAL_VQA_TYPES:
        candidates = by_type[question_type]
        if question_type == "object existence":
            current = select_joint_margins(
                candidates, ["yes", "no"], [25, 25], used_images, rng
            )
        elif question_type == "object quantity":
            current = select_joint_margins(
                candidates,
                ["low", "medium", "high"],
                [17, 16, 17],
                used_images,
                rng,
            )
        else:
            current = []
            for group, count in zip(LENGTH_ORDER, [15, 20, 15]):
                pool = [c for c in candidates if c["group"] == group]
                shuffled_ranks(pool, rng)
                current.extend(choose_prefer_new_images(pool, count, used_images))
        selected.extend(current)
    return selected


def caption_difficulty(candidate: dict[str, Any]) -> tuple[str, float, dict[str, Any]]:
    mapping = {"short": ("easy", 0), "medium": ("medium", 1), "long": ("hard", 2)}
    label, score = mapping[candidate["group"]]
    return label, score, {"length_group": candidate["group"]}


def referring_difficulty(candidate: dict[str, Any]) -> tuple[str, float, dict[str, Any]]:
    score = 0.0
    factors: dict[str, Any] = {
        "unique": candidate["unique"],
        "area_group": candidate["group"],
    }
    if candidate["unique"] is False:
        score += 1
    if candidate["group"] == "medium":
        score += 0.5
    elif candidate["group"] == "small":
        score += 1
    label = "easy" if score < 1 else "medium" if score < 2 else "hard"
    return label, score, factors


def vqa_difficulty(candidate: dict[str, Any]) -> tuple[str, int, dict[str, Any]]:
    length_score = {"short": 0, "medium": 1, "long": 2}[candidate["group"]]
    question_type = candidate["question_type"]
    type_score = (
        2
        if question_type == "reasoning"
        else 1
        if question_type
        in {"object quantity", "object size", "object position", "object direction"}
        else 0
    )
    question = candidate["record"]["question"]
    relation_score = 1 if RELATION_RE.search(question) else 0
    quantity_score = 0
    if question_type == "object quantity":
        value = candidate["quantity_value"]
        quantity_score = 0 if value <= 2 else 1 if value <= 5 else 2
    answer_length_score = 1 if candidate["answer_word_count"] >= 3 else 0
    score = length_score + type_score + relation_score + quantity_score + answer_length_score
    label = "easy" if score <= 1 else "medium" if score <= 3 else "hard"
    factors = {
        "length_group": candidate["group"],
        "question_type": question_type,
        "relation_or_comparison": bool(relation_score),
        "quantity_value": candidate.get("quantity_value"),
        "answer_word_count": candidate["answer_word_count"],
    }
    return label, score, factors


def make_manifest(
    caption: list[dict[str, Any]],
    referring: list[dict[str, Any]],
    vqa: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for candidate in caption + referring + vqa:
        if candidate["task"] == "caption":
            difficulty, score, factors = caption_difficulty(candidate)
            sampling_group = candidate["group"]
            metric = candidate["word_count"]
        elif candidate["task"] == "referring":
            difficulty, score, factors = referring_difficulty(candidate)
            sampling_group = (
                f"unique={'true' if candidate['unique'] else 'false'}|"
                f"area={candidate['group']}"
            )
            metric = candidate["area"]
        else:
            difficulty, score, factors = vqa_difficulty(candidate)
            sampling_group = f"type={candidate['question_type']}|length={candidate['group']}"
            metric = candidate["question_word_count"]
        manifest.append(
            {
                "task": candidate["task"],
                "subset": candidate["subset"],
                "source_file": candidate["source_file"],
                "source_index": candidate["source_index"],
                "image_id": candidate["image_id"],
                "question_id": candidate["record"].get("question_id"),
                "sampling_group": sampling_group,
                "length_or_area_metric": metric,
                "difficulty_proxy": difficulty,
                "difficulty_score": score,
                "difficulty_factors": factors,
            }
        )
    return manifest


def count_nested_distribution(
    manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    overall = Counter(item["difficulty_proxy"] for item in manifest)
    by_section: dict[str, Counter[str]] = defaultdict(Counter)
    by_vqa_type: dict[str, Counter[str]] = defaultdict(Counter)
    for item in manifest:
        if item["task"] == "vqa":
            section = "core_vqa" if item["subset"] == "core" else "supplemental_vqa"
            question_type = item["difficulty_factors"]["question_type"]
            by_vqa_type[question_type][item["difficulty_proxy"]] += 1
        else:
            section = item["task"]
        by_section[section][item["difficulty_proxy"]] += 1
    return {
        "total": len(manifest),
        "overall": dict(overall),
        "by_section": {key: dict(value) for key, value in by_section.items()},
        "by_vqa_type": {key: dict(value) for key, value in by_vqa_type.items()},
        "caption_length_group": dict(
            Counter(
                item["difficulty_factors"]["length_group"]
                for item in manifest
                if item["task"] == "caption"
            )
        ),
        "referring_unique": dict(
            Counter(
                str(item["difficulty_factors"]["unique"]).lower()
                for item in manifest
                if item["task"] == "referring"
            )
        ),
        "referring_area_group": dict(
            Counter(
                item["difficulty_factors"]["area_group"]
                for item in manifest
                if item["task"] == "referring"
            )
        ),
    }


def image_reuse_report(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(item["image_id"] for item in manifest)
    histogram = Counter(counts.values())
    return {
        "unique_image_count": len(counts),
        "reused_record_count": len(manifest) - len(counts),
        "reuse_ratio": (len(manifest) - len(counts)) / len(manifest),
        "usage_count_histogram": {str(k): histogram[k] for k in sorted(histogram)},
        "maximum_usage_count": max(counts.values(), default=0),
        "reused_images": {
            image_id: count for image_id, count in sorted(counts.items()) if count > 1
        },
    }


def make_sampling_report(
    source_counts: dict[str, int],
    caption: list[dict[str, Any]],
    referring: list[dict[str, Any]],
    vqa: list[dict[str, Any]],
    filtered: Counter[str],
    manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    vqa_type_counts = Counter(c["question_type"] for c in vqa)
    vqa_length_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for candidate in vqa:
        vqa_length_counts[candidate["question_type"]][candidate["group"]] += 1
    report = {
        "seed": SEED,
        "source_record_counts": source_counts,
        "filtered_counts": dict(filtered),
        "final_task_counts": {
            "caption": len(caption),
            "referring": len(referring),
            "vqa": len(vqa),
            "total": len(manifest),
        },
        "caption_length_distribution": dict(Counter(c["group"] for c in caption)),
        "referring_cross_distribution": {
            f"unique={'true' if unique else 'false'}|area={group}": sum(
                1 for c in referring if c["unique"] is unique and c["group"] == group
            )
            for unique in [True, False]
            for group in ["small", "medium", "large"]
        },
        "referring_obj_cls_distribution": dict(
            sorted(Counter(c["obj_cls"] for c in referring).items())
        ),
        "vqa_type_distribution": dict(vqa_type_counts),
        "vqa_length_distribution": {
            key: dict(value) for key, value in vqa_length_counts.items()
        },
        "object_existence_answer_distribution": dict(
            Counter(c["answer_group"] for c in vqa if c["question_type"] == "object existence")
        ),
        "object_quantity_distribution": dict(
            Counter(c["answer_group"] for c in vqa if c["question_type"] == "object quantity")
        ),
    }
    report.update(image_reuse_report(manifest))
    return report


def resolve_source_image(images_dir: Path, image_id: str) -> Path:
    relative = Path(image_id)
    if relative.is_absolute() or ".." in relative.parts:
        raise BuildError(f"Unsafe image_id path: {image_id!r}")
    path = images_dir / relative
    if not path.is_file():
        raise BuildError(f"Missing source image for image_id {image_id!r}: {path}")
    return path


def copy_images(images_dir: Path, output_images: Path, manifest: list[dict[str, Any]]) -> None:
    output_images.mkdir(parents=True, exist_ok=False)
    for image_id in sorted({item["image_id"] for item in manifest}):
        source = resolve_source_image(images_dir, image_id)
        destination = output_images / image_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise BuildError(f"Image destination collision: {destination}")
        shutil.copy2(source, destination)


def create_readme() -> str:
    return """# VRSBench_ModelCompare_1100

This directory is a deterministic 1,100-record subset of the official VRSBench
validation annotations for model comparison.

- Caption: 250 records
- Referring: 250 records
- VQA: 600 records
- Seed: 42

Records in the three official JSON files are copied without field changes.
Derived sampling and difficulty information is stored only in auxiliary files.
Image reuse is allowed; selection merely prefers image IDs not already used.
The difficulty labels are task-specific heuristics, not official VRSBench labels.

Re-run validation:

```bash
python scripts/validate_vrsbench.py \\
  --source-root "/home/user/下载/datasets/vrsbench" \\
  --dataset-root "/home/user/silverdew/VRSBench_ModelCompare_1100" \\
  --write-report
```

VRSBench text annotations are published under CC-BY-4.0. Some underlying images
come from DOTA and are restricted to academic use. This subset does not alter
the licenses of the source annotations or images.
"""


def ensure_safe_paths(source_root: Path, output_root: Path, staging_root: Path) -> None:
    if output_root.exists():
        raise BuildError(f"Final output already exists; refusing to overwrite: {output_root}")
    if staging_root.exists():
        raise BuildError(f"Staging output already exists; refusing to overwrite: {staging_root}")
    if output_root.parent.resolve() != staging_root.parent.resolve():
        raise BuildError("Final and staging directories must have the same parent directory")
    source_resolved = source_root.resolve()
    for target in [output_root.resolve(strict=False), staging_root.resolve(strict=False)]:
        if target == source_resolved or source_resolved in target.parents:
            raise BuildError(f"Output path must not be inside the source directory: {target}")


def build_selection(source_root: Path) -> dict[str, Any]:
    cap_records = load_json_array(source_root / CAP_FILE)
    ref_records = load_json_array(source_root / REF_FILE)
    vqa_records = load_json_array(source_root / VQA_FILE)

    cap_candidates, cap_boundaries = caption_candidates(cap_records)
    ref_candidates, ref_boundaries = referring_candidates(ref_records)
    vqa_by_type, vqa_boundaries, filtered = vqa_candidates(vqa_records)

    rng = random.Random(SEED)
    used_images: set[str] = set()
    selected_cap = select_caption(cap_candidates, used_images, rng)
    selected_ref = select_referring(ref_candidates, used_images, rng)
    selected_vqa = select_vqa(vqa_by_type, used_images, rng)

    selected_cap.sort(key=lambda c: c["source_index"])
    selected_ref.sort(key=lambda c: c["source_index"])
    type_order = {
        value: index for index, value in enumerate(CORE_VQA_TYPES + SUPPLEMENTAL_VQA_TYPES)
    }
    selected_vqa.sort(key=lambda c: (type_order[c["question_type"]], c["source_index"]))

    manifest = make_manifest(selected_cap, selected_ref, selected_vqa)
    if len(manifest) != 1100:
        raise BuildError(f"Internal count error: selected {len(manifest)} records, expected 1100")

    images_dir = source_root / "Images_val" / "Images_val"
    if not images_dir.is_dir():
        raise BuildError(f"Missing image directory: {images_dir}")
    for image_id in sorted({item["image_id"] for item in manifest}):
        resolve_source_image(images_dir, image_id)

    source_counts = {
        CAP_FILE: len(cap_records),
        REF_FILE: len(ref_records),
        VQA_FILE: len(vqa_records),
    }
    difficulty_method = {
        "official_difficulty": False,
        "seed": SEED,
        "random_generator": "Python random.Random(42)",
        "word_pattern": WORD_RE.pattern,
        "rank_group_rule": (
            "sort by metric then source_index; first floor(30%) is short/small; "
            "last floor(30%) is long/large; remainder is medium"
        ),
        "caption_boundaries": cap_boundaries,
        "referring_boundaries": ref_boundaries,
        "vqa_boundaries_by_type": vqa_boundaries,
        "obj_corner_rule": (
            "flatten four polygon points (8 numbers), then use min/max x/y "
            "to compute enclosing axis-aligned box area"
        ),
        "quantity_rule": (
            "accept a full nonnegative decimal integer or an English number "
            "word from zero through ninety-nine"
        ),
        "relation_phrases": RELATION_PHRASES,
        "selection_rule": (
            "fixed task/type order; seeded candidate order; prefer unused image_id; "
            "allow reuse when needed"
        ),
    }
    sampling_report = make_sampling_report(
        source_counts,
        selected_cap,
        selected_ref,
        selected_vqa,
        filtered,
        manifest,
    )
    return {
        "caption": selected_cap,
        "referring": selected_ref,
        "vqa": selected_vqa,
        "manifest": manifest,
        "difficulty_method": difficulty_method,
        "difficulty_distribution": count_nested_distribution(manifest),
        "sampling_report": sampling_report,
        "images_dir": images_dir,
    }


def write_staging(
    selection: dict[str, Any],
    source_root: Path,
    staging_root: Path,
    validator_script: Path,
) -> None:
    staging_root.mkdir(parents=False, exist_ok=False)
    try:
        dump_json(
            staging_root / CAP_FILE,
            [copy.deepcopy(c["record"]) for c in selection["caption"]],
        )
        dump_json(
            staging_root / REF_FILE,
            [copy.deepcopy(c["record"]) for c in selection["referring"]],
        )
        dump_json(
            staging_root / VQA_FILE,
            [copy.deepcopy(c["record"]) for c in selection["vqa"]],
        )
        dump_json(staging_root / "selection_manifest.json", selection["manifest"])
        dump_json(staging_root / "difficulty_method.json", selection["difficulty_method"])
        dump_json(
            staging_root / "difficulty_distribution.json",
            selection["difficulty_distribution"],
        )
        dump_json(staging_root / "sampling_report.json", selection["sampling_report"])
        with (staging_root / "README.md").open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(create_readme())

        scripts_dir = staging_root / "scripts"
        scripts_dir.mkdir()
        shutil.copy2(Path(__file__).resolve(), scripts_dir / "sample_vrsbench.py")
        shutil.copy2(validator_script, scripts_dir / "validate_vrsbench.py")
        copy_images(
            selection["images_dir"],
            staging_root / "Images_val" / "Images_val",
            selection["manifest"],
        )

        completed = subprocess.run(
            [
                sys.executable,
                str(scripts_dir / "validate_vrsbench.py"),
                "--source-root",
                str(source_root),
                "--dataset-root",
                str(staging_root),
                "--write-report",
            ],
            check=False,
        )
        if completed.returncode != 0:
            raise BuildError(
                f"Independent validation failed with exit code {completed.returncode}; "
                f"staging directory retained at {staging_root}"
            )
        report = load_json_array_or_object(staging_root / "validation_report.json")
        if not isinstance(report, dict) or report.get("validation") != "PASS":
            raise BuildError(
                f"Validator did not produce PASS; staging directory retained at {staging_root}"
            )
    except Exception:
        # Deliberately retain staging for audit; never remove it automatically.
        raise


def load_json_array_or_object(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser()
    output_root = args.output_root.expanduser()
    staging_root = args.staging_root.expanduser()
    try:
        ensure_safe_paths(source_root, output_root, staging_root)
        selection = build_selection(source_root)
        preflight = {
            "status": "FEASIBLE",
            "final_task_counts": selection["sampling_report"]["final_task_counts"],
            "unique_image_count": selection["sampling_report"]["unique_image_count"],
            "reused_record_count": selection["sampling_report"]["reused_record_count"],
            "maximum_usage_count": selection["sampling_report"]["maximum_usage_count"],
        }
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        if args.preflight_only:
            return 0

        validator_script = Path(__file__).resolve().with_name("validate_vrsbench.py")
        if not validator_script.is_file():
            raise BuildError(f"Validator script is missing: {validator_script}")
        write_staging(selection, source_root, staging_root, validator_script)
        if output_root.exists():
            raise BuildError(
                f"Final output appeared during validation; refusing to overwrite: {output_root}"
            )
        os.replace(staging_root, output_root)
        print(f"PASS: published dataset to {output_root}")
        return 0
    except BuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
