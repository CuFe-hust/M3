#!/usr/bin/env python3
"""Independently validate a VRSBench_ModelCompare_1100 dataset directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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
ALL_VQA_TYPES = CORE_VQA_TYPES + SUPPLEMENTAL_VQA_TYPES
LENGTH_ORDER = ["short", "medium", "long"]
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/user/下载/datasets/vrsbench"),
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write validation_report.json inside dataset-root.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_array(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Expected an array of objects: {path}")
    return data


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


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
    flat = flatten_numbers(obj_corner)
    if len(flat) != 8:
        raise ValueError(f"expected 8 coordinate values, got {len(flat)}")
    xs = flat[0::2]
    ys = flat[1::2]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


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


def assign_groups(metrics: list[tuple[int, float]]) -> dict[int, str]:
    ordered = sorted(metrics, key=lambda item: (item[1], item[0]))
    edge = math.floor(0.30 * len(ordered))
    result: dict[int, str] = {}
    for position, (source_index, _) in enumerate(ordered):
        if position < edge:
            result[source_index] = "short"
        elif position >= len(ordered) - edge:
            result[source_index] = "long"
        else:
            result[source_index] = "medium"
    return result


def assign_area_groups(metrics: list[tuple[int, float]]) -> dict[int, str]:
    groups = assign_groups(metrics)
    return {
        index: "small" if group == "short" else "large" if group == "long" else group
        for index, group in groups.items()
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Checks:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def equal(self, name: str, actual: Any, expected: Any) -> bool:
        passed = actual == expected
        self.items.append(
            {
                "name": name,
                "expected": expected,
                "actual": actual,
                "result": "PASS" if passed else "FAIL",
            }
        )
        return passed

    def truth(self, name: str, condition: bool, detail: Any = None) -> bool:
        passed = bool(condition)
        self.items.append(
            {
                "name": name,
                "expected": True,
                "actual": passed,
                "detail": detail,
                "result": "PASS" if passed else "FAIL",
            }
        )
        return passed

    @property
    def error_count(self) -> int:
        return sum(item["result"] == "FAIL" for item in self.items)


def load_required_files(
    source_root: Path, dataset_root: Path
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    source = {
        CAP_FILE: read_array(source_root / CAP_FILE),
        REF_FILE: read_array(source_root / REF_FILE),
        VQA_FILE: read_array(source_root / VQA_FILE),
    }
    output = {
        CAP_FILE: read_array(dataset_root / CAP_FILE),
        REF_FILE: read_array(dataset_root / REF_FILE),
        VQA_FILE: read_array(dataset_root / VQA_FILE),
    }
    manifest = read_array(dataset_root / "selection_manifest.json")
    difficulty_distribution = read_json(dataset_root / "difficulty_distribution.json")
    sampling_report = read_json(dataset_root / "sampling_report.json")
    if not isinstance(difficulty_distribution, dict):
        raise ValueError("difficulty_distribution.json must be an object")
    if not isinstance(sampling_report, dict):
        raise ValueError("sampling_report.json must be an object")
    return source, output, manifest, difficulty_distribution, sampling_report


def expected_manifest_groups(
    source: dict[str, list[dict[str, Any]]]
) -> tuple[dict[int, str], dict[int, str], dict[str, dict[int, str]]]:
    caption_metrics = []
    for index, record in enumerate(source[CAP_FILE]):
        text = record.get("ground_truth")
        if not isinstance(text, str):
            raise ValueError(f"{CAP_FILE}[{index}].ground_truth is not text")
        caption_metrics.append((index, float(word_count(text))))
    caption_groups = assign_groups(caption_metrics)

    referring_metrics = []
    for index, record in enumerate(source[REF_FILE]):
        referring_metrics.append((index, referring_area(record.get("obj_corner"))))
    referring_groups = assign_area_groups(referring_metrics)

    vqa_groups: dict[str, dict[int, str]] = {}
    for question_type in ALL_VQA_TYPES:
        metrics: list[tuple[int, float]] = []
        for index, record in enumerate(source[VQA_FILE]):
            if record.get("type") != question_type:
                continue
            question = record.get("question")
            answer = record.get("ground_truth")
            if not isinstance(question, str) or not isinstance(answer, str):
                continue
            if question_type == "object existence":
                if answer.strip().casefold() not in {"yes", "no"}:
                    continue
            if question_type == "object quantity":
                if parse_nonnegative_integer(answer) is None:
                    continue
            metrics.append((index, float(word_count(question))))
        vqa_groups[question_type] = assign_groups(metrics)
    return caption_groups, referring_groups, vqa_groups


def validate_counts(
    checks: Checks,
    output: dict[str, list[dict[str, Any]]],
    manifest: list[dict[str, Any]],
) -> None:
    checks.equal("caption record count", len(output[CAP_FILE]), 250)
    checks.equal("referring record count", len(output[REF_FILE]), 250)
    checks.equal("VQA record count", len(output[VQA_FILE]), 600)
    checks.equal(
        "total formal record count",
        sum(len(output[name]) for name in [CAP_FILE, REF_FILE, VQA_FILE]),
        1100,
    )
    checks.equal("manifest record count", len(manifest), 1100)

    vqa_type_counts = Counter(record.get("type") for record in output[VQA_FILE])
    for question_type in ALL_VQA_TYPES:
        checks.equal(f"VQA type count: {question_type}", vqa_type_counts[question_type], 50)
    unexpected = sorted(set(vqa_type_counts) - set(ALL_VQA_TYPES), key=str)
    checks.equal("unexpected VQA types", unexpected, [])


def validate_provenance(
    checks: Checks,
    source: dict[str, list[dict[str, Any]]],
    output: dict[str, list[dict[str, Any]]],
    manifest: list[dict[str, Any]],
) -> None:
    combined = output[CAP_FILE] + output[REF_FILE] + output[VQA_FILE]
    source_file_ranges = (
        [CAP_FILE] * len(output[CAP_FILE])
        + [REF_FILE] * len(output[REF_FILE])
        + [VQA_FILE] * len(output[VQA_FILE])
    )
    all_equal = len(combined) == len(manifest)
    problems: list[str] = []
    if all_equal:
        for position, (record, item, expected_file) in enumerate(
            zip(combined, manifest, source_file_ranges)
        ):
            source_file = item.get("source_file")
            source_index = item.get("source_index")
            if source_file != expected_file:
                all_equal = False
                problems.append(f"manifest[{position}] source_file={source_file!r}")
                continue
            if isinstance(source_index, bool) or not isinstance(source_index, int):
                all_equal = False
                problems.append(f"manifest[{position}] invalid source_index")
                continue
            if not 0 <= source_index < len(source[source_file]):
                all_equal = False
                problems.append(f"manifest[{position}] source_index out of range")
                continue
            source_record = source[source_file][source_index]
            if record != source_record:
                all_equal = False
                problems.append(f"formal record {position} differs from source")
            if item.get("image_id") != record.get("image_id"):
                all_equal = False
                problems.append(f"manifest[{position}] image_id mismatch")
            if item.get("question_id") != record.get("question_id"):
                all_equal = False
                problems.append(f"manifest[{position}] question_id mismatch")
    checks.truth(
        "formal records and manifest exactly match source records",
        all_equal,
        problems[:20],
    )


def validate_order(
    checks: Checks,
    output: dict[str, list[dict[str, Any]]],
    manifest: list[dict[str, Any]],
) -> None:
    cap_manifest = manifest[: len(output[CAP_FILE])]
    ref_start = len(output[CAP_FILE])
    ref_end = ref_start + len(output[REF_FILE])
    ref_manifest = manifest[ref_start:ref_end]
    vqa_manifest = manifest[ref_end:]
    checks.truth(
        "Caption output order is source_index ascending",
        [x.get("source_index") for x in cap_manifest]
        == sorted(x.get("source_index") for x in cap_manifest),
    )
    checks.truth(
        "Referring output order is source_index ascending",
        [x.get("source_index") for x in ref_manifest]
        == sorted(x.get("source_index") for x in ref_manifest),
    )
    type_order = {value: index for index, value in enumerate(ALL_VQA_TYPES)}
    vqa_keys = [
        (type_order.get(record.get("type"), 999), item.get("source_index"))
        for record, item in zip(output[VQA_FILE], vqa_manifest)
    ]
    checks.truth("VQA output order is fixed type then source_index", vqa_keys == sorted(vqa_keys))


def validate_groups_and_quotas(
    checks: Checks,
    source: dict[str, list[dict[str, Any]]],
    output: dict[str, list[dict[str, Any]]],
    manifest: list[dict[str, Any]],
) -> None:
    caption_groups, referring_groups, vqa_groups = expected_manifest_groups(source)

    caption_manifest = manifest[:250]
    caption_actual = Counter()
    caption_manifest_ok = True
    for item in caption_manifest:
        index = item.get("source_index")
        group = caption_groups.get(index)
        caption_actual[group] += 1
        if item.get("sampling_group") != group:
            caption_manifest_ok = False
    checks.equal(
        "Caption length distribution",
        {key: caption_actual[key] for key in LENGTH_ORDER},
        {"short": 75, "medium": 100, "long": 75},
    )
    checks.truth("Caption manifest groups match recomputed groups", caption_manifest_ok)

    ref_manifest = manifest[250:500]
    ref_cross = Counter()
    ref_classes = Counter()
    ref_manifest_ok = True
    for record, item in zip(output[REF_FILE], ref_manifest):
        index = item.get("source_index")
        group = referring_groups.get(index)
        unique = record.get("unique")
        ref_cross[(unique, group)] += 1
        ref_classes[record.get("obj_cls")] += 1
        expected_sampling_group = (
            f"unique={'true' if unique is True else 'false'}|area={group}"
        )
        if (
            (unique is not True and unique is not False)
            or item.get("sampling_group") != expected_sampling_group
        ):
            ref_manifest_ok = False
    expected_ref = {
        (True, "small"): 37,
        (True, "medium"): 50,
        (True, "large"): 38,
        (False, "small"): 38,
        (False, "medium"): 50,
        (False, "large"): 37,
    }
    checks.equal(
        "Referring unique-area cross distribution",
        {str(key): ref_cross[key] for key in expected_ref},
        {str(key): value for key, value in expected_ref.items()},
    )
    checks.truth("Referring manifest groups match recomputed groups", ref_manifest_ok)
    checks.truth(
        "Referring obj_cls cap",
        max(ref_classes.values(), default=0) <= 37,
        {"maximum": max(ref_classes.values(), default=0), "counts": dict(ref_classes)},
    )

    vqa_manifest = manifest[500:]
    vqa_length = defaultdict(Counter)
    vqa_manifest_ok = True
    existence = Counter()
    quantity = Counter()
    for record, item in zip(output[VQA_FILE], vqa_manifest):
        question_type = record.get("type")
        source_index = item.get("source_index")
        group = vqa_groups.get(question_type, {}).get(source_index)
        vqa_length[question_type][group] += 1
        expected_sampling_group = f"type={question_type}|length={group}"
        if item.get("sampling_group") != expected_sampling_group:
            vqa_manifest_ok = False
        answer = record.get("ground_truth")
        if question_type == "object existence" and isinstance(answer, str):
            existence[answer.strip().casefold()] += 1
        if question_type == "object quantity" and isinstance(answer, str):
            parsed = parse_nonnegative_integer(answer)
            quantity[quantity_group(parsed) if parsed is not None else "invalid"] += 1

    for question_type in ALL_VQA_TYPES:
        actual = {key: vqa_length[question_type][key] for key in LENGTH_ORDER}
        checks.equal(
            f"VQA length distribution: {question_type}",
            actual,
            {"short": 15, "medium": 20, "long": 15},
        )
    checks.truth("VQA manifest groups match recomputed groups", vqa_manifest_ok)
    checks.equal(
        "object existence Yes/No distribution",
        {"yes": existence["yes"], "no": existence["no"]},
        {"yes": 25, "no": 25},
    )
    checks.equal(
        "object existence has no other answers",
        sum(existence.values()) - existence["yes"] - existence["no"],
        0,
    )
    checks.equal(
        "object quantity low/medium/high distribution",
        {key: quantity[key] for key in ["low", "medium", "high"]},
        {"low": 17, "medium": 16, "high": 17},
    )
    checks.equal("object quantity has no invalid answers", quantity["invalid"], 0)


def validate_images(
    checks: Checks,
    source_root: Path,
    dataset_root: Path,
    manifest: list[dict[str, Any]],
) -> None:
    source_images = source_root / "Images_val" / "Images_val"
    target_images = dataset_root / "Images_val" / "Images_val"
    expected_ids = {item.get("image_id") for item in manifest if isinstance(item.get("image_id"), str)}
    actual_relative = {
        path.relative_to(target_images).as_posix()
        for path in target_images.rglob("*")
        if path.is_file()
    } if target_images.is_dir() else set()
    expected_relative = {Path(image_id).as_posix() for image_id in expected_ids}
    checks.equal("target image file set", sorted(actual_relative), sorted(expected_relative))

    problems: list[str] = []
    verified = 0
    try:
        from PIL import Image
    except ImportError:
        checks.truth("Pillow is installed", False, "Install Pillow before validation")
        return
    checks.truth("Pillow is installed", True)

    for image_id in sorted(expected_ids):
        relative = Path(image_id)
        if relative.is_absolute() or ".." in relative.parts:
            problems.append(f"unsafe image_id: {image_id}")
            continue
        source = source_images / relative
        target = target_images / relative
        if not source.is_file() or not target.is_file():
            problems.append(f"missing source or target: {image_id}")
            continue
        if sha256(source) != sha256(target):
            problems.append(f"SHA-256 mismatch: {image_id}")
            continue
        try:
            with Image.open(target) as image:
                detected_format = image.format
                detected_size = image.size
                image.verify()
            with Image.open(target) as image:
                image.load()
            if detected_format != "PNG":
                problems.append(f"not PNG: {image_id} ({detected_format})")
                continue
            if detected_size != (512, 512):
                problems.append(f"wrong size: {image_id} ({detected_size})")
                continue
        except Exception as exc:
            problems.append(f"cannot decode {image_id}: {exc}")
            continue
        verified += 1
    checks.equal("verified image count", verified, len(expected_ids))
    checks.equal("image validation problems", problems, [])


def recompute_difficulty_distribution(manifest: list[dict[str, Any]]) -> dict[str, Any]:
    overall = Counter(item.get("difficulty_proxy") for item in manifest)
    by_section: dict[str, Counter[str]] = defaultdict(Counter)
    by_vqa_type: dict[str, Counter[str]] = defaultdict(Counter)
    for item in manifest:
        task = item.get("task")
        difficulty = item.get("difficulty_proxy")
        factors = item.get("difficulty_factors")
        if not isinstance(factors, dict):
            factors = {}
        if task == "vqa":
            section = "core_vqa" if item.get("subset") == "core" else "supplemental_vqa"
            by_vqa_type[factors.get("question_type")][difficulty] += 1
        else:
            section = task
        by_section[section][difficulty] += 1
    return {
        "total": len(manifest),
        "overall": dict(overall),
        "by_section": {key: dict(value) for key, value in by_section.items()},
        "by_vqa_type": {key: dict(value) for key, value in by_vqa_type.items()},
        "caption_length_group": dict(
            Counter(
                item.get("difficulty_factors", {}).get("length_group")
                for item in manifest
                if item.get("task") == "caption"
            )
        ),
        "referring_unique": dict(
            Counter(
                str(item.get("difficulty_factors", {}).get("unique")).lower()
                for item in manifest
                if item.get("task") == "referring"
            )
        ),
        "referring_area_group": dict(
            Counter(
                item.get("difficulty_factors", {}).get("area_group")
                for item in manifest
                if item.get("task") == "referring"
            )
        ),
    }


def validate_reports(
    checks: Checks,
    manifest: list[dict[str, Any]],
    difficulty_distribution: dict[str, Any],
    sampling_report: dict[str, Any],
) -> None:
    expected_difficulty = recompute_difficulty_distribution(manifest)
    checks.equal(
        "difficulty_distribution matches manifest",
        difficulty_distribution,
        expected_difficulty,
    )
    image_counts = Counter(item.get("image_id") for item in manifest)
    histogram = Counter(image_counts.values())
    expected_reuse = {
        "unique_image_count": len(image_counts),
        "reused_record_count": len(manifest) - len(image_counts),
        "reuse_ratio": (len(manifest) - len(image_counts)) / len(manifest),
        "usage_count_histogram": {str(k): histogram[k] for k in sorted(histogram)},
        "maximum_usage_count": max(image_counts.values(), default=0),
        "reused_images": {
            key: value for key, value in sorted(image_counts.items(), key=lambda x: str(x[0]))
            if value > 1
        },
    }
    for field, expected in expected_reuse.items():
        checks.equal(
            f"sampling_report image statistic: {field}",
            sampling_report.get(field),
            expected,
        )


def run_validation(
    source_root: Path, dataset_root: Path
) -> tuple[Checks, dict[str, Any]]:
    checks = Checks()
    source, output, manifest, difficulty_distribution, sampling_report = load_required_files(
        source_root, dataset_root
    )
    validate_counts(checks, output, manifest)
    validate_provenance(checks, source, output, manifest)
    validate_order(checks, output, manifest)
    validate_groups_and_quotas(checks, source, output, manifest)
    validate_images(checks, source_root, dataset_root, manifest)
    validate_reports(checks, manifest, difficulty_distribution, sampling_report)
    image_count = len({item.get("image_id") for item in manifest})
    summary = {
        "validation": "PASS" if checks.error_count == 0 else "FAIL",
        "checks": checks.items,
        "error_count": checks.error_count,
        "warning_count": 0,
        "total_record_count": len(manifest),
        "unique_image_count": image_count,
        "formal_records_match_source": next(
            (
                item["result"] == "PASS"
                for item in checks.items
                if item["name"] == "formal records and manifest exactly match source records"
            ),
            False,
        ),
        "images_sha256_and_format_verified": all(
            item["result"] == "PASS"
            for item in checks.items
            if item["name"] in {"verified image count", "image validation problems"}
        ),
    }
    return checks, summary


def failure_report(message: str) -> dict[str, Any]:
    return {
        "validation": "FAIL",
        "checks": [],
        "error_count": 1,
        "warning_count": 0,
        "fatal_error": message,
    }


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser()
    dataset_root = args.dataset_root.expanduser()
    report_path = dataset_root / "validation_report.json"
    try:
        _, report = run_validation(source_root, dataset_root)
    except Exception as exc:
        report = failure_report(f"{type(exc).__name__}: {exc}")
    if args.write_report:
        try:
            write_json(report_path, report)
        except Exception as exc:
            print(f"ERROR: cannot write validation report: {exc}", file=sys.stderr)
            return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("validation") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
