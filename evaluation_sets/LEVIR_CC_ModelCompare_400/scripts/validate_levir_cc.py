#!/usr/bin/env python3
"""Independently validate a LEVIR_CC_ModelCompare_400 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SEED = 42
SOURCE_JSON = "LevirCCcaptions.json"
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")
GROUP_ORDER = ["change_short", "change_medium", "change_long", "no_change"]
GROUP_QUOTAS = {
    "change_short": 60,
    "change_medium": 80,
    "change_long": 60,
    "no_change": 200,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/user/下载/datasets/levir_cc/Levir-CC-dataset"),
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write validation_report.json in the dataset directory.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def ordered_deep_equal(left: Any, right: Any) -> bool:
    """Compare JSON values while treating object key order as significant."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        if list(left.keys()) != list(right.keys()):
            return False
        return all(ordered_deep_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            ordered_deep_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def validate_source_structure(data: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(data, dict) or list(data.keys()) != ["images"]:
        raise ValueError(f"{label} top level must contain only 'images'")
    images = data["images"]
    if not isinstance(images, list) or not all(isinstance(item, dict) for item in images):
        raise ValueError(f"{label}.images must be an array of objects")
    return images


def safe_filename(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute() or len(path.parts) != 1 or value in {".", ".."}:
        return None
    return value


def recompute_selection(source_images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for source_index, record in enumerate(source_images):
        if record.get("split") != "val":
            continue
        flag = record.get("changeflag")
        if type(flag) is not int or flag not in {0, 1}:
            raise ValueError(f"source images[{source_index}].changeflag is invalid")
        filename = safe_filename(record.get("filename"))
        if filename is None:
            raise ValueError(f"source images[{source_index}].filename is invalid")
        sentences = record.get("sentences")
        if not isinstance(sentences, list) or len(sentences) != 5:
            raise ValueError(f"source images[{source_index}] does not have 5 sentences")
        candidate = {
            "source_index": source_index,
            "record": record,
            "filename": filename,
            "changeflag": flag,
            "caption_median_words": None,
            "sampling_group": "no_change",
        }
        if flag == 1:
            raw_values: list[str] = []
            for sentence in sentences:
                if not isinstance(sentence, dict) or not isinstance(sentence.get("raw"), str):
                    raise ValueError(f"source images[{source_index}] has an invalid sentence")
                raw_values.append(sentence["raw"])
            candidate["caption_median_words"] = statistics.median(
                word_count(text) for text in raw_values
            )
        candidates.append(candidate)

    changed = sorted(
        [candidate for candidate in candidates if candidate["changeflag"] == 1],
        key=lambda candidate: (
            candidate["caption_median_words"],
            candidate["source_index"],
        ),
    )
    edge = math.floor(0.30 * len(changed))
    for position, candidate in enumerate(changed):
        if position < edge:
            candidate["sampling_group"] = "change_short"
        elif position >= len(changed) - edge:
            candidate["sampling_group"] = "change_long"
        else:
            candidate["sampling_group"] = "change_medium"

    rng = random.Random(SEED)
    selected: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        pool = sorted(
            [candidate for candidate in candidates if candidate["sampling_group"] == group],
            key=lambda candidate: candidate["source_index"],
        )
        quota = GROUP_QUOTAS[group]
        if len(pool) < quota:
            raise ValueError(f"source has too few candidates for {group}")
        shuffled = list(pool)
        rng.shuffle(shuffled)
        selected.extend(shuffled[:quota])
    selected.sort(key=lambda candidate: candidate["source_index"])
    return selected


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


def validate_formal_annotations(
    checks: Checks,
    source_images: list[dict[str, Any]],
    output_images: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    expected_selection: list[dict[str, Any]],
) -> None:
    checks.equal("formal record count", len(output_images), 400)
    checks.equal("manifest record count", len(manifest), 400)
    checks.equal(
        "all formal records use val split",
        sum(record.get("split") == "val" for record in output_images),
        400,
    )
    flags = Counter(record.get("changeflag") for record in output_images)
    checks.equal("change record count", flags[1], 200)
    checks.equal("no-change record count", flags[0], 200)
    checks.equal("records with other changeflag values", len(output_images) - flags[0] - flags[1], 0)

    sentence_lengths = [
        len(record.get("sentences", []))
        if isinstance(record.get("sentences"), list)
        else -1
        for record in output_images
    ]
    checks.equal("records with exactly five sentences", sentence_lengths.count(5), 400)
    checks.equal("reference sentence count", sum(x for x in sentence_lengths if x >= 0), 2000)

    expected_indices = [candidate["source_index"] for candidate in expected_selection]
    actual_indices = [item.get("source_index") for item in manifest]
    checks.equal("seed=42 selected source_index sequence", actual_indices, expected_indices)
    checks.truth(
        "formal records are ordered by source_index",
        actual_indices == sorted(actual_indices),
    )

    problems: list[str] = []
    if len(output_images) != len(manifest):
        problems.append("formal and manifest lengths differ")
    else:
        for position, (record, item) in enumerate(zip(output_images, manifest)):
            source_index = item.get("source_index")
            if type(source_index) is not int or not 0 <= source_index < len(source_images):
                problems.append(f"manifest[{position}] has invalid source_index")
                continue
            source_record = source_images[source_index]
            if not ordered_deep_equal(record, source_record):
                problems.append(
                    f"formal images[{position}] is not an ordered exact copy of source[{source_index}]"
                )
            expected_manifest = {
                "source_index": source_index,
                "imgid": record.get("imgid"),
                "filename": record.get("filename"),
                "split": record.get("split"),
                "changeflag": record.get("changeflag"),
                "sampling_group": expected_selection[position]["sampling_group"]
                if position < len(expected_selection)
                else None,
                "caption_median_words": expected_selection[position]["caption_median_words"]
                if position < len(expected_selection)
                else None,
                "image_a": f"images/val/A/{record.get('filename')}",
                "image_b": f"images/val/B/{record.get('filename')}",
            }
            if not ordered_deep_equal(item, expected_manifest):
                problems.append(f"manifest[{position}] does not match the formal record")
    checks.equal("formal records and manifest strictly match source", problems, [])

    groups = Counter(item.get("sampling_group") for item in manifest)
    checks.equal(
        "sampling group counts",
        {group: groups[group] for group in GROUP_ORDER},
        GROUP_QUOTAS,
    )


def validate_images(
    checks: Checks,
    source_root: Path,
    dataset_root: Path,
    output_images: list[dict[str, Any]],
) -> None:
    expected_filenames = {
        record.get("filename")
        for record in output_images
        if isinstance(record.get("filename"), str)
    }
    format_problems: list[str] = []
    hash_problems: list[str] = []
    try:
        from PIL import Image
    except ImportError:
        checks.truth("Pillow is installed", False)
        return
    checks.truth("Pillow is installed", True)

    for phase in ["A", "B"]:
        source_dir = source_root / "images" / "val" / phase
        target_dir = dataset_root / "images" / "val" / phase
        actual_files = (
            {path.name for path in target_dir.iterdir() if path.is_file()}
            if target_dir.is_dir()
            else set()
        )
        checks.equal(
            f"{phase} image file set",
            sorted(actual_files),
            sorted(expected_filenames),
        )
        for filename in sorted(expected_filenames):
            source = source_dir / filename
            target = target_dir / filename
            if not source.is_file() or not target.is_file():
                hash_problems.append(f"{phase}/{filename}: source or target missing")
                continue
            if sha256(source) != sha256(target):
                hash_problems.append(f"{phase}/{filename}: SHA-256 mismatch")
                continue
            try:
                with Image.open(target) as image:
                    detected_format = image.format
                    detected_size = image.size
                    detected_mode = image.mode
                    image.verify()
                with Image.open(target) as image:
                    image.load()
                if detected_format != "PNG":
                    format_problems.append(f"{phase}/{filename}: format={detected_format}")
                if detected_size != (256, 256):
                    format_problems.append(f"{phase}/{filename}: size={detected_size}")
                if detected_mode != "RGB":
                    format_problems.append(f"{phase}/{filename}: mode={detected_mode}")
            except Exception as exc:
                format_problems.append(f"{phase}/{filename}: decode error: {exc}")

    checks.equal("image SHA-256 problems", hash_problems, [])
    checks.equal("image format/size/mode problems", format_problems, [])
    checks.equal("total expected image files", 2 * len(expected_filenames), 800)


def validate_sampling_report(
    checks: Checks,
    report: dict[str, Any],
    manifest: list[dict[str, Any]],
) -> None:
    groups = Counter(item.get("sampling_group") for item in manifest)
    flags = Counter(item.get("changeflag") for item in manifest)
    expected_fields = {
        "selected_record_count": 400,
        "selected_change_count": 200,
        "selected_no_change_count": 200,
        "selected_group_counts": {
            group: groups[group] for group in GROUP_ORDER
        },
        "records_with_exactly_five_sentences": 400,
        "reference_sentence_count": 2000,
        "image_a_count": 400,
        "image_b_count": 400,
        "seed": SEED,
    }
    for field, expected in expected_fields.items():
        checks.equal(
            f"sampling_report field: {field}",
            report.get(field),
            expected,
        )
    checks.equal("manifest change count", flags[1], 200)
    checks.equal("manifest no-change count", flags[0], 200)


def run_validation(source_root: Path, dataset_root: Path) -> dict[str, Any]:
    checks = Checks()
    source_data = read_json(source_root / SOURCE_JSON)
    output_data = read_json(dataset_root / SOURCE_JSON)
    source_images = validate_source_structure(source_data, "source")
    output_images = validate_source_structure(output_data, "formal output")

    manifest = read_json(dataset_root / "selection_manifest.json")
    if not isinstance(manifest, list) or not all(isinstance(item, dict) for item in manifest):
        raise ValueError("selection_manifest.json must be an array of objects")
    sampling_report = read_json(dataset_root / "sampling_report.json")
    if not isinstance(sampling_report, dict):
        raise ValueError("sampling_report.json must be an object")

    expected_selection = recompute_selection(source_images)
    validate_formal_annotations(
        checks,
        source_images,
        output_images,
        manifest,
        expected_selection,
    )
    validate_images(checks, source_root, dataset_root, output_images)
    validate_sampling_report(checks, sampling_report, manifest)

    return {
        "validation": "PASS" if checks.error_count == 0 else "FAIL",
        "checks": checks.items,
        "error_count": checks.error_count,
        "warning_count": 0,
        "formal_record_count": len(output_images),
        "reference_sentence_count": sum(
            len(record.get("sentences", []))
            for record in output_images
            if isinstance(record.get("sentences"), list)
        ),
        "image_a_count": len(output_images),
        "image_b_count": len(output_images),
        "formal_records_match_source": next(
            (
                item["result"] == "PASS"
                for item in checks.items
                if item["name"] == "formal records and manifest strictly match source"
            ),
            False,
        ),
        "images_sha256_and_format_verified": all(
            item["result"] == "PASS"
            for item in checks.items
            if item["name"]
            in {
                "image SHA-256 problems",
                "image format/size/mode problems",
                "A image file set",
                "B image file set",
            }
        ),
    }


def failure_report(exc: Exception) -> dict[str, Any]:
    return {
        "validation": "FAIL",
        "checks": [],
        "error_count": 1,
        "warning_count": 0,
        "fatal_error": f"{type(exc).__name__}: {exc}",
    }


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser()
    dataset_root = args.dataset_root.expanduser()
    try:
        report = run_validation(source_root, dataset_root)
    except Exception as exc:
        report = failure_report(exc)
    if args.write_report:
        try:
            write_json(dataset_root / "validation_report.json", report)
        except Exception as exc:
            print(f"ERROR: cannot write validation_report.json: {exc}", file=sys.stderr)
            return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("validation") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
