#!/usr/bin/env python3
"""Build a deterministic 400-pair LEVIR-CC validation subset."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
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


class BuildError(RuntimeError):
    """A clear build failure that should be shown to the user."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/user/下载/datasets/levir_cc/Levir-CC-dataset"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/user/silverdew/LEVIR_CC_ModelCompare_400"),
    )
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path("/home/user/silverdew/.LEVIR_CC_ModelCompare_400_staging"),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Select in memory and print a summary without creating directories.",
    )
    return parser.parse_args()


def load_source(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.is_file():
        raise BuildError(f"Missing source annotation: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise BuildError(f"Cannot read source annotation {path}: {exc}") from exc
    if not isinstance(data, dict) or list(data.keys()) != ["images"]:
        raise BuildError("Source JSON top level must contain only the field 'images'")
    images = data["images"]
    if not isinstance(images, list) or not all(isinstance(item, dict) for item in images):
        raise BuildError("Source field 'images' must be an array of objects")
    return data, images


def dump_json(path: Path, value: Any) -> None:
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


def require_exact_int(value: Any, allowed: set[int], label: str) -> int:
    if type(value) is not int or value not in allowed:
        raise BuildError(f"{label} must be one of {sorted(allowed)}, got {value!r}")
    return value


def validate_sentences(record: dict[str, Any], label: str) -> list[dict[str, Any]]:
    sentences = record.get("sentences")
    if not isinstance(sentences, list) or len(sentences) != 5:
        raise BuildError(f"{label} must contain exactly 5 sentences")
    for sentence_index, sentence in enumerate(sentences):
        if not isinstance(sentence, dict):
            raise BuildError(f"{label}.sentences[{sentence_index}] is not an object")
        if not isinstance(sentence.get("raw"), str):
            raise BuildError(f"{label}.sentences[{sentence_index}].raw is not text")
        if not isinstance(sentence.get("tokens"), list):
            raise BuildError(f"{label}.sentences[{sentence_index}].tokens is not an array")
    return sentences


def safe_filename(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise BuildError(f"{label}.filename must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or len(path.parts) != 1 or value in {".", ".."}:
        raise BuildError(f"{label}.filename is unsafe: {value!r}")
    return value


def build_candidates(images: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    for source_index, record in enumerate(images):
        split = record.get("split")
        split_counts[str(split)] += 1
        if split != "val":
            continue
        label = f"images[{source_index}]"
        changeflag = require_exact_int(record.get("changeflag"), {0, 1}, f"{label}.changeflag")
        filename = safe_filename(record.get("filename"), label)
        sentences = validate_sentences(record, label)
        candidate = {
            "source_index": source_index,
            "record": record,
            "filename": filename,
            "changeflag": changeflag,
            "caption_median_words": None,
            "sampling_group": "no_change",
        }
        if changeflag == 1:
            counts = [word_count(sentence["raw"]) for sentence in sentences]
            candidate["caption_median_words"] = statistics.median(counts)
        candidates.append(candidate)
    return candidates, dict(split_counts)


def assign_change_length_groups(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    changed = [candidate for candidate in candidates if candidate["changeflag"] == 1]
    ordered = sorted(
        changed,
        key=lambda candidate: (
            candidate["caption_median_words"],
            candidate["source_index"],
        ),
    )
    edge = math.floor(0.30 * len(ordered))
    for position, candidate in enumerate(ordered):
        if position < edge:
            candidate["sampling_group"] = "change_short"
        elif position >= len(ordered) - edge:
            candidate["sampling_group"] = "change_long"
        else:
            candidate["sampling_group"] = "change_medium"

    boundaries: dict[str, Any] = {
        "change_candidate_count": len(ordered),
        "edge_count": edge,
    }
    for group in ["change_short", "change_medium", "change_long"]:
        values = [
            candidate["caption_median_words"]
            for candidate in ordered
            if candidate["sampling_group"] == group
        ]
        boundaries[group] = {
            "count": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    return boundaries


def select_samples(
    images: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates, split_counts = build_candidates(images)
    boundaries = assign_change_length_groups(candidates)
    rng = random.Random(SEED)
    selected: list[dict[str, Any]] = []
    candidate_counts: dict[str, int] = {}

    for group in GROUP_ORDER:
        pool = sorted(
            [candidate for candidate in candidates if candidate["sampling_group"] == group],
            key=lambda candidate: candidate["source_index"],
        )
        candidate_counts[group] = len(pool)
        quota = GROUP_QUOTAS[group]
        if len(pool) < quota:
            raise BuildError(
                f"Sampling group {group} needs {quota} records but only {len(pool)} are available"
            )
        shuffled = list(pool)
        rng.shuffle(shuffled)
        selected.extend(shuffled[:quota])

    selected.sort(key=lambda candidate: candidate["source_index"])
    if len(selected) != 400:
        raise BuildError(f"Internal selection error: expected 400, got {len(selected)}")
    if len({candidate["source_index"] for candidate in selected}) != 400:
        raise BuildError("Internal selection error: duplicate source_index")

    report = {
        "seed": SEED,
        "source_split_counts": split_counts,
        "val_candidate_count": len(candidates),
        "val_change_candidate_count": sum(c["changeflag"] == 1 for c in candidates),
        "val_no_change_candidate_count": sum(c["changeflag"] == 0 for c in candidates),
        "sampling_group_candidate_counts": candidate_counts,
        "change_length_boundaries": boundaries,
    }
    return selected, report


def manifest_for(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source_index": candidate["source_index"],
            "imgid": candidate["record"].get("imgid"),
            "filename": candidate["filename"],
            "split": candidate["record"].get("split"),
            "changeflag": candidate["changeflag"],
            "sampling_group": candidate["sampling_group"],
            "caption_median_words": candidate["caption_median_words"],
            "image_a": f"images/val/A/{candidate['filename']}",
            "image_b": f"images/val/B/{candidate['filename']}",
        }
        for candidate in selected
    ]


def final_sampling_report(
    base_report: dict[str, Any],
    selected: list[dict[str, Any]],
) -> dict[str, Any]:
    group_counts = Counter(candidate["sampling_group"] for candidate in selected)
    change_counts = Counter(candidate["changeflag"] for candidate in selected)
    report = dict(base_report)
    report.update(
        {
            "selected_record_count": len(selected),
            "selected_change_count": change_counts[1],
            "selected_no_change_count": change_counts[0],
            "selected_group_counts": {
                group: group_counts[group] for group in GROUP_ORDER
            },
            "records_with_exactly_five_sentences": sum(
                len(candidate["record"]["sentences"]) == 5 for candidate in selected
            ),
            "reference_sentence_count": sum(
                len(candidate["record"]["sentences"]) for candidate in selected
            ),
            "image_a_count": len(selected),
            "image_b_count": len(selected),
            "filtered_records": 0,
            "failure_reasons": {},
        }
    )
    return report


def ensure_safe_paths(source_root: Path, output_root: Path, staging_root: Path) -> None:
    if output_root.exists():
        raise BuildError(f"Final output already exists; refusing to overwrite: {output_root}")
    if staging_root.exists():
        raise BuildError(f"Staging output already exists; refusing to overwrite: {staging_root}")
    if output_root.parent.resolve() != staging_root.parent.resolve():
        raise BuildError("Final and staging directories must have the same parent")
    source_resolved = source_root.resolve()
    for target in [output_root.resolve(strict=False), staging_root.resolve(strict=False)]:
        if target == source_resolved or source_resolved in target.parents:
            raise BuildError(f"Output must not be inside the source directory: {target}")


def check_selected_images(source_root: Path, selected: list[dict[str, Any]]) -> None:
    for candidate in selected:
        filename = candidate["filename"]
        for phase in ["A", "B"]:
            path = source_root / "images" / "val" / phase / filename
            if not path.is_file():
                raise BuildError(f"Missing selected {phase} image: {path}")


def copy_selected_images(
    source_root: Path,
    staging_root: Path,
    selected: list[dict[str, Any]],
) -> None:
    for phase in ["A", "B"]:
        source_dir = source_root / "images" / "val" / phase
        target_dir = staging_root / "images" / "val" / phase
        target_dir.mkdir(parents=True, exist_ok=False)
        for candidate in selected:
            filename = candidate["filename"]
            shutil.copy2(source_dir / filename, target_dir / filename)


def generated_readme() -> str:
    return """# LEVIR_CC_ModelCompare_400

Fixed LEVIR-CC validation subset for early model selection.

- 400 bi-temporal image pairs
- 200 change pairs
- 200 no-change pairs
- 5 original reference sentences per pair
- seed: 42

The formal `LevirCCcaptions.json` contains unmodified source records. Derived
sampling information appears only in auxiliary files. Image A is the pre-phase
image and image B is the post-phase image.

Run independent validation:

```bash
python scripts/validate_levir_cc.py \\
  --source-root "/home/user/下载/datasets/levir_cc/Levir-CC-dataset" \\
  --dataset-root "/home/user/silverdew/LEVIR_CC_ModelCompare_400" \\
  --write-report
```
"""


def write_staging(
    source_root: Path,
    staging_root: Path,
    selected: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
    sampling_report: dict[str, Any],
    validator_script: Path,
) -> None:
    staging_root.mkdir(parents=False, exist_ok=False)
    try:
        formal_records = [copy.deepcopy(candidate["record"]) for candidate in selected]
        for candidate, copied in zip(selected, formal_records):
            if not ordered_deep_equal(candidate["record"], copied):
                raise BuildError(
                    f"Deep copy changed source record at index {candidate['source_index']}"
                )
        dump_json(staging_root / SOURCE_JSON, {"images": formal_records})
        dump_json(staging_root / "selection_manifest.json", manifest)
        dump_json(staging_root / "sampling_report.json", sampling_report)
        with (staging_root / "README.md").open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(generated_readme())

        scripts_dir = staging_root / "scripts"
        scripts_dir.mkdir()
        shutil.copy2(Path(__file__).resolve(), scripts_dir / "sample_levir_cc.py")
        shutil.copy2(validator_script, scripts_dir / "validate_levir_cc.py")
        copy_selected_images(source_root, staging_root, selected)

        completed = subprocess.run(
            [
                sys.executable,
                str(scripts_dir / "validate_levir_cc.py"),
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
                f"Independent validation failed; staging retained at {staging_root}"
            )
        with (staging_root / "validation_report.json").open("r", encoding="utf-8") as handle:
            validation_report = json.load(handle)
        if validation_report.get("validation") != "PASS":
            raise BuildError(
                f"Validator did not produce PASS; staging retained at {staging_root}"
            )
    except Exception:
        # Staging is intentionally retained for audit after a failure.
        raise


def main() -> int:
    args = parse_args()
    source_root = args.source_root.expanduser()
    output_root = args.output_root.expanduser()
    staging_root = args.staging_root.expanduser()
    try:
        ensure_safe_paths(source_root, output_root, staging_root)
        _, source_images = load_source(source_root / SOURCE_JSON)
        selected, base_report = select_samples(source_images)
        check_selected_images(source_root, selected)
        manifest = manifest_for(selected)
        sampling_report = final_sampling_report(base_report, selected)
        preflight = {
            "status": "FEASIBLE",
            "selected_record_count": len(selected),
            "change_count": sampling_report["selected_change_count"],
            "no_change_count": sampling_report["selected_no_change_count"],
            "group_counts": sampling_report["selected_group_counts"],
            "reference_sentence_count": sampling_report["reference_sentence_count"],
            "image_file_count": 2 * len(selected),
        }
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        if args.preflight_only:
            return 0

        validator_script = Path(__file__).resolve().with_name("validate_levir_cc.py")
        if not validator_script.is_file():
            raise BuildError(f"Missing validator script: {validator_script}")
        write_staging(
            source_root,
            staging_root,
            selected,
            manifest,
            sampling_report,
            validator_script,
        )
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
