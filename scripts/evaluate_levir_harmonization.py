"""Offline LEVIR-CC harmonization evaluation and calibration.
离线 LEVIR-CC 一致化评测与校准。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from statistics import mean, median, pstdev

import cv2
import numpy as np
from PIL import Image

from spacers_agent.agents.change.harmonizer import PairHarmonizer
from spacers_agent.settings import ChangeHarmonizationSettings
from spacers_agent.workflows.artifact_writer import atomic_write_json


def build_parser() -> argparse.ArgumentParser:
    """Build the reproducible offline evaluator CLI. / 构建可复现离线评测 CLI。"""

    parser = argparse.ArgumentParser(description="Evaluate auditable LEVIR-CC temporal harmonization.")
    parser.add_argument("--dataset", choices=["LEVIR-CC"], default="LEVIR-CC")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "val", "test"], required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-file", type=Path)
    parser.add_argument("--write-calibration", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    split = "val" if args.split == "validation" else args.split
    pair_root = args.root / "images" / split
    first_dir, second_dir = pair_root / "A", pair_root / "B"
    names = sorted({path.name for path in first_dir.glob("*.png")} & {path.name for path in second_dir.glob("*.png")})
    names = names[args.offset : args.offset + args.max_pairs]
    if not names:
        raise SystemExit(f"No paired PNG images found under {pair_root}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = ChangeHarmonizationSettings(calibration_file=args.calibration_file)
    harmonizer = PairHarmonizer(settings)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    mask_dir = args.root / "labels" / split
    for name in names:
        try:
            raw1, raw2 = _read_rgb(first_dir / name), _read_rgb(second_dir / name)
            result = harmonizer.run(raw1, raw2)
            metrics = result.decision.metrics
            row: dict[str, object] = {
                "pair": Path(name).stem,
                "status": result.decision.status,
                "reason_codes": "|".join(result.decision.reason_codes),
                "raw_fallback_used": not result.decision.used_for_proposal,
                "sharpness_adjustment_used": bool(result.transform_summary.get("sharpness_adjustment_used", False)),
                "has_change": None,
            }
            if metrics is not None:
                row.update(metrics.model_dump())
                transforms = result.transform_summary.get("lab_transforms", [])
                gains = [abs(float(item[key])) for item in transforms for key in ("t1_gain", "t2_gain")]
                offsets = [abs(float(item[key])) for item in transforms for key in ("t1_offset", "t2_offset")]
                row["transform_gain_abs_max"] = max(gains, default=0.0)
                row["transform_offset_abs_max"] = max(offsets, default=0.0)
                row["sharpness_ratio_before"] = result.transform_summary.get("sharpness_ratio_before")
                row["sharpness_ratio_after"] = result.transform_summary.get("sharpness_ratio_after")
            label_path = mask_dir / name
            if label_path.is_file() and metrics is not None:
                change_mask = np.asarray(Image.open(label_path).convert("L")) > 0
                row["has_change"] = bool(change_mask.any())
                row.update(_masked_change_metrics(raw1, raw2, result.t1, result.t2, change_mask))
            rows.append(row)
        except Exception as error:  # Keep every failed pair visible. / 保持每个失败图对可见。
            failures.append({"pair": name, "error_type": type(error).__name__, "message": str(error)})

    _write_csv(args.output_dir / "metrics.csv", rows)
    summary = {
        "dataset": "LEVIR-CC", "root": str(args.root), "split": split,
        "offset": args.offset, "requested_pairs": args.max_pairs,
        "sample_ids": [str(row["pair"]) for row in rows],
        "processed": len(rows), "failed": len(failures),
        "algorithm_version": settings.version, "code_version": _git_revision(),
        "status_counts": _counts(str(row["status"]) for row in rows),
        "metrics": _summaries(rows),
    }
    atomic_write_json(args.output_dir / "summary.json", summary)
    atomic_write_json(args.output_dir / "grouped_summary.json", {
        "has_change": _summaries([row for row in rows if row.get("has_change") is True]),
        "no_change": _summaries([row for row in rows if row.get("has_change") is False]),
        "labels_available": any(row.get("has_change") is not None for row in rows),
    })
    atomic_write_json(args.output_dir / "failed_pairs.json", failures)
    if args.write_calibration:
        calibration = _calibration(rows, settings.version, split)
        atomic_write_json(args.output_dir / "calibration.json", calibration)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _masked_change_metrics(raw1: np.ndarray, raw2: np.ndarray, out1: np.ndarray, out2: np.ndarray, change: np.ndarray) -> dict[str, float | None]:
    before = np.abs(cv2.cvtColor(raw1, cv2.COLOR_RGB2GRAY).astype(np.float32) - cv2.cvtColor(raw2, cv2.COLOR_RGB2GRAY).astype(np.float32))
    after = np.abs(cv2.cvtColor(out1, cv2.COLOR_RGB2GRAY).astype(np.float32) - cv2.cvtColor(out2, cv2.COLOR_RGB2GRAY).astype(np.float32))
    unchanged = ~change
    changed_before = float(before[change].mean()) if change.any() else None
    changed_after = float(after[change].mean()) if change.any() else None
    unchanged_before = float(before[unchanged].mean()) if unchanged.any() else None
    unchanged_after = float(after[unchanged].mean()) if unchanged.any() else None
    return {
        "changed_mad_before": changed_before, "changed_mad_after": changed_after,
        "unchanged_mad_before": unchanged_before, "unchanged_mad_after": unchanged_after,
        "background_suppression": _one_minus_ratio(unchanged_after, unchanged_before),
        "change_retention": _ratio(changed_after, changed_before),
    }


def _summaries(rows: list[dict[str, object]]) -> dict[str, dict[str, float | int | list[float]]]:
    result: dict[str, dict[str, float | int | list[float]]] = {}
    keys = (
        "pif_ratio", "mad_full_before", "mad_full_after", "mad_pif_before", "mad_pif_after",
        "corr_full_before", "corr_full_after", "pct_diff_gt20_before", "pct_diff_gt20_after",
        "background_suppression", "change_retention",
    )
    for key in keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            result[key] = {"n": len(values), "mean": mean(values), "median": median(values), "std": pstdev(values), "bootstrap_95_ci": _bootstrap_ci(values)}
    return result


def _bootstrap_ci(values: list[float]) -> list[float]:
    generator = np.random.default_rng(20260805)
    array = np.asarray(values, dtype=np.float64)
    samples = generator.choice(array, size=(1000, array.size), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _calibration(rows: list[dict[str, object]], version: str, split: str) -> dict[str, object]:
    def quantiles(key: str) -> dict[str, float]:
        values = np.asarray([float(row[key]) for row in rows if isinstance(row.get(key), (int, float))])
        return {label: float(np.quantile(values, q)) for label, q in (("p01", .01), ("p05", .05), ("p50", .5), ("p95", .95), ("p99", .99))} if values.size else {}
    improvements = []
    for row in rows:
        before, after = row.get("mad_pif_before"), row.get("mad_pif_after")
        if isinstance(before, (int, float)) and before > 0 and isinstance(after, (int, float)):
            improvements.append(1.0 - float(after) / float(before))
    return {
        "algorithm_version": version, "code_version": _git_revision(), "split": split,
        "sample_count": len(rows), "pif_ratio": quantiles("pif_ratio"),
        "pif_mad_improvement": _quantile_values(improvements),
        "transform_gain_abs_max": quantiles("transform_gain_abs_max"),
        "transform_offset_abs_max": quantiles("transform_offset_abs_max"),
        "sharpness_ratio_before": quantiles("sharpness_ratio_before"),
        "sharpness_ratio_after": quantiles("sharpness_ratio_after"),
    }


def _quantile_values(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {label: float(np.quantile(array, q)) for label, q in (("p01", .01), ("p05", .05), ("p50", .5), ("p95", .95), ("p99", .99))} if array.size else {}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _counts(values: object) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    return numerator / denominator if numerator is not None and denominator not in (None, 0.0) else None


def _one_minus_ratio(numerator: float | None, denominator: float | None) -> float | None:
    ratio = _ratio(numerator, denominator)
    return 1.0 - ratio if ratio is not None else None


def _git_revision() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
