"""Deterministic history rebuild from immutable run packages."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from m3rs_eval.contracts import MetricRecord, RunManifest, read_jsonl


@dataclass
class HistoryIndex:
    """Immutable snapshot of all run history rebuilt from run packages."""

    ranked_run_ids: list[str]
    file_hashes: dict[str, str]
    runs_root: Path
    history_root: Path
    invalid_runs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranked_run_ids": self.ranked_run_ids,
            "file_hashes": self.file_hashes,
            "runs_root": str(self.runs_root),
            "history_root": str(self.history_root),
            "invalid_runs": self.invalid_runs,
        }


_RUNS_CSV_COLUMNS = [
    "run_id",
    "status",
    "mode",
    "protocol_id",
    "created_at",
    "config_hash",
    "request_manifest_hash",
    "eligible_for_history",
    "protocol_hash",
    "command_hash",
    "system_version_hash",
    "system_version",
    "limit",
    "doctor_passed",
    "environment_platform",
    "environment_cpu",
    "environment_gpus",
    "dataset_manifest_hashes",
    "xlrs_protocol",
]

_METRICS_LONG_COLUMNS = [
    "run_id",
    "metric_id",
    "availability",
    "provenance",
    "value_canonical",
    "n_samples",
    "n_failures",
    "ci95_low",
    "ci95_high",
    "dataset",
    "task",
    "slice",
    "language",
    "protocol_id",
    "benchmark_version",
    "recorded_at",
    "source_log_path",
    "notes",
    "baseline_run_id",
]

_COVERAGE_COLUMNS = [
    "run_id",
    "dataset",
    "status",
    "expected",
    "requested",
    "predicted",
    "failed",
    "coverage",
]


def rebuild_history(runs_root: Path, history_root: Path) -> HistoryIndex:
    """Rebuild the canonical history index from every run package under *runs_root*.

    Only run packages with a valid *run_manifest.json* are accepted.  Runs whose
    manifest cannot be read are recorded in ``invalid_runs`` and skipped.

    History is written atomically to *history_root*:

    * ``runs.csv``      – one row per run, stable column order
    * ``metrics_long.csv`` – one row per run × metric_id
    * ``coverage.csv``  – per-dataset, per-task coverage status
    * ``history_manifest.json`` – generation timestamp, file hashes, rankings

    CSV files are UTF-8 with BOM for Excel compatibility.
    The rebuild is deterministic and idempotent for the same input.
    """
    runs_root = Path(runs_root)
    history_root = Path(history_root)
    history_root.mkdir(parents=True, exist_ok=True)

    # Scan every subdirectory of runs_root for run packages
    run_dirs = sorted(
        [p for p in runs_root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )
    runs: list[tuple[RunManifest, Path]] = []
    invalid_runs: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        run_id = run_dir.name
        manifest_file = run_dir / "run_manifest.json"
        if not manifest_file.is_file():
            invalid_runs.append({
                "run_id": run_id,
                "reason": "run_manifest.json missing",
            })
            continue
        try:
            manifest_raw = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            invalid_runs.append({
                "run_id": run_id,
                "reason": f"run_manifest.json parse error: {exc}",
            })
            continue
        if not isinstance(manifest_raw, dict):
            invalid_runs.append({
                "run_id": run_id,
                "reason": "run_manifest.json is not a JSON object",
            })
            continue
        try:
            manifest = RunManifest.from_dict(manifest_raw)
        except Exception as exc:
            invalid_runs.append({
                "run_id": run_id,
                "reason": f"invalid manifest: {exc}",
            })
            continue
        runs.append((manifest, run_dir))

    # Stable sort: created_at ascending, then run_id ascending (tiebreaker)
    runs.sort(key=lambda item: (item[0].created_at, item[0].run_id))

    # Build CSV content in memory
    runs_csv_rows = _build_runs_csv(runs)
    metrics_csv_rows = _build_metrics_csv(runs)
    coverage_csv_rows = _build_coverage_csv(runs)

    # Compute ranked run ids
    ranked_run_ids = [
        manifest.run_id
        for manifest, _ in runs
        if manifest.mode == "full"
        and manifest.status == "complete"
        and manifest.eligible_for_history
    ]

    # Write CSV files atomically
    file_hashes: dict[str, str] = {}

    file_hashes["runs.csv"] = _write_csv_atomic(
        history_root, "runs.csv", _RUNS_CSV_COLUMNS, runs_csv_rows
    )
    file_hashes["metrics_long.csv"] = _write_csv_atomic(
        history_root, "metrics_long.csv", _METRICS_LONG_COLUMNS, metrics_csv_rows
    )
    file_hashes["coverage.csv"] = _write_csv_atomic(
        history_root, "coverage.csv", _COVERAGE_COLUMNS, coverage_csv_rows
    )

    # Write history manifest
    manifest_payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "ranked_run_ids": ranked_run_ids,
        "file_hashes": file_hashes,
        "invalid_runs": invalid_runs,
    }
    manifest_path = history_root / "history_manifest.json"
    temporary = history_root / f".history_manifest.json.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest_payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, manifest_path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

    return HistoryIndex(
        ranked_run_ids=ranked_run_ids,
        file_hashes=file_hashes,
        runs_root=runs_root,
        history_root=history_root,
        invalid_runs=invalid_runs,
    )


def _build_runs_csv(runs: list[tuple[RunManifest, Path]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for manifest, _ in runs:
        metadata = manifest.metadata
        environment = metadata.get("environment", {}) if isinstance(metadata, dict) else {}
        gpus = environment.get("gpus", [])
        gpu_list = ", ".join(gpus) if isinstance(gpus, list) else str(gpus)
        dataset_hashes = metadata.get("dataset_manifest_hashes", {})
        dataset_hashes_str = json.dumps(dataset_hashes, ensure_ascii=False, sort_keys=True)
        doctor = metadata.get("doctor", {}) if isinstance(metadata, dict) else {}
        doctor_passed = str(doctor.get("passed", "")) if isinstance(doctor, dict) else ""
        limit_val = metadata.get("limit", "") if isinstance(metadata, dict) else ""
        rows.append({
            "run_id": manifest.run_id,
            "status": manifest.status,
            "mode": manifest.mode,
            "protocol_id": manifest.protocol_id,
            "created_at": manifest.created_at,
            "config_hash": manifest.config_hash,
            "request_manifest_hash": manifest.request_manifest_hash,
            "eligible_for_history": str(manifest.eligible_for_history),
            "protocol_hash": manifest.protocol_hash,
            "command_hash": manifest.command_hash,
            "system_version_hash": manifest.system_version_hash,
            "system_version": str(metadata.get("system_version", "")) if isinstance(metadata, dict) else "",
            "limit": str(limit_val),
            "doctor_passed": str(doctor_passed),
            "environment_platform": str(environment.get("platform", "")) if isinstance(environment, dict) else "",
            "environment_cpu": str(environment.get("cpu", "")) if isinstance(environment, dict) else "",
            "environment_gpus": gpu_list,
            "dataset_manifest_hashes": dataset_hashes_str,
            "xlrs_protocol": str(metadata.get("xlrs_protocol", "")) if isinstance(metadata, dict) else "",
        })
    return rows


def _build_metrics_csv(runs: list[tuple[RunManifest, Path]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for manifest, run_dir in runs:
        metrics_path = run_dir / "metrics.jsonl"
        if not metrics_path.is_file():
            continue
        try:
            records = read_jsonl(metrics_path, MetricRecord)
        except Exception:
            continue
        for record in records:
            rows.append({
                "run_id": record.run_id,
                "metric_id": record.metric_id,
                "availability": record.availability,
                "provenance": record.provenance,
                "value_canonical": str(record.value_canonical) if record.value_canonical is not None else "",
                "n_samples": str(record.n_samples),
                "n_failures": str(record.n_failures),
                "ci95_low": str(record.ci95_low) if record.ci95_low is not None else "",
                "ci95_high": str(record.ci95_high) if record.ci95_high is not None else "",
                "dataset": record.dataset or "",
                "task": record.task or "",
                "slice": record.slice or "",
                "language": record.language or "",
                "protocol_id": record.protocol_id,
                "benchmark_version": record.benchmark_version,
                "recorded_at": record.recorded_at,
                "source_log_path": record.source_log_path or "",
                "notes": record.notes or "",
                "baseline_run_id": record.baseline_run_id or "",
            })
    return rows


def _build_coverage_csv(runs: list[tuple[RunManifest, Path]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for manifest, run_dir in runs:
        coverage_path = run_dir / "coverage.json"
        if not coverage_path.is_file():
            continue
        try:
            coverage_data = json.loads(coverage_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        datasets = coverage_data.get("datasets", {})
        if isinstance(datasets, dict):
            for dataset_name, ds_info in datasets.items():
                if isinstance(ds_info, dict):
                    rows.append({
                        "run_id": manifest.run_id,
                        "dataset": dataset_name,
                        "status": str(ds_info.get("status", "")),
                        "expected": str(ds_info.get("expected", "")),
                        "requested": str(ds_info.get("requested", "")),
                        "predicted": str(ds_info.get("predicted", "")),
                        "failed": str(ds_info.get("failed", "")),
                        "coverage": str(ds_info.get("coverage", "")),
                    })
    return rows


def _write_csv_atomic(
    history_root: Path, filename: str, columns: list[str], rows: list[dict[str, str]]
) -> str:
    """Write CSV with UTF-8 BOM atomically; return SHA-256 hash."""
    import io

    # Build CSV content in memory
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

    # Prepend BOM
    content_bytes = ("﻿" + buf.getvalue()).encode("utf-8")

    # Compute hash
    file_hash = hashlib.sha256(content_bytes).hexdigest()

    target = history_root / filename
    temporary = history_root / f".{filename}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("wb") as handle:
            handle.write(content_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass

    return file_hash
