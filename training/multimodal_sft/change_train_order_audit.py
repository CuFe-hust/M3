"""Exact content and source-mixing audit for formal ChangeAgent SFT corpora."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

from training.multimodal_sft.change_corpus import (
    FORMAL_TRAIN_ORDERING_POLICY,
    FORMAL_TRAIN_ORDERING_SEED,
    VALIDATION_ORDERING_POLICY,
    formal_train_order_key,
)


class ChangeTrainOrderAuditError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ChangeTrainOrderAuditError(f"CORPUS_FILE_MISSING: {path}")
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ChangeTrainOrderAuditError(
                    f"CORPUS_JSON_INVALID: {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ChangeTrainOrderAuditError(
                    f"CORPUS_ROW_INVALID: {path}:{line_number}"
                )
            result.append(row)
    return result


def _by_id(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        episode_id = str(row.get("episode_id") or "")
        if not episode_id or episode_id in result:
            raise ChangeTrainOrderAuditError(
                f"DUPLICATE_OR_MISSING_EPISODE_ID: {episode_id}"
            )
        result[episode_id] = row
    return result


def _source(row: dict[str, Any]) -> str:
    provenance = row.get("provenance")
    return str(provenance.get("source_id") or "") if isinstance(provenance, dict) else ""


def _maximum_run(rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], str]) -> int:
    maximum = current = 0
    previous: str | None = None
    for row in rows:
        value = key(row)
        current = current + 1 if value == previous else 1
        maximum = max(maximum, current)
        previous = value
    return maximum


def _window_source_distribution(rows: list[dict[str, Any]], start: int, size: int) -> dict[str, int]:
    return dict(sorted(Counter(_source(row) for row in rows[start : start + size]).items()))


def _order_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    window = min(1000, len(rows))
    middle_start = max(0, (len(rows) - window) // 2)
    last_start = max(0, len(rows) - window)
    return {
        "episode_count": len(rows),
        "source_counts": dict(sorted(Counter(_source(row) for row in rows).items())),
        "task_counts": dict(sorted(Counter(str(row.get("task") or "") for row in rows).items())),
        "maximum_contiguous_same_source_run": _maximum_run(rows, _source),
        "maximum_contiguous_same_task_run": _maximum_run(rows, lambda row: str(row.get("task") or "")),
        "first_1000_source_distribution": _window_source_distribution(rows, 0, window),
        "middle_1000_source_distribution": _window_source_distribution(rows, middle_start, window),
        "last_1000_source_distribution": _window_source_distribution(rows, last_start, window),
    }


def audit_train_order(*, old_dir: str | Path, new_dir: str | Path) -> dict[str, Any]:
    old_root, new_root = Path(old_dir), Path(new_dir)
    old_train = _rows(old_root / "train.jsonl")
    new_train = _rows(new_root / "train.jsonl")
    old_by_id, new_by_id = _by_id(old_train), _by_id(new_train)
    missing = sorted(set(old_by_id) - set(new_by_id))
    extra = sorted(set(new_by_id) - set(old_by_id))
    mismatches = [
        episode_id
        for episode_id in sorted(set(old_by_id) & set(new_by_id))
        if old_by_id[episode_id] != new_by_id[episode_id]
    ]
    expected_ids = [
        row["episode_id"]
        for row in sorted(new_train, key=formal_train_order_key)
    ]
    actual_ids = [row["episode_id"] for row in new_train]
    old_ids = [row["episode_id"] for row in old_train]

    old_manifest = json.loads((old_root / "manifest.json").read_text(encoding="utf-8"))
    new_manifest = json.loads((new_root / "manifest.json").read_text(encoding="utf-8"))
    expected_ordering = {
        "train": {
            "policy": FORMAL_TRAIN_ORDERING_POLICY,
            "seed": FORMAL_TRAIN_ORDERING_SEED,
            "key": "episode_id",
        },
        "validation": {"policy": VALIDATION_ORDERING_POLICY},
    }
    validation_old_sha = _sha256(old_root / "validation.jsonl")
    validation_new_sha = _sha256(new_root / "validation.jsonl")
    artifacts: dict[str, Any] = {}
    for name in (
        "validation.jsonl",
        "pair_registry.jsonl",
        "changechat_row_map.jsonl",
        "source_summary.json",
        "target_contract.json",
        "rejected.jsonl",
    ):
        old_sha = _sha256(old_root / name)
        new_sha = _sha256(new_root / name)
        artifacts[name] = {
            "old_sha256": old_sha,
            "new_sha256": new_sha,
            "identical": old_sha == new_sha,
        }

    old_stats, new_stats = _order_stats(old_train), _order_stats(new_train)
    gates = {
        "episode_count_equal": len(old_train) == len(new_train),
        "episode_set_equal": not missing and not extra,
        "content_by_id_equal": not mismatches,
        "train_order_changed": old_ids != actual_ids,
        "new_order_matches_contract": actual_ids == expected_ids,
        "validation_byte_identical": validation_old_sha == validation_new_sha,
        "supporting_artifacts_identical": all(item["identical"] for item in artifacts.values()),
        "target_contract_unchanged": old_manifest.get("target_contract") == new_manifest.get("target_contract"),
        "ordering_manifest_exact": new_manifest.get("ordering") == expected_ordering,
        "source_block_bias_removed": (
            new_stats["maximum_contiguous_same_source_run"] < 10_000
            and new_stats["maximum_contiguous_same_source_run"]
            < old_stats["maximum_contiguous_same_source_run"]
        ),
    }
    return {
        "schema_version": 1,
        "status": "PASS" if all(gates.values()) else "CORPUS_ORDER_AUDIT_FAILED",
        "ordering": expected_ordering,
        "old": {
            "root": str(old_root),
            "train_sha256": _sha256(old_root / "train.jsonl"),
            "validation_sha256": validation_old_sha,
            **old_stats,
        },
        "new": {
            "root": str(new_root),
            "train_sha256": _sha256(new_root / "train.jsonl"),
            "validation_sha256": validation_new_sha,
            **new_stats,
        },
        "comparison": {
            "missing_episode_ids": len(missing),
            "extra_episode_ids": len(extra),
            "content_mismatches_by_episode_id": len(mismatches),
            "missing_examples": missing[:20],
            "extra_examples": extra[:20],
            "content_mismatch_examples": mismatches[:20],
        },
        "supporting_artifacts": artifacts,
        "gates": gates,
    }
