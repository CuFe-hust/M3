from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training.multimodal_sft.change_target_migration import (
    ChangeTargetMigrationError,
    audit_target_contract,
    compare_corpora,
    migrate_reference_corpus,
)


def _row(*, answer: str = "A building appeared.", evidence=None) -> dict:
    return {
        "schema_version": 1,
        "episode_id": "source/levir:train:one.png/0",
        "parent_sample_id": "levir:train:one.png",
        "split": "train",
        "task": "change_caption",
        "target": {
            "response_schema": "ChangeInitialResult",
            "result": {
                "agent_name": "change_agent", "answer": answer, "boxes": [],
                "evidence": [] if evidence is None else evidence,
                "evidence_items": [], "geometry": {}, "status": "completed",
            },
        },
    }


def _old_corpus(root: Path, row: dict | None = None) -> Path:
    root.mkdir()
    payload = json.dumps(row or _row(), separators=(",", ":")) + "\n"
    (root / "train.jsonl").write_text(payload, encoding="utf-8")
    (root / "validation.jsonl").write_text("", encoding="utf-8")
    for name in ("rejected.jsonl", "pair_registry.jsonl", "changechat_row_map.jsonl"):
        (root / name).write_text("", encoding="utf-8")
    (root / "source_summary.json").write_text("[]\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "change_prompt": {"ref": "x", "sha256": "x"},
        "outputs": {"train.jsonl_sha256": hashlib.sha256(payload.encode()).hexdigest()},
    }
    (root / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return root


def test_audit_and_safe_reference_migration_allow_only_legacy_field_removal(tmp_path: Path) -> None:
    old = _old_corpus(tmp_path / "old")
    audit = audit_target_contract(
        train=old / "train.jsonl", validation=old / "validation.jsonl", manifest=old / "manifest.json",
    )
    assert audit["status"] == "PASS"
    assert audit["counts"]["result_evidence_rows"] == 1
    assert audit["counts"]["evidence_nonempty_rows"] == 0
    assert audit["diff_paths"] == {"/target/result/evidence": 1}

    new = tmp_path / "reference"
    migrate_reference_corpus(old_dir=old, output_dir=new)
    migrated = json.loads((new / "train.jsonl").read_text(encoding="utf-8"))
    assert migrated["schema_version"] == 2
    assert "evidence" not in migrated["target"]["result"]
    assert compare_corpora(old_dir=old, new_dir=new)["status"] == "PASS"
    with pytest.raises(ChangeTargetMigrationError, match="OUTPUT_EXISTS"):
        migrate_reference_corpus(old_dir=old, output_dir=new)


def test_nonempty_evidence_and_unexpected_changes_fail_closed(tmp_path: Path) -> None:
    old = _old_corpus(tmp_path / "old", _row(evidence=[{"claim": "semantic content"}]))
    audit = audit_target_contract(
        train=old / "train.jsonl", validation=old / "validation.jsonl", manifest=old / "manifest.json",
    )
    assert audit["migration_allowed"] is False
    assert audit["counts"]["evidence_nonempty_rows"] == 1
    with pytest.raises(ChangeTargetMigrationError, match="TARGET_CONTRACT_MIGRATION_REVIEW_REQUIRED"):
        migrate_reference_corpus(old_dir=old, output_dir=tmp_path / "blocked")

    safe_old = _old_corpus(tmp_path / "safe")
    reference = tmp_path / "reference"
    migrate_reference_corpus(old_dir=safe_old, output_dir=reference)
    row = json.loads((reference / "train.jsonl").read_text(encoding="utf-8"))
    row["target"]["result"]["answer"] = "tampered"
    (reference / "train.jsonl").write_text(json.dumps(row, separators=(",", ":")) + "\n", encoding="utf-8")
    result = compare_corpora(old_dir=safe_old, new_dir=reference)
    assert result["status"] == "CORPUS_MIGRATION_UNEXPECTED_DIFF"
    assert result["unexpected_paths"] == {"/target/result/answer": 1}


def test_confidence_only_is_safely_removed_and_pair_split_changes_are_blocked(tmp_path: Path) -> None:
    row = _row()
    row["target"]["result"]["evidence_items"] = [
        {
            "label": "building", "box": [1, 2, 3, 4], "point": None,
            "image_id": None, "coordinate_frame": "normalized_0_999_top_left",
            "confidence": 0.8,
        },
    ]
    row["target"]["result"].pop("evidence")
    row["target"]["result"]["boxes"] = [[1, 2, 3, 4]]
    row["target"]["result"]["geometry"] = {
        "evidence_quality": ["trusted_box"], "repair_severity": "none",
    }
    old = _old_corpus(tmp_path / "old", row)
    audit = audit_target_contract(
        train=old / "train.jsonl", validation=old / "validation.jsonl", manifest=old / "manifest.json",
    )
    assert audit["migration_allowed"] is True
    assert audit["counts"]["evidence_item_confidence_fields"] == 1
    assert audit["diff_paths"]["/target/result/evidence_items/0/confidence"] == 1
    reference = tmp_path / "reference"
    migrate_reference_corpus(old_dir=old, output_dir=reference)
    migrated = json.loads((reference / "train.jsonl").read_text(encoding="utf-8"))
    assert "confidence" not in migrated["target"]["result"]["evidence_items"][0]

    for field, value, expected_path in (
        ("parent_sample_id", "other-pair", "/parent_sample_id"),
        ("split", "validation", "/split"),
    ):
        changed = dict(migrated)
        changed[field] = value
        (reference / "train.jsonl").write_text(json.dumps(changed, separators=(",", ":")) + "\n", encoding="utf-8")
        result = compare_corpora(old_dir=old, new_dir=reference)
        assert result["status"] == "CORPUS_MIGRATION_UNEXPECTED_DIFF"
        assert result["unexpected_paths"] == {expected_path: 1}
