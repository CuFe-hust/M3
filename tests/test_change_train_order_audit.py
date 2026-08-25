from __future__ import annotations

import json
from pathlib import Path

from training.multimodal_sft.change_corpus import (
    FORMAL_TRAIN_ORDERING_POLICY,
    FORMAL_TRAIN_ORDERING_SEED,
    formal_train_order_key,
)
from training.multimodal_sft.change_train_order_audit import audit_train_order


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def _corpus(root: Path, train: list[dict], *, ordering: dict | None = None) -> None:
    root.mkdir()
    _write_jsonl(root / "train.jsonl", train)
    validation = [{"episode_id": "validation/one", "task": "change_caption", "provenance": {"source_id": "validation"}}]
    _write_jsonl(root / "validation.jsonl", validation)
    for name in ("pair_registry.jsonl", "changechat_row_map.jsonl", "rejected.jsonl"):
        (root / name).write_text("{}\n", encoding="utf-8")
    (root / "source_summary.json").write_text("[]\n", encoding="utf-8")
    (root / "target_contract.json").write_text('{"contract":"v2"}\n', encoding="utf-8")
    manifest = {"target_contract": {"version": "v2"}}
    if ordering is not None:
        manifest["ordering"] = ordering
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_order_audit_proves_only_train_sequence_changed(tmp_path: Path) -> None:
    old_rows = [
        {
            "episode_id": f"{source}/{index}", "task": "change_caption" if index % 2 else "change_qa",
            "provenance": {"source_id": source}, "payload": f"{source}-{index}",
        }
        for source in ("a", "b", "c")
        for index in range(12)
    ]
    new_rows = sorted(old_rows, key=formal_train_order_key)
    ordering = {
        "train": {"policy": FORMAL_TRAIN_ORDERING_POLICY, "seed": FORMAL_TRAIN_ORDERING_SEED, "key": "episode_id"},
        "validation": {"policy": "builder_source_order_v1"},
    }
    old, new = tmp_path / "old", tmp_path / "new"
    _corpus(old, old_rows)
    _corpus(new, new_rows, ordering=ordering)

    report = audit_train_order(old_dir=old, new_dir=new)
    assert report["status"] == "PASS"
    assert all(report["gates"].values())
    assert report["comparison"] == {
        "missing_episode_ids": 0,
        "extra_episode_ids": 0,
        "content_mismatches_by_episode_id": 0,
        "missing_examples": [],
        "extra_examples": [],
        "content_mismatch_examples": [],
    }
    assert report["old"]["maximum_contiguous_same_source_run"] == 12
    assert report["new"]["maximum_contiguous_same_source_run"] < 12


def test_order_audit_blocks_content_or_validation_changes(tmp_path: Path) -> None:
    rows = [
        {"episode_id": f"a/{index}", "task": "change_caption", "provenance": {"source_id": "a"}}
        for index in range(3)
    ]
    ordering = {
        "train": {"policy": FORMAL_TRAIN_ORDERING_POLICY, "seed": FORMAL_TRAIN_ORDERING_SEED, "key": "episode_id"},
        "validation": {"policy": "builder_source_order_v1"},
    }
    old, new = tmp_path / "old", tmp_path / "new"
    _corpus(old, rows)
    changed = [dict(row) for row in sorted(rows, key=formal_train_order_key)]
    changed[0]["task"] = "change_qa"
    _corpus(new, changed, ordering=ordering)
    (new / "validation.jsonl").write_text('{"episode_id":"changed"}\n', encoding="utf-8")
    report = audit_train_order(old_dir=old, new_dir=new)
    assert report["status"] == "CORPUS_ORDER_AUDIT_FAILED"
    assert report["gates"]["content_by_id_equal"] is False
    assert report["gates"]["validation_byte_identical"] is False
