from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from training.multimodal_sft.change_corpus import (
    FORMAL_TRAIN_ORDERING_POLICY,
    FORMAL_TRAIN_ORDERING_SEED,
    ChangeCorpusBuildError,
    build_corpus,
    formal_train_order_key,
)
from training.multimodal_sft.change_target_contract import change_target_contract_identity


def test_authoritative_builder_preserves_pair_split_and_emits_v2(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    captions = tmp_path / "levir.json"
    captions.write_text(json.dumps({"images": [
        {"filename": "train.png", "split": "train", "captions": ["New building."]},
        {"filename": "val.png", "split": "val", "captions": ["NO_CHANGE"]},
        {"filename": "test.png", "split": "test", "captions": ["Road removed."]},
    ]}), encoding="utf-8")
    exclusions = tmp_path / "exclusions.txt"
    exclusions.write_text("", encoding="utf-8")
    spec = tmp_path / "source_spec.yaml"
    spec.write_text(yaml.safe_dump({
        "schema_version": 1,
        "canonical_dataset": {"type": "levir", "captions": str(captions.resolve()), "image_root": str(image_root.resolve())},
        "exclusions": {"file": str(exclusions.resolve())},
        "split_policy": {"authority": "levir_official", "include_test": False},
        "sources": [{"id": "levir", "kind": "levir_caption", "task": "change_caption", "path": str(captions.resolve()), "enabled": True}],
    }, sort_keys=False), encoding="utf-8")
    prompt = tmp_path / "prompt.md"
    prompt.write_text("prompt", encoding="utf-8")
    output = tmp_path / "v2"
    manifest = build_corpus(spec, output, str(prompt))

    assert manifest["schema_version"] == 2
    assert len(manifest["builder_git"]["commit"]) == 40
    assert len(manifest["builder_git"]["tree"]) == 40
    assert isinstance(manifest["builder_git"]["working_tree_clean"], bool)
    assert manifest["target_contract"] == change_target_contract_identity()
    assert manifest["ordering"] == {
        "train": {"policy": FORMAL_TRAIN_ORDERING_POLICY, "seed": FORMAL_TRAIN_ORDERING_SEED, "key": "episode_id"},
        "validation": {"policy": "builder_source_order_v1"},
    }
    assert manifest["counts"]["unique_train_pairs"] == 1
    assert manifest["counts"]["unique_validation_pairs"] == 1
    assert manifest["counts"]["by_split"] == {"train": 1, "validation": 1, "test": 0}
    train = json.loads((output / "train.jsonl").read_text(encoding="utf-8"))
    result = train["target"]["result"]
    assert "evidence" not in result
    assert train["target"]["contract_version"] == manifest["target_contract"]["version"]
    assert (output / "target_contract.json").is_file()
    repeat = tmp_path / "v2-repeat"
    repeat_manifest = build_corpus(spec, repeat, str(prompt))
    assert (output / "train.jsonl").read_bytes() == (repeat / "train.jsonl").read_bytes()
    assert manifest["outputs"]["train.jsonl_sha256"] == repeat_manifest["outputs"]["train.jsonl_sha256"]
    assert manifest["ordering"] == repeat_manifest["ordering"]
    with pytest.raises(ChangeCorpusBuildError, match="OUTPUT_EXISTS"):
        build_corpus(spec, output, str(prompt))


def test_formal_train_order_is_hash_sorted_not_source_blocks(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    canonical_rows = [
        {"filename": f"pair-{index:02d}.png", "split": "train"}
        for index in range(12)
    ]
    captions = tmp_path / "canonical.json"
    captions.write_text(json.dumps(canonical_rows), encoding="utf-8")
    exclusions = tmp_path / "exclusions.txt"
    exclusions.write_text("", encoding="utf-8")
    sources = []
    source_block_ids = []
    for source_index, source_id in enumerate(("source_a", "source_b", "source_c")):
        rows = []
        for local_index in range(4):
            pair_index = source_index * 4 + local_index
            rows.append({
                "filename": f"pair-{pair_index:02d}.png",
                "question": f"question {source_id} {local_index}",
                "answer": f"answer {source_id} {local_index}",
            })
            source_block_ids.append(f"{source_id}/levir:train:pair-{pair_index:02d}.png/{local_index}")
        source_path = tmp_path / f"{source_id}.json"
        source_path.write_text(json.dumps(rows), encoding="utf-8")
        sources.append({
            "id": source_id, "kind": "changechat", "task": "change_qa",
            "path": str(source_path.resolve()), "enabled": True,
        })
    spec = tmp_path / "source_spec.yaml"
    spec.write_text(yaml.safe_dump({
        "schema_version": 1,
        "canonical_dataset": {"type": "levir", "captions": str(captions.resolve()), "image_root": str(image_root.resolve())},
        "exclusions": {"file": str(exclusions.resolve())},
        "split_policy": {"authority": "levir_official", "include_test": False},
        "sources": sources,
    }, sort_keys=False), encoding="utf-8")
    prompt = tmp_path / "prompt.md"
    prompt.write_text("prompt", encoding="utf-8")
    output = tmp_path / "mixed"
    build_corpus(spec, output, str(prompt))
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    actual_ids = [row["episode_id"] for row in rows]
    assert actual_ids == [row["episode_id"] for row in sorted(rows, key=formal_train_order_key)]
    assert actual_ids != source_block_ids


def test_formal_train_order_requires_episode_id() -> None:
    with pytest.raises(ChangeCorpusBuildError, match="EPISODE_ID_REQUIRED_FOR_ORDERING"):
        formal_train_order_key({})
