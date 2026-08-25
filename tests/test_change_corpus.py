from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from training.multimodal_sft.change_corpus import ChangeCorpusBuildError, build_corpus
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
    assert manifest["target_contract"] == change_target_contract_identity()
    assert manifest["counts"]["unique_train_pairs"] == 1
    assert manifest["counts"]["unique_validation_pairs"] == 1
    assert manifest["counts"]["by_split"] == {"train": 1, "validation": 1, "test": 0}
    train = json.loads((output / "train.jsonl").read_text(encoding="utf-8"))
    result = train["target"]["result"]
    assert "evidence" not in result
    assert train["target"]["contract_version"] == manifest["target_contract"]["version"]
    assert (output / "target_contract.json").is_file()
    with pytest.raises(ChangeCorpusBuildError, match="OUTPUT_EXISTS"):
        build_corpus(spec, output, str(prompt))
