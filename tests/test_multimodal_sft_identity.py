import json
from pathlib import Path

import pytest

from training.multimodal_sft.checkpoint import (
    checkpoint_complete,
    build_training_manifest,
    write_completion_marker,
    write_manifest,
)
from training.multimodal_sft.identity import (
    base_weight_identity,
    processor_content_identity,
)


class _Processor:
    chat_template = "{{ messages }}"
    special_tokens_map = {"eos_token": "</s>"}
    eos_token_id = 2

    def save_pretrained(self, path: str | Path) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        (root / "tokenizer_config.json").write_text('{"chat_template":"{{ messages }}"}\n', encoding="utf-8")
        (root / "special_tokens_map.json").write_text('{"eos_token":"</s>"}\n', encoding="utf-8")


def _base(root: Path, payload: bytes) -> Path:
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"tiny"}\n', encoding="utf-8")
    (root / "model.safetensors").write_bytes(payload)
    return root


def test_base_weight_identity_uses_content_not_config_or_path(tmp_path: Path) -> None:
    left = base_weight_identity(_base(tmp_path / "a", b"A"))
    right = base_weight_identity(_base(tmp_path / "b", b"B"))
    assert left["scheme"] == "hf_local_weight_files_v1"
    assert left["sha256"] != right["sha256"]
    assert base_weight_identity(tmp_path / "a") == left


def test_safetensors_index_rejects_path_escape(tmp_path: Path) -> None:
    root = tmp_path / "indexed"
    root.mkdir()
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "../outside.safetensors"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="BASE_WEIGHT_IDENTITY_INVALID_INDEX"):
        base_weight_identity(root)


def test_processor_identity_contains_semantic_and_content_hashes(tmp_path: Path) -> None:
    processor = _Processor()
    processor.save_pretrained(tmp_path)
    identity = processor_content_identity(tmp_path, processor)
    assert identity["chat_template_sha256"]
    assert identity["special_tokens_sha256"]
    assert identity["content_sha256"]
    assert {item["path"] for item in identity["files"]} == {
        "special_tokens_map.json",
        "tokenizer_config.json",
    }


def test_completion_marker_v2_covers_all_mutable_artifacts(tmp_path: Path) -> None:
    (tmp_path / "adapter").mkdir()
    (tmp_path / "processor").mkdir()
    (tmp_path / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
    (tmp_path / "processor" / "processor_config.json").write_text("{}", encoding="utf-8")
    plan = {"adapter_name": "tiny", "full_train_parameter_names": []}
    manifest = build_training_manifest(
        adapter_name="tiny",
        model_identity={"model_type": "tiny"},
        task_profile="phase2",
        data_contract={},
        tuning_policy={},
        parameter_plan=plan,
        processor_identity={"content_sha256": "fixture"},
    )
    write_manifest(tmp_path, manifest)
    (tmp_path / "parameter_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    for name in (
        "model_trainable_state.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "rng_state.pt",
        "training_log.jsonl",
    ):
        (tmp_path / name).write_bytes(name.encode())
    write_completion_marker(tmp_path, global_step=1)
    assert checkpoint_complete(tmp_path)
    (tmp_path / "optimizer.pt").write_bytes(b"mutated")
    assert not checkpoint_complete(tmp_path)
