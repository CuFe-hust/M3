from __future__ import annotations

import pytest

from scripts import finetune_change_qwen_sft as wrapper
from scripts import finetune_qwen3vl_phase2 as shared


def test_profiles_are_explicit_and_resume_isolated(tmp_path) -> None:
    assert shared.DataArguments.__dataclass_fields__["data_profile"].default == "phase2"
    context = {
        "training_profile": "change_agent", "data_contract": {"name": "change_qwen_sft"}, "change_prompt": {"sha256": "a"},
        "base_model": {"fingerprint": "base", "revision": None}, "processor": {"fingerprint": "processor"},
        "data": {"train_sha256": "train", "eval_sha256": None, "train_upstream_manifest_sha256": None},
        "lora": {"rank": 1, "alpha": 1, "dropout": 0.0, "bias": "none", "target_modules": []},
        "merger": {"parameters": []}, "optimizer": {"groups": []},
        "augmentation": {"seed": "s", "config": {}},
        "training": {"max_seq_length": 16, "image_min_pixels": 1, "image_max_pixels": 1, "image_pixels_applied": False, "torch_dtype": "float32"},
        "data_sampling": {"group_key": "task", "repeat_weights": {}},
    }
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / shared.TRAINING_MANIFEST_FILENAME).write_text('{"schema_version":1,"training_profile":"phase2"}', encoding="utf-8")
    with pytest.raises(shared.ResumeConflictError):
        shared.validate_resume_checkpoint(checkpoint, context)


def test_wrapper_refuses_profile_override() -> None:
    with pytest.raises(ValueError):
        wrapper.main(["--data_profile", "phase2"])
