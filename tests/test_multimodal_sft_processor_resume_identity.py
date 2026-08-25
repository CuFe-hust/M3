from __future__ import annotations

import pytest

from training.multimodal_sft.checkpoint import (
    CheckpointContractError,
    build_training_manifest,
    validate_resume_compatibility,
)
from training.multimodal_sft.identity import artifact_tree_identity


def _identity(*, content: str = "content-a", encoding: str = "contract-v1", template: str = "template-a", special: str = "special-a") -> dict:
    return {
        "class": "fixture.Processor",
        "tokenizer_class": "fixture.Tokenizer",
        "chat_template_sha256": template,
        "special_tokens_sha256": special,
        "special_token_ids": {"eos_token_id": 2},
        "encoding_contract_version": encoding,
        "content_sha256": content,
    }


def _manifest(processor: dict) -> dict:
    return build_training_manifest(
        adapter_name="fixture",
        model_identity={"model_type": "fixture"},
        task_profile="phase2",
        data_contract={},
        tuning_policy={"name": "lora_only"},
        parameter_plan={"parameter_names": ["language.weight"]},
        processor_identity=processor,
    )


def _validate(manifest: dict, processor: dict) -> None:
    validate_resume_compatibility(
        manifest,
        adapter_name="fixture",
        model_identity={"model_type": "fixture"},
        task_profile="phase2",
        tuning_policy={"name": "lora_only"},
        parameter_plan={"parameter_names": ["language.weight"]},
        processor_identity=processor,
    )


def test_same_processor_content_and_encoding_resume_pass() -> None:
    processor = _identity()
    _validate(_manifest(processor), processor)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("content_sha256", "content-b", "RESUME_PROCESSOR_CONTENT_MISMATCH"),
        ("encoding_contract_version", "contract-v2", "RESUME_PROCESSOR_ENCODING_CONTRACT_MISMATCH"),
        ("chat_template_sha256", "template-b", "RESUME_PROCESSOR_IDENTITY_MISMATCH"),
        ("special_tokens_sha256", "special-b", "RESUME_PROCESSOR_IDENTITY_MISMATCH"),
    ),
)
def test_processor_resume_identity_drift_is_rejected(field: str, value: str, error: str) -> None:
    expected = _identity()
    actual = dict(expected)
    actual[field] = value
    with pytest.raises(CheckpointContractError, match=error):
        _validate(_manifest(expected), actual)


def test_missing_processor_content_is_unproven() -> None:
    expected = _identity()
    actual = dict(expected)
    actual.pop("content_sha256")
    with pytest.raises(CheckpointContractError, match="RESUME_PROCESSOR_IDENTITY_UNPROVEN"):
        _validate(_manifest(expected), actual)


def test_tokenizer_artifact_content_drift_is_rejected(tmp_path) -> None:
    left = tmp_path / "processor_a"
    right = tmp_path / "processor_b"
    left.mkdir()
    right.mkdir()
    (left / "tokenizer.json").write_text('{"vocab":{"a":1}}\n', encoding="utf-8")
    (right / "tokenizer.json").write_text('{"vocab":{"a":2}}\n', encoding="utf-8")
    expected = _identity(content=artifact_tree_identity(left)["sha256"])
    actual = dict(expected)
    actual["content_sha256"] = artifact_tree_identity(right)["sha256"]
    with pytest.raises(CheckpointContractError, match="RESUME_PROCESSOR_CONTENT_MISMATCH"):
        _validate(_manifest(expected), actual)
