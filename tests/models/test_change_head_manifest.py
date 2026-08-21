from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agents.change.settings import AgentChangeSettings
from models.base import (
    LEARNED_CHANGE_INPUT_CONTRACT_VERSION,
    LEARNED_CHANGE_OUTPUT_CONTRACT_VERSION,
    ModelCacheIdentity,
)
from models.change_head.fingerprint import build_change_input_pipeline_fingerprint
from models.change_head.manifest import ChangeHeadManifest, hash_class_names


_SHA = "a" * 64


def _manifest(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "input_contract_version": LEARNED_CHANGE_INPUT_CONTRACT_VERSION,
        "output_contract_version": LEARNED_CHANGE_OUTPUT_CONTRACT_VERSION,
        "architecture": {
            "name": "multi_expert_siamese_change_head_v1",
            "hidden_dim": 32,
            "semantic_dim": 16,
            "decoder_dim": 16,
            "optional_expert_dropout_supported": True,
            "use_pif_mask": True,
            "use_rgb_pair": False,
        },
        "experts": ({
            "expert_id": "segmenter_mitb2_001",
            "logical_model_id": "SegFormer-MiT-B2:iSAID:local",
            "weights_sha256": _SHA,
            "class_names_sha256": "b" * 64,
            "feature_stages": (1, 2),
            "feature_channels_by_stage": {1: 32, 2: 64},
            "required": True,
            "missing_policy": "error",
        },),
        "pipeline_fingerprint": "c" * 64,
        "model_weights_sha256": "d" * 64,
        "created_from_git_commit": "deadbeef",
        "training_manifest_sha256": "e" * 64,
    }
    data.update(overrides)
    return data


def test_valid_manifest_and_input_spec() -> None:
    manifest = ChangeHeadManifest.model_validate(_manifest())
    assert manifest.input_spec().requirement_by_expert_id()["segmenter_mitb2_001"].required


def test_duplicate_expert_id_rejected() -> None:
    data = _manifest()
    expert = dict(data["experts"][0])  # type: ignore[index]
    data["experts"] = (data["experts"][0], expert)
    with pytest.raises(ValueError, match="LEARNED_CHANGE_EXPERT_ID_MISMATCH"):
        ChangeHeadManifest.model_validate(data)


def test_bad_sha_empty_stage_and_required_zero_policy_rejected() -> None:
    data = _manifest()
    data["experts"] = ({
        **data["experts"][0],  # type: ignore[index]
        "weights_sha256": "not-a-sha",
        "feature_stages": (),
        "missing_policy": "zero_with_presence_mask",
    },)
    with pytest.raises(ValidationError):
        ChangeHeadManifest.model_validate(data)


def test_optional_zero_policy_requires_dropout_support() -> None:
    data = _manifest()
    data["architecture"] = {**data["architecture"], "optional_expert_dropout_supported": False}  # type: ignore[index]
    data["experts"] = ({
        **data["experts"][0],  # type: ignore[index]
        "expert_id": "segmenter_mitb2_002",
        "required": False,
        "missing_policy": "zero_with_presence_mask",
    }, data["experts"][0])
    with pytest.raises(ValueError, match="dropout"):
        ChangeHeadManifest.model_validate(data)


def test_contract_version_mismatch_rejected() -> None:
    with pytest.raises(ValueError, match="LEARNED_CHANGE_CONTRACT_MISMATCH"):
        ChangeHeadManifest.model_validate({**_manifest(), "input_contract_version": "old"})


def test_class_names_hash_is_deterministic() -> None:
    assert hash_class_names(("background", "building")) == hash_class_names(
        ["background", "building"]
    )


def test_fingerprint_excludes_io_paths_and_changes_with_semantics(tmp_path: Path) -> None:
    settings = AgentChangeSettings()
    settings.harmonization.calibration_file = tmp_path / "calibration.json"
    identity = ModelCacheIdentity(
        model="SegFormer-MiT-B2:iSAID:local",
        generation={"weights_sha256": _SHA},
        client_version="segformer-v1",
    )
    digest_a, payload_a = build_change_input_pipeline_fingerprint(
        settings=settings,
        semantic_client_identities={"segmenter_mitb2_001": identity},
    )
    settings.semantic.tile_size += 1
    digest_b, payload_b = build_change_input_pipeline_fingerprint(
        settings=settings,
        semantic_client_identities={"segmenter_mitb2_001": identity},
    )
    assert digest_a != digest_b
    encoded = json.dumps(payload_a, ensure_ascii=False)
    assert "save_artifacts" not in encoded
    assert str(tmp_path) not in encoded
    assert payload_a != payload_b
