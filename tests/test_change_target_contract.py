from __future__ import annotations

from training.multimodal_sft.change_target_contract import (
    CHANGE_SFT_EPISODE_SCHEMA_VERSION,
    CHANGE_TARGET_CONTRACT_VERSION,
    canonical_change_initial_result,
    change_target_contract_descriptor,
    change_target_contract_identity,
)


def test_contract_identity_matches_public_serialization() -> None:
    descriptor = change_target_contract_descriptor()
    identity = change_target_contract_identity()
    assert CHANGE_SFT_EPISODE_SCHEMA_VERSION == 2
    assert identity["version"] == CHANGE_TARGET_CONTRACT_VERSION
    assert identity["result_fields"] == [
        "agent_name", "answer", "boxes", "evidence_items", "geometry", "status",
    ]
    assert identity["visual_evidence_fields"] == [
        "label", "box", "point", "image_id", "coordinate_frame",
    ]
    assert descriptor["legacy_input_only_fields"] == {
        "result": ["evidence"], "visual_evidence": ["confidence"],
    }
    assert identity == change_target_contract_identity()


def test_legacy_fields_are_readable_but_never_canonical_output() -> None:
    result = canonical_change_initial_result({
        "agent_name": "change_agent",
        "answer": "A building appeared.",
        "boxes": [],
        "evidence": [],
        "evidence_items": [{"label": "building", "box": [1, 2, 3, 4], "confidence": 0.75}],
        "geometry": {},
        "status": "completed",
    })
    assert "evidence" not in result
    assert "confidence" not in result["evidence_items"][0]
    assert result["evidence_items"][0]["label"] == "building"
