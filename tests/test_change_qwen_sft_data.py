from __future__ import annotations

import pytest
from scripts.change_qwen_sft_data import ChangeSFTDataError, validate_change_episode
from training.multimodal_sft.change_target_contract import CHANGE_TARGET_CONTRACT_VERSION


def _episode() -> dict:
    return {
        "schema_version": 2, "episode_id": "x", "parent_sample_id": "p", "task": "change_caption",
        "input_contract": "semantic_pair_v1", "question": "",
        "images": [{"image_source": "x", "path": "a.png", "role": "raw_full_t1"}, {"image_source": "x", "path": "b.png", "role": "raw_full_t2"}],
        "request_payload": {"image_manifest": [{"index": "0", "role": "raw_full_t1"}, {"index": "1", "role": "raw_full_t2"}]},
        "target": {"response_schema": "ChangeInitialResult", "contract_version": CHANGE_TARGET_CONTRACT_VERSION, "result": {"agent_name": "change_agent", "answer": "A building appeared.", "boxes": [], "evidence_items": [], "geometry": {}, "status": "completed"}},
    }


def test_schema_rejects_empty_qa_and_reversed_pair() -> None:
    episode = _episode()
    episode["task"] = "change_qa"
    with pytest.raises(ChangeSFTDataError, match="missing_question"):
        validate_change_episode(episode)
    episode["question"] = "What changed?"
    episode["images"][:2] = list(reversed(episode["images"][:2]))
    with pytest.raises(ChangeSFTDataError, match="invalid_role_order"):
        validate_change_episode(episode)


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda row: row.update(schema_version=1), "schema_version"),
        (lambda row: row["target"].update(contract_version="legacy"), "target_contract_version_mismatch"),
        (lambda row: row["target"]["result"].update(evidence=[]), "noncanonical_target_result"),
        (lambda row: row["target"]["result"].update(evidence_items=[{"label": "x", "box": [1, 2, 3, 4], "point": None, "image_id": None, "coordinate_frame": "normalized_0_999_top_left", "confidence": 0.9}]), "noncanonical_target_result"),
    ],
)
def test_target_contract_is_strict_for_training_rows(mutate, code: str) -> None:
    episode = _episode()
    mutate(episode)
    with pytest.raises(ChangeSFTDataError, match=code):
        validate_change_episode(episode)
