from __future__ import annotations

import pytest
from scripts.change_qwen_sft_data import ChangeSFTDataError, validate_change_episode


def _episode() -> dict:
    return {
        "schema_version": 1, "episode_id": "x", "parent_sample_id": "p", "task": "change_caption",
        "input_contract": "semantic_pair_v1", "question": "",
        "images": [{"image_source": "x", "path": "a.png", "role": "raw_full_t1"}, {"image_source": "x", "path": "b.png", "role": "raw_full_t2"}],
        "request_payload": {"image_manifest": [{"index": "0", "role": "raw_full_t1"}, {"index": "1", "role": "raw_full_t2"}]},
        "target": {"response_schema": "ChangeInitialResult", "result": {"agent_name": "change_agent", "answer": "A building appeared.", "boxes": [], "evidence": [], "evidence_items": [], "geometry": {}, "status": "completed"}},
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
