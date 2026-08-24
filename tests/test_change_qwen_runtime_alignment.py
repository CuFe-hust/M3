from __future__ import annotations

from agents.change.prompt_contract import INITIAL_RESPONSE_SUFFIX, evidence_label


def test_initial_prompt_contract_matches_runtime_raw_labels() -> None:
    assert INITIAL_RESPONSE_SUFFIX.startswith("Decision stage is initial.")
    assert evidence_label("raw_full_t1") == "AUTHORITATIVE RAW T1 - earlier full scene"
    assert evidence_label("raw_full_t2") == "AUTHORITATIVE RAW T2 - later full scene"
