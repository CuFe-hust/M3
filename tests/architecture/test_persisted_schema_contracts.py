"""Freeze current persisted artifact contracts. / 固化当前持久化产物契约。"""

from __future__ import annotations

from pathlib import Path

import pytest

from spacers_agent.schemas import AgentResult, CountingResult, SampleRunStatus, VisualEvidence


def test_counting_result_schema_enforces_count() -> None:
    """final_count must equal accepted global_points. / final_count 必须等于已接受点数。"""
    result = CountingResult(
        sample_id="test", target="building", question="How many?", source_width=1000,
        source_height=1000, tile_count=1, succeeded_tiles=["r000_c000"], failed_tiles=[],
        global_points=[], merged_groups=[], unresolved_conflicts=[], warnings=[], final_count=0,
        status="completed",
    )
    assert result.final_count == 0


def test_counting_result_rejects_count_mismatch() -> None:
    """Counting validation remains strict. / 计数结果校验仍然严格。"""
    from spacers_agent.schemas import GlobalPointObservation

    with pytest.raises(Exception):
        CountingResult(
            sample_id="test", target="building", question="How many?", source_width=1000,
            source_height=1000, tile_count=1, succeeded_tiles=["r000_c000"], failed_tiles=[],
            global_points=[GlobalPointObservation(
                global_id="test:r000_c000:p001", target="building", source_tile_id="r000_c000",
                local_id="p001", local_x_norm=500, local_y_norm=500, global_x_px=500,
                global_y_px=500, global_x_norm=500, global_y_norm=500, radius_px=0.0,
                confidence=0.9, ownership_valid=True, near_core_boundary=False, accepted=True,
                short_evidence="test",
            )],
            merged_groups=[], unresolved_conflicts=[], warnings=[], final_count=5, status="completed",
        )


def test_visual_evidence_requires_exactly_one_geometry() -> None:
    """Evidence keeps its original geometry invariant. / 证据保留原有几何不变量。"""
    assert VisualEvidence(label="car", box=[100, 200, 300, 400]).point is None
    assert VisualEvidence(label="car", point=[500, 600]).box is None
    with pytest.raises(Exception):
        VisualEvidence(label="car", box=[100, 200, 300, 400], point=[500, 600])
    with pytest.raises(Exception):
        VisualEvidence(label="car")


def test_agent_result_serialization_roundtrip() -> None:
    """Agent result preserves answer, boxes, and evidence. / Agent 结果保留回答、框和证据。"""
    result = AgentResult(
        agent_name="general_vqa_agent", answer="yes", boxes=[[100, 200, 300, 400]],
        evidence=["visible building at centre"],
        evidence_items=[VisualEvidence(label="building", box=[100, 200, 300, 400], confidence=0.9)],
        status="completed",
    )
    reloaded = AgentResult.model_validate_json(result.model_dump_json())
    assert reloaded.answer == "yes"
    assert reloaded.agent_name == "general_vqa_agent"


def test_sample_run_status_uses_agent_result_path() -> None:
    """Status records the canonical non-count artifact. / 状态记录规范的非计数产物。"""
    status = SampleRunStatus(
        sample_id="s1", task="general_vqa", state="succeeded",
        result_path=Path("samples/s1/agent_result.json"), updated_at="2026-07-25T18:00:00Z",
    )
    assert SampleRunStatus.model_validate_json(status.model_dump_json()).result_path == status.result_path
