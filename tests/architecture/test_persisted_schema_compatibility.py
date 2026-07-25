"""Freeze persisted artifact schema compatibility.
冻结持久化产物的 schema 兼容性。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spacers_agent.schemas import (
    CountingResult,
    ExpertResult,
    SampleRunStatus,
    VisualEvidence,
)


FIXTURES = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "legacy"


# ── counting_result.json / counting_result.json ─────────────────────────


def test_counting_result_schema_enforces_count():
    """final_count must equal accepted global_points."""
    # Construct a minimal valid result
    result = CountingResult(
        sample_id="test",
        target="building",
        question="How many?",
        source_width=1000,
        source_height=1000,
        tile_count=1,
        succeeded_tiles=["r000_c000"],
        failed_tiles=[],
        global_points=[],
        merged_groups=[],
        unresolved_conflicts=[],
        warnings=[],
        final_count=0,
        status="completed",
    )
    assert result.final_count == 0
    assert sum(1 for p in result.global_points if p.accepted) == 0


def test_counting_result_rejects_count_mismatch():
    """final_count != accepted points raises ValidationError."""
    from spacers_agent.schemas import GlobalPointObservation
    with pytest.raises(Exception):  # pydantic ValidationError
        CountingResult(
            sample_id="test",
            target="building",
            question="How many?",
            source_width=1000,
            source_height=1000,
            tile_count=1,
            succeeded_tiles=["r000_c000"],
            failed_tiles=[],
            global_points=[
                GlobalPointObservation(
                    global_id="test:r000_c000:p001",
                    target="building",
                    source_tile_id="r000_c000",
                    local_id="p001",
                    local_x_norm=500,
                    local_y_norm=500,
                    global_x_px=500,
                    global_y_px=500,
                    global_x_norm=500,
                    global_y_norm=500,
                    radius_px=0.0,
                    confidence=0.9,
                    ownership_valid=True,
                    near_core_boundary=False,
                    accepted=True,
                    short_evidence="test",
                )
            ],
            merged_groups=[],
            unresolved_conflicts=[],
            warnings=[],
            final_count=5,  # MISMATCH / 不匹配
            status="completed",
        )


def test_counting_result_failed_tiles_cannot_be_completed():
    """Failed tiles require partial or failed status."""
    with pytest.raises(Exception):
        CountingResult(
            sample_id="test",
            target="building",
            question="How many?",
            source_width=1000,
            source_height=1000,
            tile_count=1,
            succeeded_tiles=[],
            failed_tiles=["r000_c000"],
            global_points=[],
            merged_groups=[],
            unresolved_conflicts=[],
            warnings=[],
            final_count=0,
            status="completed",  # INVALID / 无效
        )


# ── VisualEvidence / VisualEvidence ──────────────────────────────────────


def test_visual_evidence_requires_exactly_one_of_box_or_point():
    """VisualEvidence must have exactly one of box or point."""
    # Valid: box only
    evidence = VisualEvidence(label="car", box=[100, 200, 300, 400])
    assert evidence.box == [100, 200, 300, 400]
    assert evidence.point is None

    # Valid: point only
    evidence = VisualEvidence(label="car", point=[500, 600])
    assert evidence.point == [500, 600]
    assert evidence.box is None

    # Invalid: both
    with pytest.raises(Exception):
        VisualEvidence(label="car", box=[100, 200, 300, 400], point=[500, 600])

    # Invalid: neither
    with pytest.raises(Exception):
        VisualEvidence(label="car")


def test_visual_evidence_box_bounds():
    """Box coordinates must be 0..999 and x1<x2, y1<y2."""
    with pytest.raises(Exception):
        VisualEvidence(label="car", box=[300, 400, 100, 200])  # reversed / 颠倒

    with pytest.raises(Exception):
        VisualEvidence(label="car", box=[-1, 0, 100, 100])  # negative / 负数

    with pytest.raises(Exception):
        VisualEvidence(label="car", box=[0, 0, 1000, 100])  # out of range / 超出范围


# ── ExpertResult / ExpertResult ──────────────────────────────────────────


def test_expert_result_serialization_roundtrip():
    """ExpertResult can serialize and deserialize."""
    result = ExpertResult(
        expert="general_vqa_expert",
        answer="yes",
        boxes=[[100, 200, 300, 400]],
        evidence=["visible building at centre"],
        evidence_items=[
            VisualEvidence(label="building", box=[100, 200, 300, 400], confidence=0.9),
        ],
        status="completed",
    )
    dumped = result.model_dump_json()
    reloaded = ExpertResult.model_validate_json(dumped)
    assert reloaded.answer == "yes"
    assert reloaded.status == "completed"


# ── SampleRunStatus / SampleRunStatus ────────────────────────────────────


def test_sample_run_status_json_roundtrip():
    """SampleRunStatus (the legacy compatible format) roundtrips."""
    status = SampleRunStatus(
        sample_id="s1",
        task="general_vqa",
        state="succeeded",
        result_path=Path("samples/s1/expert_result.json"),
        updated_at="2026-07-25T18:00:00Z",
    )
    dumped = status.model_dump_json()
    reloaded = SampleRunStatus.model_validate_json(dumped)
    assert reloaded.sample_id == "s1"
    assert reloaded.state == "succeeded"


# ── Legacy fixture existence / 旧 fixture 存在性 ─────────────────────────


def test_fixtures_directory_exists():
    """Legacy fixtures directory exists."""
    assert FIXTURES.is_dir(), f"Missing fixtures directory: {FIXTURES}"
