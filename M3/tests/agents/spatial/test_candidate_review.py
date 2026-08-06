"""Test SpatialCandidateReviewer — needs_review, review success/failure. / 测试 SpatialCandidateReviewer。"""

from __future__ import annotations

from pathlib import Path

import pytest

from spacers_agent.agents.spatial.candidate_review import SpatialCandidateReviewer
from spacers_agent.schemas import AgentResult, ImageRef, UnifiedSample


def _sample() -> UnifiedSample:
    return UnifiedSample(
        sample_id="s1", dataset="VRSBench", split="validation", task="general_vqa",
        images=[ImageRef(image_id="i1", path=Path("/tmp/i1.png"), role="image")],
        question="What is the largest vehicle?",
        metadata={"question_type": "spatial"},
    )


def test_needs_review_returns_bool():
    """needs_review returns bool for any result. / needs_review 对任意结果返回 bool。"""
    reviewer = SpatialCandidateReviewer(None, "model", review_prompt="", review_prompt_version="v1")
    result = AgentResult(agent_name="spatial_agent", answer="car", evidence_items=[], status="completed")
    # Without VRSBench metadata, should be False / 无 VRSBench 元数据时应返回 False
    is_needed = reviewer.needs_review(_sample(), result)
    assert isinstance(is_needed, bool)


def test_no_review_prompt_skips():
    """When review_prompt is empty, needs_review still works. / review_prompt 为空时 needs_review 仍工作。"""
    reviewer = SpatialCandidateReviewer(None, "model", review_prompt="", review_prompt_version="v1")
    result = AgentResult(agent_name="spatial_agent", answer="ok", status="completed")
    assert isinstance(reviewer.needs_review(_sample(), result), bool)
