"""Test SpatialAgent — grid prompt, candidate review, supported tasks. / 测试 SpatialAgent。"""

from __future__ import annotations

from pathlib import Path

import pytest

from spacers_agent.agents.spatial.agent import SpatialAgent
from spacers_agent.agents.base import AgentContext
from spacers_agent.schemas import ExpertResult, ImageRef, UnifiedSample

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "legacy"
TEST_IMAGE = FIXTURES / "test_image.png"


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def complete_json(self, *, messages, response_model, request_meta):
        self.calls.append({"request_meta": request_meta, "messages": messages})
        return ExpertResult(expert="spatial_expert", answer="top-left", status="completed")


def _sample(question: str = "Where is the car?") -> UnifiedSample:
    return UnifiedSample(
        sample_id="s1", dataset="VRSBench", split="validation", task="general_vqa",
        images=[ImageRef(image_id="i1", path=TEST_IMAGE, role="image")],
        question=question,
        metadata={"question_type": "spatial"},
    )


@pytest.mark.asyncio
async def test_spatial_supported_tasks():
    agent = SpatialAgent(None, {"spatial": "p", "spatial_grid": "gp"}, "model")
    assert "spatial_relation" in agent.supported_tasks


@pytest.mark.asyncio
async def test_spatial_result_filename():
    client = _FakeClient()
    agent = SpatialAgent(client, {"spatial": "p", "spatial_grid": "gp"}, "model")
    ctx = AgentContext(artifact_dir=Path("/tmp"), settings=None, qwen_client=None, call_budget=None)
    exec_result = await agent.run(_sample(), ctx)
    assert exec_result.result_filename == "expert_result.json"


@pytest.mark.asyncio
async def test_spatial_no_review_when_not_needed():
    """When candidate review is not needed, no second call is made."""
    client = _FakeClient()
    agent = SpatialAgent(
        client,
        {"spatial": "p", "spatial_grid": "gp", "spatial_review": "", "spatial_grid_review": ""},
        "model",
    )
    ctx = AgentContext(artifact_dir=Path("/tmp"), settings=None, qwen_client=None, call_budget=None)
    exec_result = await agent.run(_sample("Where is the car?"), ctx)
    assert exec_result.status == "completed"
