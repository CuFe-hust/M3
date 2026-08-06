"""Test GroundingAgent — request schema, supported tasks. / 测试 GroundingAgent。"""

from __future__ import annotations

from pathlib import Path

import pytest

from spacers_agent.agents.grounding.agent import GroundingAgent
from spacers_agent.agents.base import AgentContext
from spacers_agent.prompt_catalog import PromptAsset
from spacers_agent.schemas import AgentResult, ImageRef, UnifiedSample

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "legacy"
TEST_IMAGE = FIXTURES / "test_image.png"
GROUNDING_PROMPT = PromptAsset("grounding", Path("general_vqa_v2.md"), "general-vqa-v2", "prompt")


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def complete_json(self, *, messages, response_model, request_meta):
        self.calls.append({"request_meta": request_meta})
        return AgentResult(agent_name="grounding_agent", answer="box at [100,200,300,400]", status="completed")


def _sample() -> UnifiedSample:
    return UnifiedSample(
        sample_id="s1", dataset="test", split="test", task="grounding",
        images=[ImageRef(image_id="i1", path=TEST_IMAGE, role="image")],
        question="Locate the building.",
    )


@pytest.mark.asyncio
async def test_grounding_supported_tasks():
    agent = GroundingAgent(None, GROUNDING_PROMPT, "model")
    assert "grounding" in agent.supported_tasks


@pytest.mark.asyncio
async def test_grounding_result_filename():
    client = _FakeClient()
    agent = GroundingAgent(client, GROUNDING_PROMPT, "model")
    ctx = AgentContext(artifact_dir=Path("/tmp"), settings=None, qwen_client=None, call_budget=None)
    exec_result = await agent.run(_sample(), ctx)
    assert exec_result.result_filename == "agent_result.json"
