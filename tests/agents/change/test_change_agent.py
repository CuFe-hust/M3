"""Test ChangeAgent — request schema, prompt version, supported tasks. / 测试 ChangeAgent。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spacers_agent.agents.change.agent import ChangeAgent
from spacers_agent.agents.base import AgentContext
from spacers_agent.schemas import ExpertResult, ImageRef, UnifiedSample

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "legacy"
TEST_IMAGE = FIXTURES / "test_image.png"


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def complete_json(self, *, messages, response_model, request_meta):
        self.calls.append({"messages": messages, "request_meta": request_meta})
        return ExpertResult(expert="change_expert", answer="A building appeared.", status="completed")


def _sample() -> UnifiedSample:
    return UnifiedSample(
        sample_id="s1", dataset="LEVIR-CC", split="test", task="change_caption",
        images=[
            ImageRef(image_id="t1", path=TEST_IMAGE, role="t1"),
            ImageRef(image_id="t2", path=TEST_IMAGE, role="t2"),
        ],
        question="Describe the change.",
    )


@pytest.mark.asyncio
async def test_change_agent_supported_tasks():
    agent = ChangeAgent(None, {"change": "prompt"}, "model")
    assert "change_caption" in agent.supported_tasks
    assert "change_qa" in agent.supported_tasks


@pytest.mark.asyncio
async def test_change_agent_result_filename():
    client = _FakeClient()
    agent = ChangeAgent(client, {"change": "prompt"}, "model")
    ctx = AgentContext(artifact_dir=Path("/tmp"), settings=None, qwen_client=None, call_budget=None)
    exec_result = await agent.run(_sample(), ctx)
    assert exec_result.result_filename == "expert_result.json"


@pytest.mark.asyncio
async def test_change_agent_prompt_version_in_trace():
    client = _FakeClient()
    agent = ChangeAgent(client, {"change": "prompt"}, "model")
    ctx = AgentContext(artifact_dir=Path("/tmp"), settings=None, qwen_client=None, call_budget=None)
    exec_result = await agent.run(_sample(), ctx)
    assert exec_result.trace["prompt_version"] == "change-v1"


@pytest.mark.asyncio
async def test_change_agent_coordinate_frame():
    client = _FakeClient()
    agent = ChangeAgent(client, {"change": "prompt"}, "model")
    ctx = AgentContext(artifact_dir=Path("/tmp"), settings=None, qwen_client=None, call_budget=None)
    await agent.run(_sample(), ctx)
    user_content = json.loads(client.calls[0]["messages"][1]["content"][2]["text"])
    assert user_content["coordinate_frame"] == "normalized_0_999_top_left"
