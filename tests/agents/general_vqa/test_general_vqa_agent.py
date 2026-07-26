"""Test GeneralVQAAgent — supported tasks, result filename, prompt version. / 测试 GeneralVQAAgent。"""

from __future__ import annotations

from pathlib import Path

import pytest

from spacers_agent.agents.general_vqa.agent import GeneralVQAAgent
from spacers_agent.agents.base import AgentContext
from spacers_agent.prompt_catalog import PromptAsset
from spacers_agent.schemas import ExpertResult, ImageRef, UnifiedSample

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "legacy"
TEST_IMAGE = FIXTURES / "test_image.png"
GENERAL_PROMPT = PromptAsset("general", Path("general_vqa_v2.md"), "general-vqa-v2", "prompt")


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def complete_json(self, *, messages, response_model, request_meta):
        self.calls.append({"request_meta": request_meta})
        return ExpertResult(expert="general_vqa_expert", answer="yes", status="completed")


def _sample() -> UnifiedSample:
    return UnifiedSample(
        sample_id="s1", dataset="test", split="test", task="general_vqa",
        images=[ImageRef(image_id="i1", path=TEST_IMAGE, role="image")],
        question="Is there a building?",
    )


@pytest.mark.asyncio
async def test_general_vqa_supported_tasks():
    agent = GeneralVQAAgent(None, GENERAL_PROMPT, "model")
    assert "general_vqa" in agent.supported_tasks
    assert "scene_classification" in agent.supported_tasks
    assert "multiple_choice_vqa" in agent.supported_tasks


@pytest.mark.asyncio
async def test_general_vqa_result_filename():
    client = _FakeClient()
    agent = GeneralVQAAgent(client, GENERAL_PROMPT, "model")
    ctx = AgentContext(artifact_dir=Path("/tmp"), settings=None, qwen_client=None, call_budget=None)
    exec_result = await agent.run(_sample(), ctx)
    assert exec_result.result_filename == "expert_result.json"


@pytest.mark.asyncio
async def test_general_vqa_prompt_version():
    client = _FakeClient()
    agent = GeneralVQAAgent(client, GENERAL_PROMPT, "model")
    ctx = AgentContext(artifact_dir=Path("/tmp"), settings=None, qwen_client=None, call_budget=None)
    exec_result = await agent.run(_sample(), ctx)
    assert exec_result.trace["prompt_version"] == "general-vqa-v2"
