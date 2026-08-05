"""Test CaptionAgent — supported tasks, result filename, dedicated prompt. / 测试 CaptionAgent。"""

from __future__ import annotations

from pathlib import Path

import pytest

from spacers_agent.agents.caption.agent import CaptionAgent
from spacers_agent.agents.base import AgentContext
from spacers_agent.prompt_catalog import PromptAsset
from spacers_agent.schemas import AgentResult, ImageRef, UnifiedSample

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "legacy"
TEST_IMAGE = FIXTURES / "test_image.png"
CAPTION_PROMPT = PromptAsset("caption", Path("caption_v1.md"), "caption-v1", "caption prompt")


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def complete_json(self, *, messages, response_model, request_meta):
        self.calls.append({"request_meta": request_meta, "messages": messages})
        return AgentResult(agent_name="caption_agent", answer="A rural landscape with buildings.", status="completed")


def _sample() -> UnifiedSample:
    return UnifiedSample(
        sample_id="s1", dataset="test", split="test", task="caption",
        images=[ImageRef(image_id="i1", path=TEST_IMAGE, role="image")],
        question="Describe the image.",
    )


@pytest.mark.asyncio
async def test_caption_supported_tasks():
    agent = CaptionAgent(None, CAPTION_PROMPT, "model")
    assert "caption" in agent.supported_tasks


@pytest.mark.asyncio
async def test_caption_result_filename():
    client = _FakeClient()
    agent = CaptionAgent(client, CAPTION_PROMPT, "model")
    ctx = AgentContext(artifact_dir=Path("/tmp"), settings=None, qwen_client=None, call_budget=None)
    exec_result = await agent.run(_sample(), ctx)
    assert exec_result.result_filename == "agent_result.json"


@pytest.mark.asyncio
async def test_caption_prompt_version():
    client = _FakeClient()
    agent = CaptionAgent(client, CAPTION_PROMPT, "model")
    ctx = AgentContext(artifact_dir=Path("/tmp"), settings=None, qwen_client=None, call_budget=None)
    exec_result = await agent.run(_sample(), ctx)
    assert exec_result.trace["prompt_version"] == "caption-v1"
