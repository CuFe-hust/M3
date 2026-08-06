"""Test ChangeAgent — request schema, prompt version, supported tasks. / 测试 ChangeAgent。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spacers_agent.agents.change.agent import ChangeAgent
from spacers_agent.agents.base import AgentContext
from spacers_agent.prompt_catalog import PromptAsset
from spacers_agent.routing import CallBudget
from spacers_agent.schemas import AgentResult, ImageRef, UnifiedSample
from spacers_agent.settings import AppSettings

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "legacy"
TEST_IMAGE = FIXTURES / "test_image.png"
CHANGE_PROMPT = PromptAsset("change", Path("change_dual_path_v1.md"), "change-dual-path-v1", "prompt")


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def complete_json(self, *, messages, response_model, request_meta):
        self.calls.append({"messages": messages, "request_meta": request_meta})
        return AgentResult(agent_name="change_agent", answer="A building appeared.", status="completed")


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
    agent = ChangeAgent(None, CHANGE_PROMPT, "model")
    assert "change_caption" in agent.supported_tasks
    assert "change_qa" in agent.supported_tasks


@pytest.mark.asyncio
async def test_change_agent_result_filename(tmp_path):
    client = _FakeClient()
    agent = ChangeAgent(client, CHANGE_PROMPT, "model")
    ctx = AgentContext(artifact_dir=tmp_path, settings=None, qwen_client=None, call_budget=None)
    exec_result = await agent.run(_sample(), ctx)
    assert exec_result.result_filename == "agent_result.json"


@pytest.mark.asyncio
async def test_change_agent_prompt_version_in_trace(tmp_path):
    client = _FakeClient()
    agent = ChangeAgent(client, CHANGE_PROMPT, "model")
    ctx = AgentContext(artifact_dir=tmp_path, settings=None, qwen_client=None, call_budget=None)
    exec_result = await agent.run(_sample(), ctx)
    assert exec_result.trace["prompt_version"] == "change-dual-path-v1"


@pytest.mark.asyncio
async def test_change_agent_coordinate_frame(tmp_path):
    client = _FakeClient()
    agent = ChangeAgent(client, CHANGE_PROMPT, "model")
    ctx = AgentContext(artifact_dir=tmp_path, settings=None, qwen_client=None, call_budget=None)
    await agent.run(_sample(), ctx)
    user_content = json.loads(client.calls[0]["messages"][1]["content"][-1]["text"])
    assert user_content["coordinate_frame"] == "normalized_0_999_top_left"


@pytest.mark.asyncio
async def test_raw_fallback_consumes_budget_and_keeps_model_path(tmp_path):
    client = _FakeClient()
    settings = AppSettings.model_validate({
        "agents": {"change": {"harmonization": {"min_pif_pixels": 1_000_000}}}
    })
    budget = CallBudget(max_qwen_calls=1)
    agent = ChangeAgent(client, CHANGE_PROMPT, "model", settings=settings.agents.change)
    ctx = AgentContext(artifact_dir=tmp_path, settings=settings, qwen_client=client, call_budget=budget)
    execution = await agent.run(_sample(), ctx)
    assert budget.qwen_calls_used == 1
    assert execution.trace["raw_fallback_used"] is True
    assert "RAW_FALLBACK_USED" in execution.trace["harmonization_reason_codes"]
    assert execution.agent_name == "change_agent"
    assert (tmp_path / "change_preprocess" / "harmonization_report.json").is_file()
    payload = json.loads(client.calls[0]["messages"][1]["content"][-1]["text"])
    assert [item["role"] for item in payload["image_manifest"]] == ["raw_full_t1", "raw_full_t2"]


@pytest.mark.asyncio
async def test_applied_dual_path_contains_raw_and_harmonized_full_evidence(tmp_path):
    client = _FakeClient()
    settings = AppSettings()
    budget = CallBudget(max_qwen_calls=1)
    agent = ChangeAgent(client, CHANGE_PROMPT, "model", settings=settings.agents.change)
    context = AgentContext(artifact_dir=tmp_path, settings=settings, qwen_client=client, call_budget=budget)
    execution = await agent.run(_sample(), context)
    payload = json.loads(client.calls[0]["messages"][1]["content"][-1]["text"])
    roles = [item["role"] for item in payload["image_manifest"]]
    assert roles[:4] == ["raw_full_t1", "raw_full_t2", "harmonized_t1", "harmonized_t2"]
    assert "proposal_overlay" in roles
    assert execution.trace["harmonization_status"] == "applied"
