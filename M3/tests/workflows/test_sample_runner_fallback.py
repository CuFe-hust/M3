"""Phase 5 — SampleRunner fallback behavior.
Phase 5 — SampleRunner fallback 行为。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spacers_agent.agents.base import Agent, AgentContext, AgentExecution, AgentName
from spacers_agent.agents.registry import AgentRegistry
from spacers_agent.prompt_catalog import PromptCatalog
from spacers_agent.routing import CallBudgetFactory, TaskRouter
from spacers_agent.routing.schemas import RoutingDecision
from spacers_agent.schemas import AgentResult, ImageRef, UnifiedSample
from spacers_agent.settings import AppSettings
from spacers_agent.workflows.artifact_writer import ArtifactWriter
from spacers_agent.workflows.judge_service import JudgeService
from spacers_agent.workflows.sample_runner import SampleRunner

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "legacy"
TEST_IMAGE = FIXTURES / "test_image.png"
PROMPT_ROOT = Path(__file__).resolve().parents[2] / "prompts"


class _PrimaryFailingAgent:
    name: AgentName = "change_agent"
    supported_tasks = frozenset({"change_qa"})

    async def run(self, sample, context: AgentContext) -> AgentExecution:
        raise RuntimeError("primary agent simulated failure")


class _FallbackSucceedingAgent:
    name: AgentName = "general_vqa_agent"
    supported_tasks = frozenset({"general_vqa", "change_qa"})

    async def run(self, sample, context: AgentContext) -> AgentExecution:
        return AgentExecution(
            agent_name=self.name,
            payload=AgentResult(agent_name="general_vqa_agent", answer="fallback answer", status="completed"),
            result_filename="agent_result.json",
            trace={"fallback": True},
        )


class _FallbackAlsoFailingAgent:
    name: AgentName = "general_vqa_agent"
    supported_tasks = frozenset({"general_vqa"})

    async def run(self, sample, context: AgentContext) -> AgentExecution:
        raise RuntimeError("fallback agent also failed")


def _runner(registry: AgentRegistry) -> SampleRunner:
    settings = AppSettings()
    catalog = PromptCatalog(PROMPT_ROOT)
    return SampleRunner(
        settings, registry, None, catalog,
        router=TaskRouter(),
        judge_service=JudgeService(
            settings,
            judge_prompt=catalog["count_judge"],
            vqa_judge_prompt=catalog["vqa_judge"],
            repair_prompt=catalog["json_repair"],
        ),
        artifact_writer=ArtifactWriter(),
        call_budget_factory=CallBudgetFactory(),
    )


def _sample() -> UnifiedSample:
    return UnifiedSample(
        sample_id="s1", dataset="test", split="test", task="change_qa",
        images=[
            ImageRef(image_id="i1", path=TEST_IMAGE, role="t1"),
            ImageRef(image_id="i2", path=TEST_IMAGE, role="t2"),
        ],
        question="Did a building appear?",
    )


@pytest.mark.asyncio
async def test_fallback_from_failing_primary_to_vqa(tmp_path: Path):
    reg = AgentRegistry()
    reg.register(_PrimaryFailingAgent())
    reg.register(_FallbackSucceedingAgent())
    runner = _runner(reg)
    sample = _sample()
    sample_dir = tmp_path / "samples" / sample.sample_id
    # The router will route change_qa to change_agent with fallback general_vqa_agent
    outcome = await runner.run_one(sample, sample_dir)
    assert outcome.fallback_used is True
    assert outcome.status.state == "succeeded"
    assert outcome.execution is not None
    assert outcome.execution.payload.answer == "fallback answer"


@pytest.mark.asyncio
async def test_all_agents_fail_raises_exception(tmp_path: Path):
    reg = AgentRegistry()
    reg.register(_PrimaryFailingAgent())
    reg.register(_FallbackAlsoFailingAgent())
    runner = _runner(reg)
    sample = _sample()
    sample_dir = tmp_path / "samples" / sample.sample_id
    with pytest.raises(RuntimeError, match="All agents failed"):
        await runner.run_one(sample, sample_dir)
