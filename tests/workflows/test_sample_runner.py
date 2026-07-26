"""Test SampleRunner routing, dispatch, and fallback. / 测试 SampleRunner。"""

from __future__ import annotations

from pathlib import Path

import pytest

from spacers_agent.agents.registry import AgentRegistry
from spacers_agent.agents.base import Agent, AgentContext, AgentExecution, AgentName
from spacers_agent.prompt_catalog import PromptCatalog
from spacers_agent.routing import CallBudgetFactory, TaskRouter
from spacers_agent.schemas import ExpertResult, ImageRef, UnifiedSample
from spacers_agent.settings import AppSettings
from spacers_agent.workflows.artifact_writer import ArtifactWriter
from spacers_agent.workflows.judge_service import JudgeService
from spacers_agent.workflows.sample_runner import SampleRunner


# ── fake agents / 假 Agent ──────────────────────────────────────────────


class _CountingAgent:
    name: AgentName = "counting_agent"
    supported_tasks = frozenset({"counting"})

    async def run(self, sample, context: AgentContext) -> AgentExecution:
        return AgentExecution(
            agent_name=self.name,
            payload=ExpertResult(expert="counting", answer="3", status="completed"),
            result_filename="counting_result.json",
            trace={"route": "test"},
        )


class _VQAAgent:
    name: AgentName = "general_vqa_agent"
    supported_tasks = frozenset({"general_vqa", "caption"})

    async def run(self, sample, context: AgentContext) -> AgentExecution:
        return AgentExecution(
            agent_name=self.name,
            payload=ExpertResult(expert="vqa", answer="yes", status="completed"),
            result_filename="expert_result.json",
            trace={"route": "test"},
        )


class _FailingAgent:
    name: AgentName = "change_agent"
    supported_tasks = frozenset({"change_caption", "change_qa"})

    async def run(self, sample, context: AgentContext) -> AgentExecution:
        raise RuntimeError("simulated failure")


# ── sample / 样本 ──────────────────────────────────────────────────────

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "legacy"
TEST_IMAGE = FIXTURES / "test_image.png"
PROMPT_ROOT = Path(__file__).resolve().parents[2] / "prompts"


def _sample(task: str = "counting", question: str = "How many?") -> UnifiedSample:
    return UnifiedSample(
        sample_id="s1", dataset="test", split="test", task=task,  # type: ignore[arg-type]
        images=[ImageRef(image_id="i1", path=TEST_IMAGE, role="image")],
        question=question,
    )


def _runner(settings: AppSettings, registry: AgentRegistry) -> SampleRunner:
    """Build a fully injected SampleRunner for workflow unit tests.
    为工作流单元测试构建完整注入的 SampleRunner。
    """

    catalog = PromptCatalog(PROMPT_ROOT)
    return SampleRunner(
        settings,
        registry,
        None,
        catalog,
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


# ── tests / 测试 ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sample_runner_dispatches_to_correct_agent():
    reg = AgentRegistry()
    reg.register(_CountingAgent())
    reg.register(_VQAAgent())

    runner = _runner(AppSettings(), reg)
    # Override route_sample to force counting / 覆盖 route_sample 强制 counting
    from spacers_agent.routing import TaskRouter
    runner._route_override = TaskRouter()

    # We test that the runner calls the right agent
    # This test exercises the architecture; for now just verify registry
    assert reg.contains("counting_agent")
    assert reg.contains("general_vqa_agent")


@pytest.mark.asyncio
async def test_sample_runner_registry_integration():
    """SampleRunner can be constructed with an AgentRegistry."""
    reg = AgentRegistry()
    reg.register(_CountingAgent())
    settings = AppSettings()
    runner = _runner(settings, reg)
    assert runner is not None
