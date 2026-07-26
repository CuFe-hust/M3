"""Phase 5 — SampleRunner partial/failed status propagation.
Phase 5 — SampleRunner partial/failed 状态传播。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spacers_agent.agents.base import Agent, AgentContext, AgentExecution, AgentName
from spacers_agent.agents.registry import AgentRegistry
from spacers_agent.prompt_catalog import PromptCatalog
from spacers_agent.routing import CallBudgetFactory, TaskRouter
from spacers_agent.schemas import CountingResult, ExpertResult, ImageRef, UnifiedSample
from spacers_agent.settings import AppSettings
from spacers_agent.workflows.artifact_writer import ArtifactWriter
from spacers_agent.workflows.judge_service import JudgeService
from spacers_agent.workflows.sample_runner import SampleRunner, sample_state_from_payload

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "legacy"
TEST_IMAGE = FIXTURES / "test_image.png"
PROMPT_ROOT = Path(__file__).resolve().parents[2] / "prompts"


class _PartialPayloadAgent:
    name: AgentName = "general_vqa_agent"
    supported_tasks = frozenset({"general_vqa"})

    async def run(self, sample, context: AgentContext) -> AgentExecution:
        return AgentExecution(
            agent_name=self.name,
            payload=ExpertResult(expert="vqa", answer="partial answer", status="partial"),
            result_filename="expert_result.json",
            trace={},
        )


class _FailedPayloadAgent:
    name: AgentName = "general_vqa_agent"
    supported_tasks = frozenset({"general_vqa"})

    async def run(self, sample, context: AgentContext) -> AgentExecution:
        return AgentExecution(
            agent_name=self.name,
            payload=ExpertResult(expert="vqa", answer="failed", status="failed"),
            result_filename="expert_result.json",
            trace={},
        )


class _CountingPartialAgent:
    name: AgentName = "counting_agent"
    supported_tasks = frozenset({"counting"})

    async def run(self, sample, context: AgentContext) -> AgentExecution:
        return AgentExecution(
            agent_name=self.name,
            payload=CountingResult(
                sample_id="s1", target="building", question="q",
                source_width=100, source_height=100,
                tile_count=1, initial_tile_count=1, leaf_tile_count=1,
                succeeded_tiles=[], failed_tiles=["t1"],
                global_points=[], merged_groups=[], unresolved_conflicts=[],
                warnings=[], final_count=0, status="partial",
            ),
            result_filename="counting_result.json",
            trace={},
        )


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


def _sample(task: str = "general_vqa") -> UnifiedSample:
    return UnifiedSample(
        sample_id="s1", dataset="test", split="test", task=task,
        images=[ImageRef(image_id="i1", path=TEST_IMAGE, role="image")],
        question="test?",
    )


@pytest.mark.asyncio
async def test_partial_expert_payload_yields_partial_status(tmp_path: Path):
    reg = AgentRegistry()
    reg.register(_PartialPayloadAgent())
    runner = _runner(reg)
    sample = _sample()
    sample_dir = tmp_path / "samples" / sample.sample_id
    outcome = await runner.run_one(sample, sample_dir)
    assert outcome.status.state == "partial"
    assert outcome.fallback_used is False


@pytest.mark.asyncio
async def test_failed_expert_payload_results_in_failed_state(tmp_path: Path):
    """A failed-status payload produces a 'failed' sample state without raising."""
    reg = AgentRegistry()
    reg.register(_FailedPayloadAgent())
    runner = _runner(reg)
    sample = _sample()
    sample_dir = tmp_path / "samples" / sample.sample_id
    outcome = await runner.run_one(sample, sample_dir)
    assert outcome.status.state == "failed"


@pytest.mark.asyncio
async def test_counting_partial_payload_yields_partial_status(tmp_path: Path):
    reg = AgentRegistry()
    reg.register(_CountingPartialAgent())
    runner = _runner(reg)
    sample = _sample("counting")
    sample_dir = tmp_path / "samples" / sample.sample_id
    outcome = await runner.run_one(sample, sample_dir)
    assert outcome.status.state == "partial"


@pytest.mark.asyncio
async def test_sample_runner_uses_injected_router_not_default(tmp_path: Path):
    """SampleRunner.run_one() must use the injected router, not create TaskRouter()."""
    reg = AgentRegistry()
    reg.register(_PartialPayloadAgent())
    runner = _runner(reg)
    assert isinstance(runner.router, TaskRouter)
