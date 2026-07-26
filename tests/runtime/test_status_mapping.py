"""Payload-to-sample status mapping tests. / 载荷到样本状态的映射测试。"""

from pathlib import Path

import pytest

from spacers_agent.agents.base import AgentExecution
from spacers_agent.agents.registry import AgentRegistry
from spacers_agent.dataset_adapters import AdapterProbe
from spacers_agent.prompt_catalog import PromptCatalog
from spacers_agent.routing import CallBudgetFactory, TaskRouter
from spacers_agent.schemas import CountingResult, ExpertResult, ImageRef, UnifiedSample
from spacers_agent.settings import AppSettings
from spacers_agent.workflows.artifact_writer import ArtifactWriter
from spacers_agent.workflows.dataset_runner import DatasetRunner
from spacers_agent.workflows.sample_runner import SampleRunOutcome, SampleRunner, sample_state_from_payload


def _counting(status: str) -> CountingResult:
    failed_tiles = ["tile-1"] if status in {"partial", "failed"} else []
    return CountingResult(
        sample_id="sample-1",
        target="building",
        question="How many buildings?",
        source_width=8,
        source_height=8,
        tile_count=1,
        succeeded_tiles=[] if failed_tiles else ["tile-1"],
        failed_tiles=failed_tiles,
        final_count=0,
        status=status,
    )


@pytest.mark.parametrize(
    ("payload_status", "expected"),
    [
        ("completed", "succeeded"),
        ("completed_with_warnings", "succeeded"),
        ("partial", "partial"),
        ("failed", "failed"),
    ],
)
def test_counting_status_mapping(payload_status: str, expected: str) -> None:
    assert sample_state_from_payload(_counting(payload_status)) == expected


@pytest.mark.parametrize(
    ("payload_status", "expected"),
    [("completed", "succeeded"), ("partial", "partial"), ("failed", "failed")],
)
def test_expert_status_mapping(payload_status: str, expected: str) -> None:
    payload = ExpertResult(expert="general_vqa_expert", answer="answer", status=payload_status)
    assert sample_state_from_payload(payload) == expected


class _Adapter:
    name = "status-test"

    def __init__(self, sample: UnifiedSample) -> None:
        self.sample = sample

    def probe(self, root: Path) -> AdapterProbe:
        return AdapterProbe(self.name, "1", root / "samples.json", ("sample_id",), 1)

    def iter_samples(self, root: Path, split: str, task: str):
        yield self.sample


class _OutcomeRunner:
    def __init__(self, outcome: SampleRunOutcome) -> None:
        self.outcome = outcome
        self.artifact_writer = ArtifactWriter()
        self.judge_service = type("Judge", (), {"judge_client": None})()

    async def run_one(self, sample, sample_dir, *, judge_policy):
        return self.outcome


@pytest.mark.asyncio
async def test_dataset_runner_uses_outcome_status_without_default_success(tmp_path: Path) -> None:
    sample = UnifiedSample(
        sample_id="sample-1",
        dataset="status-test",
        split="test",
        task="general_vqa",
        images=[ImageRef(image_id="image-1", path=tmp_path / "image.png", role="image")],
        question="What is visible?",
    )
    payload = ExpertResult(expert="general_vqa_expert", answer="partial", status="partial")
    execution = AgentExecution(
        agent_name="general_vqa_agent",
        payload=payload,
        result_filename="expert_result.json",
    )
    from spacers_agent.schemas import SampleRunStatus

    status = SampleRunStatus(
        sample_id=sample.sample_id,
        task=sample.task,
        state="partial",
        result_path=tmp_path / "expert_result.json",
        updated_at="2026-07-26T00:00:00+00:00",
    )
    outcome = SampleRunOutcome(execution, status, None, None, False)
    settings = AppSettings.model_validate({"paths": {"dataset_root": tmp_path}})
    runner = DatasetRunner(
        _Adapter(sample),
        _OutcomeRunner(outcome),
        run_dir=tmp_path / "run",
        settings=settings,
    )

    summary = await runner.run(split="test", task="general_vqa")

    assert summary.succeeded == 0
    assert summary.partial == 1


class _CompletedAgent:
    name = "general_vqa_agent"
    supported_tasks = frozenset({"general_vqa"})

    async def run(self, sample, context):
        return AgentExecution(
            agent_name=self.name,
            payload=ExpertResult(
                expert="general_vqa_expert",
                answer="retained answer",
                status="completed",
            ),
            result_filename="expert_result.json",
        )


class _ExplodingJudgeService:
    judge_client = object()

    async def judge_vqa(self, **kwargs):
        raise RuntimeError("visible judge failure")


@pytest.mark.asyncio
async def test_judge_error_is_visible_without_deleting_agent_result(tmp_path: Path) -> None:
    sample = UnifiedSample(
        sample_id="judge-error",
        dataset="test",
        split="test",
        task="general_vqa",
        images=[ImageRef(image_id="image-1", path=tmp_path / "image.png", role="image")],
        question="What is visible?",
    )
    registry = AgentRegistry()
    registry.register(_CompletedAgent())
    settings = AppSettings()
    catalog = PromptCatalog(Path(__file__).resolve().parents[2] / "prompts")
    runner = SampleRunner(
        settings,
        registry,
        object(),
        catalog,
        router=TaskRouter(),
        judge_service=_ExplodingJudgeService(),
        artifact_writer=ArtifactWriter(),
        call_budget_factory=CallBudgetFactory(),
    )

    outcome = await runner.run_one(sample, tmp_path / "sample", judge_policy="all")

    assert outcome.status.state == "succeeded"
    assert (tmp_path / "sample" / "expert_result.json").is_file()
    evaluation = outcome.evaluation
    assert isinstance(evaluation, dict)
    assert evaluation["judge_status"] == "failed"
    assert "visible judge failure" in evaluation["judge_error"]
