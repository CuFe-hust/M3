"""Contract tests for the v4 SampleRunner execution kernel.

SampleRunner 契约测试：它消费已物化的 VisualTaskPlan v4 与 visual views，
验证确定性路由、fallback、artifact basename、稳定错误码和评测行为。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import pytest

from agents.base import AgentExecution
from agents.counting.target_parser import CountTargetResolutionError
from agents.errors import AgentTaskMismatchError
from agents.registry import AgentRegistry
from agents.schema import AgentResult, MaterializedVisualView, VisualTaskPlan
from data.schema import GroundTruth, ImageRef, UnifiedSample
from routing.router import TaskRouter
from workflows.artifact_writer import ArtifactWriter
from workflows.call_budget import CallBudgetFactory
from workflows.sample_runner import SampleRunner, sample_state_from_payload


class _FakeAgent:
    def __init__(
        self,
        name: str,
        tasks: tuple[str, ...],
        *,
        answer: str = "ok",
        status: str = "completed",
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.supported_tasks = frozenset(tasks)
        self.answer = answer
        self.status = status
        self.error = error
        self.calls: list[tuple[UnifiedSample, object]] = []

    async def run(self, sample: UnifiedSample, context: object) -> AgentExecution:
        self.calls.append((sample, context))
        if sample.task not in self.supported_tasks:
            raise AgentTaskMismatchError(self.name, sample.task, supported=self.supported_tasks)
        if self.error is not None:
            raise self.error
        return AgentExecution(
            agent_name=self.name,
            payload=AgentResult(
                agent_name=self.name,
                answer=self.answer,
                status=self.status,  # type: ignore[arg-type]
            ),
            result_filename="agent_result.json",
        )


def _sample(*, task: str = "general_vqa", sample_id: str = "s1") -> UnifiedSample:
    images = (
        [
            ImageRef(image_id="t1", path="t1.png", role="t1"),
            ImageRef(image_id="t2", path="t2.png", role="t2"),
        ]
        if task in {"change_caption", "change_qa"}
        else [ImageRef(image_id="img1", path="img.png", role="image")]
    )
    return UnifiedSample(
        sample_id=sample_id,
        dataset="demo",
        split="test",
        task=task,  # type: ignore[arg-type]
        images=images,
        question="Is there a road?",
        ground_truth=GroundTruth(answers=["ok"]),
    )


def _plan(task: str = "general_vqa") -> VisualTaskPlan:
    return VisualTaskPlan(
        version="visual-task-plan-v5",
        task=task,  # type: ignore[arg-type]
        reason_codes=["test"],
    )


def _view() -> MaterializedVisualView:
    return MaterializedVisualView(
        image_id="img1",
        view_mode="full_image",
        source_size=(100, 80),
        crop_xyxy=(0, 0, 100, 80),
        crop_size=(100, 80),
    )


def _runner(agents: list[_FakeAgent], *, fallback_on_partial: bool = False) -> SampleRunner:
    registry = AgentRegistry()
    for agent in agents:
        registry.register(agent)
    return SampleRunner(
        registry=registry,
        router=TaskRouter(),
        qwen_client=object(),  # fake agents never touch the client
        artifact_writer=ArtifactWriter(),
        call_budget_factory=CallBudgetFactory(),
        fallback_on_partial=fallback_on_partial,
        data_root=Path("/data"),
    )


def _run(
    runner: SampleRunner,
    sample: UnifiedSample,
    sample_dir: Path,
    *,
    plan: VisualTaskPlan | None = None,
    views: tuple[MaterializedVisualView, ...] = (),
    evaluate: bool = True,
):
    return asyncio.run(
        runner.run_one(
            sample,
            sample_dir,
            visual_task_plan=plan,
            visual_views=views,
            judge_policy="none",
            evaluate=evaluate,
        )
    )


def test_v4_success_writes_only_canonical_plan_artifact(tmp_path: Path) -> None:
    agent = _FakeAgent("general_vqa_agent", ("general_vqa",), answer="ok")
    runner = _runner([agent])
    sample_dir = tmp_path / "sample"
    outcome = _run(runner, _sample(), sample_dir, plan=_plan(), views=(_view(),))

    assert outcome.status.state == "succeeded"
    assert outcome.status.result_path == Path("agent_result.json")
    assert (sample_dir / "visual_task_plan.json").is_file()
    assert not (sample_dir / "visual_plan.json").exists()
    assert not (sample_dir / "joint_visual_plan.json").exists()
    trace = json.loads((sample_dir / "agent_trace.json").read_text(encoding="utf-8"))
    assert trace["planning_mode"] == "visual-task-plan-v5"
    assert trace["resolution_source"] == "visual-task-plan-v5"
    assert "low_confidence" not in trace
    assert trace["candidate_tasks"] == ["general_vqa"]
    assert agent.calls[0][1].visual_task_plan.version == "visual-task-plan-v5"
    assert agent.calls[0][1].visual_views == (_view(),)


def test_v4_plan_task_mismatch_fails_before_execution(tmp_path: Path) -> None:
    agent = _FakeAgent("general_vqa_agent", ("general_vqa",))
    with pytest.raises(ValueError, match="must equal"):
        _run(
            _runner([agent]),
            _sample(),
            tmp_path / "sample",
            plan=_plan("caption"),
            views=(_view(),),
        )
    assert agent.calls == []


def test_declared_router_fallback_rebuilds_only_change_qa_task(tmp_path: Path) -> None:
    primary = _FakeAgent("change_agent", ("change_qa",), error=RuntimeError("boom"))
    fallback = _FakeAgent("general_vqa_agent", ("general_vqa",), answer="ok")
    sample = _sample(task="change_qa")
    outcome = _run(
        _runner([primary, fallback]),
        sample,
        tmp_path / "sample",
    )
    assert outcome.status.state == "succeeded"
    assert outcome.fallback_used is True
    assert fallback.calls[0][0].task == "general_vqa"
    assert sample.task == "change_qa"


def test_partial_policy_uses_declared_fallback(tmp_path: Path) -> None:
    primary = _FakeAgent("change_agent", ("change_qa",), status="partial")
    fallback = _FakeAgent("general_vqa_agent", ("general_vqa",), answer="ok")
    outcome = _run(
        _runner([primary, fallback], fallback_on_partial=True),
        _sample(task="change_qa"),
        tmp_path / "sample",
    )
    assert outcome.status.state == "succeeded"
    assert outcome.fallback_used is True
    assert len(primary.calls) == 1 and len(fallback.calls) == 1


def test_failure_artifacts_contain_stable_code_only(tmp_path: Path) -> None:
    agent = _FakeAgent(
        "general_vqa_agent",
        ("general_vqa",),
        error=RuntimeError("/private/model.pt sk-secret"),
    )
    outcome = _run(_runner([agent]), _sample(), tmp_path / "sample")
    assert outcome.status.state == "failed"
    assert outcome.status.error_code == "RuntimeError"
    trace = json.loads((tmp_path / "sample" / "agent_trace.json").read_text(encoding="utf-8"))
    text = json.dumps(trace)
    assert "/private/model.pt" not in text
    assert "sk-secret" not in text


def test_count_target_error_code_survives_persisted_sample_status(tmp_path: Path) -> None:
    agent = _FakeAgent(
        "counting_agent",
        ("counting",),
        error=CountTargetResolutionError("COUNT_TARGET_CONFLICT"),
    )
    outcome = _run(
        _runner([agent]),
        _sample(task="counting"),
        tmp_path / "sample",
    )
    assert outcome.status.state == "failed"
    assert outcome.status.error_code == "COUNT_TARGET_CONFLICT"
    persisted = json.loads(
        (tmp_path / "sample" / "status.json").read_text(encoding="utf-8")
    )
    assert persisted["error_code"] == "COUNT_TARGET_CONFLICT"


def test_sample_state_mapping_is_closed() -> None:
    for status, expected in {
        "completed": "succeeded",
        "completed_with_warnings": "succeeded",
        "partial": "partial",
        "failed": "failed",
        "unexpected": "failed",
    }.items():
        payload = SimpleNamespace(status=status)
        assert sample_state_from_payload(payload) == expected
