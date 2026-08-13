"""Contract tests for SampleRunner: routing, candidate fallback, agent
fallback, partial policy, shared budget, evaluation, and optional judge.

SampleRunner 契约测试：路由、候选兜底、Agent 兜底、partial 策略、共享预算、
评测与可选 judge。所有测试离线：使用注入的 fake Agent 与 fake judge client。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from agents.base import AgentExecution, VisualPlanBindings
from agents.counting.schema import CountingResult, GlobalPointObservation
from agents.errors import AgentTaskMismatchError
from agents.registry import AgentRegistry
from agents.schema import (
    AgentResult,
    FirstQwenVisualPlan,
    JointQwenVisualPlan,
    RoiPlan,
)
from data.schema import GroundTruth, ImageRef, TaskNormalization, UnifiedSample
from evaluation.judges.base import VQAAnswerJudgeResult
from evaluation.records import EvaluationRecord
from routing.router import TaskRouter
from routing.schema import TaskResolution
from workflows.artifact_writer import ArtifactWriter
from workflows.call_budget import CallBudget, CallBudgetFactory
from workflows.judge_service import JudgeService
from workflows.sample_runner import SampleRunner, sample_state_from_payload
from workflows.schema import SampleRunOutcome
from workflows.visual_planner import VisualPlanError, VisualPlanningGate


# ── helpers / 测试辅助 ──────────────────────────────────────────────────────


class _DummyClient:
    """Placeholder VisionLanguageClient; fake agents never touch it.
    占位 VisionLanguageClient；fake Agent 绝不使用它。"""

    async def complete_json(self, **kwargs):
        raise AssertionError("dummy client must not be called")


class _FakeAgent:
    """Protocol-compatible fake agent with a configurable outcome.
    可配置结果的协议兼容 fake Agent。"""

    def __init__(
        self,
        name: str,
        tasks: tuple[str, ...],
        *,
        payload: Any | None = None,
        error: Exception | None = None,
        status: str = "completed",
        reserve_qwen: int = 0,
        result_filename: str = "agent_result.json",
    ) -> None:
        self.name = name
        self.supported_tasks = frozenset(tasks)
        self._payload = payload
        self._error = error
        self._status = status
        self._reserve_qwen = reserve_qwen
        self._result_filename = result_filename
        self.calls: list[tuple[UnifiedSample, object]] = []

    async def run(self, sample: UnifiedSample, context: object) -> AgentExecution:
        self.calls.append((sample, context))
        for _ in range(self._reserve_qwen):
            context.call_budget.reserve_qwen()  # type: ignore[attr-defined]
        if self._error is not None:
            raise self._error
        payload = self._payload
        if payload is None:
            payload = AgentResult(agent_name=self.name, answer="ok", status=self._status)
        return AgentExecution(
            agent_name=self.name,
            payload=payload,
            result_filename=self._result_filename,
        )


class _TaskCheckedAgent(_FakeAgent):
    """Fake that enforces the same fail-closed task guard as real agents.
    与真实 Agent 一样执行 fail-closed task guard 的 fake。"""

    async def run(self, sample: UnifiedSample, context: object) -> AgentExecution:
        if sample.task not in self.supported_tasks:
            raise AgentTaskMismatchError(
                self.name,
                sample.task,
                supported=self.supported_tasks,
            )
        return await super().run(sample, context)


class _FakeJudgeClient:
    """Records calls and returns or raises a configured outcome.
    记录调用并返回/抛出配置结果的 judge client fake。"""

    def __init__(self, verdict=None, error: Exception | None = None) -> None:
        self.verdict = verdict
        self.error = error
        self.calls = 0

    def judge(self, payload, *, request_meta):
        return self.judge_json(
            payload,
            response_model=type(self.verdict),
            request_meta=request_meta,
        )

    def judge_json(self, payload, *, response_model, request_meta, system_prompt=None):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return response_model.model_validate(self.verdict.model_dump())


def _image(image_id: str, path: str, role: str) -> ImageRef:
    return ImageRef(image_id=image_id, path=path, role=role)  # type: ignore[arg-type]


def _sample(
    *,
    task: str = "general_vqa",
    sample_id: str = "s1",
    question: str = "Is there a road?",
    answers: list[str] | None = None,
    images: list[ImageRef] | None = None,
    normalization: TaskNormalization | None = None,
    ground_truth: GroundTruth | None = None,
) -> UnifiedSample:
    if images is None:
        images = [_image("i0", "img0.png", "image")]
    return UnifiedSample(
        sample_id=sample_id,
        dataset="parity",
        split="test",
        task=task,  # type: ignore[arg-type]
        images=images,
        question=question,
        ground_truth=ground_truth or GroundTruth(answers=answers or ["ok"]),
        normalization=normalization,
    )


def _change_sample() -> UnifiedSample:
    return _sample(
        task="change_qa",
        sample_id="change-1",
        question="What changed between the two images?",
        answers=["road added"],
        images=[
            _image("i0", "t1.png", "t1"),
            _image("i1", "t2.png", "t2"),
        ],
    )


def _counting_result(final_count: int = 2) -> CountingResult:
    points = []
    for index in range(final_count):
        points.append(
            GlobalPointObservation(
                global_id=f"p{index}",
                target="car",
                source_tile_id="t0",
                local_id=f"p{index}",
                local_x_norm=100 + index,
                local_y_norm=100,
                local_radius_norm=5,
                global_x_px=100 + index,
                global_y_px=100,
                global_x_norm=100 + index,
                global_y_norm=100,
                radius_px=5.0,
                confidence=0.9,
                ownership_valid=True,
                near_core_boundary=False,
                accepted=True,
                short_evidence="visible",
            )
        )
    return CountingResult(
        sample_id="s1",
        target="car",
        question="How many cars?",
        source_width=1000,
        source_height=1000,
        tile_count=1,
        succeeded_tiles=["t0"],
        failed_tiles=[],
        global_points=points,
        merged_groups=[],
        unresolved_conflicts=[],
        final_count=final_count,
        status="completed",
    )


def _runner(
    agents: list[_FakeAgent],
    *,
    judge_service: JudgeService | None = None,
    fallback_on_partial: bool = False,
    router: TaskRouter | None = None,
    visual_planning: VisualPlanningGate | None = None,
    joint_bindings: VisualPlanBindings | None = None,
    data_root: Path | None = None,
) -> SampleRunner:
    registry = AgentRegistry()
    for agent in agents:
        registry.register(agent)
    return SampleRunner(
        registry=registry,
        router=router or TaskRouter(),
        qwen_client=_DummyClient(),
        artifact_writer=ArtifactWriter(),
        call_budget_factory=CallBudgetFactory(),
        judge_service=judge_service,
        fallback_on_partial=fallback_on_partial,
        visual_planning=visual_planning,
        joint_bindings=joint_bindings,
        data_root=data_root,
    )


def _run(
    runner: SampleRunner,
    sample: UnifiedSample,
    sample_dir: Path,
    *,
    resolution: TaskResolution | None = None,
    joint_plan: JointQwenVisualPlan | None = None,
    judge_policy: str = "none",
    budget: CallBudget | None = None,
) -> SampleRunOutcome:
    return asyncio.run(
        runner.run_one(
            sample,
            sample_dir,
            resolution=resolution,
            joint_plan=joint_plan,
            judge_policy=judge_policy,
            budget=budget,
        )
    )


def _resolution(
    task: str,
    candidates: list[str],
    *,
    low_confidence: bool = True,
    source: str = "model",
) -> TaskResolution:
    return TaskResolution(
        task=task,  # type: ignore[arg-type]
        confidence=0.4 if low_confidence else 0.95,
        candidate_tasks=candidates,  # type: ignore[arg-type]
        needs_candidate_fallback=low_confidence,
        source=source,  # type: ignore[arg-type]
        reason_codes=["low_confidence" if low_confidence else "model_high_confidence"],
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_dir(tmp_path: Path, name: str = "s1") -> Path:
    return tmp_path / "samples" / name


# ── primary success / 主路径成功 ────────────────────────────────────────────


def test_primary_success_writes_all_artifacts(tmp_path: Path) -> None:
    agent = _FakeAgent("general_vqa_agent", ("general_vqa",))
    runner = _runner([agent])
    sample = _sample(answers=["ok"])
    outcome = _run(runner, sample, _sample_dir(tmp_path))
    assert outcome.status.state == "succeeded"
    assert outcome.status.result_path is not None
    assert outcome.routing is not None
    assert outcome.routing.primary_agent == "general_vqa_agent"
    assert isinstance(outcome.evaluation, EvaluationRecord)
    assert outcome.evaluation.judge_status == "not_requested"
    assert outcome.evaluation.deterministic_metrics is not None
    assert outcome.evaluation.deterministic_metrics.exact_match is True
    assert outcome.fallback_used is False
    assert len(agent.calls) == 1
    directory = _sample_dir(tmp_path)
    assert (directory / "sample.json").is_file()
    assert (directory / "routing_decision.json").is_file()
    assert (directory / "agent_result.json").is_file()
    assert (directory / "agent_trace.json").is_file()
    assert (directory / "vqa_evaluation.json").is_file()
    status = _read_json(directory / "status.json")
    assert status["state"] == "succeeded"
    assert status["error_code"] is None
    trace = _read_json(directory / "agent_trace.json")
    assert trace["router_used"] is True
    assert trace["task_type"] == "general_vqa"
    assert trace["resolution_source"] == "dataset_task"
    assert trace["judge_status"] == "not_requested"
    assert trace["execution_agent"] == "general_vqa_agent"


def test_sample_state_mapping() -> None:
    assert sample_state_from_payload(_counting_result()) == "succeeded"
    assert (
        sample_state_from_payload(
            AgentResult(agent_name="general_vqa_agent", answer="x", status="partial")
        )
        == "partial"
    )
    assert (
        sample_state_from_payload(
            AgentResult(agent_name="general_vqa_agent", answer="x", status="failed")
        )
        == "failed"
    )


def test_run_one_accepts_external_shared_budget(tmp_path: Path) -> None:
    agent = _FakeAgent("general_vqa_agent", ("general_vqa",), reserve_qwen=2)
    runner = _runner([agent])
    external = CallBudget(max_qwen_calls=10, max_deepseek_calls=0)
    outcome = _run(runner, _sample(), _sample_dir(tmp_path), budget=external)
    assert outcome.status.state == "succeeded"
    recorded = agent.calls[0][1].call_budget
    assert recorded is external
    assert external.qwen_calls_used == 2


# ── routing fallback / 路由兜底 ─────────────────────────────────────────────


def test_routing_fallback_on_primary_exception(tmp_path: Path) -> None:
    change_agent = _FakeAgent(
        "change_agent", ("change_qa", "change_caption"), error=RuntimeError("boom")
    )
    vqa_agent = _FakeAgent("general_vqa_agent", ("general_vqa",))
    runner = _runner([change_agent, vqa_agent])
    outcome = _run(runner, _change_sample(), _sample_dir(tmp_path))
    assert outcome.status.state == "succeeded"
    assert outcome.fallback_used is True
    assert len(change_agent.calls) == 1
    assert len(vqa_agent.calls) == 1
    trace = _read_json(_sample_dir(tmp_path) / "agent_trace.json")
    assert trace["fallback_used"] is True
    assert trace["fallback_agents"] == ["general_vqa_agent"]
    assert trace["execution_mode"] == "fallback"


def test_change_qa_real_fallback_rematerializes_general_vqa(tmp_path: Path) -> None:
    original = _change_sample()
    original_payload = original.model_dump(mode="json")
    change_agent = _TaskCheckedAgent(
        "change_agent",
        ("change_qa", "change_caption"),
        error=RuntimeError("primary failed"),
    )
    generic_agent = _TaskCheckedAgent(
        "general_vqa_agent",
        ("general_vqa",),
    )
    runner = _runner([change_agent, generic_agent])

    outcome = _run(runner, original, _sample_dir(tmp_path))

    assert outcome.status.state == "succeeded"
    assert outcome.fallback_used is True
    assert len(generic_agent.calls) == 1
    fallback_sample = generic_agent.calls[0][0]
    assert fallback_sample.task == "general_vqa"
    assert [item.role for item in fallback_sample.images] == ["image", "context"]
    assert fallback_sample.normalization is None
    assert original.model_dump(mode="json") == original_payload
    trace = _read_json(_sample_dir(tmp_path) / "agent_trace.json")
    assert trace["resolved_task"] == "change_qa"
    assert trace["execution_task"] == "general_vqa"
    assert trace["execution_agent"] == "general_vqa_agent"
    assert trace["fallback_used"] is True
    assert trace["fallback_from_task"] == "change_qa"


def test_routing_fallback_not_run_when_primary_succeeds(tmp_path: Path) -> None:
    change_agent = _FakeAgent("change_agent", ("change_qa", "change_caption"))
    vqa_agent = _FakeAgent("general_vqa_agent", ("general_vqa",))
    runner = _runner([change_agent, vqa_agent])
    outcome = _run(runner, _change_sample(), _sample_dir(tmp_path))
    assert outcome.status.state == "succeeded"
    assert outcome.fallback_used is False
    assert len(change_agent.calls) == 1
    assert len(vqa_agent.calls) == 0
    trace = _read_json(_sample_dir(tmp_path) / "agent_trace.json")
    assert trace["execution_task"] == "change_qa"


# ── partial policy / partial 策略 ───────────────────────────────────────────


def test_partial_with_fallback_policy_runs_fallback(tmp_path: Path) -> None:
    change_agent = _FakeAgent(
        "change_agent", ("change_qa", "change_caption"), status="partial"
    )
    vqa_agent = _FakeAgent("general_vqa_agent", ("general_vqa",))
    runner = _runner([change_agent, vqa_agent], fallback_on_partial=True)
    outcome = _run(runner, _change_sample(), _sample_dir(tmp_path))
    assert outcome.status.state == "succeeded"
    assert outcome.fallback_used is True
    assert len(vqa_agent.calls) == 1
    trace = _read_json(_sample_dir(tmp_path) / "agent_trace.json")
    assert trace["primary_reason"] == "PRIMARY_PARTIAL"


def test_partial_without_policy_stays_partial(tmp_path: Path) -> None:
    change_agent = _FakeAgent(
        "change_agent", ("change_qa", "change_caption"), status="partial"
    )
    vqa_agent = _FakeAgent("general_vqa_agent", ("general_vqa",))
    runner = _runner([change_agent, vqa_agent])
    outcome = _run(runner, _change_sample(), _sample_dir(tmp_path))
    assert outcome.status.state == "partial"
    assert outcome.fallback_used is False
    assert len(vqa_agent.calls) == 0


# ── low-confidence candidates / 低置信度候选 ────────────────────────────────


def test_low_confidence_candidate_plan_dedups_agents(tmp_path: Path) -> None:
    agent = _FakeAgent(
        "general_vqa_agent",
        ("general_vqa", "scene_classification", "multiple_choice_vqa"),
    )
    runner = _runner([agent])
    sample = _sample()
    resolution = _resolution(
        "general_vqa",
        ["general_vqa", "scene_classification", "multiple_choice_vqa"],
    )
    outcome = _run(runner, sample, _sample_dir(tmp_path), resolution=resolution)
    assert outcome.status.state == "succeeded"
    assert len(agent.calls) == 1  # all three tasks route to one agent / 三任务同一 Agent
    trace = _read_json(_sample_dir(tmp_path) / "agent_trace.json")
    assert trace["low_confidence"] is True
    assert trace["resolution_source"] == "model"
    assert trace["candidate_tasks"] == ["general_vqa"]
    assert trace["attempt_agents"] == [["general_vqa_agent"]]


def test_candidate_fallback_runs_next_task_after_primary_failure(tmp_path: Path) -> None:
    vqa_agent = _FakeAgent(
        "general_vqa_agent", ("general_vqa",), error=RuntimeError("primary broke")
    )
    caption_agent = _FakeAgent("caption_agent", ("caption",))
    runner = _runner([vqa_agent, caption_agent])
    resolution = _resolution("general_vqa", ["general_vqa", "caption"])
    outcome = _run(
        runner, _sample(question="Describe the scene."), _sample_dir(tmp_path),
        resolution=resolution,
    )
    assert outcome.status.state == "succeeded"
    # Fix G: a candidate-task fallback counts as fallback_used.
    # Fix G：候选任务兜底计入 fallback_used。
    assert outcome.fallback_used is True
    assert len(vqa_agent.calls) == 1
    assert len(caption_agent.calls) == 1
    assert outcome.execution is not None
    assert outcome.execution.agent_name == "caption_agent"
    # Fix F: the routing artifact reflects the executed task, the trace keeps
    # both the resolved and the executed task.
    # Fix F：routing 产物反映实际执行任务，trace 同时保留解析与执行任务。
    routing = _read_json(_sample_dir(tmp_path) / "routing_decision.json")
    assert routing["primary_agent"] == "caption_agent"
    assert routing["task"] == "caption"
    trace = _read_json(_sample_dir(tmp_path) / "agent_trace.json")
    assert trace["resolved_task"] == "general_vqa"
    assert trace["execution_task"] == "caption"
    assert trace["task_type"] == "general_vqa"  # fixed semantics: resolved task
    assert trace["candidate_tasks"] == ["general_vqa", "caption"]
    assert trace["attempt_agents"] == [["general_vqa_agent"], ["caption_agent"]]
    assert trace["execution_agent"] == "caption_agent"


def test_failed_payload_status_continues_to_next_candidate(tmp_path: Path) -> None:
    vqa_agent = _FakeAgent(
        "general_vqa_agent", ("general_vqa",), status="failed"
    )
    caption_agent = _FakeAgent("caption_agent", ("caption",))
    runner = _runner([vqa_agent, caption_agent])
    resolution = _resolution("general_vqa", ["general_vqa", "caption"])
    outcome = _run(
        runner, _sample(question="Describe the scene."), _sample_dir(tmp_path),
        resolution=resolution,
    )
    assert outcome.status.state == "succeeded"
    assert outcome.execution is not None
    assert outcome.execution.agent_name == "caption_agent"


def test_high_confidence_runs_only_top_task(tmp_path: Path) -> None:
    vqa_agent = _FakeAgent("general_vqa_agent", ("general_vqa",))
    caption_agent = _FakeAgent("caption_agent", ("caption",))
    runner = _runner([vqa_agent, caption_agent])
    resolution = _resolution(
        "general_vqa", ["general_vqa", "caption"], low_confidence=False
    )
    outcome = _run(
        runner, _sample(question="Describe the scene."), _sample_dir(tmp_path),
        resolution=resolution,
    )
    assert outcome.status.state == "succeeded"
    assert len(vqa_agent.calls) == 1
    assert len(caption_agent.calls) == 0


# ── incompatible candidates / 不兼容候选 ────────────────────────────────────


def test_incompatible_change_candidate_is_skipped(tmp_path: Path) -> None:
    vqa_agent = _FakeAgent("general_vqa_agent", ("general_vqa",))
    change_agent = _FakeAgent("change_agent", ("change_caption", "change_qa"))
    runner = _runner([vqa_agent, change_agent])
    resolution = _resolution("general_vqa", ["general_vqa", "change_caption"])
    sample = _sample(question="Describe the scene.")
    outcome = _run(runner, sample, _sample_dir(tmp_path), resolution=resolution)
    assert outcome.status.state == "succeeded"
    assert len(change_agent.calls) == 0
    trace = _read_json(_sample_dir(tmp_path) / "agent_trace.json")
    assert {"task": "change_caption", "reason": "INCOMPATIBLE_SAMPLE"} in trace[
        "skipped_candidates"
    ]


def test_change_candidate_rebuilds_sample_without_mutating_original(tmp_path: Path) -> None:
    vqa_agent = _FakeAgent(
        "general_vqa_agent", ("general_vqa",), error=RuntimeError("fail")
    )
    change_agent = _FakeAgent("change_agent", ("change_qa",))
    runner = _runner([vqa_agent, change_agent])
    normalization = TaskNormalization(
        source_task="general_vqa",
        normalized_task="general_vqa",  # type: ignore[arg-type]
        normalizer="test",
        version="1",
    )
    sample = _sample(
        question="What changed?",
        images=[
            _image("i0", "img0.png", "image"),
            _image("i1", "img1.png", "context"),
        ],
        normalization=normalization,
    )
    original_dump = sample.model_dump(mode="json")
    resolution = _resolution("general_vqa", ["general_vqa", "change_qa"])
    outcome = _run(runner, sample, _sample_dir(tmp_path), resolution=resolution)
    assert outcome.status.state == "succeeded"
    assert len(change_agent.calls) == 1
    candidate_sample = change_agent.calls[0][0]
    assert candidate_sample.task == "change_qa"
    assert candidate_sample.normalization is None
    assert [image.role for image in candidate_sample.images] == ["t1", "t2"]
    # The original sample is untouched. / 原样本未被修改。
    assert sample.model_dump(mode="json") == original_dump
    assert sample.task == "general_vqa"
    assert sample.normalization is not None
    assert [image.role for image in sample.images] == ["image", "context"]


def test_no_executable_attempts_fails_with_stable_code(tmp_path: Path) -> None:
    change_agent = _FakeAgent("change_agent", ("change_caption", "change_qa"))
    runner = _runner([change_agent])
    resolution = _resolution("change_caption", ["change_caption", "change_qa"])
    sample = _sample(question="Describe the scene.")  # single image / 单图
    outcome = _run(runner, sample, _sample_dir(tmp_path), resolution=resolution)
    assert outcome.status.state == "failed"
    assert outcome.status.error_code == "NO_EXECUTABLE_ATTEMPTS"
    trace = _read_json(_sample_dir(tmp_path) / "agent_trace.json")
    assert trace["failure_code"] == "NO_EXECUTABLE_ATTEMPTS"
    assert len(change_agent.calls) == 0


def test_unroutable_candidate_is_skipped(tmp_path: Path) -> None:
    class _StrictRouter(TaskRouter):
        def route(self, task, *, capabilities=None):
            if task == "caption":
                raise KeyError("caption not routable")
            return super().route(task, capabilities=capabilities)

    vqa_agent = _FakeAgent("general_vqa_agent", ("general_vqa",))
    caption_agent = _FakeAgent("caption_agent", ("caption",))
    runner = _runner([vqa_agent, caption_agent], router=_StrictRouter())
    resolution = _resolution("general_vqa", ["general_vqa", "caption"])
    outcome = _run(
        runner, _sample(question="Describe the scene."), _sample_dir(tmp_path),
        resolution=resolution,
    )
    assert outcome.status.state == "succeeded"
    assert len(vqa_agent.calls) == 1
    assert len(caption_agent.calls) == 0
    trace = _read_json(_sample_dir(tmp_path) / "agent_trace.json")
    assert {"task": "caption", "reason": "UNROUTABLE_TASK"} in trace["skipped_candidates"]


# ── shared budget / 共享预算 ────────────────────────────────────────────────


def test_shared_budget_across_attempts_and_judge(tmp_path: Path) -> None:
    caption_agent = _FakeAgent(
        "caption_agent", ("caption",), error=RuntimeError("fail"), reserve_qwen=2
    )
    vqa_agent = _FakeAgent("general_vqa_agent", ("general_vqa",), reserve_qwen=2)
    judge_client = _FakeJudgeClient(verdict=VQAAnswerJudgeResult(score=1))
    judge_service = JudgeService(
        judge_prompt="counting prompt",
        judge_prompt_version="v1",
        vqa_judge_prompt="vqa prompt",
        vqa_judge_prompt_version="v2",
        judge_client=judge_client,
    )
    runner = _runner(
        [caption_agent, vqa_agent],
        judge_service=judge_service,
    )
    # caption 是 top task：先失败，随后 general_vqa 候选兜底成功并触发 judge。
    resolution = _resolution("caption", ["caption", "general_vqa"])
    sample = _sample(question="Describe the scene.")
    outcome = _run(
        runner,
        sample,
        _sample_dir(tmp_path),
        resolution=resolution,
        judge_policy="all",
    )
    assert outcome.status.state == "succeeded"
    budgets = [call[1].call_budget for call in caption_agent.calls + vqa_agent.calls]
    assert len(budgets) == 2
    assert budgets[0] is budgets[1]
    budget = budgets[0]
    # 2 qwen reservations per attempt, plus one deepseek reservation from the
    # VQA judge on the same budget object. / 每次尝试 2 次 qwen 预留，外加
    # VQA judge 在同一预算对象上的 1 次 deepseek 预留。
    assert budget.qwen_calls_used == 4
    assert budget.deepseek_calls_used == 1
    assert judge_client.calls == 1


# ── deterministic evaluation / 确定性评估 ───────────────────────────────────


def test_counting_deterministic_evaluation(tmp_path: Path) -> None:
    counting_agent = _FakeAgent(
        "counting_agent", ("counting",), payload=_counting_result(final_count=2)
    )
    runner = _runner([counting_agent])
    sample = _sample(
        task="counting",
        question="How many cars?",
        ground_truth=GroundTruth(count=2),
    )
    outcome = _run(runner, sample, _sample_dir(tmp_path))
    assert outcome.status.state == "succeeded"
    assert isinstance(outcome.evaluation, EvaluationRecord)
    assert outcome.evaluation.task == "counting"
    assert outcome.evaluation.deterministic_metrics is not None
    assert outcome.evaluation.deterministic_metrics.exact_match == 1
    evaluation = _read_json(_sample_dir(tmp_path) / "counting_evaluation.json")
    assert evaluation["task"] == "counting"
    assert evaluation["judge_status"] == "not_requested"


def test_counting_deterministic_evaluation_mismatch(tmp_path: Path) -> None:
    counting_agent = _FakeAgent(
        "counting_agent", ("counting",), payload=_counting_result(final_count=3)
    )
    runner = _runner([counting_agent])
    sample = _sample(
        task="counting",
        question="How many cars?",
        ground_truth=GroundTruth(count=2),
    )
    outcome = _run(runner, sample, _sample_dir(tmp_path))
    assert outcome.evaluation is not None
    assert outcome.evaluation.deterministic_metrics.exact_match == 0


def test_vqa_evaluation_without_judge_service(tmp_path: Path) -> None:
    agent = _FakeAgent("general_vqa_agent", ("general_vqa",))
    runner = _runner([agent])
    sample = _sample(answers=["yes"])
    outcome = _run(runner, sample, _sample_dir(tmp_path))
    assert isinstance(outcome.evaluation, EvaluationRecord)
    assert outcome.evaluation.judge_status == "not_requested"
    evaluation = _read_json(_sample_dir(tmp_path) / "vqa_evaluation.json")
    assert evaluation["task"] == "general_vqa"
    assert evaluation["judge_status"] == "not_requested"


# ── VQA judge / VQA 判卷 ────────────────────────────────────────────────────


def test_vqa_judge_succeeds(tmp_path: Path) -> None:
    judge_client = _FakeJudgeClient(verdict=VQAAnswerJudgeResult(score=1))
    judge_service = JudgeService(
        judge_prompt="counting prompt",
        judge_prompt_version="v1",
        vqa_judge_prompt="vqa prompt",
        vqa_judge_prompt_version="v2",
        judge_client=judge_client,
    )
    agent = _FakeAgent("general_vqa_agent", ("general_vqa",))
    runner = _runner([agent], judge_service=judge_service)
    outcome = _run(
        runner,
        _sample(answers=["yes"]),
        _sample_dir(tmp_path),
        judge_policy="all",
    )
    assert outcome.status.state == "succeeded"
    assert outcome.evaluation is not None
    assert outcome.evaluation.judge_status == "succeeded"
    assert outcome.evaluation.judge_parsed.score == 1
    evaluation = _read_json(_sample_dir(tmp_path) / "vqa_evaluation.json")
    assert evaluation["judge_status"] == "succeeded"
    trace = _read_json(_sample_dir(tmp_path) / "agent_trace.json")
    assert trace["judge_status"] == "succeeded"


def test_vqa_judge_failure_keeps_deterministic(tmp_path: Path) -> None:
    judge_client = _FakeJudgeClient(error=RuntimeError("secret-raw-detail"))
    judge_service = JudgeService(
        judge_prompt="counting prompt",
        judge_prompt_version="v1",
        vqa_judge_prompt="vqa prompt",
        vqa_judge_prompt_version="v2",
        judge_client=judge_client,
    )
    agent = _FakeAgent(
        "general_vqa_agent",
        ("general_vqa",),
        payload=AgentResult(agent_name="general_vqa_agent", answer="yes", status="completed"),
    )
    runner = _runner([agent], judge_service=judge_service)
    outcome = _run(
        runner,
        _sample(answers=["yes"]),
        _sample_dir(tmp_path),
        judge_policy="all",
    )
    assert outcome.status.state == "succeeded"  # judge never fails the sample
    assert outcome.evaluation is not None
    assert outcome.evaluation.judge_status == "failed"
    assert outcome.evaluation.judge_error == "RuntimeError"
    assert outcome.evaluation.deterministic_metrics is not None
    assert outcome.evaluation.deterministic_metrics.exact_match is True
    evaluation_text = (
        _sample_dir(tmp_path) / "vqa_evaluation.json"
    ).read_text(encoding="utf-8")
    assert "secret-raw-detail" not in evaluation_text


def test_vqa_judge_policy_none_records_not_requested(tmp_path: Path) -> None:
    judge_client = _FakeJudgeClient(verdict=VQAAnswerJudgeResult(score=1))
    judge_service = JudgeService(
        judge_prompt="counting prompt",
        judge_prompt_version="v1",
        vqa_judge_prompt="vqa prompt",
        vqa_judge_prompt_version="v2",
        judge_client=judge_client,
    )
    agent = _FakeAgent("general_vqa_agent", ("general_vqa",))
    runner = _runner([agent], judge_service=judge_service)
    outcome = _run(
        runner,
        _sample(answers=["yes"]),
        _sample_dir(tmp_path),
        judge_policy="none",
    )
    assert outcome.evaluation is not None
    assert outcome.evaluation.judge_status == "not_requested"
    assert judge_client.calls == 0


@pytest.mark.parametrize(
    ("task", "agent_name"),
    [
        ("general_vqa", "general_vqa_agent"),
        ("multiple_choice_vqa", "general_vqa_agent"),
        ("scene_classification", "general_vqa_agent"),
        ("spatial_relation", "general_vqa_agent"),
        ("change_qa", "change_agent"),
    ],
)
def test_every_vqa_family_mismatch_uses_semantic_judge(
    tmp_path: Path,
    task: str,
    agent_name: str,
) -> None:
    judge_client = _FakeJudgeClient(verdict=VQAAnswerJudgeResult(score=1))
    judge_service = JudgeService(
        judge_prompt="counting prompt",
        judge_prompt_version="v1",
        vqa_judge_prompt="vqa prompt",
        vqa_judge_prompt_version="v2",
        judge_client=judge_client,
    )
    agent = _FakeAgent(
        agent_name,
        (task,),
        payload=AgentResult(
            agent_name=agent_name,
            answer="semantic paraphrase",
            status="completed",
        ),
    )
    sample = (
        _change_sample()
        if task == "change_qa"
        else _sample(task=task, answers=["official answer"])
    )
    outcome = _run(
        _runner([agent], judge_service=judge_service),
        sample,
        _sample_dir(tmp_path),
        judge_policy="errors-only",
    )
    assert judge_client.calls == 1
    assert outcome.evaluation is not None
    assert outcome.evaluation.task == "general_vqa"
    assert outcome.evaluation.deterministic_metrics.exact_match is False
    assert outcome.evaluation.judge_status == "succeeded"


def test_non_vqa_family_never_uses_vqa_semantic_judge(tmp_path: Path) -> None:
    judge_client = _FakeJudgeClient(verdict=VQAAnswerJudgeResult(score=1))
    judge_service = JudgeService(
        judge_prompt="counting prompt",
        judge_prompt_version="v1",
        vqa_judge_prompt="vqa prompt",
        vqa_judge_prompt_version="v2",
        judge_client=judge_client,
    )
    agent = _FakeAgent(
        "caption_agent",
        ("caption",),
        payload=AgentResult(
            agent_name="caption_agent", answer="candidate", status="completed"
        ),
    )
    outcome = _run(
        _runner([agent], judge_service=judge_service),
        _sample(task="caption", answers=["reference"]),
        _sample_dir(tmp_path),
        judge_policy="errors-only",
    )
    assert outcome.status.state == "succeeded"
    assert judge_client.calls == 0


# ── failure stability / 失败稳定性 ──────────────────────────────────────────


def test_failure_records_only_stable_codes(tmp_path: Path) -> None:
    agent = _FakeAgent(
        "general_vqa_agent",
        ("general_vqa",),
        error=RuntimeError("C:\\secret\\path sk-secret-raw boom"),
    )
    runner = _runner([agent])
    outcome = _run(runner, _sample(), _sample_dir(tmp_path))
    assert outcome.status.state == "failed"
    assert outcome.status.error_code == "RuntimeError"
    assert outcome.status.error_message == "RuntimeError"
    assert outcome.execution is None
    status_text = (_sample_dir(tmp_path) / "status.json").read_text(encoding="utf-8")
    assert "secret" not in status_text
    assert "sk-" not in status_text
    assert "C:\\" not in status_text
    trace_text = (_sample_dir(tmp_path) / "agent_trace.json").read_text(encoding="utf-8")
    assert "secret" not in trace_text
    assert "sk-" not in trace_text
    trace = json.loads(trace_text)
    assert trace["failure_code"] == "RuntimeError"
    assert trace["judge_status"] == "not_requested"


def test_all_attempts_failed_records_last_stable_code(tmp_path: Path) -> None:
    vqa_agent = _FakeAgent(
        "general_vqa_agent", ("general_vqa",), error=ValueError("first secret")
    )
    caption_agent = _FakeAgent(
        "caption_agent", ("caption",), error=RuntimeError("second secret")
    )
    runner = _runner([vqa_agent, caption_agent])
    resolution = _resolution("general_vqa", ["general_vqa", "caption"])
    outcome = _run(
        runner,
        _sample(question="Describe the scene."),
        _sample_dir(tmp_path),
        resolution=resolution,
    )
    assert outcome.status.state == "failed"
    assert outcome.status.error_code == "RuntimeError"
    trace = _read_json(_sample_dir(tmp_path) / "agent_trace.json")
    assert trace["failure_code"] == "RuntimeError"
    assert "secret" not in json.dumps(trace)


# ── evaluation coverage (Fix E) / 评估覆盖 ──────────────────────────────────


def test_fine_grained_counting_uses_counting_evaluation(tmp_path: Path) -> None:
    counting_agent = _FakeAgent(
        "counting_agent",
        ("counting", "fine_grained_counting"),
        payload=_counting_result(final_count=2),
    )
    runner = _runner([counting_agent])
    sample = _sample(
        task="fine_grained_counting",
        question="How many cars?",
        ground_truth=GroundTruth(count=2),
    )
    outcome = _run(runner, sample, _sample_dir(tmp_path))
    assert outcome.evaluation is not None
    assert outcome.evaluation.task == "counting"
    assert outcome.evaluation.deterministic_metrics is not None
    assert outcome.evaluation.deterministic_metrics.exact_match == 1
    assert (_sample_dir(tmp_path) / "counting_evaluation.json").is_file()


def test_fine_grained_counting_non_counting_payload_fails_closed(tmp_path: Path) -> None:
    """A non-CountingResult payload on a counting task must not fabricate
    metrics. 计数任务上非 CountingResult 载荷绝不伪造指标。"""
    agent = _FakeAgent(
        "counting_agent", ("counting", "fine_grained_counting")
    )  # AgentResult payload
    runner = _runner([agent])
    sample = _sample(
        task="fine_grained_counting",
        question="How many cars?",
        ground_truth=GroundTruth(count=2),
    )
    outcome = _run(runner, sample, _sample_dir(tmp_path))
    assert outcome.status.state == "succeeded"
    assert outcome.evaluation is None
    assert not (_sample_dir(tmp_path) / "counting_evaluation.json").exists()


def test_multiple_choice_vqa_uses_vqa_evaluation(tmp_path: Path) -> None:
    agent = _FakeAgent(
        "general_vqa_agent",
        ("general_vqa", "multiple_choice_vqa"),
        payload=AgentResult(
            agent_name="general_vqa_agent", answer="A", status="completed"
        ),
    )
    runner = _runner([agent])
    sample = _sample(
        task="multiple_choice_vqa",
        question="Pick one.",
        answers=["A"],
    )
    outcome = _run(runner, sample, _sample_dir(tmp_path))
    assert outcome.evaluation is not None
    assert outcome.evaluation.task == "general_vqa"  # canonical metric task
    assert outcome.evaluation.deterministic_metrics is not None
    assert outcome.evaluation.deterministic_metrics.exact_match is True
    assert (_sample_dir(tmp_path) / "vqa_evaluation.json").is_file()


def test_scene_classification_uses_vqa_evaluation(tmp_path: Path) -> None:
    agent = _FakeAgent(
        "general_vqa_agent",
        ("general_vqa", "scene_classification"),
        payload=AgentResult(
            agent_name="general_vqa_agent", answer="ok", status="completed"
        ),
    )
    runner = _runner([agent])
    sample = _sample(task="scene_classification", question="What is this?", answers=["ok"])
    outcome = _run(runner, sample, _sample_dir(tmp_path))
    assert outcome.evaluation is not None
    assert outcome.evaluation.task == "general_vqa"
    assert outcome.evaluation.deterministic_metrics.exact_match is True


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [("road added", True), ("no visible change", False)],
)
def test_change_qa_uses_generic_vqa_exact_match(
    tmp_path: Path,
    candidate: str,
    expected: bool,
) -> None:
    agent = _FakeAgent(
        "change_agent",
        ("change_qa",),
        payload=AgentResult(
            agent_name="change_agent", answer=candidate, status="completed"
        ),
    )
    outcome = _run(_runner([agent]), _change_sample(), _sample_dir(tmp_path))
    assert outcome.evaluation is not None
    assert outcome.evaluation.task == "general_vqa"
    assert outcome.evaluation.deterministic_metrics.exact_match is expected
    assert (_sample_dir(tmp_path) / "vqa_evaluation.json").is_file()


def test_spatial_relation_uses_vqa_evaluation(tmp_path: Path) -> None:
    agent = _FakeAgent(
        "general_vqa_agent",
        ("spatial_relation",),
        payload=AgentResult(
            agent_name="general_vqa_agent", answer="north", status="completed"
        ),
    )
    sample = _sample(
        task="spatial_relation",
        question="Where is A relative to B?",
        answers=["north"],
    )
    outcome = _run(_runner([agent]), sample, _sample_dir(tmp_path))
    assert outcome.evaluation is not None
    assert outcome.evaluation.task == "general_vqa"
    assert outcome.evaluation.deterministic_metrics.exact_match is True
    assert (_sample_dir(tmp_path) / "vqa_evaluation.json").is_file()


def test_grounding_valid_geometry_writes_grounding_evaluation(tmp_path: Path) -> None:
    agent = _FakeAgent(
        "grounding_agent",
        ("grounding",),
        payload=AgentResult(
            agent_name="grounding_agent",
            answer="located",
            boxes=[[10.0, 20.0, 110.0, 120.0]],
            status="completed",
        ),
    )
    runner = _runner([agent])
    sample = _sample(
        task="grounding",
        question="Where is the car?",
        ground_truth=GroundTruth(
            boxes=[[10.0, 20.0, 110.0, 120.0]],
            coordinate_frame="normalized_0_999_top_left",
        ),
    )
    outcome = _run(runner, sample, _sample_dir(tmp_path))
    assert outcome.evaluation is not None
    assert outcome.evaluation.task == "grounding"
    assert outcome.evaluation.deterministic_metrics is not None
    assert outcome.evaluation.deterministic_metrics.iou == 1.0
    assert outcome.evaluation.deterministic_metrics.iou_at_0_5 is True
    evaluation = _read_json(_sample_dir(tmp_path) / "grounding_evaluation.json")
    assert evaluation["task"] == "grounding"


def test_grounding_missing_gt_no_fake_metric(tmp_path: Path) -> None:
    agent = _FakeAgent(
        "grounding_agent",
        ("grounding",),
        payload=AgentResult(
            agent_name="grounding_agent",
            answer="located",
            boxes=[[10.0, 20.0, 110.0, 120.0]],
            status="completed",
        ),
    )
    runner = _runner([agent])
    sample = _sample(
        task="grounding",
        question="Where is the car?",
        ground_truth=GroundTruth(boxes=[]),
    )
    outcome = _run(runner, sample, _sample_dir(tmp_path))
    assert outcome.status.state == "succeeded"
    assert outcome.evaluation is None
    assert not (_sample_dir(tmp_path) / "grounding_evaluation.json").exists()


def test_grounding_missing_prediction_no_fake_metric(tmp_path: Path) -> None:
    runner = _runner([_grounding_agent([])])
    sample = _sample(
        task="grounding",
        question="Where is the car?",
        ground_truth=GroundTruth(
            boxes=[[10.0, 20.0, 110.0, 120.0]],
            coordinate_frame="normalized_0_999_top_left",
        ),
    )
    outcome = _run(runner, sample, _sample_dir(tmp_path))
    assert outcome.status.state == "succeeded"
    assert outcome.evaluation is None
    assert not (_sample_dir(tmp_path) / "grounding_evaluation.json").exists()


def test_caption_writes_per_sample_caption_evaluation(tmp_path: Path) -> None:
    agent = _FakeAgent(
        "caption_agent",
        ("caption",),
        payload=AgentResult(
            agent_name="caption_agent", answer="a street scene", status="completed"
        ),
    )
    runner = _runner([agent])
    sample = _sample(
        task="caption",
        question="",
        answers=["a street scene"],
    )
    outcome = _run(runner, sample, _sample_dir(tmp_path))
    assert outcome.status.state == "succeeded"
    assert outcome.evaluation is not None
    assert outcome.evaluation.task == "caption"
    metrics = outcome.evaluation.deterministic_metrics
    assert metrics.candidate == "a street scene"
    assert metrics.references == ["a street scene"]
    evaluation = _read_json(_sample_dir(tmp_path) / "caption_evaluation.json")
    assert evaluation["task"] == "caption"
    assert evaluation["deterministic_metrics"]["candidate"] == "a street scene"


def test_caption_without_references_no_record(tmp_path: Path) -> None:
    agent = _FakeAgent("caption_agent", ("caption",))
    runner = _runner([agent])
    sample = _sample(
        task="caption", question="", ground_truth=GroundTruth(answers=[])
    )
    outcome = _run(runner, sample, _sample_dir(tmp_path))
    assert outcome.evaluation is None
    assert not (_sample_dir(tmp_path) / "caption_evaluation.json").exists()


def test_change_caption_writes_caption_evaluation(tmp_path: Path) -> None:
    agent = _FakeAgent(
        "change_agent",
        ("change_caption",),
        payload=AgentResult(
            agent_name="change_agent", answer="a road was added", status="completed"
        ),
    )
    sample = _sample(
        task="change_caption",
        sample_id="change-caption-1",
        question="",
        answers=["a road was added"],
        images=[
            _image("i0", "t1.png", "t1"),
            _image("i1", "t2.png", "t2"),
        ],
    )
    outcome = _run(_runner([agent]), sample, _sample_dir(tmp_path))
    assert outcome.evaluation is not None
    assert outcome.evaluation.task == "caption"
    metrics = outcome.evaluation.deterministic_metrics
    assert metrics.candidate == "a road was added"
    assert metrics.references == ["a road was added"]
    assert (_sample_dir(tmp_path) / "caption_evaluation.json").is_file()


def test_change_caption_without_references_no_record(tmp_path: Path) -> None:
    agent = _FakeAgent("change_agent", ("change_caption",))
    sample = _sample(
        task="change_caption",
        sample_id="change-caption-no-reference",
        question="",
        images=[
            _image("i0", "t1.png", "t1"),
            _image("i1", "t2.png", "t2"),
        ],
        ground_truth=GroundTruth(answers=[]),
    )
    outcome = _run(_runner([agent]), sample, _sample_dir(tmp_path))
    assert outcome.evaluation is None
    assert not (_sample_dir(tmp_path) / "caption_evaluation.json").exists()


# ── grounding frame safety (Fix A) / grounding 坐标系安全 ───────────────────


def _grounding_agent(boxes: list[list[float]]) -> _FakeAgent:
    return _FakeAgent(
        "grounding_agent",
        ("grounding",),
        payload=AgentResult(
            agent_name="grounding_agent",
            answer="located",
            boxes=boxes,
            status="completed",
        ),
    )


def test_grounding_normalized_four_box_produces_iou(tmp_path: Path) -> None:
    runner = _runner([_grounding_agent([[100.0, 200.0, 400.0, 500.0]])])
    sample = _sample(
        task="grounding",
        question="Where is the car?",
        ground_truth=GroundTruth(
            boxes=[[100.0, 200.0, 400.0, 500.0]],
            coordinate_frame="normalized_0_999_top_left",
        ),
    )
    outcome = _run(runner, sample, _sample_dir(tmp_path))
    assert outcome.evaluation is not None
    assert outcome.evaluation.deterministic_metrics.iou == 1.0
    assert (_sample_dir(tmp_path) / "grounding_evaluation.json").is_file()


def test_grounding_frame_mismatch_fails_closed(tmp_path: Path) -> None:
    """normalized prediction vs source_pixels ground truth must never be
    IoU-ed together. normalized 预测与 source_pixels 真值绝不直接计算 IoU。"""
    runner = _runner([_grounding_agent([[100.0, 200.0, 400.0, 500.0]])])
    sample = _sample(
        task="grounding",
        question="Where is the car?",
        ground_truth=GroundTruth(
            boxes=[[620.0, 1100.0, 1400.0, 1800.0]],
            coordinate_frame="source_pixels_top_left",
        ),
    )
    outcome = _run(runner, sample, _sample_dir(tmp_path))
    assert outcome.status.state == "succeeded"
    assert outcome.evaluation is None
    assert not (_sample_dir(tmp_path) / "grounding_evaluation.json").exists()


def test_grounding_eight_value_polygon_fails_closed(tmp_path: Path) -> None:
    runner = _runner([_grounding_agent([[100.0, 200.0, 400.0, 500.0]])])
    sample = _sample(
        task="grounding",
        question="Where is the car?",
        ground_truth=GroundTruth(
            boxes=[[620.0, 1100.0, 1400.0, 1800.0, 700.0, 1200.0, 800.0, 1300.0]],
            coordinate_frame="normalized_0_999_top_left",
        ),
    )
    outcome = _run(runner, sample, _sample_dir(tmp_path))
    assert outcome.evaluation is None
    assert not (_sample_dir(tmp_path) / "grounding_evaluation.json").exists()


def test_grounding_non_four_prediction_box_fails_closed(tmp_path: Path) -> None:
    runner = _runner([_grounding_agent([[10.0, 20.0, 110.0]])])
    sample = _sample(
        task="grounding",
        question="Where is the car?",
        ground_truth=GroundTruth(
            boxes=[[10.0, 20.0, 110.0, 120.0]],
            coordinate_frame="normalized_0_999_top_left",
        ),
    )
    outcome = _run(runner, sample, _sample_dir(tmp_path))
    assert outcome.evaluation is None
    assert not (_sample_dir(tmp_path) / "grounding_evaluation.json").exists()


def test_grounding_missing_frame_fails_closed(tmp_path: Path) -> None:
    """A ground truth without a declared coordinate frame is never guessed.
    未声明坐标系的真值绝不猜测。"""
    runner = _runner([_grounding_agent([[100.0, 200.0, 400.0, 500.0]])])
    sample = _sample(
        task="grounding",
        question="Where is the car?",
        ground_truth=GroundTruth(boxes=[[100.0, 200.0, 400.0, 500.0]]),
    )
    outcome = _run(runner, sample, _sample_dir(tmp_path))
    assert outcome.evaluation is None
    assert not (_sample_dir(tmp_path) / "grounding_evaluation.json").exists()


def test_grounding_vrsbench_source_pixels_regression(tmp_path: Path) -> None:
    """VRSBench-style ground truth in source_pixels_top_left must not produce
    a grounding evaluation against a normalized prediction.
    VRSBench 风格 source_pixels_top_left 真值绝不与 normalized 预测产出
    grounding 评估。"""
    runner = _runner([_grounding_agent([[100.0, 200.0, 400.0, 500.0]])])
    sample = _sample(
        task="grounding",
        question="Where is the car?",
        ground_truth=GroundTruth(
            boxes=[[620.0, 1100.0, 1400.0, 1800.0]],
            coordinate_frame="source_pixels_top_left",
        ),
    )
    outcome = _run(runner, sample, _sample_dir(tmp_path))
    assert outcome.status.state == "succeeded"
    assert outcome.evaluation is None
    assert not (_sample_dir(tmp_path) / "grounding_evaluation.json").exists()


# ── portable result paths (Fix D) / 可移植结果路径 ──────────────────────────


def test_status_result_path_is_sample_relative_basename(tmp_path: Path) -> None:
    agent = _FakeAgent("general_vqa_agent", ("general_vqa",))
    runner = _runner([agent])
    outcome = _run(runner, _sample(), _sample_dir(tmp_path))
    assert outcome.status.result_path == Path("agent_result.json")


def test_counting_status_result_path_is_basename(tmp_path: Path) -> None:
    counting_agent = _FakeAgent(
        "counting_agent",
        ("counting",),
        payload=_counting_result(final_count=2),
        result_filename="counting_result.json",
    )
    runner = _runner([counting_agent])
    sample = _sample(
        task="counting",
        question="How many cars?",
        ground_truth=GroundTruth(count=2),
    )
    outcome = _run(runner, sample, _sample_dir(tmp_path))
    assert outcome.status.result_path == Path("counting_result.json")


# ── visual planning gate (C7, 14A2) / 视觉规划门 ────────────────────────────


class _FakePlanner:
    """Duck-typed VisualPlanner with a configurable outcome and call record.
    The configured plan lives under result so it never shadows the plan
    method the gate invokes. 可配置结果并记录调用的 VisualPlanner 鸭子类型；
    配置的计划放在 result 下，避免遮蔽 gate 调用的 plan 方法。"""

    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[UnifiedSample, object, object, object]] = []

    async def plan(self, sample, *, data_root, artifact_dir, budget):
        self.calls.append((sample, data_root, artifact_dir, budget))
        if self.error is not None:
            raise self.error
        return self.result


def _visual_plan() -> FirstQwenVisualPlan:
    """A valid direct_vqa plan (feature on, legacy-equivalent family).
    一条合法 direct_vqa 计划（特性开启但等价于旧路径的家族）。"""
    return FirstQwenVisualPlan(
        version="first-qwen-plan-v1",
        execution_family="direct_vqa",
        confidence=0.9,
        roi_plan=RoiPlan(rois=[]),
    )


def test_visual_planning_gate_runs_planner_once_for_planning_task(
    tmp_path: Path,
) -> None:
    """A planning-task sample gets exactly one planner call, a persisted
    visual_plan.json, and the plan/bindings inside AgentContext; the budget
    is shared between planner and agent.
    规划任务样本恰好一次规划调用、持久化 visual_plan.json、plan/bindings 进入
    AgentContext；预算在规划器与 Agent 间共享。"""
    planner = _FakePlanner(result=_visual_plan())
    bindings = VisualPlanBindings()
    gate = VisualPlanningGate(planner, bindings=bindings)
    agent = _FakeAgent("general_vqa_agent", ("general_vqa",))
    runner = _runner([agent], visual_planning=gate)
    sample = _sample()
    outcome = _run(runner, sample, _sample_dir(tmp_path))
    assert outcome.status.state == "succeeded"
    assert len(planner.calls) == 1
    planned_sample, _data_root, _artifact_dir, planned_budget = planner.calls[0]
    assert planned_sample.sample_id == sample.sample_id
    assert planned_budget is agent.calls[0][1].call_budget  # shared / 共享预算
    plan_json = _read_json(_sample_dir(tmp_path) / "visual_plan.json")
    assert plan_json["version"] == "first-qwen-plan-v1"
    assert plan_json["execution_family"] == "direct_vqa"
    context = agent.calls[0][1]
    assert context.visual_plan == _visual_plan()
    assert context.visual_bindings is bindings
    # Legacy artifacts still land unchanged. / 旧产物仍然原样落盘。
    assert (_sample_dir(tmp_path) / "routing_decision.json").is_file()
    assert (_sample_dir(tmp_path) / "agent_result.json").is_file()


def test_visual_planning_gate_skips_non_planning_task(tmp_path: Path) -> None:
    """caption/change/counting samples never reach the planner: no call, no
    plan artifact, and no plan/bindings leak into AgentContext.
    caption/change/counting 样本绝不触达规划器：无调用、无计划产物、
    plan/bindings 不泄漏进 AgentContext。"""
    planner = _FakePlanner(result=_visual_plan())
    gate = VisualPlanningGate(planner)
    caption_agent = _FakeAgent("caption_agent", ("caption",))
    runner = _runner([caption_agent], visual_planning=gate)
    sample = _sample(task="caption", question="Describe the scene.")
    outcome = _run(runner, sample, _sample_dir(tmp_path))
    assert outcome.status.state == "succeeded"
    assert planner.calls == []
    assert not (_sample_dir(tmp_path) / "visual_plan.json").exists()
    context = caption_agent.calls[0][1]
    assert context.visual_plan is None
    assert context.visual_bindings is None


def test_visual_planning_gate_absent_writes_no_plan_artifact(tmp_path: Path) -> None:
    """The frozen flag-off state (no gate wired) must not produce any
    visual-plan artifact. 冻结的 flag-off 状态（未接 gate）不得产生任何
    visual-plan 产物。"""
    agent = _FakeAgent("general_vqa_agent", ("general_vqa",))
    runner = _runner([agent])
    outcome = _run(runner, _sample(), _sample_dir(tmp_path))
    assert outcome.status.state == "succeeded"
    assert not (_sample_dir(tmp_path) / "visual_plan.json").exists()
    context = agent.calls[0][1]
    assert context.visual_plan is None
    assert context.visual_bindings is None


def test_visual_planning_gate_plan_error_is_strict_failure(tmp_path: Path) -> None:
    """Frozen planner failure policy: VisualPlanError becomes a failed sample
    with the stable VISUAL_PLAN_FAILED:<CODE> error code, no retry, no legacy
    fallback, and no agent call. 冻结规划失败策略：VisualPlanError 变为携带
    稳定 VISUAL_PLAN_FAILED:<CODE> 错误码的 failed 样本，无重试、无旧路径
    回退、不调用 Agent。"""
    planner = _FakePlanner(error=VisualPlanError("LOW_CONFIDENCE"))
    gate = VisualPlanningGate(planner)
    agent = _FakeAgent("general_vqa_agent", ("general_vqa",))
    runner = _runner([agent], visual_planning=gate)
    outcome = _run(runner, _sample(), _sample_dir(tmp_path))
    assert outcome.status.state == "failed"
    assert outcome.status.error_code == "VISUAL_PLAN_FAILED:LOW_CONFIDENCE"
    assert outcome.status.error_message == "VisualPlanError"
    assert len(agent.calls) == 0
    assert not (_sample_dir(tmp_path) / "visual_plan.json").exists()
    assert not (_sample_dir(tmp_path) / "routing_decision.json").exists()
    trace = _read_json(_sample_dir(tmp_path) / "agent_trace.json")
    assert trace["failure_code"] == "VISUAL_PLAN_FAILED:LOW_CONFIDENCE"


def test_visual_planning_gate_unexpected_error_is_stable_code(
    tmp_path: Path,
) -> None:
    """An unexpected gate exception maps to its type name; the raw message
    never leaks into persisted artifacts. 意外 gate 异常映射为类型名；原始
    消息绝不泄漏进持久化产物。"""
    planner = _FakePlanner(error=RuntimeError("raw secret detail"))
    gate = VisualPlanningGate(planner)
    runner = _runner([_FakeAgent("general_vqa_agent", ("general_vqa",))], visual_planning=gate)
    outcome = _run(runner, _sample(), _sample_dir(tmp_path))
    assert outcome.status.state == "failed"
    assert outcome.status.error_code == "RuntimeError"
    status_text = (_sample_dir(tmp_path) / "status.json").read_text(encoding="utf-8")
    trace_text = (_sample_dir(tmp_path) / "agent_trace.json").read_text(encoding="utf-8")
    assert "secret" not in status_text
    assert "secret" not in trace_text


# ── joint mode (doc 15) / 联合模式（doc 15） ────────────────────────────────


def _joint_plan(*, task: str = "general_vqa") -> JointQwenVisualPlan:
    """A valid joint plan: model-selected task plus a direct_vqa plan.
    一条合法联合计划：模型选定 task 加 direct_vqa 计划。"""
    return JointQwenVisualPlan(
        version="joint-qwen-plan-v1",
        task=task,  # type: ignore[arg-type]
        visual_plan=_visual_plan(),
    )


def test_run_one_joint_writes_artifact_and_derives_resolution(
    tmp_path: Path,
) -> None:
    """joint_plan persists joint_visual_plan.json, derives the resolution
    deterministically (source model, single candidate), injects the plan and
    bindings into AgentContext, and records joint_plan in the trace.
    joint_plan 持久化 joint_visual_plan.json、确定性派生 resolution（source
    model、单一候选）、把计划与绑定注入 AgentContext，并在 trace 记录
    joint_plan。"""
    bindings = VisualPlanBindings()
    agent = _FakeAgent("general_vqa_agent", ("general_vqa",))
    runner = _runner([agent], joint_bindings=bindings)
    plan = _joint_plan()
    outcome = _run(runner, _sample(), _sample_dir(tmp_path), joint_plan=plan)
    assert outcome.status.state == "succeeded"
    joint_json = _read_json(_sample_dir(tmp_path) / "joint_visual_plan.json")
    assert joint_json["version"] == "joint-qwen-plan-v1"
    assert joint_json["task"] == "general_vqa"
    assert joint_json["visual_plan"]["execution_family"] == "direct_vqa"
    trace = _read_json(_sample_dir(tmp_path) / "agent_trace.json")
    assert trace["joint_plan"] is True
    assert trace["task_type"] == "general_vqa"
    assert trace["resolution_source"] == "model"
    assert trace["candidate_tasks"] == ["general_vqa"]
    context = agent.calls[0][1]
    assert context.visual_plan == _visual_plan()
    assert context.visual_bindings is bindings
    # The routing artifact reflects the model task. / routing 产物反映模型任务。
    routing = _read_json(_sample_dir(tmp_path) / "routing_decision.json")
    assert routing["task"] == "general_vqa"
    # No separate visual_plan.json from the old gate. / 旧 gate 的独立
    # visual_plan.json 不出现。
    assert not (_sample_dir(tmp_path) / "visual_plan.json").exists()


def test_run_one_joint_resolution_and_joint_plan_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    """Supplying both resolution and joint_plan fails closed instead of
    guessing which one wins. 同时提供 resolution 与 joint_plan 严格失败，
    绝不猜测哪个生效。"""
    runner = _runner([_FakeAgent("general_vqa_agent", ("general_vqa",))])
    with pytest.raises(ValueError, match="mutually exclusive"):
        _run(
            runner,
            _sample(),
            _sample_dir(tmp_path),
            resolution=_resolution("general_vqa", ["general_vqa"], low_confidence=False),
            joint_plan=_joint_plan(),
        )


def test_run_one_joint_task_mismatch_fails_closed(tmp_path: Path) -> None:
    """The executed sample must already carry the model-selected task; a
    mismatch raises instead of silently re-routing. 被执行样本必须已携带模型
    选定 task；不一致直接抛出而不是静默改路由。"""
    runner = _runner([_FakeAgent("general_vqa_agent", ("general_vqa",))])
    plan = _joint_plan(task="caption")
    with pytest.raises(ValueError, match="must equal the model-selected task"):
        _run(runner, _sample(), _sample_dir(tmp_path), joint_plan=plan)


def test_run_one_joint_shares_external_budget(tmp_path: Path) -> None:
    """The budget that already paid for the planner call is the same budget
    the agent sees; no second budget is minted inside run_one.
    已为规划调用付费的预算与 Agent 所见为同一对象；run_one 内部不新建预算。"""
    agent = _FakeAgent("general_vqa_agent", ("general_vqa",))
    runner = _runner([agent])
    budget = CallBudget(max_qwen_calls=100)
    _run(
        runner,
        _sample(),
        _sample_dir(tmp_path),
        joint_plan=_joint_plan(),
        budget=budget,
    )
    assert agent.calls[0][1].call_budget is budget


def test_run_one_joint_unroutable_task_fails_with_joint_mode_trace(
    tmp_path: Path,
) -> None:
    """A model-selected task with no registered agent collapses to a stable
    UnsupportedAgentError failure; the trace still records joint_plan
    honestly. 模型选定 task 无注册 Agent 时收敛为稳定 UnsupportedAgentError
    失败；trace 仍如实记录 joint_plan。"""
    agent = _FakeAgent("general_vqa_agent", ("general_vqa",))
    runner = _runner([agent])
    plan = _joint_plan(task="counting")
    outcome = _run(
        runner,
        _sample(task="counting"),
        _sample_dir(tmp_path),
        joint_plan=plan,
    )
    assert outcome.status.state == "failed"
    assert outcome.status.error_code == "UnsupportedAgentError"
    trace = _read_json(_sample_dir(tmp_path) / "agent_trace.json")
    assert trace["joint_plan"] is True
    assert trace["failure_code"] == "UnsupportedAgentError"
    assert len(agent.calls) == 0


# ── joint mode × real agent ROI consumption (doc 15 §4.5, Phase D) ─────────


class _EvidenceRecordingClient:
    """Minimal VisionLanguageClient recording calls and returning a stable
    AgentResult; cache_identity satisfies the evidence identity guard.
    记录调用并返回稳定 AgentResult 的最小 VisionLanguageClient；
    cache_identity 满足证据身份守卫。"""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def cache_identity(self):
        from models.base import ModelCacheIdentity

        return ModelCacheIdentity(
            model="fake-model",
            generation={"temperature": 0.0, "do_sample": False, "max_tokens": 128},
            client_version="1",
        )

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls += 1
        return response_model.model_validate(
            {"agent_name": "general_vqa_agent", "answer": "yes", "status": "completed"}
        )


def test_run_one_joint_object_evidence_plan_drives_real_agent_evidence_path(
    tmp_path: Path,
) -> None:
    """A joint object-evidence plan reaches the real general_vqa agent through
    AgentContext and selects the evidence path: the injected service executes
    exactly once and the final Qwen call happens exactly once — the plan is
    consumed, not ignored. 联合 object-evidence 计划经 AgentContext 到达真实
    general_vqa Agent 并选择证据路径：注入服务恰好执行一次、最终 Qwen 恰好
    调用一次——计划被消费而非忽略。"""
    import numpy as np

    from agents.general_vqa.agent import GeneralVQAAgent
    from agents.general_vqa.evidence.executor import EvidenceExecution
    from agents.general_vqa.evidence.schema import RoiEvidenceRecord, VqaEvidenceBundle
    from agents.schema import ObjectEvidenceRequest, RoiRegion

    # A 200x160 image under the preview shrink floor: the full-image ROI crop
    # keeps its native size, matching the presence-mask shape (H, W).
    # 200x160 图像低于预览缩放下限：整图 ROI 裁切保持原始尺寸，与 presence
    # mask 形状 (H, W) 一致。
    image_path = tmp_path / "img0.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (200, 160), (1, 2, 3)).save(image_path, format="PNG")

    client = _EvidenceRecordingClient()
    agent = GeneralVQAAgent(client)
    service_calls: list[tuple[object, dict, str]] = []

    class _FakeVqaEvidenceService:
        """VqaEvidenceService protocol fake: records the call and returns a
        minimal evidence execution. VqaEvidenceService 协议 fake：记录调用并
        返回最小证据执行结果。"""

        def execute(self, plan, images, *, fallback_image_id):
            service_calls.append((plan, dict(images), fallback_image_id))
            return EvidenceExecution(
                bundle=VqaEvidenceBundle(
                    catalog_version="first-qwen-plan-v1",
                    rois=[
                        RoiEvidenceRecord(
                            roi_id="full",
                            image_id="i0",
                            source_size=(200, 160),
                            core_xyxy=(0, 0, 200, 160),
                            expanded_xyxy=(0, 0, 200, 160),
                            crop_size=(200, 160),
                        )
                    ],
                    leaf_states={"building_outline": "hit"},
                ),
                layer_states=(),
                outcomes=(),
                masks={("full", "building_outline"): np.ones((160, 200), dtype=bool)},
            )

    bindings = VisualPlanBindings(vqa_evidence=_FakeVqaEvidenceService())
    plan = JointQwenVisualPlan(
        version="joint-qwen-plan-v1",
        task="general_vqa",
        visual_plan=FirstQwenVisualPlan(
            version="first-qwen-plan-v1",
            execution_family="object_evidence_vqa",
            confidence=0.9,
            roi_plan=RoiPlan(rois=[]),
            evidence_request=ObjectEvidenceRequest(
                composite_categories=["building_outline"]
            ),
        ),
    )
    runner = _runner([agent], joint_bindings=bindings, data_root=tmp_path)
    sample = _sample(sample_id="s1")
    budget = CallBudget(max_qwen_calls=10)
    budget.reserve_qwen()  # the joint planner call already consumed one / 联合
    # 规划调用已消费一次
    outcome = _run(runner, sample, _sample_dir(tmp_path), joint_plan=plan, budget=budget)
    assert outcome.status.state == "succeeded"
    # The evidence path fired: one service execution and one final Qwen call.
    # 证据路径触发：一次服务执行与一次最终 Qwen 调用。
    assert len(service_calls) == 1
    assert service_calls[0][0].execution_family == "object_evidence_vqa"
    assert set(service_calls[0][1]) == {"i0"}
    assert service_calls[0][2] == "i0"
    assert client.calls == 1
    assert budget.qwen_calls_used == 2  # planner + final agent call / 规划+最终
    # The joint artifact and the standard result land together.
    # 联合产物与标准结果一起落盘。
    assert (_sample_dir(tmp_path) / "joint_visual_plan.json").is_file()
    assert (_sample_dir(tmp_path) / "agent_result.json").is_file()
