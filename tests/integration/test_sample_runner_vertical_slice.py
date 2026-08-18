"""Vertical slice: SampleRunner + real GeneralVQAAgent + fake model client.

垂直切片：SampleRunner + 真实 GeneralVQAAgent + fake 模型客户端。离线打通
单样本执行内核（路由→Agent→产物→确定性评估→trace→状态），不接入
DatasetRunner，不导入任何旧包。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from PIL import Image

from agents.general_vqa import GeneralVQAAgent
from agents.registry import AgentRegistry
from data.schema import GroundTruth, ImageRef, UnifiedSample
from models.base import ModelCacheIdentity
from routing.router import TaskRouter
from agents.schema import MaterializedVisualView, VisualTaskPlan
from workflows.artifact_writer import ArtifactWriter
from workflows.call_budget import CallBudgetFactory
from workflows.sample_runner import SampleRunner
from workflows.schema import SampleRunOutcome


class _FakeBudget:
    def reserve_qwen(self) -> None:
        pass

    def reserve_deepseek(self) -> None:
        pass


class _FakeClient:
    """Minimal VisionLanguageClient with a stable cache identity.
    带稳定缓存身份的最小 VisionLanguageClient。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="fake-model",
            generation={"temperature": 0.0, "do_sample": False, "max_tokens": 128},
            client_version="1",
        )

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls.append({"messages": messages, "request_hash": request_meta.request_hash})
        return response_model.model_validate(
            {"agent_name": "general_vqa_agent", "answer": "yes", "status": "completed"}
        )


def _make_image(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (10, 20, 30)).save(root / "img.png", format="PNG")


def _sample(root: Path) -> UnifiedSample:
    _make_image(root)
    return UnifiedSample(
        sample_id="slice-1",
        dataset="parity",
        split="test",
        task="general_vqa",
        images=[ImageRef(image_id="i1", path="img.png", role="image")],
        question="Is there a road?",
        ground_truth=GroundTruth(answers=["yes"]),
    )


def _runner(root: Path) -> tuple[SampleRunner, _FakeClient]:
    client = _FakeClient()
    registry = AgentRegistry()
    registry.register(GeneralVQAAgent(client))
    runner = SampleRunner(
        registry=registry,
        router=TaskRouter(),
        qwen_client=client,
        artifact_writer=ArtifactWriter(),
        call_budget_factory=CallBudgetFactory(),
        data_root=root,
    )
    return runner, client


def _plan() -> VisualTaskPlan:
    return VisualTaskPlan(
        version="visual-task-plan-v5",
        task="general_vqa",
        reason_codes=["test"],
    )


def _view() -> MaterializedVisualView:
    return MaterializedVisualView(
        image_id="i1",
        view_mode="full_image",
        source_size=(4, 4),
        crop_xyxy=(0, 0, 4, 4),
        crop_size=(4, 4),
    )


def _run(runner: SampleRunner, sample: UnifiedSample, sample_dir: Path):
    return asyncio.run(
        runner.run_one(
            sample,
            sample_dir,
            visual_task_plan=_plan(),
            visual_views=(_view(),),
            judge_policy="none",
        )
    )


def test_sample_runner_vertical_slice_general_vqa(tmp_path: Path) -> None:
    runner, client = _runner(tmp_path)
    sample = _sample(tmp_path)
    sample_dir = tmp_path / "samples" / sample.sample_id
    outcome = _run(runner, sample, sample_dir)
    assert isinstance(outcome, SampleRunOutcome)
    assert outcome.status.state == "succeeded"
    assert outcome.status.result_path is not None
    assert outcome.routing is not None
    assert outcome.routing.primary_agent == "general_vqa_agent"
    assert len(client.calls) == 1
    # Artifact contract in order: sample → status → routing → result →
    # evaluation → trace → final status. / 产物契约顺序。
    assert (sample_dir / "sample.json").is_file()
    assert (sample_dir / "routing_decision.json").is_file()
    assert (sample_dir / "visual_task_plan.json").is_file()
    assert (sample_dir / "agent_result.json").is_file()
    assert (sample_dir / "vqa_evaluation.json").is_file()
    assert (sample_dir / "agent_trace.json").is_file()
    status = json.loads((sample_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "succeeded"
    assert status["task"] == "general_vqa"
    assert status["sample_id"] == "slice-1"
    evaluation = json.loads(
        (sample_dir / "vqa_evaluation.json").read_text(encoding="utf-8")
    )
    assert evaluation["task"] == "general_vqa"
    assert evaluation["judge_status"] == "not_requested"
    assert evaluation["deterministic_metrics"]["exact_match"] is True
    trace = json.loads((sample_dir / "agent_trace.json").read_text(encoding="utf-8"))
    assert trace["execution_agent"] == "general_vqa_agent"
    assert trace["task_type"] == "general_vqa"
    assert trace["resolution_source"] == "visual-task-plan-v5"
    assert trace["judge_status"] == "not_requested"
