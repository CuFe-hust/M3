"""Vertical slices for explicit and draft DatasetRunner v2 planning.

显式样本与 SampleDraft 两条 DatasetRunner 垂直切片：两者都经同一个 v2
planner，draft 只在 planner 返回 task 后物化为 UnifiedSample。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agents.base import AgentExecution
from agents.registry import AgentRegistry
from agents.schema import AgentResult, MaterializedVisualView, VisualTaskPlan
from data.adapters.base import AdapterProbe
from data.schema import GroundTruth, ImageRef, SampleDraft, UnifiedSample
from routing.router import TaskRouter
from workflows.artifact_writer import ArtifactWriter
from workflows.call_budget import CallBudgetFactory
from workflows.dataset_runner import DatasetRunner, storage_key
from workflows.run_store import RunStore
from workflows.sample_runner import SampleRunner


class _FakeAdapter:
    name = "auto-demo"
    supported_tasks = frozenset({"general_vqa"})

    def __init__(self, sample: UnifiedSample, draft: SampleDraft) -> None:
        self.sample = sample
        self.draft = draft

    def probe(self, root: Path, task: str | None = None) -> AdapterProbe:
        sample_file = root / "samples.jsonl"
        sample_file.touch()
        return AdapterProbe(
            dataset=self.name,
            version="1",
            sample_file=sample_file,
            observed_fields=("id", "images", "question"),
            sample_count=1,
            task=task,
            available_tasks=("general_vqa",),
        )

    def iter_samples(self, root: Path, split: str, task: str):
        yield self.sample

    def iter_drafts(self, root: Path, split: str):
        yield self.draft


class _FakePlanner:
    def __init__(self) -> None:
        self.calls: list[object] = []

    async def plan_with_views(self, view, *, data_root, artifact_dir, budget):
        self.calls.append(view)
        plan = VisualTaskPlan(
            version="visual-task-plan-v2",
            task="general_vqa",
            confidence=0.95,
            reason_codes=["test"],
        )
        image = view.images[0]
        path = (data_root / image.path).resolve()
        from PIL import Image

        size = Image.open(path).size
        views = (
            MaterializedVisualView(
                image_id=image.image_id,
                view_mode="full_image",
                source_size=size,
                crop_xyxy=(0, 0, size[0], size[1]),
                crop_size=size,
            ),
        )
        return plan, views


class _FakeAgent:
    name = "general_vqa_agent"
    supported_tasks = frozenset({"general_vqa"})

    def __init__(self) -> None:
        self.calls: list[UnifiedSample] = []

    async def run(self, sample: UnifiedSample, context: object) -> AgentExecution:
        self.calls.append(sample)
        return AgentExecution(
            agent_name=self.name,
            payload=AgentResult(agent_name=self.name, answer="ok"),
            result_filename="agent_result.json",
        )


def _setup(tmp_path: Path) -> tuple[Path, Path, _FakePlanner, _FakeAgent, _FakeAdapter]:
    root = tmp_path / "data"
    root.mkdir()
    from PIL import Image

    Image.new("RGB", (12, 8), (1, 2, 3)).save(root / "img.png", format="PNG")
    sample = UnifiedSample(
        sample_id="explicit-1",
        dataset="auto-demo",
        split="test",
        task="general_vqa",
        images=[ImageRef(image_id="img1", path="img.png", role="image")],
        question="Is there a road?",
        ground_truth=GroundTruth(answers=["ok"]),
    )
    draft = SampleDraft(
        sample_id="draft-1",
        dataset="auto-demo",
        split="test",
        images=[ImageRef(image_id="img1", path="img.png", role="image")],
        question="Is there a road?",
    )
    adapter = _FakeAdapter(sample, draft)
    planner = _FakePlanner()
    agent = _FakeAgent()
    registry = AgentRegistry()
    registry.register(agent)
    run_store = RunStore(tmp_path / "runs", tmp_path)
    run_store.create_run(
        config_payload={"planning_mode": "visual-task-plan-v2"},
        model_ids={"qwen": "logical-qwen"},
        prompt_paths=[],
        run_id="auto-run",
    )
    run_dir = tmp_path / "runs" / "auto-run"
    sample_runner = SampleRunner(
        registry=registry,
        router=TaskRouter(),
        qwen_client=object(),
        artifact_writer=ArtifactWriter(),
        call_budget_factory=CallBudgetFactory(),
        data_root=root,
    )
    dataset_runner = DatasetRunner(
        adapter=adapter,
        sample_runner=sample_runner,
        run_dir=run_dir,
        artifact_writer=ArtifactWriter(),
        call_budget_factory=CallBudgetFactory(),
        visual_task_planner=planner,
        data_root=root,
    )
    return root, run_dir, planner, agent, adapter


def test_explicit_dataset_sample_uses_v2_planner(tmp_path: Path) -> None:
    root, run_dir, planner, agent, adapter = _setup(tmp_path)
    registry = AgentRegistry()
    registry.register(agent)
    runner = SampleRunner(
        registry=registry,
        router=TaskRouter(),
        qwen_client=object(),
        artifact_writer=ArtifactWriter(),
        call_budget_factory=CallBudgetFactory(),
        data_root=root,
    )
    dataset_runner = DatasetRunner(
        adapter=adapter,
        sample_runner=runner,
        run_dir=run_dir,
        artifact_writer=ArtifactWriter(),
        call_budget_factory=CallBudgetFactory(),
        visual_task_planner=planner,
        data_root=root,
    )
    summary = asyncio.run(
        dataset_runner.run(root=root, split="test", task="general_vqa", sample_concurrency=1)
    )
    sample_dir = run_dir / "tasks" / "general_vqa" / "samples" / storage_key("explicit-1")
    assert summary.succeeded == 1
    assert len(planner.calls) == 1
    assert agent.calls[0].task == "general_vqa"
    assert (sample_dir / "visual_task_plan.json").is_file()


def test_draft_auto_task_materializes_after_v2_planning(tmp_path: Path) -> None:
    root, run_dir, planner, agent, adapter = _setup(tmp_path)
    registry = AgentRegistry()
    registry.register(agent)
    runner = SampleRunner(
        registry=registry,
        router=TaskRouter(),
        qwen_client=object(),
        artifact_writer=ArtifactWriter(),
        call_budget_factory=CallBudgetFactory(),
        data_root=root,
    )
    dataset_runner = DatasetRunner(
        adapter=adapter,
        sample_runner=runner,
        run_dir=run_dir,
        artifact_writer=ArtifactWriter(),
        call_budget_factory=CallBudgetFactory(),
        visual_task_planner=planner,
        data_root=root,
    )
    summary = asyncio.run(
        dataset_runner.run(root=root, split="test", task=None, sample_concurrency=1)
    )
    sample_dir = run_dir / "tasks" / "auto" / "samples" / storage_key("draft-1")
    assert summary.succeeded == 1
    assert len(planner.calls) == 1
    assert isinstance(planner.calls[0], SampleDraft)
    assert agent.calls[0].task == "general_vqa"
    plan_payload = json.loads((sample_dir / "visual_task_plan.json").read_text(encoding="utf-8"))
    assert plan_payload["version"] == "visual-task-plan-v2"
    assert not (sample_dir / "visual_plan.json").exists()
    assert not (sample_dir / "joint_visual_plan.json").exists()
