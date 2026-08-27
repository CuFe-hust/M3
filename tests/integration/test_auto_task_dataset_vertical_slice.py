"""Vertical slices for explicit and draft DatasetRunner v3 planning.

显式样本与 SampleDraft 两条 DatasetRunner 垂直切片：两者都经同一个 v3
planner，draft 只在 planner 返回 task 后物化为 UnifiedSample。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agents.base import AgentExecution, VisualPlanBindings
from agents.general_vqa import GeneralVQAAgent
from agents.general_vqa.evidence.executor import EvidenceExecution
from agents.general_vqa.evidence.schema import RoiEvidenceRecord, VqaEvidenceBundle
from agents.registry import AgentRegistry
from agents.schema import AgentResult, MaterializedVisualView, VisualTaskPlan
from data.adapters.base import AdapterProbe
from data.schema import GroundTruth, ImageRef, SampleDraft, TaskNormalization, UnifiedSample
from models.base import ModelCacheIdentity
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
            version="visual-task-plan-v5",
            task="general_vqa",
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
        config_payload={"planning_mode": "visual-task-plan-v5"},
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


def test_explicit_dataset_sample_uses_v3_planner(tmp_path: Path) -> None:
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


def test_draft_auto_task_materializes_after_v3_planning(tmp_path: Path) -> None:
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
    assert plan_payload["version"] == "visual-task-plan-v5"
    assert "confidence" not in plan_payload
    assert not (sample_dir / "visual_plan.json").exists()
    assert not (sample_dir / "joint_visual_plan.json").exists()


class _EvidenceClient:
    """Record final-Qwen calls and return a schema-valid AgentResult.
    记录 final-Qwen 调用并返回 schema 合法的 AgentResult。"""

    def __init__(self, answer: str = "road") -> None:
        self.answer = answer
        self.calls = 0

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="fake-evidence",
            generation={"temperature": 0.0},
            client_version="test",
        )

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls += 1
        return response_model.model_validate(
            {
                "agent_name": "general_vqa_agent",
                "answer": self.answer,
                "status": "completed",
            }
        )


class _FakeEvidenceService:
    """One empty full-image ROI bundle; no model runs inside.
    一个空整图 ROI bundle；内部不运行任何模型。"""

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, plan, images, *, fallback_image_id, materialized_views):
        self.calls += 1
        size = images[fallback_image_id].size
        roi = RoiEvidenceRecord(
            roi_id="full",
            image_id=fallback_image_id,
            source_size=size,
            core_xyxy=(0, 0, *size),
            expanded_xyxy=(0, 0, *size),
            crop_size=size,
        )
        bundle = VqaEvidenceBundle(
            catalog_version="test-catalog-v1",
            preprocessing_version="greedy-1024-stretch-v1",
            rois=[roi],
            missing_leaves=["small-vehicle"],
            leaf_states={"small-vehicle": "missing"},
        )
        return EvidenceExecution(
            bundle=bundle,
            layer_states=(),
            outcomes=(),
            preview_evidence=(),
            palette={},
        )


def _evidence_runner(
    run_dir: Path,
    root: Path,
    client: _EvidenceClient,
    service: _FakeEvidenceService,
    *,
    sample: UnifiedSample,
    planned_task: str = "scene_classification",
) -> DatasetRunner:
    """Assemble a DatasetRunner whose SampleRunner carries the real
    GeneralVQAAgent plus the injected evidence service.
    组装 DatasetRunner：其 SampleRunner 携带真实 GeneralVQAAgent 与注入的
    evidence 服务。"""
    registry = AgentRegistry()
    registry.register(GeneralVQAAgent(client))
    runner = SampleRunner(
        registry=registry,
        router=TaskRouter(),
        qwen_client=client,
        artifact_writer=ArtifactWriter(),
        call_budget_factory=CallBudgetFactory(),
        data_root=root,
        visual_bindings=VisualPlanBindings(vqa_evidence=service),
    )
    return DatasetRunner(
        adapter=_FakeAdapter(sample, _draft_sample(root)),
        sample_runner=runner,
        run_dir=run_dir,
        artifact_writer=ArtifactWriter(),
        call_budget_factory=CallBudgetFactory(),
        visual_task_planner=_PlannedPlanner(planned_task),
        data_root=root,
    )


def _draft_sample(root: Path) -> SampleDraft:
    return SampleDraft(
        sample_id="draft-1",
        dataset="auto-demo",
        split="test",
        images=[ImageRef(image_id="img1", path="img.png", role="image")],
        question="Is there a road?",
    )


class _PlannedPlanner:
    """Planner that always selects one declared task with evidence on.
    总是选择声明 task 且开启 evidence 的规划器。"""

    def __init__(self, task: str) -> None:
        self.task = task
        self.calls = 0

    async def plan_with_views(self, view, *, data_root, artifact_dir, budget):
        self.calls += 1
        plan = VisualTaskPlan(
            version="visual-task-plan-v5",
            task=self.task,  # type: ignore[arg-type]
            needs_visual_assistance=True,
            object_categories=["small-vehicle"],
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


def test_scene_classification_evidence_vertical_slice(tmp_path: Path) -> None:
    """A non-general_vqa GeneralVQAAgent task runs the full chain planner ->
    router -> evidence -> result -> evaluation, and the persisted status task
    stays the executed task with the general_vqa evaluation family.
    非 general_vqa 的 GeneralVQAAgent task 运行完整链路 planner -> router ->
    evidence -> result -> evaluation，持久化状态 task 保持执行 task，评测族为\n    general_vqa。"""
    root = tmp_path / "data"
    root.mkdir()
    from PIL import Image

    Image.new("RGB", (12, 8), (1, 2, 3)).save(root / "img.png", format="PNG")
    sample = UnifiedSample(
        sample_id="scene-1",
        dataset="auto-demo",
        split="test",
        task="scene_classification",
        images=[ImageRef(image_id="img1", path="img.png", role="image")],
        question="What land use is shown?",
        ground_truth=GroundTruth(answers=["road"]),
    )
    run_store = RunStore(tmp_path / "runs", tmp_path)
    run_store.create_run(
        config_payload={"planning_mode": "visual-task-plan-v5"},
        model_ids={"qwen": "logical-qwen"},
        prompt_paths=[],
        run_id="scene-run",
    )
    run_dir = tmp_path / "runs" / "scene-run"
    client = _EvidenceClient(answer="road")
    service = _FakeEvidenceService()
    dataset_runner = _evidence_runner(
        run_dir,
        root,
        client,
        service,
        sample=sample,
    )
    summary = asyncio.run(
        dataset_runner.run(root=root, split="test", task="scene_classification", sample_concurrency=1)
    )
    assert summary.succeeded == 1
    assert service.calls == 1
    assert client.calls == 1  # exactly one final Qwen / 恰好一次 final Qwen
    sample_dir = (
        run_dir / "tasks" / "scene_classification" / "samples" / storage_key("scene-1")
    )
    status = json.loads((sample_dir / "status.json").read_text(encoding="utf-8"))
    assert status["task"] == "scene_classification"
    assert status["state"] == "succeeded"
    assert (sample_dir / "vqa_evidence.json").is_file()
    assert (sample_dir / "vqa_evaluation.json").is_file()
    plan_payload = json.loads(
        (sample_dir / "visual_task_plan.json").read_text(encoding="utf-8")
    )
    assert plan_payload["task"] == "scene_classification"


def test_multiple_choice_evidence_vertical_slice_enforces_choices(
    tmp_path: Path,
) -> None:
    """End-to-end: a multiple_choice_vqa evidence sample whose final answer is
    not among the choices ends partial with the constraint violation recorded,
    and the evidence bundle is still persisted. 端到端：最终答案不在选项中的\n    multiple_choice_vqa evidence 样本以 partial 结束并记录约束违规，证据包\n    仍然持久化。"""
    root = tmp_path / "data"
    root.mkdir()
    from PIL import Image

    Image.new("RGB", (12, 8), (1, 2, 3)).save(root / "img.png", format="PNG")
    sample = UnifiedSample(
        sample_id="mc-1",
        dataset="auto-demo",
        split="test",
        task="multiple_choice_vqa",
        images=[ImageRef(image_id="img1", path="img.png", role="image")],
        question="Which class is shown?",
        ground_truth=GroundTruth(answers=["road"]),
        normalization=TaskNormalization(
            source_task="multiple_choice_vqa",
            normalized_task="multiple_choice_vqa",  # type: ignore[arg-type]
            normalizer="test",
            version="1",
            choices=["road", "water"],
        ),
    )
    run_store = RunStore(tmp_path / "runs", tmp_path)
    run_store.create_run(
        config_payload={"planning_mode": "visual-task-plan-v5"},
        model_ids={"qwen": "logical-qwen"},
        prompt_paths=[],
        run_id="mc-run",
    )
    run_dir = tmp_path / "runs" / "mc-run"
    client = _EvidenceClient(answer="grass")  # not among the choices / 不在选项中
    service = _FakeEvidenceService()
    dataset_runner = _evidence_runner(
        run_dir,
        root,
        client,
        service,
        sample=sample,
        planned_task="multiple_choice_vqa",
    )
    summary = asyncio.run(
        dataset_runner.run(root=root, split="test", task="multiple_choice_vqa", sample_concurrency=1)
    )
    assert summary.partial == 1
    assert service.calls == 1
    assert client.calls == 1
    sample_dir = run_dir / "tasks" / "multiple_choice_vqa" / "samples" / storage_key("mc-1")
    status = json.loads((sample_dir / "status.json").read_text(encoding="utf-8"))
    assert status["task"] == "multiple_choice_vqa"
    assert status["state"] == "partial"
    assert (sample_dir / "vqa_evidence.json").is_file()
    agent_result = json.loads(
        (sample_dir / "agent_result.json").read_text(encoding="utf-8")
    )
    assert agent_result["status"] == "partial"
    assert (
        agent_result["geometry"]["answer_constraint_violation"]
        == "answer 'grass' does not map to a single choice"
    )
