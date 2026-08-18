"""Resume integration for the versioned DatasetRunner path.

版本化 DatasetRunner resume 集成测试：succeeded 只做无模型补评测，v5
partial/failed 可重走 planner，历史 v4/v2/legacy 非成功重跑稳定拒绝。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from PIL import Image

from agents.general_vqa import GeneralVQAAgent
from agents.registry import AgentRegistry
from agents.schema import MaterializedVisualView, VisualTaskPlan
from data.adapters.base import AdapterProbe
from data.schema import GroundTruth, ImageRef, SampleDraft, UnifiedSample
from routing.router import TaskRouter
from workflows.artifact_writer import ArtifactWriter
from workflows.call_budget import CallBudgetFactory
from workflows.dataset_runner import DatasetRunner, storage_key
from workflows.run_store import RunStore
from workflows.sample_runner import SampleRunner
from workflows.schema import SampleRunStatus
from models.base import ModelCacheIdentity


class _FakeClient:
    def __init__(self, status: str = "completed") -> None:
        self.status = status
        self.calls = 0

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(model="fake-model", generation={"temperature": 0.0}, client_version="test")

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls += 1
        return response_model.model_validate(
            {
                "agent_name": "general_vqa_agent",
                "answer": "yes",
                "status": self.status,
            }
        )


class _FakePlanner:
    def __init__(self) -> None:
        self.calls = 0

    async def plan_with_views(self, sample, *, data_root, artifact_dir, budget):
        self.calls += 1
        ref = sample.images[0]
        size = Image.open(data_root / ref.path).size
        return (
            VisualTaskPlan(
                version="visual-task-plan-v5",
                task="general_vqa",
                reason_codes=["test"],
            ),
            (
                MaterializedVisualView(
                    image_id=ref.image_id,
                    view_mode="full_image",
                    source_size=size,
                    crop_xyxy=(0, 0, size[0], size[1]),
                    crop_size=size,
                ),
            ),
        )


class _FakeAdapter:
    name = "fake"
    supported_tasks = frozenset({"general_vqa"})

    def __init__(self, samples: list[UnifiedSample]) -> None:
        self.samples = samples

    def probe(self, root: Path, task: str | None = None) -> AdapterProbe:
        return AdapterProbe(
            dataset=self.name,
            version="1",
            sample_file=root / "samples.jsonl",
            observed_fields=("id",),
            sample_count=len(self.samples),
            task=task,
            available_tasks=("general_vqa",),
        )

    def iter_samples(self, root: Path, split: str, task: str):
        yield from self.samples


class _FakeDraftAdapter:
    name = "fake-draft"
    supported_tasks = frozenset()

    def __init__(self, drafts: list[SampleDraft]) -> None:
        self.drafts = drafts

    def probe(self, root: Path, task: str | None = None) -> AdapterProbe:
        return AdapterProbe(
            dataset=self.name,
            version="1",
            sample_file=root / "drafts.jsonl",
            observed_fields=("id",),
            sample_count=len(self.drafts),
            task=task,
            available_tasks=(),
        )

    def iter_drafts(self, root: Path, split: str):
        yield from self.drafts


def _sample(sample_id: str = "resume-1") -> UnifiedSample:
    return UnifiedSample(
        sample_id=sample_id,
        dataset="fake",
        split="test",
        task="general_vqa",
        images=[ImageRef(image_id="img1", path="img.png", role="image")],
        question="Is there a road?",
        ground_truth=GroundTruth(answers=["yes"]),
    )


def _draft(sample_id: str = "resume-draft-1") -> SampleDraft:
    return SampleDraft(
        sample_id=sample_id,
        dataset="fake-draft",
        split="test",
        images=[ImageRef(image_id="img1", path="img.png", role="image")],
        question="Is there a road?",
    )


def _setup(
    tmp_path: Path,
    *,
    client_status: str = "completed",
    planning_mode: str = "visual-task-plan-v5",
    adapter=None,
):
    root = tmp_path / "data"
    root.mkdir()
    Image.new("RGB", (16, 10), (1, 2, 3)).save(root / "img.png", format="PNG")
    (root / "samples.jsonl").write_text("{}\n", encoding="utf-8")
    client = _FakeClient(client_status)
    planner = _FakePlanner()
    registry = AgentRegistry()
    registry.register(GeneralVQAAgent(client))
    sample_runner = SampleRunner(
        registry=registry,
        router=TaskRouter(),
        qwen_client=client,
        artifact_writer=ArtifactWriter(),
        call_budget_factory=CallBudgetFactory(),
        data_root=root,
    )
    run_store = RunStore(tmp_path / "runs", tmp_path)
    run_store.create_run(
        config_payload={"planning_mode": planning_mode},
        model_ids={"qwen": "logical-qwen"},
        prompt_paths=[],
        run_id="resume-run",
    )
    run_dir = tmp_path / "runs" / "resume-run"
    adapter = adapter if adapter is not None else _FakeAdapter([_sample()])
    runner = DatasetRunner(
        adapter=adapter,
        sample_runner=sample_runner,
        run_dir=run_dir,
        artifact_writer=ArtifactWriter(),
        call_budget_factory=CallBudgetFactory(),
        visual_task_planner=planner,
        planning_mode=planning_mode,
        data_root=root,
    )
    return root, run_dir, runner, client, planner


def _run(
    runner: DatasetRunner,
    root: Path,
    *,
    task: str | None = "general_vqa",
    resume: bool = False,
):
    return asyncio.run(
        runner.run(root=root, split="test", task=task, resume=resume, sample_concurrency=1)
    )


def _sample_dir(run_dir: Path) -> Path:
    return run_dir / "tasks" / "general_vqa" / "samples" / storage_key("resume-1")


def test_resume_succeeded_supplements_missing_evaluation_without_model(tmp_path: Path) -> None:
    root, run_dir, runner, client, planner = _setup(tmp_path)
    assert _run(runner, root).succeeded == 1
    assert client.calls == 1 and planner.calls == 1
    evaluation = _sample_dir(run_dir) / "vqa_evaluation.json"
    evaluation.unlink()

    assert _run(runner, root, resume=True).succeeded == 1
    assert evaluation.is_file()
    assert client.calls == 1
    assert planner.calls == 1


def test_v2_succeeded_resume_supplements_without_model(tmp_path: Path) -> None:
    root, run_dir, runner, client, planner = _setup(tmp_path)
    assert _run(runner, root).succeeded == 1
    runner.planning_mode = "visual-task-plan-v2"
    evaluation = _sample_dir(run_dir) / "vqa_evaluation.json"
    evaluation.unlink()

    assert _run(runner, root, resume=True).succeeded == 1
    assert evaluation.is_file()
    assert client.calls == 1 and planner.calls == 1


def test_resume_partial_reruns_v5_planner_and_agent(tmp_path: Path) -> None:
    root, run_dir, runner, client, planner = _setup(tmp_path)
    _run(runner, root)
    status_path = _sample_dir(run_dir) / "status.json"
    status = SampleRunStatus.model_validate(json.loads(status_path.read_text(encoding="utf-8")))
    ArtifactWriter().write_final_status(
        _sample_dir(run_dir),
        status.model_copy(update={"state": "partial", "error_code": None, "error_message": None}),
    )
    assert _run(runner, root, resume=True).succeeded == 1
    assert client.calls == 2
    assert planner.calls == 2


def test_predictions_are_append_only_across_resume(tmp_path: Path) -> None:
    root, run_dir, runner, _client, _planner = _setup(tmp_path)
    _run(runner, root)
    _run(runner, root, resume=True)
    rows = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 2
    assert rows[-1]["status"] == "succeeded"


def test_legacy_resume_rejects_non_successful_rerun_stably(tmp_path: Path) -> None:
    root, run_dir, runner, client, planner = _setup(tmp_path, client_status="failed")
    first = _run(runner, root)
    assert first.failed == 1
    persisted = SampleRunStatus.model_validate(
        json.loads((_sample_dir(run_dir) / "status.json").read_text(encoding="utf-8"))
    )
    ArtifactWriter().write_final_status(
        _sample_dir(run_dir),
        persisted.model_copy(update={"task": "caption", "state": "partial"}),
    )
    legacy = DatasetRunner(
        adapter=runner.adapter,
        sample_runner=runner.sample_runner,
        run_dir=run_dir,
        artifact_writer=ArtifactWriter(),
        planning_mode="legacy",
        data_root=root,
    )
    summary = _run(legacy, root, resume=True)
    status = json.loads((_sample_dir(run_dir) / "status.json").read_text(encoding="utf-8"))
    assert summary.failed == 1
    assert status["error_code"] == "LEGACY_PLANNING_RESUME_UNSUPPORTED"
    assert status["task"] == "caption"
    rows = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert rows[-1]["task"] == "caption"
    assert client.calls == 1
    assert planner.calls == 1


def test_v2_resume_rejects_non_successful_rerun_without_model(tmp_path: Path) -> None:
    root, run_dir, runner, client, planner = _setup(tmp_path, client_status="failed")
    first = _run(runner, root)
    assert first.failed == 1
    assert client.calls == 1 and planner.calls == 1


def test_v4_resume_rejects_non_successful_rerun_without_reinterpreting_roi(
    tmp_path: Path,
) -> None:
    root, run_dir, runner, client, planner = _setup(tmp_path, client_status="failed")
    first = _run(runner, root)
    assert first.failed == 1
    persisted = SampleRunStatus.model_validate(
        json.loads((_sample_dir(run_dir) / "status.json").read_text(encoding="utf-8"))
    )
    ArtifactWriter().write_final_status(
        _sample_dir(run_dir),
        persisted.model_copy(update={"task": "caption", "state": "partial"}),
    )
    runner.planning_mode = "visual-task-plan-v4"
    status = _run(runner, root, resume=True)
    persisted_payload = json.loads(
        (_sample_dir(run_dir) / "status.json").read_text(encoding="utf-8")
    )
    assert status.failed == 1
    assert persisted_payload["error_code"] == "LEGACY_PLANNING_RESUME_UNSUPPORTED"
    assert persisted_payload["task"] == "caption"
    assert client.calls == 1 and planner.calls == 1
    persisted = SampleRunStatus.model_validate(
        json.loads((_sample_dir(run_dir) / "status.json").read_text(encoding="utf-8"))
    )
    ArtifactWriter().write_final_status(
        _sample_dir(run_dir),
        persisted.model_copy(update={"task": "caption", "state": "partial"}),
    )
    runner.planning_mode = "visual-task-plan-v2"
    status = _run(runner, root, resume=True)
    persisted = json.loads((_sample_dir(run_dir) / "status.json").read_text(encoding="utf-8"))
    assert status.failed == 1
    assert persisted["error_code"] == "LEGACY_PLANNING_RESUME_UNSUPPORTED"
    assert persisted["task"] == "caption"
    assert client.calls == 1 and planner.calls == 1


def test_legacy_resume_preserves_persisted_task_for_draft(tmp_path: Path) -> None:
    adapter = _FakeDraftAdapter([_draft()])
    root, run_dir, runner, client, planner = _setup(
        tmp_path,
        client_status="failed",
        adapter=adapter,
    )
    first = _run(runner, root, task=None)
    assert first.failed == 1
    sample_dir = run_dir / "tasks" / "auto" / "samples" / storage_key("resume-draft-1")
    persisted = SampleRunStatus.model_validate(
        json.loads((sample_dir / "status.json").read_text(encoding="utf-8"))
    )
    ArtifactWriter().write_final_status(
        sample_dir,
        persisted.model_copy(update={"task": "caption", "state": "partial"}),
    )
    legacy = DatasetRunner(
        adapter=runner.adapter,
        sample_runner=runner.sample_runner,
        run_dir=run_dir,
        artifact_writer=ArtifactWriter(),
        planning_mode="legacy",
        data_root=root,
    )

    summary = _run(legacy, root, task=None, resume=True)
    status = json.loads((sample_dir / "status.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert summary.failed == 1
    assert status["error_code"] == "LEGACY_PLANNING_RESUME_UNSUPPORTED"
    assert status["task"] == "caption"
    assert rows[-1]["run_task"] == "auto"
    assert rows[-1]["task"] == "caption"
    assert client.calls == 1
    assert planner.calls == 1
