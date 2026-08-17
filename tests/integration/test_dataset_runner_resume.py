"""Resume integration for the v2 DatasetRunner path.

v2 DatasetRunner resume 集成测试：succeeded 只做无模型补评测，partial/failed
重新走 planner，append-only predictions 保留历史，legacy 非成功重跑稳定拒绝。
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
from data.schema import GroundTruth, ImageRef, UnifiedSample
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
                version="visual-task-plan-v2",
                task="general_vqa",
                confidence=0.95,
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


def _setup(tmp_path: Path, *, client_status: str = "completed"):
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
        config_payload={"planning_mode": "visual-task-plan-v2"},
        model_ids={"qwen": "logical-qwen"},
        prompt_paths=[],
        run_id="resume-run",
    )
    run_dir = tmp_path / "runs" / "resume-run"
    adapter = _FakeAdapter([_sample()])
    runner = DatasetRunner(
        adapter=adapter,
        sample_runner=sample_runner,
        run_dir=run_dir,
        artifact_writer=ArtifactWriter(),
        call_budget_factory=CallBudgetFactory(),
        visual_task_planner=planner,
        data_root=root,
    )
    return root, run_dir, runner, client, planner


def _run(runner: DatasetRunner, root: Path, *, resume: bool = False):
    return asyncio.run(
        runner.run(root=root, split="test", task="general_vqa", resume=resume, sample_concurrency=1)
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

def test_resume_partial_reruns_v2_planner_and_agent(tmp_path: Path) -> None:
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
    assert client.calls == 1
    assert planner.calls == 1
