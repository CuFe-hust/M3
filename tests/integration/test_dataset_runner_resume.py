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
from workflows.schema import (
    EvidencePreprocessingIdentity,
    SampleRunStatus,
    VQA_ASSISTANCE_SCOPE,
)
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
    evidence_preprocessing: EvidencePreprocessingIdentity | None = (
        EvidencePreprocessingIdentity()
    ),
    vqa_assistance_scope: str | None = VQA_ASSISTANCE_SCOPE,
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
        evidence_preprocessing=evidence_preprocessing,
        vqa_assistance_scope=vqa_assistance_scope,
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


def test_legacy_vqa_evidence_rerun_fails_closed_without_model(tmp_path: Path) -> None:
    """A legacy run (no frozen evidence preprocessing identity) that needs to
    rerun VQA evidence fails stably with the legacy code and zero model calls.
    历史运行（无冻结 evidence 预处理身份）需要重跑 VQA evidence 时以 legacy
    code 稳定失败，且零模型调用。"""
    root, run_dir, runner, client, planner = _setup(
        tmp_path, evidence_preprocessing=None
    )
    assert _run(runner, root).succeeded == 1
    status_path = _sample_dir(run_dir) / "status.json"
    status = SampleRunStatus.model_validate(json.loads(status_path.read_text(encoding="utf-8")))
    ArtifactWriter().write_final_status(
        _sample_dir(run_dir),
        status.model_copy(update={"state": "partial", "error_code": None, "error_message": None}),
    )

    summary = _run(runner, root, resume=True)

    assert summary.failed == 1
    assert client.calls == 1
    assert planner.calls == 1
    rerun_status = SampleRunStatus.model_validate(
        json.loads(status_path.read_text(encoding="utf-8"))
    )
    assert rerun_status.state == "failed"
    assert rerun_status.error_code == "LEGACY_VQA_EVIDENCE_PREPROCESSING_UNSUPPORTED"


def test_legacy_vqa_evidence_succeeded_resume_repairs_nothing(tmp_path: Path) -> None:
    """An old (pre-identity) succeeded sample resumes with zero model calls and
    no evidence repair: only the missing evaluation is supplemented, and no
    vqa_evidence.json is ever fabricated for the legacy sample (14.14).
    历史（无身份）succeeded 样本 resume 时零模型调用且不做 evidence 修复：只
    补判缺失的 evaluation，绝不为此历史样本虚构 vqa_evidence.json（14.14）。"""
    root, run_dir, runner, client, planner = _setup(
        tmp_path, evidence_preprocessing=None
    )
    assert _run(runner, root).succeeded == 1
    assert client.calls == 1 and planner.calls == 1
    sample_dir = _sample_dir(run_dir)
    assert not (sample_dir / "vqa_evidence.json").exists()
    evaluation = sample_dir / "vqa_evaluation.json"
    evaluation.unlink()

    assert _run(runner, root, resume=True).succeeded == 1

    assert client.calls == 1 and planner.calls == 1
    assert evaluation.is_file()
    assert not (sample_dir / "vqa_evidence.json").exists()


def test_legacy_vqa_evidence_draft_resume_fails_closed_before_replanning(
    tmp_path: Path,
) -> None:
    """A draft whose persisted execution task is general_vqa cannot replan
    without an evidence identity. 持久化执行 task 为 general_vqa 的 draft 在缺少
    evidence identity 时不得重新规划。"""
    root, run_dir, runner, client, planner = _setup(
        tmp_path,
        client_status="failed",
        adapter=_FakeDraftAdapter([_draft()]),
        evidence_preprocessing=None,
    )
    assert _run(runner, root, task=None).failed == 1
    sample_dir = run_dir / "tasks" / "auto" / "samples" / storage_key("resume-draft-1")
    status_path = sample_dir / "status.json"
    persisted = SampleRunStatus.model_validate(
        json.loads(status_path.read_text(encoding="utf-8"))
    )
    assert persisted.task == "general_vqa"
    planner_calls = planner.calls
    client_calls = client.calls

    summary = _run(runner, root, task=None, resume=True)

    assert summary.failed == 1
    status = SampleRunStatus.model_validate(json.loads(status_path.read_text(encoding="utf-8")))
    assert status.error_code == "LEGACY_VQA_EVIDENCE_PREPROCESSING_UNSUPPORTED"
    assert planner.calls == planner_calls
    assert client.calls == client_calls


def test_legacy_vqa_assistance_scope_rerun_fails_closed_for_vqa_tasks(
    tmp_path: Path,
) -> None:
    """A legacy run without the frozen VQA assistance scope that needs to
    replan any GeneralVQAAgent task fails stably with the legacy scope code
    and zero model calls — the new evidence behavior is never silently
    adopted. 缺少冻结 VQA assistance scope 的历史运行需要重新规划任一
    GeneralVQAAgent task 时，以 legacy scope code 稳定失败且零模型调用——绝不
    静默采用新 evidence 行为。"""
    sample = _sample().model_copy(update={"task": "scene_classification"})
    root, run_dir, runner, client, planner = _setup(
        tmp_path,
        client_status="failed",
        adapter=_FakeAdapter([sample]),
        vqa_assistance_scope=None,
    )
    assert _run(runner, root).failed == 1
    sample_dir = _sample_dir(run_dir)
    status_path = sample_dir / "status.json"
    persisted = SampleRunStatus.model_validate(
        json.loads(status_path.read_text(encoding="utf-8"))
    )
    ArtifactWriter().write_final_status(
        sample_dir,
        persisted.model_copy(
            update={"state": "partial", "error_code": None, "error_message": None}
        ),
    )
    planner_calls = planner.calls
    client_calls = client.calls

    summary = _run(runner, root, resume=True)

    assert summary.failed == 1
    status = SampleRunStatus.model_validate(
        json.loads(status_path.read_text(encoding="utf-8"))
    )
    assert status.error_code == "LEGACY_VQA_ASSISTANCE_SCOPE_UNSUPPORTED"
    # The persisted execution task (planner-selected general_vqa for the fake
    # planner) stays authoritative and is itself a GeneralVQAAgent task.
    # 持久化执行 task（fake planner 选定的 general_vqa）保持权威，且本身是
    # GeneralVQAAgent task。
    assert status.task == "general_vqa"
    assert planner.calls == planner_calls
    assert client.calls == client_calls
    assert not (sample_dir / "vqa_evidence.json").exists()


def test_legacy_scope_gate_uses_persisted_task_not_source_task(tmp_path: Path) -> None:
    """Regression (review): the legacy scope gate must use the persisted
    execution task, not the adapter source task — the planner may rewrite the
    source. Source caption + persisted general_vqa + legacy scope must fail
    closed instead of replanning under the new evidence behavior.
    回归（评审）：legacy scope 门禁必须使用持久化 execution task 而非 adapter
    source task——planner 可能改写 source。source=caption + persisted=
    general_vqa + legacy scope 必须严格失败，绝不按新 evidence 行为重规划。"""
    # Source task is caption (NOT in the VQA task set); the fake planner
    # rewrites it to general_vqa on the fresh run, so the persisted execution
    # task is general_vqa. source task 是 caption（不在 VQA task 集合）；fresh
    # 运行时 fake planner 将其改写为 general_vqa，因此持久化 execution task
    # 是 general_vqa。
    sample = _sample().model_copy(update={"task": "caption"})
    root, run_dir, runner, client, planner = _setup(
        tmp_path,
        adapter=_FakeAdapter([sample]),
        vqa_assistance_scope=None,
    )
    assert _run(runner, root).succeeded == 1
    sample_dir = _sample_dir(run_dir)
    status_path = sample_dir / "status.json"
    persisted = SampleRunStatus.model_validate(
        json.loads(status_path.read_text(encoding="utf-8"))
    )
    assert persisted.task == "general_vqa"
    ArtifactWriter().write_final_status(
        sample_dir,
        persisted.model_copy(
            update={"state": "partial", "error_code": None, "error_message": None}
        ),
    )
    planner_calls = planner.calls
    client_calls = client.calls

    summary = _run(runner, root, resume=True)

    assert summary.failed == 1
    status = SampleRunStatus.model_validate(
        json.loads(status_path.read_text(encoding="utf-8"))
    )
    assert status.error_code == "LEGACY_VQA_ASSISTANCE_SCOPE_UNSUPPORTED"
    assert status.task == "general_vqa"
    assert planner.calls == planner_calls
    assert client.calls == client_calls
    assert not (sample_dir / "vqa_evidence.json").exists()


def test_legacy_preprocessing_gate_uses_persisted_task_not_source_task(
    tmp_path: Path,
) -> None:
    """Regression (review): the legacy evidence-preprocessing gate also uses
    the persisted execution task. Source caption + persisted general_vqa +
    missing preprocessing identity fails with the preprocessing legacy code.
    回归（评审）：legacy evidence-preprocessing 门禁同样使用持久化 execution
    task。source=caption + persisted=general_vqa + 缺 preprocessing 身份时以
    preprocessing legacy code 失败。"""
    sample = _sample().model_copy(update={"task": "caption"})
    root, run_dir, runner, client, planner = _setup(
        tmp_path,
        adapter=_FakeAdapter([sample]),
        evidence_preprocessing=None,
    )
    assert _run(runner, root).succeeded == 1
    sample_dir = _sample_dir(run_dir)
    status_path = sample_dir / "status.json"
    persisted = SampleRunStatus.model_validate(
        json.loads(status_path.read_text(encoding="utf-8"))
    )
    assert persisted.task == "general_vqa"
    ArtifactWriter().write_final_status(
        sample_dir,
        persisted.model_copy(
            update={"state": "partial", "error_code": None, "error_message": None}
        ),
    )
    planner_calls = planner.calls
    client_calls = client.calls

    summary = _run(runner, root, resume=True)

    assert summary.failed == 1
    status = SampleRunStatus.model_validate(
        json.loads(status_path.read_text(encoding="utf-8"))
    )
    assert status.error_code == "LEGACY_VQA_EVIDENCE_PREPROCESSING_UNSUPPORTED"
    assert status.task == "general_vqa"
    assert planner.calls == planner_calls
    assert client.calls == client_calls


def test_legacy_scope_gate_fails_closed_on_unknown_persisted_task(
    tmp_path: Path,
) -> None:
    """A legacy run whose persisted execution task is the honest 'unknown'
    sentinel (pre-task failure) cannot prove the replan stays outside the VQA
    family and fails closed with the legacy scope code, zero planner calls.
    持久化 execution task 为诚实 'unknown' 哨兵（预 task 失败）的历史运行无法
    证明重规划会留在 VQA 族之外，以 legacy scope code 严格失败且零规划调用。"""
    sample = _sample().model_copy(update={"task": "caption"})
    root, run_dir, runner, client, planner = _setup(
        tmp_path,
        adapter=_FakeAdapter([sample]),
        vqa_assistance_scope=None,
    )
    assert _run(runner, root).succeeded == 1
    sample_dir = _sample_dir(run_dir)
    status_path = sample_dir / "status.json"
    persisted = SampleRunStatus.model_validate(
        json.loads(status_path.read_text(encoding="utf-8"))
    )
    ArtifactWriter().write_final_status(
        sample_dir,
        persisted.model_copy(
            update={"task": "unknown", "state": "partial", "error_code": None, "error_message": None}
        ),
    )
    planner_calls = planner.calls
    client_calls = client.calls

    summary = _run(runner, root, resume=True)

    assert summary.failed == 1
    status = SampleRunStatus.model_validate(
        json.loads(status_path.read_text(encoding="utf-8"))
    )
    assert status.error_code == "LEGACY_VQA_ASSISTANCE_SCOPE_UNSUPPORTED"
    assert status.task == "unknown"
    assert planner.calls == planner_calls
    assert client.calls == client_calls


def test_legacy_scope_gate_allows_replan_for_persisted_non_vqa_task(
    tmp_path: Path,
) -> None:
    """A legacy run whose persisted execution task is not a GeneralVQAAgent
    task (e.g. caption) keeps the documented replan behavior: the scope gate
    must not over-block non-VQA execution tasks.
    持久化 execution task 不是 GeneralVQAAgent task（如 caption）的历史运行保
    持文档化重规划行为：scope 门禁不得过度拦截非 VQA 执行任务。"""
    root, run_dir, runner, client, planner = _setup(
        tmp_path,
        vqa_assistance_scope=None,
    )
    sample_dir = _sample_dir(run_dir)
    status_path = sample_dir / "status.json"
    sample_dir.mkdir(parents=True, exist_ok=True)
    ArtifactWriter().write_final_status(
        sample_dir,
        SampleRunStatus(
            sample_id="resume-1",
            task="caption",
            state="partial",
            updated_at="2026-08-21T00:00:00+00:00",
        ),
    )
    planner_calls = planner.calls
    client_calls = client.calls

    summary = _run(runner, root, resume=True)

    # The gate does not fire; the sample replans through the v5 seam.
    # 门禁不触发；样本经 v5 seam 重新规划。
    assert summary.succeeded == 1
    assert planner.calls == planner_calls + 1
    assert client.calls == client_calls + 1
    status = SampleRunStatus.model_validate(
        json.loads(status_path.read_text(encoding="utf-8"))
    )
    assert status.task == "general_vqa"  # planner rewrote it again / planner 再次改写


def test_legacy_vqa_assistance_scope_draft_resume_fails_closed_before_replanning(
    tmp_path: Path,
) -> None:
    """A draft whose persisted execution task is a GeneralVQAAgent task cannot
    replan without the frozen assistance scope. 持久化执行 task 为
    GeneralVQAAgent task 的 draft 在缺少冻结 assistance scope 时不得重新规划。"""
    root, run_dir, runner, client, planner = _setup(
        tmp_path,
        client_status="failed",
        adapter=_FakeDraftAdapter([_draft()]),
        vqa_assistance_scope=None,
        evidence_preprocessing=EvidencePreprocessingIdentity(),
    )
    assert _run(runner, root, task=None).failed == 1
    sample_dir = run_dir / "tasks" / "auto" / "samples" / storage_key("resume-draft-1")
    status_path = sample_dir / "status.json"
    persisted = SampleRunStatus.model_validate(
        json.loads(status_path.read_text(encoding="utf-8"))
    )
    assert persisted.task == "general_vqa"
    planner_calls = planner.calls
    client_calls = client.calls

    summary = _run(runner, root, task=None, resume=True)

    assert summary.failed == 1
    status = SampleRunStatus.model_validate(
        json.loads(status_path.read_text(encoding="utf-8"))
    )
    assert status.error_code == "LEGACY_VQA_ASSISTANCE_SCOPE_UNSUPPORTED"
    assert planner.calls == planner_calls
    assert client.calls == client_calls


def test_legacy_scope_gate_draft_fails_closed_on_unknown_persisted_task(
    tmp_path: Path,
) -> None:
    """A legacy draft whose persisted execution task is the 'unknown' sentinel
    (the fresh run failed before task resolution) cannot prove the replan
    stays outside the VQA family and fails closed with the legacy scope code.
    持久化 execution task 为 'unknown' 哨兵（fresh 运行在任务解析前失败）的
    历史 draft 无法证明重规划会留在 VQA 族之外，以 legacy scope code 严格
    失败。"""
    root, run_dir, runner, client, planner = _setup(
        tmp_path,
        client_status="failed",
        adapter=_FakeDraftAdapter([_draft()]),
        vqa_assistance_scope=None,
        evidence_preprocessing=EvidencePreprocessingIdentity(),
    )
    assert _run(runner, root, task=None).failed == 1
    sample_dir = run_dir / "tasks" / "auto" / "samples" / storage_key("resume-draft-1")
    status_path = sample_dir / "status.json"
    persisted = SampleRunStatus.model_validate(
        json.loads(status_path.read_text(encoding="utf-8"))
    )
    ArtifactWriter().write_final_status(
        sample_dir,
        persisted.model_copy(
            update={"task": "unknown", "state": "partial", "error_code": None, "error_message": None}
        ),
    )
    planner_calls = planner.calls
    client_calls = client.calls

    summary = _run(runner, root, task=None, resume=True)

    assert summary.failed == 1
    status = SampleRunStatus.model_validate(
        json.loads(status_path.read_text(encoding="utf-8"))
    )
    assert status.error_code == "LEGACY_VQA_ASSISTANCE_SCOPE_UNSUPPORTED"
    assert status.task == "unknown"
    assert planner.calls == planner_calls
    assert client.calls == client_calls


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
