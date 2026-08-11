"""Integration: DatasetRunner resume semantics with the real SampleRunner.

集成：真实 SampleRunner 下的 DatasetRunner resume 语义。离线：fake 模型
客户端与 fake judge 客户端，真实 GeneralVQAAgent。覆盖 succeeded 不重新
推理、缺失确定性评估只补、缺失/失败 judge 只补、补判异常降级 skipped、
partial/failed/running/损坏状态重新执行。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from PIL import Image

from agents.counting.schema import CountingResult, GlobalPointObservation
from agents.general_vqa import GeneralVQAAgent
from agents.registry import AgentRegistry
from agents.schema import AgentResult
from data.adapters.base import AdapterProbe
from data.schema import GroundTruth, ImageRef, UnifiedSample
from evaluation.judges.base import VQAAnswerJudgeResult
from models.base import ModelCacheIdentity
from routing.router import TaskRouter
from workflows.artifact_writer import ArtifactWriter
from workflows.call_budget import CallBudgetFactory
from workflows.dataset_runner import DatasetRunner, storage_key
from workflows.judge_service import JudgeService
from workflows.run_store import RunStore
from workflows.sample_runner import SampleRunner


class _FakeClient:
    """Minimal VisionLanguageClient with configurable answer status.
    可配置答案状态的最小 VisionLanguageClient。"""

    def __init__(self, status: str = "completed") -> None:
        self.status = status
        self.calls = 0

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="fake-model",
            generation={"temperature": 0.0, "do_sample": False, "max_tokens": 128},
            client_version="1",
        )

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls += 1
        return response_model.model_validate(
            {
                "agent_name": "general_vqa_agent",
                "answer": "yes",
                "status": self.status,
            }
        )


class _FakeJudgeClient:
    def __init__(self, verdict: VQAAnswerJudgeResult, error: Exception | None = None) -> None:
        self.verdict = verdict
        self.error = error
        self.calls = 0

    def judge(self, payload, *, request_meta):
        return self.judge_json(
            payload, response_model=type(self.verdict), request_meta=request_meta
        )

    def judge_json(self, payload, *, response_model, request_meta, system_prompt=None):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return response_model.model_validate(self.verdict.model_dump())


class _FakeAdapter:
    name = "fake"
    supported_tasks = frozenset({"general_vqa", "counting"})

    def __init__(self, samples: list[UnifiedSample]) -> None:
        self._samples = samples

    def probe(self, root: Path, task: str | None = None) -> AdapterProbe:
        return AdapterProbe(
            dataset="fake",
            version="1",
            sample_file=root / "samples.jsonl",  # root-anchored / 锚定 root
            observed_fields=("id",),
            sample_count=len(self._samples),
            task=task,
            available_tasks=("general_vqa", "counting"),
        )

    def iter_samples(self, root: Path, split: str, task: str):
        for sample in self._samples:
            yield sample


def _vqa_sample() -> UnifiedSample:
    return UnifiedSample(
        sample_id="resume-1",
        dataset="fake",
        split="test",
        task="general_vqa",
        images=[ImageRef(image_id="i0", path="img.png", role="image")],
        question="Is there a road?",
        ground_truth=GroundTruth(answers=["yes"]),
    )


def _vqa_sample2() -> UnifiedSample:
    return UnifiedSample(
        sample_id="resume-2",
        dataset="fake",
        split="test",
        task="general_vqa",
        images=[ImageRef(image_id="i0", path="img.png", role="image")],
        question="Is there a road?",
        ground_truth=GroundTruth(answers=["yes"]),
    )


def _counting_result() -> CountingResult:
    points = []
    for index in range(2):
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
        sample_id="resume-count",
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
        final_count=2,
        status="completed",
    )


def _counting_sample() -> UnifiedSample:
    return UnifiedSample(
        sample_id="resume-count",
        dataset="fake",
        split="test",
        task="counting",
        images=[ImageRef(image_id="i0", path="img.png", role="image")],
        question="How many cars?",
        ground_truth=GroundTruth(count=2),
    )


def _setup(
    tmp_path: Path,
    *,
    client_status: str = "completed",
    judge_service: JudgeService | None = None,
    judge_policy: str = "none",
    adapter_samples: list[UnifiedSample] | None = None,
):
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (10, 20, 30)).save(root / "img.png", format="PNG")
    client = _FakeClient(status=client_status)
    registry = AgentRegistry()
    registry.register(GeneralVQAAgent(client))
    run_id = "resume-run"
    RunStore(tmp_path / "runs", tmp_path).create_run(
        config_payload={"k": "v"},
        model_ids={"qwen": "q"},
        prompt_paths=[],
        run_id=run_id,
    )
    run_dir = tmp_path / "runs" / run_id
    sample_runner = SampleRunner(
        registry=registry,
        router=TaskRouter(),
        qwen_client=client,
        artifact_writer=ArtifactWriter(),
        call_budget_factory=CallBudgetFactory(),
        judge_service=judge_service,
        data_root=root,
    )
    adapter = _FakeAdapter(adapter_samples or [_vqa_sample()])
    dataset_runner = DatasetRunner(
        adapter=adapter,
        sample_runner=sample_runner,
        run_dir=run_dir,
        artifact_writer=ArtifactWriter(),
        judge_policy=judge_policy,
    )
    return root, client, adapter, sample_runner, dataset_runner, run_dir


def _run(
    dataset_runner: DatasetRunner,
    *,
    root: Path,
    task: str = "general_vqa",
    resume: bool = False,
    sample_concurrency: int = 1,
):
    return asyncio.run(
        dataset_runner.run(
            root=root,
            split="test",
            task=task,
            resume=resume,
            sample_concurrency=sample_concurrency,
        )
    )


def _sample_dir(run_dir: Path, task: str, sample_id: str) -> Path:
    return run_dir / "tasks" / task / "samples" / storage_key(sample_id)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _status_of(run_dir: Path, task: str, sample_id: str) -> dict:
    return _read_json(_sample_dir(run_dir, task, sample_id) / "status.json")


# ── resume: succeeded / resume：succeeded ───────────────────────────────────


def test_resume_succeeded_does_not_reinfer(tmp_path: Path) -> None:
    root, client, _, _, dataset_runner, run_dir = _setup(tmp_path)
    first = _run(dataset_runner, root=root)
    assert first.succeeded == 1
    assert client.calls == 1
    second = _run(dataset_runner, root=root, resume=True)
    assert second.succeeded == 1
    assert client.calls == 1  # no re-inference / 不重新推理
    assert _status_of(run_dir, "general_vqa", "resume-1")["state"] == "succeeded"


def test_resume_supplements_missing_deterministic_evaluation(tmp_path: Path) -> None:
    root, client, _, _, dataset_runner, run_dir = _setup(tmp_path)
    _run(dataset_runner, root=root)
    evaluation_path = (
        _sample_dir(run_dir, "general_vqa", "resume-1") / "vqa_evaluation.json"
    )
    assert evaluation_path.is_file()
    evaluation_path.unlink()
    summary = _run(dataset_runner, root=root, resume=True)
    assert summary.succeeded == 1
    assert client.calls == 1  # supplement only, no re-inference / 只补不重推
    evaluation = _read_json(evaluation_path)
    assert evaluation["task"] == "general_vqa"
    assert evaluation["judge_status"] == "not_requested"
    assert evaluation["deterministic_metrics"]["exact_match"] is True
    assert _status_of(run_dir, "general_vqa", "resume-1")["state"] == "succeeded"


def test_resume_supplements_missing_counting_evaluation(tmp_path: Path) -> None:
    """A succeeded counting sample without its evaluation gets the
    deterministic evaluation written without running any agent.
    succeeded 计数样本缺少评估时，不运行任何 Agent 直接补写确定性评估。"""
    root, client, _, _, dataset_runner, run_dir = _setup(
        tmp_path, adapter_samples=[_counting_sample()]
    )
    sample_dir = _sample_dir(run_dir, "counting", "resume-count")
    artifact_writer = ArtifactWriter()
    artifact_writer.write_sample(sample_dir, _counting_sample())
    artifact_writer.write_final_status(
        sample_dir,
        SampleRunStatusForTest.succeeded(sample_id="resume-count", task="counting"),
    )
    artifact_writer.write_execution(
        sample_dir,
        AgentExecutionForTest.of(_counting_result()),
    )
    summary = _run(dataset_runner, root=root, task="counting", resume=True)
    assert summary.succeeded == 1
    assert client.calls == 0
    evaluation = _read_json(sample_dir / "counting_evaluation.json")
    assert evaluation["task"] == "counting"
    assert evaluation["deterministic_metrics"]["exact_match"] == 1


def test_resume_supplement_missing_result_marks_skipped(tmp_path: Path) -> None:
    root, client, _, _, dataset_runner, run_dir = _setup(tmp_path)
    _run(dataset_runner, root=root)
    sample_dir = _sample_dir(run_dir, "general_vqa", "resume-1")
    (sample_dir / "agent_result.json").unlink()
    (sample_dir / "vqa_evaluation.json").unlink()
    summary = _run(dataset_runner, root=root, resume=True)
    assert summary.skipped == 1
    assert client.calls == 1  # still no re-inference / 仍然不重新推理
    status = _status_of(run_dir, "general_vqa", "resume-1")
    assert status["state"] == "skipped"
    assert status["error_code"] == "PERSISTED_RESULT_MISSING"


# ── resume: judge / resume：judge 补判 ──────────────────────────────────────


def test_resume_rejudges_failed_judge(tmp_path: Path) -> None:
    judge_client = _FakeJudgeClient(
        VQAAnswerJudgeResult(score=1), error=RuntimeError("judge secret")
    )
    judge_service = JudgeService(
        judge_prompt="p",
        judge_prompt_version="v1",
        vqa_judge_prompt="v",
        vqa_judge_prompt_version="v2",
        judge_client=judge_client,
    )
    root, client, _, _, dataset_runner, run_dir = _setup(
        tmp_path, judge_service=judge_service, judge_policy="all"
    )
    first = _run(dataset_runner, root=root)
    assert first.succeeded == 1
    assert judge_client.calls == 1
    evaluation = _read_json(
        _sample_dir(run_dir, "general_vqa", "resume-1") / "vqa_evaluation.json"
    )
    assert evaluation["judge_status"] == "failed"
    judge_client.error = None  # the endpoint recovers / 端点恢复
    second = _run(dataset_runner, root=root, resume=True)
    assert second.succeeded == 1
    assert client.calls == 1  # judge-only supplement / 只补 judge
    assert judge_client.calls == 2
    evaluation = _read_json(
        _sample_dir(run_dir, "general_vqa", "resume-1") / "vqa_evaluation.json"
    )
    assert evaluation["judge_status"] == "succeeded"
    assert evaluation["judge_parsed"]["score"] == 1


def test_resume_judge_policy_none_does_not_rejudge(tmp_path: Path) -> None:
    judge_client = _FakeJudgeClient(
        VQAAnswerJudgeResult(score=1), error=RuntimeError("judge secret")
    )
    judge_service = JudgeService(
        judge_prompt="p",
        judge_prompt_version="v1",
        vqa_judge_prompt="v",
        vqa_judge_prompt_version="v2",
        judge_client=judge_client,
    )
    root, client, _, _, dataset_runner, run_dir = _setup(
        tmp_path, judge_service=judge_service, judge_policy="all"
    )
    _run(dataset_runner, root=root)
    assert judge_client.calls == 1
    # A second runner with judge_policy=none must not re-judge.
    # judge_policy=none 的第二个 runner 不得重判。
    no_judge_runner = DatasetRunner(
        adapter=dataset_runner.adapter,
        sample_runner=dataset_runner.sample_runner,
        run_dir=run_dir,
        artifact_writer=ArtifactWriter(),
        judge_policy="none",
    )
    summary = _run(no_judge_runner, root=root, resume=True)
    assert summary.succeeded == 1
    assert client.calls == 1
    assert judge_client.calls == 1


# ── resume: partial / failed / running / corrupt ────────────────────────────


def test_resume_partial_reruns(tmp_path: Path) -> None:
    root, client, _, _, dataset_runner, run_dir = _setup(
        tmp_path, client_status="partial"
    )
    first = _run(dataset_runner, root=root)
    assert first.partial == 1
    assert client.calls == 1
    client.status = "completed"
    second = _run(dataset_runner, root=root, resume=True)
    assert second.succeeded == 1
    assert client.calls == 2  # re-run / 重跑
    assert _status_of(run_dir, "general_vqa", "resume-1")["state"] == "succeeded"


def test_resume_failed_reruns(tmp_path: Path) -> None:
    root, client, _, _, dataset_runner, run_dir = _setup(
        tmp_path, client_status="failed"
    )
    first = _run(dataset_runner, root=root)
    assert first.failed == 1
    client.status = "completed"
    second = _run(dataset_runner, root=root, resume=True)
    assert second.succeeded == 1
    assert client.calls == 2
    assert _status_of(run_dir, "general_vqa", "resume-1")["state"] == "succeeded"


def test_resume_stale_running_reruns(tmp_path: Path) -> None:
    root, client, _, _, dataset_runner, run_dir = _setup(tmp_path)
    _run(dataset_runner, root=root)
    status = _status_of(run_dir, "general_vqa", "resume-1")
    status["state"] = "running"
    status_path = _sample_dir(run_dir, "general_vqa", "resume-1") / "status.json"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    summary = _run(dataset_runner, root=root, resume=True)
    assert summary.succeeded == 1
    assert client.calls == 2  # stale running is re-run / 陈旧 running 重跑
    assert _status_of(run_dir, "general_vqa", "resume-1")["state"] == "succeeded"


def test_resume_corrupt_status_reruns(tmp_path: Path) -> None:
    root, client, _, _, dataset_runner, run_dir = _setup(tmp_path)
    _run(dataset_runner, root=root)
    status_path = _sample_dir(run_dir, "general_vqa", "resume-1") / "status.json"
    status_path.write_text("{corrupt", encoding="utf-8")
    summary = _run(dataset_runner, root=root, resume=True)
    assert summary.succeeded == 1
    assert client.calls == 2
    assert _status_of(run_dir, "general_vqa", "resume-1")["state"] == "succeeded"


# ── resume parity: shared deterministic dispatch (Fix B) ────────────────────


def _seed_succeeded_sample(
    run_dir: Path,
    *,
    task: str,
    sample: UnifiedSample,
    payload: Any,
    result_filename: str,
    execution_task: str | None = None,
) -> Path:
    """Manually construct a succeeded sample dir (sample.json + status.json +
    result artifact) as if a fresh run had completed it.
    手工构造 succeeded 样本目录（sample.json + status.json + 结果产物），
    模拟 fresh run 已完成。"""
    from agents.base import AgentExecution
    from workflows.schema import SampleRunStatus

    sample_dir = run_dir / "tasks" / task / "samples" / storage_key(sample.sample_id)
    writer = ArtifactWriter()
    writer.write_sample(sample_dir, sample)
    writer.write_final_status(
        sample_dir,
        SampleRunStatus(
            sample_id=sample.sample_id,
            task=execution_task or task,  # type: ignore[arg-type]
            state="succeeded",
            updated_at="2026-01-01T00:00:00Z",
        ),
    )
    writer.write_execution(
        sample_dir,
        AgentExecution(
            agent_name=getattr(payload, "agent_name", "general_vqa_agent"),
            payload=payload,
            result_filename=result_filename,
        ),
    )
    return sample_dir


def _vqa_bucket_sample(task: str, answers: list[str]) -> UnifiedSample:
    return UnifiedSample(
        sample_id=f"parity-{task}",
        dataset="fake",
        split="test",
        task=task,  # type: ignore[arg-type]
        images=[ImageRef(image_id="i0", path="img.png", role="image")],
        question="Question?",
        ground_truth=GroundTruth(answers=answers),
    )


def _change_bucket_sample(task: str, answers: list[str]) -> UnifiedSample:
    return UnifiedSample(
        sample_id=f"parity-{task}",
        dataset="fake",
        split="test",
        task=task,  # type: ignore[arg-type]
        images=[
            ImageRef(image_id="i0", path="t1.png", role="t1"),
            ImageRef(image_id="i1", path="t2.png", role="t2"),
        ],
        question="" if task == "change_caption" else "What changed?",
        ground_truth=GroundTruth(answers=answers),
    )


def test_resume_supplements_multiple_choice_vqa(tmp_path: Path) -> None:
    sample = _vqa_bucket_sample("multiple_choice_vqa", ["A"])
    root, client, _, _, dataset_runner, run_dir = _setup(
        tmp_path, adapter_samples=[sample]
    )
    sample_dir = _seed_succeeded_sample(
        run_dir,
        task="multiple_choice_vqa",
        sample=sample,
        payload=AgentResult(
            agent_name="general_vqa_agent", answer="A", status="completed"
        ),
        result_filename="agent_result.json",
    )
    summary = _run(dataset_runner, root=root, task="multiple_choice_vqa", resume=True)
    assert summary.succeeded == 1
    assert client.calls == 0  # no re-inference / 不重新推理
    evaluation = _read_json(sample_dir / "vqa_evaluation.json")
    assert evaluation["task"] == "general_vqa"
    assert evaluation["deterministic_metrics"]["exact_match"] is True
    assert evaluation["judge_status"] == "not_requested"  # no judge for mcq


def test_resume_supplements_scene_classification(tmp_path: Path) -> None:
    sample = _vqa_bucket_sample("scene_classification", ["ok"])
    root, client, _, _, dataset_runner, run_dir = _setup(
        tmp_path, adapter_samples=[sample]
    )
    sample_dir = _seed_succeeded_sample(
        run_dir,
        task="scene_classification",
        sample=sample,
        payload=AgentResult(
            agent_name="general_vqa_agent", answer="ok", status="completed"
        ),
        result_filename="agent_result.json",
    )
    summary = _run(dataset_runner, root=root, task="scene_classification", resume=True)
    assert summary.succeeded == 1
    assert client.calls == 0
    evaluation = _read_json(sample_dir / "vqa_evaluation.json")
    assert evaluation["task"] == "general_vqa"
    assert evaluation["deterministic_metrics"]["exact_match"] is True


def test_resume_supplements_change_qa_without_qwen(tmp_path: Path) -> None:
    sample = _change_bucket_sample("change_qa", ["road added"])
    root, client, _, _, dataset_runner, run_dir = _setup(
        tmp_path, adapter_samples=[sample]
    )
    sample_dir = _seed_succeeded_sample(
        run_dir,
        task="change_qa",
        sample=sample,
        payload=AgentResult(
            agent_name="change_agent", answer="road added", status="completed"
        ),
        result_filename="agent_result.json",
    )
    summary = _run(dataset_runner, root=root, task="change_qa", resume=True)
    assert summary.succeeded == 1
    assert client.calls == 0
    evaluation = _read_json(sample_dir / "vqa_evaluation.json")
    assert evaluation["task"] == "general_vqa"
    assert evaluation["deterministic_metrics"]["exact_match"] is True


def test_resume_uses_persisted_spatial_execution_task(tmp_path: Path) -> None:
    sample = _vqa_bucket_sample("general_vqa", ["north"])
    root, client, _, _, dataset_runner, run_dir = _setup(
        tmp_path, adapter_samples=[sample]
    )
    sample_dir = _seed_succeeded_sample(
        run_dir,
        task="general_vqa",
        execution_task="spatial_relation",
        sample=sample,
        payload=AgentResult(
            agent_name="general_vqa_agent", answer="north", status="completed"
        ),
        result_filename="agent_result.json",
    )
    summary = _run(dataset_runner, root=root, task="general_vqa", resume=True)
    assert summary.succeeded == 1
    assert client.calls == 0
    evaluation = _read_json(sample_dir / "vqa_evaluation.json")
    assert evaluation["task"] == "general_vqa"
    assert evaluation["deterministic_metrics"]["exact_match"] is True


def test_resume_supplements_fine_grained_counting(tmp_path: Path) -> None:
    sample = UnifiedSample(
        sample_id="parity-fgc",
        dataset="fake",
        split="test",
        task="fine_grained_counting",
        images=[ImageRef(image_id="i0", path="img.png", role="image")],
        question="How many cars?",
        ground_truth=GroundTruth(count=2),
    )
    root, client, _, _, dataset_runner, run_dir = _setup(
        tmp_path, adapter_samples=[sample]
    )
    sample_dir = _seed_succeeded_sample(
        run_dir,
        task="fine_grained_counting",
        sample=sample,
        payload=_counting_result(),
        result_filename="counting_result.json",
    )
    summary = _run(dataset_runner, root=root, task="fine_grained_counting", resume=True)
    assert summary.succeeded == 1
    assert client.calls == 0
    evaluation = _read_json(sample_dir / "counting_evaluation.json")
    assert evaluation["task"] == "counting"
    assert evaluation["deterministic_metrics"]["exact_match"] == 1


def test_resume_supplements_caption(tmp_path: Path) -> None:
    sample = UnifiedSample(
        sample_id="parity-caption",
        dataset="fake",
        split="test",
        task="caption",
        images=[ImageRef(image_id="i0", path="img.png", role="image")],
        question="",
        ground_truth=GroundTruth(answers=["a street scene"]),
    )
    root, client, _, _, dataset_runner, run_dir = _setup(
        tmp_path, adapter_samples=[sample]
    )
    sample_dir = _seed_succeeded_sample(
        run_dir,
        task="caption",
        sample=sample,
        payload=AgentResult(
            agent_name="general_vqa_agent",
            answer="a street scene",
            status="completed",
        ),
        result_filename="agent_result.json",
    )
    summary = _run(dataset_runner, root=root, task="caption", resume=True)
    assert summary.succeeded == 1
    assert client.calls == 0
    evaluation = _read_json(sample_dir / "caption_evaluation.json")
    assert evaluation["task"] == "caption"
    assert evaluation["deterministic_metrics"]["candidate"] == "a street scene"


def test_resume_supplements_change_caption_without_qwen(tmp_path: Path) -> None:
    sample = _change_bucket_sample("change_caption", ["a road was added"])
    root, client, _, _, dataset_runner, run_dir = _setup(
        tmp_path, adapter_samples=[sample]
    )
    sample_dir = _seed_succeeded_sample(
        run_dir,
        task="change_caption",
        sample=sample,
        payload=AgentResult(
            agent_name="change_agent",
            answer="a road was added",
            status="completed",
        ),
        result_filename="agent_result.json",
    )
    summary = _run(dataset_runner, root=root, task="change_caption", resume=True)
    assert summary.succeeded == 1
    assert client.calls == 0
    evaluation = _read_json(sample_dir / "caption_evaluation.json")
    assert evaluation["task"] == "caption"
    assert evaluation["deterministic_metrics"]["candidate"] == "a road was added"


def test_resume_supplements_grounding_compatible_frame(tmp_path: Path) -> None:
    sample = UnifiedSample(
        sample_id="parity-grounding",
        dataset="fake",
        split="test",
        task="grounding",
        images=[ImageRef(image_id="i0", path="img.png", role="image")],
        question="Where is the car?",
        ground_truth=GroundTruth(
            boxes=[[10.0, 20.0, 110.0, 120.0]],
            coordinate_frame="normalized_0_999_top_left",
        ),
    )
    root, client, _, _, dataset_runner, run_dir = _setup(
        tmp_path, adapter_samples=[sample]
    )
    sample_dir = _seed_succeeded_sample(
        run_dir,
        task="grounding",
        sample=sample,
        payload=AgentResult(
            agent_name="general_vqa_agent",
            answer="located",
            boxes=[[10.0, 20.0, 110.0, 120.0]],
            status="completed",
        ),
        result_filename="agent_result.json",
    )
    summary = _run(dataset_runner, root=root, task="grounding", resume=True)
    assert summary.succeeded == 1
    assert client.calls == 0
    evaluation = _read_json(sample_dir / "grounding_evaluation.json")
    assert evaluation["task"] == "grounding"
    assert evaluation["deterministic_metrics"]["iou_at_0_5"] is True


def test_resume_grounding_incompatible_frame_no_fake_metric(tmp_path: Path) -> None:
    """Resume must not fabricate a grounding metric for an incompatible
    frame; the sample stays succeeded. resume 对不兼容坐标系绝不伪造 grounding
    指标；样本保持 succeeded。"""
    sample = UnifiedSample(
        sample_id="parity-grounding-px",
        dataset="fake",
        split="test",
        task="grounding",
        images=[ImageRef(image_id="i0", path="img.png", role="image")],
        question="Where is the car?",
        ground_truth=GroundTruth(
            boxes=[[620.0, 1100.0, 1400.0, 1800.0]],
            coordinate_frame="source_pixels_top_left",
        ),
    )
    root, client, _, _, dataset_runner, run_dir = _setup(
        tmp_path, adapter_samples=[sample]
    )
    sample_dir = _seed_succeeded_sample(
        run_dir,
        task="grounding",
        sample=sample,
        payload=AgentResult(
            agent_name="general_vqa_agent",
            answer="located",
            boxes=[[100.0, 200.0, 400.0, 500.0]],
            status="completed",
        ),
        result_filename="agent_result.json",
    )
    summary = _run(dataset_runner, root=root, task="grounding", resume=True)
    assert summary.succeeded == 1
    assert client.calls == 0
    assert not (sample_dir / "grounding_evaluation.json").exists()
    assert _status_of(run_dir, "grounding", "parity-grounding-px")["state"] == "succeeded"


# ── predictions history contract (Fix E) / 预测历史契约 ─────────────────────


def test_predictions_are_append_only_history_with_current_state(tmp_path: Path) -> None:
    """fresh + resume produce two rows for the same (run_task, sample_id);
    the last row is the current state and updated_at never moves backwards.
    fresh + resume 对同一 (run_task, sample_id) 产生两行；最后一行是当前状态，
    updated_at 不回退。"""
    root, client, _, _, dataset_runner, run_dir = _setup(tmp_path)
    _run(dataset_runner, root=root)
    _run(dataset_runner, root=root, resume=True)
    assert client.calls == 1
    rows = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert all(row["run_task"] == "general_vqa" for row in rows)
    assert all(row["sample_id"] == "resume-1" for row in rows)
    assert rows[1]["status"] == "succeeded"
    assert rows[1]["updated_at"] >= rows[0]["updated_at"]
    assert rows[1]["result_path"].startswith("tasks/general_vqa/samples/")


# ── legacy absolute result path regression (06.6.1) ─────────────────────────


def test_resume_legacy_absolute_result_path_rejected_and_rerun(tmp_path: Path) -> None:
    """A persisted status with an absolute result_path fails schema
    validation; resume treats it as invalid and re-runs the sample, producing
    a clean basename status and a run-relative prediction path.
    带绝对 result_path 的持久化状态无法通过 schema 校验；resume 视为无效并
    重新执行样本，产出干净的 basename 状态与 run 相对预测路径。"""
    root, client, _, _, dataset_runner, run_dir = _setup(tmp_path)
    sample = _vqa_sample()
    sample_dir = run_dir / "tasks" / "general_vqa" / "samples" / storage_key("resume-1")
    writer = ArtifactWriter()
    writer.write_sample(sample_dir, sample)
    (sample_dir / "status.json").write_text(
        json.dumps(
            {
                "sample_id": "resume-1",
                "task": "general_vqa",
                "state": "succeeded",
                "result_path": "C:/old/run/agent_result.json",
                "updated_at": "2026-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    summary = _run(dataset_runner, root=root, resume=True)
    assert summary.succeeded == 1
    assert client.calls == 1  # the invalid status forces a re-run / 重新执行
    status = _read_json(sample_dir / "status.json")
    assert status["result_path"] == "agent_result.json"
    rows = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["result_path"].startswith("tasks/general_vqa/samples/")
    assert "C:/old" not in json.dumps(rows)


# ── judge off the event loop (Fix D) / judge 不阻塞事件循环 ─────────────────


def test_judge_runs_off_the_event_loop(tmp_path: Path) -> None:
    """A blocking judge must not stall the asyncio loop: with
    sample_concurrency=2 both samples must enter the judge in parallel, which
    a threading barrier proves deterministically. 阻塞 judge 不得卡住事件
    循环：sample_concurrency=2 时两个样本必须并行进入 judge，用 threading
    barrier 确定性证明。"""
    import threading

    class _BlockingJudgeClient(_FakeJudgeClient):
        def __init__(self, verdict) -> None:
            super().__init__(verdict)
            self.barrier = threading.Barrier(2, timeout=10)
            self.passed = False

        def judge_json(self, payload, *, response_model, request_meta, system_prompt=None):
            self.barrier.wait(timeout=10)
            self.passed = True
            return super().judge_json(
                payload,
                response_model=response_model,
                request_meta=request_meta,
                system_prompt=system_prompt,
            )

    judge_client = _BlockingJudgeClient(VQAAnswerJudgeResult(score=1))
    judge_service = JudgeService(
        judge_prompt="p",
        judge_prompt_version="v1",
        vqa_judge_prompt="v",
        vqa_judge_prompt_version="v2",
        judge_client=judge_client,
    )
    samples = [_vqa_sample(), _vqa_sample2()]
    root, _, _, _, dataset_runner, _ = _setup(
        tmp_path,
        judge_service=judge_service,
        judge_policy="all",
        adapter_samples=samples,
    )
    summary = _run(dataset_runner, root=root, sample_concurrency=2)
    assert summary.succeeded == 2
    assert summary.failed == 0
    assert judge_client.passed is True  # both judges entered concurrently


# ── helpers: minimal schema objects for manual sample dirs ──────────────────


class SampleRunStatusForTest:
    @staticmethod
    def succeeded(*, sample_id: str, task: str):
        from workflows.schema import SampleRunStatus

        return SampleRunStatus(
            sample_id=sample_id,
            task=task,  # type: ignore[arg-type]
            state="succeeded",
            updated_at="2026-08-08T00:00:00+00:00",
        )


class AgentExecutionForTest:
    @staticmethod
    def of(payload: Any):
        from agents.base import AgentExecution

        return AgentExecution(
            agent_name="counting_agent",
            payload=payload,
            result_filename="counting_result.json",
        )
