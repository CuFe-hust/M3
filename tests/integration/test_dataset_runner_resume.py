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
            sample_file=Path("samples.jsonl"),
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


def _run(dataset_runner: DatasetRunner, *, root: Path, task: str = "general_vqa", resume: bool = False):
    return asyncio.run(
        dataset_runner.run(root=root, split="test", task=task, resume=resume, sample_concurrency=1)
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
    assert status["error_code"] == "AGENT_RESULT_MISSING"


# ── resume: judge / resume：judge 补判 ──────────────────────────────────────


def test_resume_rejudges_failed_judge(tmp_path: Path) -> None:
    judge_client = _FakeJudgeClient(
        VQAAnswerJudgeResult(score=1), error=RuntimeError("judge secret")
    )
    judge_service = JudgeService(
        judge_prompt="p", vqa_judge_prompt="v", judge_client=judge_client
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
        judge_prompt="p", vqa_judge_prompt="v", judge_client=judge_client
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
