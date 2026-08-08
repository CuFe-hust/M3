"""Vertical slice: manifest-driven draft adapter → TaskResolver →
materialize → SampleRunner, orchestrated by DatasetRunner in draft mode.

垂直切片：manifest 驱动的 draft 适配器 → TaskResolver → 物化 →
SampleRunner，由 DatasetRunner draft 模式编排。离线：fake resolver 客户端
与 fake agents，真实 TaskResolver / SampleRunner / DatasetRunner /
ManifestDraftAdapter。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from PIL import Image

from agents.base import AgentExecution
from agents.registry import AgentRegistry
from agents.schema import AgentResult
from data.adapters.manifest import ManifestDraftAdapter
from data.schema import UnifiedSample
from models.base import ModelCacheIdentity
from routing.router import TaskRouter
from workflows.artifact_writer import ArtifactWriter
from workflows.call_budget import CallBudget, CallBudgetFactory
from workflows.dataset_runner import DatasetRunner, storage_key
from workflows.run_store import RunStore
from workflows.sample_runner import SampleRunner
from workflows.task_resolver import TaskResolver


class _DummyQwenClient:
    """Placeholder qwen client; fake agents never touch it.
    占位 qwen 客户端；fake Agent 绝不使用它。"""

    async def complete_json(self, **kwargs):
        raise AssertionError("dummy qwen client must not be called")


class _ResolverClient:
    """Fake model client for the TaskResolver; records calls and returns
    queued resolutions. TaskResolver 的 fake 模型客户端；记录调用并按队列返回
    解析结果。"""

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.calls = 0

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="fake-resolver",
            generation={"temperature": 0.0, "do_sample": False, "max_tokens": 64},
            client_version="1",
        )

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls += 1
        if not self.responses:
            raise AssertionError("resolver client called more times than configured")
        return response_model.model_validate(self.responses.pop(0))


class _FakeAgent:
    def __init__(
        self,
        name: str,
        tasks: tuple[str, ...],
        *,
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.supported_tasks = frozenset(tasks)
        self.error = error
        self.calls: list[tuple[UnifiedSample, object]] = []

    async def run(self, sample: UnifiedSample, context: object) -> AgentExecution:
        self.calls.append((sample, context))
        if self.error is not None:
            raise self.error
        payload = AgentResult(agent_name=self.name, answer="ok", status="completed")
        return AgentExecution(
            agent_name=self.name,
            payload=payload,
            result_filename="agent_result.json",
        )


def _make_dataset(root: Path, *, with_task: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (1, 2, 3)).save(root / "img.png", format="PNG")
    Image.new("RGB", (4, 4), (1, 2, 3)).save(root / "img2.png", format="PNG")
    fields = {"id": "id", "split": "split", "question": "question", "images": "images"}
    if with_task:
        fields["task"] = "task"
    (root / "spacers_adapter.json").write_text(
        json.dumps(
            {
                "dataset": "auto-demo",
                "version": "1",
                "samples_file": "samples.jsonl",
                "fields": fields,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_rows(root: Path, rows: list[dict]) -> None:
    (root / "samples.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _setup(
    tmp_path: Path,
    *,
    rows: list[dict],
    with_task: bool = True,
    resolver_responses: list[dict] | None = None,
    agent_errors: dict[str, Exception] | None = None,
):
    root = tmp_path / "data"
    _make_dataset(root, with_task=with_task)
    _write_rows(root, rows)
    agent_errors = agent_errors or {}
    agents = [
        _FakeAgent("general_vqa_agent", ("general_vqa",), error=agent_errors.get("general_vqa_agent")),
        _FakeAgent("caption_agent", ("caption",), error=agent_errors.get("caption_agent")),
        _FakeAgent("change_agent", ("change_caption", "change_qa"), error=agent_errors.get("change_agent")),
    ]
    registry = AgentRegistry()
    for agent in agents:
        registry.register(agent)
    run_id = "auto-run"
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
        qwen_client=_DummyQwenClient(),
        artifact_writer=ArtifactWriter(),
        call_budget_factory=CallBudgetFactory(),
        data_root=root,
    )
    resolver = TaskResolver(
        _ResolverClient(resolver_responses or []),
        system_prompt="resolve the task",
        confidence_threshold=0.7,
    )
    dataset_runner = DatasetRunner(
        adapter=ManifestDraftAdapter("auto-demo", {"general_vqa", "caption", "change_caption", "change_qa"}),
        sample_runner=sample_runner,
        run_dir=run_dir,
        artifact_writer=ArtifactWriter(),
        task_resolver=resolver,
        call_budget_factory=CallBudgetFactory(),
    )
    return root, run_dir, sample_runner, dataset_runner, agents, resolver


def _run(dataset_runner: DatasetRunner, *, root: Path, resume: bool = False):
    return asyncio.run(
        dataset_runner.run(root=root, split="test", task=None, resume=resume, sample_concurrency=1)
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _trace_of(run_dir: Path, sample_id: str) -> dict:
    return _read_json(
        run_dir / "tasks" / "auto" / "samples" / storage_key(sample_id) / "agent_trace.json"
    )


def _status_of(run_dir: Path, sample_id: str) -> dict:
    return _read_json(
        run_dir / "tasks" / "auto" / "samples" / storage_key(sample_id) / "status.json"
    )


def _resolution_response(
    task: str,
    *,
    confidence: float = 0.95,
    candidates: list[str] | None = None,
) -> dict:
    candidates = candidates or [task]
    return {
        "task": task,
        "confidence": confidence,
        "candidate_tasks": candidates,
        "reason_codes": [
            "low_confidence" if confidence < 0.7 else "model_high_confidence"
        ],
    }


# ── explicit task / 显式 task ───────────────────────────────────────────────


def test_explicit_task_zero_resolver_calls(tmp_path: Path) -> None:
    rows = [
        {"id": "a1", "split": "test", "question": "Is there a road?", "images": ["img.png"], "task": "general_vqa"},
        {"id": "a2", "split": "test", "question": "Describe.", "images": ["img.png"], "task": "caption"},
    ]
    root, run_dir, _, dataset_runner, agents, resolver = _setup(
        tmp_path, rows=rows, with_task=True
    )
    summary = _run(dataset_runner, root=root)
    assert summary.succeeded == 2
    assert resolver._client.calls == 0  # explicit tasks never call the model
    assert len(agents[0].calls) == 1  # general_vqa_agent
    assert len(agents[1].calls) == 1  # caption_agent
    assert _status_of(run_dir, "a1")["task"] == "general_vqa"
    assert _status_of(run_dir, "a2")["task"] == "caption"


# ── missing task / 缺失 task ────────────────────────────────────────────────


def test_missing_task_exactly_one_resolver_call_per_draft(tmp_path: Path) -> None:
    rows = [
        {"id": "a1", "split": "test", "question": "Is there a road?", "images": ["img.png"]},
        {"id": "a2", "split": "test", "question": "Describe.", "images": ["img.png"]},
    ]
    root, run_dir, _, dataset_runner, agents, resolver = _setup(
        tmp_path,
        rows=rows,
        with_task=False,
        resolver_responses=[
            _resolution_response("general_vqa"),
            _resolution_response("caption"),
        ],
    )
    summary = _run(dataset_runner, root=root)
    assert summary.succeeded == 2
    assert resolver._client.calls == 2  # exactly one call per missing-task draft
    assert _status_of(run_dir, "a1")["task"] == "general_vqa"
    assert _status_of(run_dir, "a2")["task"] == "caption"


def test_high_confidence_runs_only_top_task(tmp_path: Path) -> None:
    rows = [
        {"id": "a1", "split": "test", "question": "Is there a road?", "images": ["img.png"]},
    ]
    root, run_dir, _, dataset_runner, agents, resolver = _setup(
        tmp_path,
        rows=rows,
        with_task=False,
        resolver_responses=[
            _resolution_response("general_vqa", candidates=["general_vqa", "caption"])
        ],
    )
    summary = _run(dataset_runner, root=root)
    assert summary.succeeded == 1
    assert len(agents[0].calls) == 1  # general_vqa_agent only
    assert len(agents[1].calls) == 0  # caption never runs
    trace = _trace_of(run_dir, "a1")
    assert trace["low_confidence"] is False
    assert trace["resolution_source"] == "model"
    assert trace["candidate_tasks"] == ["general_vqa"]


def test_low_confidence_candidate_fallback(tmp_path: Path) -> None:
    rows = [
        {"id": "a1", "split": "test", "question": "Is there a road?", "images": ["img.png"]},
    ]
    root, run_dir, _, dataset_runner, agents, resolver = _setup(
        tmp_path,
        rows=rows,
        with_task=False,
        resolver_responses=[
            _resolution_response(
                "general_vqa", confidence=0.4, candidates=["general_vqa", "caption"]
            )
        ],
        agent_errors={"general_vqa_agent": RuntimeError("primary broke")},
    )
    summary = _run(dataset_runner, root=root)
    assert summary.succeeded == 1
    assert len(agents[0].calls) == 1  # primary candidate failed
    assert len(agents[1].calls) == 1  # caption candidate rescued the sample
    trace = _trace_of(run_dir, "a1")
    assert trace["low_confidence"] is True
    assert trace["candidate_tasks"] == ["general_vqa", "caption"]
    assert trace["attempt_agents"] == [["general_vqa_agent"], ["caption_agent"]]
    assert trace["execution_agent"] == "caption_agent"


# ── deterministic rules / 确定性规则 ────────────────────────────────────────


def test_blank_single_image_resolves_to_caption(tmp_path: Path) -> None:
    rows = [{"id": "a1", "split": "test", "question": "", "images": ["img.png"]}]
    root, run_dir, _, dataset_runner, agents, resolver = _setup(
        tmp_path, rows=rows, with_task=False
    )
    summary = _run(dataset_runner, root=root)
    assert summary.succeeded == 1
    assert resolver._client.calls == 0  # blank-question rule, no model call
    assert len(agents[1].calls) == 1  # caption_agent
    status = _status_of(run_dir, "a1")
    assert status["task"] == "caption"
    assert _trace_of(run_dir, "a1")["resolution_source"] == "rule"


def test_blank_two_images_resolves_to_change_caption(tmp_path: Path) -> None:
    rows = [
        {"id": "a1", "split": "test", "question": "", "images": ["img.png", "img2.png"]}
    ]
    root, run_dir, _, dataset_runner, agents, resolver = _setup(
        tmp_path, rows=rows, with_task=False
    )
    summary = _run(dataset_runner, root=root)
    assert summary.succeeded == 1
    assert resolver._client.calls == 0
    assert len(agents[2].calls) == 1  # change_agent
    sample = agents[2].calls[0][0]
    assert sample.task == "change_caption"
    assert [image.role for image in sample.images] == ["t1", "t2"]
    assert _status_of(run_dir, "a1")["task"] == "change_caption"


def test_incompatible_image_count_fails_honestly(tmp_path: Path) -> None:
    rows = [
        {
            "id": "a1",
            "split": "test",
            "question": "",
            "images": ["img.png", "img2.png", "img.png"],
        }
    ]
    root, run_dir, _, dataset_runner, agents, resolver = _setup(
        tmp_path, rows=rows, with_task=False
    )
    summary = _run(dataset_runner, root=root)
    assert summary.failed == 1
    assert resolver._client.calls == 0  # the rule fails without guessing
    assert all(len(agent.calls) == 0 for agent in agents)
    status = _status_of(run_dir, "a1")
    assert status["state"] == "failed"
    assert status["error_code"] == "EMPTY_UNRESOLVABLE_REQUEST"
    assert status["task"] == "unknown"  # honest sentinel, never general_vqa
    assert "general_vqa" not in status["task"]


# ── shared budget / 共享预算 ────────────────────────────────────────────────


def test_shared_budget_spans_resolver_and_agents(tmp_path: Path) -> None:
    rows = [
        {"id": "a1", "split": "test", "question": "Is there a road?", "images": ["img.png"]},
    ]
    root, _, _, dataset_runner, agents, resolver = _setup(
        tmp_path,
        rows=rows,
        with_task=False,
        resolver_responses=[_resolution_response("general_vqa")],
    )
    summary = _run(dataset_runner, root=root)
    assert summary.succeeded == 1
    recorded = agents[0].calls[0][1].call_budget
    assert isinstance(recorded, CallBudget)
    # 1 qwen reservation from the resolver plus 0 from this fake agent (it
    # does not reserve) would be 1; this fake agent reserves nothing, so the
    # resolver's reservation is the only one — proving the same object was
    # shared. / resolver 的 1 次 qwen 预留即证明同一预算对象贯穿全程。
    assert recorded.qwen_calls_used >= 1


# ── contracts / 契约 ────────────────────────────────────────────────────────


def test_unified_sample_task_still_required() -> None:
    from pydantic import ValidationError

    import pytest

    payload = {
        "sample_id": "x",
        "dataset": "d",
        "split": "test",
        "images": [{"image_id": "i0", "path": "img.png", "role": "image"}],
        "question": "q",
    }
    with pytest.raises(ValidationError):
        UnifiedSample.model_validate(payload)


# ── candidate fallback resume consistency (Fix F) ───────────────────────────


def test_candidate_fallback_resume_consistency(tmp_path: Path) -> None:
    """resolved general_vqa + failed primary + caption candidate success: the
    canonical sample keeps the resolved task, the status/trace/routing record
    the executed task, and resume never re-resolves, never re-infers, and
    never applies VQA judge semantics to the caption execution.
    解析 general_vqa + primary 失败 + caption 候选成功：canonical sample 保留
    解析任务，status/trace/routing 记录执行任务；resume 不重新解析、不重新
    推理、不对 caption 执行应用 VQA judge 语义。"""
    rows = [
        {"id": "a1", "split": "test", "question": "Is there a road?", "images": ["img.png"]}
    ]
    root, run_dir, _, dataset_runner, agents, resolver = _setup(
        tmp_path,
        rows=rows,
        with_task=False,
        resolver_responses=[
            _resolution_response(
                "general_vqa", confidence=0.4, candidates=["general_vqa", "caption"]
            )
        ],
        agent_errors={"general_vqa_agent": RuntimeError("primary broke")},
    )
    summary = _run(dataset_runner, root=root)
    assert summary.succeeded == 1
    assert summary.failed == 0
    sample_dir = run_dir / "tasks" / "auto" / "samples" / storage_key("a1")
    assert _read_json(sample_dir / "sample.json")["task"] == "general_vqa"
    status = _read_json(sample_dir / "status.json")
    assert status["task"] == "caption"
    trace = _read_json(sample_dir / "agent_trace.json")
    assert trace["resolved_task"] == "general_vqa"
    assert trace["execution_task"] == "caption"
    assert trace["task_type"] == "general_vqa"
    assert trace["fallback_used"] is True
    routing = _read_json(sample_dir / "routing_decision.json")
    assert routing["primary_agent"] == "caption_agent"
    assert routing["task"] == "caption"
    # Resume: nothing re-runs and no VQA judge applies to the caption sample.
    # Resume：什么都不重跑，VQA judge 不作用于 caption 样本。
    resolver_calls_before = resolver._client.calls
    summary_resume = _run(dataset_runner, root=root, resume=True)
    assert summary_resume.succeeded == 1
    assert resolver._client.calls == resolver_calls_before
    assert len(agents[0].calls) == 1  # general_vqa_agent ran once in run 1
    assert len(agents[1].calls) == 1  # caption_agent ran once in run 1
    assert len(agents[2].calls) == 0  # change_agent never involved
    assert not (sample_dir / "vqa_evaluation.json").exists()
    assert _read_json(sample_dir / "status.json")["state"] == "succeeded"


# ── draft failure isolation (Fix I) / draft 异常隔离 ────────────────────────


def test_draft_single_sample_failure_does_not_break_dataset(tmp_path: Path) -> None:
    """A single draft whose SampleRunner raises unexpectedly must collapse
    into a stable failed status without terminating the dataset run.
    单个 draft 的 SampleRunner 意外异常必须收敛为稳定 failed 状态，不终止
    整个数据集运行。"""

    class _RaisingRunner:
        async def run_one(
            self, sample, sample_dir, *, resolution=None, judge_policy="none", budget=None
        ):
            raise RuntimeError("secret raw detail")

    rows = [
        {"id": "a1", "split": "test", "question": "Is there a road?", "images": ["img.png"], "task": "general_vqa"}
    ]
    root, run_dir, _, _, _, resolver = _setup(tmp_path, rows=rows, with_task=True)
    dataset_runner = DatasetRunner(
        adapter=ManifestDraftAdapter("auto-demo", {"general_vqa", "caption"}),
        sample_runner=_RaisingRunner(),  # type: ignore[arg-type]
        run_dir=run_dir,
        artifact_writer=ArtifactWriter(),
        task_resolver=resolver,
        call_budget_factory=CallBudgetFactory(),
    )
    summary = _run(dataset_runner, root=root)
    assert summary.failed == 1
    assert summary.total == summary.failed  # accounting stays closed
    status = _read_json(
        run_dir / "tasks" / "auto" / "samples" / storage_key("a1") / "status.json"
    )
    assert status["state"] == "failed"
    assert status["error_code"] == "RuntimeError"
    assert "secret raw" not in json.dumps(status)
