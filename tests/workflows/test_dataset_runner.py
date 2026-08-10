"""Contract tests for DatasetRunner orchestration: selection order, shard
stability, storage keys, concurrency, fail-fast, summary, and predictions.

DatasetRunner 编排契约测试：selection 顺序、分片稳定性、存储键、并发、
fail-fast、汇总与 predictions。使用 stub SampleRunner 隔离编排逻辑；离线。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from data.adapters.base import AdapterProbe
from data.schema import GroundTruth, ImageRef, UnifiedSample
from workflows.artifact_writer import ArtifactWriter
from workflows.dataset_runner import (
    DatasetRunner,
    _shard_matches,
    select_samples,
    storage_key,
)
from workflows.run_store import RunStore
from workflows.schema import SampleRunOutcome, SampleRunStatus


# ── helpers / 测试辅助 ──────────────────────────────────────────────────────


class _FakeAdapter:
    name = "fake"
    supported_tasks = frozenset({"general_vqa", "caption", "counting"})

    def __init__(self, samples: list[UnifiedSample]) -> None:
        self._samples = samples
        self.probe_calls = 0

    def probe(self, root: Path, task: str | None = None) -> AdapterProbe:
        self.probe_calls += 1
        return AdapterProbe(
            dataset="fake",
            version="1",
            sample_file=root / "samples.jsonl",  # root-anchored / 锚定 root
            observed_fields=("id",),
            sample_count=len(self._samples),
            task=task,
            available_tasks=("general_vqa", "caption", "counting"),
        )

    def iter_samples(self, root: Path, split: str, task: str):
        for sample in self._samples:
            yield sample


class _StubSampleRunner:
    """Recording SampleRunner stub: configurable state, delay, and per-sample
    failure; tracks in-flight concurrency; persists sample.json/status.json
    like the real runner. 记录型 SampleRunner stub：可配置状态、延迟与逐样本
    失败；跟踪并发深度；像真实 runner 一样持久化 sample.json/status.json。"""

    def __init__(
        self,
        state: str = "succeeded",
        *,
        delay: float = 0.0,
        delays: dict[str, float] | None = None,
        states: dict[str, str] | None = None,
        fail_ids: set[str] | None = None,
        raise_error: Exception | None = None,
    ) -> None:
        self.state = state
        self.delay = delay
        self.delays = delays or {}
        self.states = states or {}
        self.fail_ids = fail_ids or set()
        self.raise_error = raise_error
        self.calls: list[tuple[UnifiedSample, Path, str, object]] = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def run_one(self, sample, sample_dir, *, resolution=None, judge_policy="none"):
        self.calls.append((sample, sample_dir, judge_policy, resolution))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        delay = self.delays.get(sample.sample_id, self.delay)
        if delay:
            await asyncio.sleep(delay)
        self.in_flight -= 1
        if self.raise_error is not None:
            raise self.raise_error
        if sample.sample_id in self.fail_ids:
            state = "failed"
        else:
            state = self.states.get(sample.sample_id, self.state)
        status = SampleRunStatus(
            sample_id=sample.sample_id,
            task=sample.task,
            state=state,
            result_path=Path("result.json"),
            updated_at="2026-08-08T00:00:00+00:00",
        )
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "sample.json").write_text(
            json.dumps(sample.model_dump(mode="json"), ensure_ascii=False),
            encoding="utf-8",
        )
        (sample_dir / "status.json").write_text(
            json.dumps(status.model_dump(mode="json"), ensure_ascii=False),
            encoding="utf-8",
        )
        return SampleRunOutcome(
            execution=None,
            status=status,
            routing=None,
            evaluation=None,
            fallback_used=False,
        )


def _sample(sample_id: str, task: str = "general_vqa") -> UnifiedSample:
    return UnifiedSample(
        sample_id=sample_id,
        dataset="fake",
        split="test",
        task=task,  # type: ignore[arg-type]
        images=[ImageRef(image_id="i0", path="img.png", role="image")],
        question="Is there a road?",
        ground_truth=GroundTruth(answers=["yes"]),
    )


def _samples(count: int, prefix: str = "s") -> list[UnifiedSample]:
    return [_sample(f"{prefix}{index}") for index in range(count)]


def _create_run(tmp_path: Path, run_id: str = "runner-run") -> tuple[Path, RunStore]:
    store = RunStore(tmp_path / "runs", tmp_path)
    store.create_run(
        config_payload={"k": "v"},
        model_ids={"qwen": "q"},
        prompt_paths=[],
        run_id=run_id,
    )
    return tmp_path / "runs" / run_id, store


def _runner(
    adapter: _FakeAdapter,
    sample_runner: _StubSampleRunner,
    run_dir: Path,
    *,
    judge_policy: str = "none",
) -> DatasetRunner:
    return DatasetRunner(
        adapter=adapter,
        sample_runner=sample_runner,
        run_dir=run_dir,
        artifact_writer=ArtifactWriter(),
        judge_policy=judge_policy,
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _predictions(run_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _run(
    runner: DatasetRunner,
    *,
    split: str = "test",
    task: str = "general_vqa",
    resume: bool = False,
    limit: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    start_index: int = 0,
    sample_ids: set[str] | None = None,
    fail_fast: bool = False,
    sample_concurrency: int = 1,
):
    return asyncio.run(
        runner.run(
            root=Path("."),
            split=split,
            task=task,
            resume=resume,
            limit=limit,
            shard_index=shard_index,
            shard_count=shard_count,
            start_index=start_index,
            sample_ids=sample_ids,
            fail_fast=fail_fast,
            sample_concurrency=sample_concurrency,
        )
    )


# ── selection / 选择 ────────────────────────────────────────────────────────


def test_selection_order_and_filters() -> None:
    samples = _samples(6)  # s0..s5
    selected = select_samples(
        samples,
        start_index=2,
        sample_ids={"s3", "s5"},
        limit=1,
    )
    assert [item.sample_id for item in selected] == ["s3"]
    selected = select_samples(samples, start_index=2, sample_ids={"s3", "s5"})
    assert [item.sample_id for item in selected] == ["s3", "s5"]


def test_selection_preserves_adapter_order() -> None:
    samples = _samples(6)
    selected = select_samples(samples, start_index=1, limit=3)
    assert [item.sample_id for item in selected] == ["s1", "s2", "s3"]


def test_selection_empty_with_zero_limit() -> None:
    assert select_samples(_samples(3), limit=0) == []


# ── shard / 分片 ────────────────────────────────────────────────────────────


def test_shard_partition_is_stable_and_coverable() -> None:
    samples = _samples(10)
    first_run = select_samples(samples, shard_index=0, shard_count=2)
    second_run = select_samples(samples, shard_index=0, shard_count=2)
    other = select_samples(samples, shard_index=1, shard_count=2)
    assert [item.sample_id for item in first_run] == [item.sample_id for item in second_run]
    union = {item.sample_id for item in first_run} | {item.sample_id for item in other}
    assert union == {item.sample_id for item in samples}
    assert not ({item.sample_id for item in first_run} & {item.sample_id for item in other})


def test_shard_is_sha256_based_and_process_stable() -> None:
    """The shard assignment must not depend on Python's randomized hash, so it
    stays identical across processes with different PYTHONHASHSEED.
    分片分配不依赖 Python 随机化哈希：不同 PYTHONHASHSEED 的进程结果一致。"""
    sample_ids = ["s0", "s1", "s2", "CON", "a/b"]
    script = (
        "from workflows.dataset_runner import _shard_matches, storage_key; "
        "print([(_shard_matches(i, 1, 3), storage_key(i)) for i in __import__('json').loads(__import__('sys').argv[1])])"
    )
    outputs = []
    for seed in ("0", "42"):
        result = subprocess.run(
            [sys.executable, "-c", script, json.dumps(sample_ids)],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, "PYTHONHASHSEED": seed},
            check=True,
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
    # The in-process selection agrees with the subprocess result.
    # 进程内选择与子进程结果一致。
    in_process = [(_shard_matches(i, 1, 3), storage_key(i)) for i in sample_ids]
    assert repr(in_process) in outputs[0]


# ── storage keys / 存储键 ───────────────────────────────────────────────────


def test_storage_key_is_short_hex_and_path_safe() -> None:
    for sample_id in (
        "CON",
        "a/b",
        "..",
        "a:b",
        "C:\\x",
        "name.",
        "name ",
        "样本",
        "样本 with 空格",
        "a" * 200,
    ):
        key = storage_key(sample_id)
        assert len(key) == 24
        assert all(character in "0123456789abcdef" for character in key)
    assert storage_key("s1") != storage_key("s2")


def test_windows_dangerous_sample_ids_stay_inside_samples_root(tmp_path: Path) -> None:
    dangerous = ["CON", "a/b", "..", "a:b", "name.", "a" * 200]
    samples = [_sample(sample_id) for sample_id in dangerous]
    run_dir, _ = _create_run(tmp_path)
    runner = _runner(_FakeAdapter(samples), _StubSampleRunner(), run_dir)
    summary = _run(runner, task="general_vqa")
    assert summary.total == len(dangerous)
    samples_root = run_dir / "tasks" / "general_vqa" / "samples"
    for sample in samples:
        sample_dir = samples_root / storage_key(sample.sample_id)
        assert sample_dir.is_dir()
        assert sample_dir.resolve().is_relative_to(samples_root.resolve())


def test_multi_task_same_sample_id_no_conflict(tmp_path: Path) -> None:
    sample = _sample("shared-id")
    run_dir, _ = _create_run(tmp_path)
    adapter = _FakeAdapter([sample])
    stub = _StubSampleRunner()
    runner = _runner(adapter, stub, run_dir)
    _run(runner, task="general_vqa")
    _run(runner, task="caption")
    first_dir = run_dir / "tasks" / "general_vqa" / "samples" / storage_key("shared-id")
    second_dir = run_dir / "tasks" / "caption" / "samples" / storage_key("shared-id")
    assert first_dir.is_dir() and second_dir.is_dir()
    assert (first_dir / "status.json").is_file()
    assert (second_dir / "status.json").is_file()
    assert len(stub.calls) == 2
    rows = _predictions(run_dir)
    assert len(rows) == 2
    # The execution-index key is (run_task, sample_id): the same sample id
    # under two task namespaces forms two distinct keys.
    # 执行索引键是 (run_task, sample_id)：同一 sample id 在两个 task 命名空间
    # 下构成两个互不冲突的键。
    keys = {(row["run_task"], row["sample_id"]) for row in rows}
    assert keys == {("general_vqa", "shared-id"), ("caption", "shared-id")}
    for task in ("general_vqa", "caption"):
        summary = _read_json(run_dir / "tasks" / task / "dataset_summary.json")
        assert summary["task"] == task
        assert summary["total"] == 1
        assert summary["succeeded"] == 1


# ── concurrency / 并发 ──────────────────────────────────────────────────────


def test_concurrency_is_bounded_by_semaphore(tmp_path: Path) -> None:
    samples = _samples(6)
    run_dir, _ = _create_run(tmp_path)
    stub = _StubSampleRunner(delay=0.02)
    runner = _runner(_FakeAdapter(samples), stub, run_dir)
    summary = _run(runner, sample_concurrency=2)
    assert summary.succeeded == 6
    assert stub.max_in_flight == 2


# ── fail-fast / 快速失败 ────────────────────────────────────────────────────


def test_fail_fast_stops_new_tasks_and_cancels_in_flight(tmp_path: Path) -> None:
    samples = _samples(4)  # s0..s3
    run_dir, _ = _create_run(tmp_path)
    # s0 fails immediately; s1 is slow and still in flight when s0 fails.
    # s0 立即失败；s1 较慢，在 s0 失败时仍在飞行中。
    stub = _StubSampleRunner(delays={"s1": 0.4}, fail_ids={"s0"})
    runner = _runner(_FakeAdapter(samples), stub, run_dir)
    summary = _run(runner, fail_fast=True, sample_concurrency=2)
    # Accounting is always closed: 1 failed + 1 cancelled + 2 not-started.
    # 记账永远闭合：1 failed + 1 cancelled + 2 not-started。
    assert summary.total == 4
    assert summary.failed == 1
    assert summary.skipped == 3
    assert summary.succeeded == 0
    assert summary.partial == 0
    assert summary.total == (
        summary.succeeded + summary.partial + summary.failed + summary.skipped
    )
    # Only s0 (failed) and s1 (cancelled) were ever submitted; s2/s3 never ran.
    # 只有 s0（失败）与 s1（被取消）被提交；s2/s3 从未运行。
    called_ids = [call[0].sample_id for call in stub.calls]
    assert called_ids == ["s0", "s1"]
    cancelled_status = _read_json(
        run_dir / "tasks" / "general_vqa" / "samples" / storage_key("s1") / "status.json"
    )
    assert cancelled_status["state"] == "skipped"
    assert cancelled_status["error_code"] == "FAIL_FAST_CANCELLED"
    for sample_id in ("s2", "s3"):
        not_started = _read_json(
            run_dir / "tasks" / "general_vqa" / "samples" / storage_key(sample_id) / "status.json"
        )
        assert not_started["state"] == "skipped"
        assert not_started["error_code"] == "FAIL_FAST_NOT_STARTED"
    rows = {row["sample_id"]: row for row in _predictions(run_dir)}
    assert rows["s1"]["status"] == "skipped"
    assert rows["s2"]["status"] == "skipped"
    assert rows["s3"]["status"] == "skipped"
    assert all(row["run_task"] == "general_vqa" for row in rows.values())
    assert rows["s0"]["result_path"].startswith("tasks/general_vqa/samples/")
    assert rows["s1"]["result_path"] is None
    assert rows["s2"]["result_path"] is None
    # No sample is left in a permanent running state. / 无样本遗留永久 running。
    for key in ("s0", "s1", "s2", "s3"):
        state = _read_json(
            run_dir / "tasks" / "general_vqa" / "samples" / storage_key(key) / "status.json"
        )["state"]
        assert state != "running"


def test_fail_fast_not_triggered_without_failures(tmp_path: Path) -> None:
    samples = _samples(4)
    run_dir, _ = _create_run(tmp_path)
    stub = _StubSampleRunner(delay=0.02)
    runner = _runner(_FakeAdapter(samples), stub, run_dir)
    summary = _run(runner, fail_fast=True, sample_concurrency=2)
    assert summary.succeeded == 4
    assert summary.failed == 0 and summary.skipped == 0


# ── probe / summary / predictions ───────────────────────────────────────────


def test_manifest_stays_runmanifest_valid_and_probe_is_separate(tmp_path: Path) -> None:
    """The run manifest must stay parseable by the RunManifest schema across
    a dataset run; the dataset probe lives next to the task summary and never
    extends the manifest schema. 数据集运行前后 manifest.json 必须始终可被
    RunManifest schema 解析；数据集 probe 位于 task 汇总同级，绝不扩展
    manifest schema。"""
    from workflows.run_store import RunManifest

    samples = _samples(2)
    run_dir, _ = _create_run(tmp_path)
    manifest_path = run_dir / "manifest.json"
    RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    adapter = _FakeAdapter(samples)
    runner = _runner(adapter, _StubSampleRunner(), run_dir)
    _run(runner, task="general_vqa")
    RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    assert "dataset_probe" not in _read_json(manifest_path)
    probe = _read_json(run_dir / "tasks" / "general_vqa" / "dataset_probe.json")
    assert probe["dataset"] == "fake"
    assert probe["task"] == "general_vqa"
    assert probe["sample_count"] == 2
    assert probe["observed_fields"] == ["id"]
    assert probe["sample_file"] == "samples.jsonl"  # dataset-relative / root 相对


def test_summary_counts_and_predictions_rows(tmp_path: Path) -> None:
    samples = _samples(3)
    run_dir, _ = _create_run(tmp_path)
    stub = _StubSampleRunner(states={"s1": "partial"}, fail_ids={"s2"})
    runner = _runner(_FakeAdapter(samples), stub, run_dir)
    summary = _run(runner, task="general_vqa")
    assert summary.run_id == "runner-run"
    assert summary.dataset == "fake"
    assert summary.split == "test"
    assert summary.task == "general_vqa"
    assert summary.total == 3
    assert summary.succeeded == 1
    assert summary.partial == 1
    assert summary.failed == 1
    assert summary.skipped == 0
    persisted = _read_json(run_dir / "tasks" / "general_vqa" / "dataset_summary.json")
    assert persisted == summary.model_dump(mode="json")
    rows = _predictions(run_dir)
    assert {row["sample_id"] for row in rows} == {"s0", "s1", "s2"}
    assert {row["status"] for row in rows} == {"succeeded", "partial", "failed"}
    assert all(row["task"] == "general_vqa" for row in rows)
    # Execution-index contract: run_task namespace + run-relative result paths.
    # 执行索引契约：run_task 命名空间 + run 相对结果路径。
    assert all(row["run_task"] == "general_vqa" for row in rows)
    for row in rows:
        assert row["result_path"].startswith("tasks/general_vqa/samples/")
        assert row["result_path"].endswith("/result.json")
        assert "C:\\" not in row["result_path"]
        assert "tmp_path" not in row["result_path"]
        assert row["updated_at"]


def test_probe_per_task_not_overwritten(tmp_path: Path) -> None:
    """dataset_probe.json lives per task directory and is never overwritten
    across tasks. dataset_probe.json 按 task 目录独立存放，跨 task 互不覆盖。"""
    sample = _sample("shared-id")
    run_dir, _ = _create_run(tmp_path)
    runner = _runner(_FakeAdapter([sample]), _StubSampleRunner(), run_dir)
    _run(runner, task="general_vqa")
    _run(runner, task="caption")
    first = _read_json(run_dir / "tasks" / "general_vqa" / "dataset_probe.json")
    second = _read_json(run_dir / "tasks" / "caption" / "dataset_probe.json")
    assert first["task"] == "general_vqa"
    assert second["task"] == "caption"
    assert first["sample_file"] == "samples.jsonl"  # dataset-relative / root 相对
    assert first != second


# ── argument validation / 参数校验 ──────────────────────────────────────────


def test_invalid_arguments_fail_stable(tmp_path: Path) -> None:
    run_dir, _ = _create_run(tmp_path)
    runner = _runner(_FakeAdapter([]), _StubSampleRunner(), run_dir)
    with pytest.raises(ValueError, match="shard_count"):
        _run(runner, shard_count=0)
    with pytest.raises(ValueError, match="shard_index"):
        _run(runner, shard_count=2, shard_index=2)
    with pytest.raises(ValueError, match="sample_concurrency"):
        _run(runner, sample_concurrency=0)


# ── raw exception safety / 原始异常安全 ─────────────────────────────────────


def test_unexpected_runner_exception_records_stable_code(tmp_path: Path) -> None:
    samples = _samples(1)
    run_dir, _ = _create_run(tmp_path)
    stub = _StubSampleRunner(raise_error=RuntimeError("C:\\secret\\path sk-secret-raw"))
    runner = _runner(_FakeAdapter(samples), stub, run_dir)
    summary = _run(runner, task="general_vqa")
    assert summary.failed == 1
    status = _read_json(
        run_dir / "tasks" / "general_vqa" / "samples" / storage_key("s0") / "status.json"
    )
    assert status["state"] == "failed"
    assert status["error_code"] == "RuntimeError"
    assert status["error_message"] == "RuntimeError"
    text = json.dumps(status)
    assert "secret" not in text and "sk-" not in text and "C:\\" not in text


# ── auto-task explicit contract (Fix H) / auto-task 显式契约 ────────────────


def test_dataset_run_options_auto_task_contract() -> None:
    from workflows.schema import DatasetRunOptions

    with pytest.raises(ValueError, match="auto_task"):
        DatasetRunOptions(
            dataset="d", root=Path("."), split="test", tasks=(), auto_task=False
        )
    with pytest.raises(ValueError, match="auto_task"):
        DatasetRunOptions(
            dataset="d", root=Path("."), split="test", tasks=("counting",), auto_task=True
        )
    DatasetRunOptions(
        dataset="d", root=Path("."), split="test", tasks=("counting",), auto_task=False
    )
    DatasetRunOptions(
        dataset="d", root=Path("."), split="test", tasks=(), auto_task=True
    )


def test_dataset_runner_task_none_is_internal_auto_task_mode(tmp_path: Path) -> None:
    """task=None is the explicit internal auto-task mode, never a user
    default; without a resolver it fails at configuration time.
    task=None 是内部显式 auto-task 模式而非用户缺省；缺少 resolver 时在
    配置期失败。"""
    run_dir, _ = _create_run(tmp_path)
    runner = _runner(_FakeAdapter([]), _StubSampleRunner(), run_dir)
    with pytest.raises(ValueError, match="draft task mode"):
        _run(runner, task=None)


# ── status task typing (Fix K) / 状态任务类型收紧 ───────────────────────────


def test_sample_run_status_task_rejects_typos_and_accepts_unknown() -> None:
    from pydantic import ValidationError

    from workflows.schema import SampleRunStatus

    SampleRunStatus(
        sample_id="s1",
        task="unknown",
        state="failed",
        error_code="EMPTY_UNRESOLVABLE_REQUEST",
        updated_at="2026-01-01T00:00:00Z",
    )
    for typo in ("coutning", "captino", "whatever", "general vqa"):
        with pytest.raises(ValidationError):
            SampleRunStatus(
                sample_id="s1",
                task=typo,  # type: ignore[arg-type]
                state="failed",
                updated_at="2026-01-01T00:00:00Z",
            )


# ── deterministic judge sample rate (Task 11C2) / 确定性 judge 抽样率 ────────


def _rate_sample(sample_id: str) -> UnifiedSample:
    """One minimal general_vqa sample. / 一条最小 general_vqa 样本。"""
    return UnifiedSample(
        sample_id=sample_id,
        dataset="fake",
        split="test",
        task="general_vqa",
        images=[ImageRef(image_id="i0", path="img.png", role="image")],
        question="q",
        ground_truth=GroundTruth(answers=["a"]),
    )


def test_judge_sample_rate_bounds() -> None:
    from workflows.schema import DatasetRunOptions

    with pytest.raises(ValueError, match="judge_sample_rate"):
        DatasetRunOptions(
            dataset="d", root=Path("."), split="s", tasks=("caption",),
            judge_sample_rate=1.5,
        )
    with pytest.raises(ValueError, match="judge_sample_rate"):
        DatasetRunOptions(
            dataset="d", root=Path("."), split="s", tasks=("caption",),
            judge_sample_rate=-0.1,
        )
    DatasetRunOptions(
        dataset="d", root=Path("."), split="s", tasks=("caption",),
        judge_sample_rate=0.0,
    )
    DatasetRunOptions(
        dataset="d", root=Path("."), split="s", tasks=("caption",),
        judge_sample_rate=1.0,
    )


def test_judge_sample_rate_zero_and_one(tmp_path: Path) -> None:
    samples = [_rate_sample(f"s{i}") for i in range(20)]
    run_dir, _ = _create_run(tmp_path, run_id="rate-0")
    stub = _StubSampleRunner()
    runner = _runner(_FakeAdapter(samples), stub, run_dir, judge_policy="all")
    runner.judge_sample_rate = 0.0
    summary = _run(runner)
    assert summary.judge_sample_rate == 0.0
    assert stub.calls
    assert all(call[2] == "none" for call in stub.calls)  # rate 0 disables judge

    run_dir1, _ = _create_run(tmp_path, run_id="rate-1")
    stub1 = _StubSampleRunner()
    runner1 = _runner(_FakeAdapter(samples), stub1, run_dir1, judge_policy="all")
    runner1.judge_sample_rate = 1.0
    _run(runner1)
    assert all(call[2] == "all" for call in stub1.calls)  # rate 1 keeps policy


def test_judge_sample_rate_intermediate_deterministic_across_resume(
    tmp_path: Path,
) -> None:
    samples = [_rate_sample(f"s{i}") for i in range(20)]
    run_dir, _ = _create_run(tmp_path, run_id="rate-run")
    stub = _StubSampleRunner()
    runner = _runner(_FakeAdapter(samples), stub, run_dir, judge_policy="all")
    runner.judge_sample_rate = 0.5
    summary = _run(runner)
    judged_fresh = {call[0].sample_id for call in stub.calls if call[2] == "all"}
    assert 0 < len(judged_fresh) < 20
    assert summary.judge_sample_rate == 0.5
    persisted = _read_json(
        run_dir / "tasks" / "general_vqa" / "dataset_summary.json"
    )
    assert persisted["judge_sample_rate"] == 0.5
    # A separately configured runner selects the identical subset (deterministic).
    # 独立配置的 runner 选择完全相同的子集（确定性）。
    probe = _runner(_FakeAdapter(samples), _StubSampleRunner(), run_dir, judge_policy="all")
    probe.judge_sample_rate = 0.5
    assert [probe._judge_policy_for(f"s{i}") for i in range(20)] == [
        runner._judge_policy_for(f"s{i}") for i in range(20)
    ]
    # Resume with no explicit rate restores the persisted policy.
    # resume 未给显式 rate 时恢复持久化策略。
    stub2 = _StubSampleRunner()
    runner2 = _runner(_FakeAdapter(samples), stub2, run_dir, judge_policy="all")
    summary2 = _run(runner2, resume=True)
    assert runner2.judge_sample_rate == 0.5
    assert summary2.judge_sample_rate == 0.5
