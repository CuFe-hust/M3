"""Contract tests for the report builder: mixed states, last-row-wins index,
corrupt optional artifacts, deterministic output, and secret/path safety.

报告构建器契约测试：混合状态、执行索引最后一行生效、损坏可选产物、确定性
输出与密钥/路径安全。离线：直接构造持久化运行产物。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agents.base import AgentExecution
from agents.schema import AgentResult
from data.schema import GroundTruth, ImageRef, UnifiedSample
from evaluation.records import (
    CountDeterministicMetrics,
    EvaluationRecord,
    VQADeterministicMetrics,
)
from reporting.builder import build_report
from reporting.exporters import write_csv, write_json
from workflows.artifact_writer import ArtifactWriter
from workflows.run_store import RunStore
from workflows.schema import SampleRunStatus


def _storage_key(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]


def _sample(sample_id: str, task: str = "general_vqa", question: str = "Question?") -> UnifiedSample:
    return UnifiedSample(
        sample_id=sample_id,
        dataset="parity",
        split="test",
        task=task,  # type: ignore[arg-type]
        images=[ImageRef(image_id="i0", path="img.png", role="image")],
        question=question,
        ground_truth=GroundTruth(answers=["yes"]),
    )


def _status(
    sample_id: str,
    task: str,
    state: str,
    *,
    error_code: str | None = None,
    result_path: str | None = "agent_result.json",
) -> SampleRunStatus:
    return SampleRunStatus(
        sample_id=sample_id,
        task=task,  # type: ignore[arg-type]
        state=state,
        error_code=error_code,
        result_path=Path(result_path) if result_path is not None else None,
        updated_at=f"2026-01-01T00:00:00+00:00:{sample_id}",
    )


def _trace(
    *,
    resolved_task: str,
    execution_agent: str,
    fallback_used: bool = False,
    judge_status: str = "not_requested",
    inference_seconds: float = 0.42,
) -> dict:
    return {
        "router_used": True,
        "task_type": resolved_task,
        "resolved_task": resolved_task,
        "execution_task": resolved_task,
        "execution_agent": execution_agent,
        "fallback_used": fallback_used,
        "judge_status": judge_status,
        "inference_seconds": inference_seconds,
    }


def _vqa_record(sample_id: str, exact: bool, judge: str = "not_requested") -> EvaluationRecord:
    return EvaluationRecord(
        sample_id=sample_id,
        task="general_vqa",
        deterministic_metrics=VQADeterministicMetrics(exact_match=exact),
        judge_status=judge,  # type: ignore[arg-type]
    )


def _counting_record(sample_id: str, exact: int) -> EvaluationRecord:
    return EvaluationRecord(
        sample_id=sample_id,
        task="counting",
        deterministic_metrics=CountDeterministicMetrics(
            predicted_count=2,
            gold_count=2 if exact else 5,
            exact_match=exact,
            absolute_error=0 if exact else 3,
            relative_error=0.0 if exact else 0.5,
            smooth_error_score=1.0 if exact else 0.223,
        ),
        judge_status="not_requested",
    )


def _create_run(tmp_path: Path) -> Path:
    store = RunStore(tmp_path / "runs", tmp_path)
    store.create_run(
        config_payload={"k": "v"},
        model_ids={"qwen": "q"},
        prompt_paths=[],
        run_id="report-run",
    )
    return tmp_path / "runs" / "report-run"


def _write_sample(
    run_dir: Path,
    *,
    run_task: str,
    sample: UnifiedSample,
    status: SampleRunStatus,
    trace: dict | None = None,
    evaluation: EvaluationRecord | None = None,
    payload: AgentResult | None = None,
    corrupt_trace: bool = False,
    corrupt_evaluation: bool = False,
    corrupt_status: bool = False,
) -> Path:
    writer = ArtifactWriter()
    sample_dir = run_dir / "tasks" / run_task / "samples" / _storage_key(sample.sample_id)
    writer.write_sample(sample_dir, sample)
    if corrupt_status:
        (sample_dir / "status.json").write_text("{corrupt", encoding="utf-8")
    else:
        writer.write_final_status(sample_dir, status)
    if trace is not None:
        if corrupt_trace:
            (sample_dir / "agent_trace.json").write_text("not json", encoding="utf-8")
        else:
            writer.write_trace(sample_dir, trace)
    if evaluation is not None:
        filename = {
            "general_vqa": "vqa_evaluation.json",
            "counting": "counting_evaluation.json",
        }[evaluation.task]
        if corrupt_evaluation:
            (sample_dir / filename).write_text("{corrupt", encoding="utf-8")
        else:
            writer.write_evaluation(sample_dir, evaluation, filename=filename)
    if payload is not None:
        writer.write_execution(
            sample_dir,
            AgentExecution(
                agent_name="general_vqa_agent",
                payload=payload,
                result_filename="agent_result.json",
            ),
        )
    result_path = (
        f"tasks/{run_task}/samples/{_storage_key(sample.sample_id)}/agent_result.json"
        if status.result_path is not None
        else None
    )
    writer.append_prediction(
        run_dir,
        sample_id=sample.sample_id,
        run_task=run_task,
        task=status.task,
        status=status,
        result_path=result_path,
    )
    return sample_dir


def _write_probe(run_dir: Path, run_task: str) -> None:
    task_dir = run_dir / "tasks" / run_task
    (task_dir).mkdir(parents=True, exist_ok=True)
    (task_dir / "dataset_probe.json").write_text(
        json.dumps(
            {
                "dataset": "parity",
                "version": "1",
                "sample_file": "samples.jsonl",
                "observed_fields": ["id"],
                "sample_count": 4,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _build_mixed_run(tmp_path: Path) -> Path:
    run_dir = _create_run(tmp_path)
    _write_probe(run_dir, "general_vqa")
    _write_probe(run_dir, "counting")
    _write_probe(run_dir, "auto")
    # succeeded VQA with a succeeded judge
    _write_sample(
        run_dir,
        run_task="general_vqa",
        sample=_sample("s1"),
        status=_status("s1", "general_vqa", "succeeded"),
        trace=_trace(
            resolved_task="general_vqa",
            execution_agent="general_vqa_agent",
            judge_status="succeeded",
        ),
        evaluation=_vqa_record("s1", exact=True, judge="succeeded"),
        payload=AgentResult(agent_name="general_vqa_agent", answer="yes"),
    )
    # partial counting with fallback
    _write_sample(
        run_dir,
        run_task="counting",
        sample=_sample("s2", task="counting", question="How many?"),
        status=_status("s2", "counting", "partial"),
        trace=_trace(
            resolved_task="counting",
            execution_agent="counting_agent",
            fallback_used=True,
        ),
        evaluation=_counting_record("s2", exact=1),
    )
    # failed with a stable code
    _write_sample(
        run_dir,
        run_task="general_vqa",
        sample=_sample("s3"),
        status=_status("s3", "general_vqa", "failed", error_code="RuntimeError"),
        trace=None,
        evaluation=None,
    )
    # skipped
    _write_sample(
        run_dir,
        run_task="auto",
        sample=_sample("s4", task="caption", question=""),
        status=_status("s4", "caption", "skipped", error_code="FAIL_FAST_CANCELLED", result_path=None),
    )
    return run_dir


def test_build_report_mixed_states(tmp_path: Path) -> None:
    report = build_report(_build_mixed_run(tmp_path))
    assert report.run_id == "report-run"
    assert report.dataset == "parity"
    assert report.total == 4
    assert report.succeeded == 1
    assert report.partial == 1
    assert report.failed == 1
    assert report.skipped == 1
    assert [item.run_task for item in report.tasks] == ["auto", "counting", "general_vqa"]
    by_task = {item.run_task: item for item in report.tasks}
    assert by_task["general_vqa"].total == 2
    assert by_task["general_vqa"].succeeded == 1
    assert by_task["general_vqa"].failed == 1
    assert by_task["counting"].fallback_count == 1
    assert by_task["counting"].fallback_rate == 1.0
    assert by_task["general_vqa"].fallback_count == 0
    assert by_task["general_vqa"].agent_usage == {"general_vqa_agent": 1}
    assert by_task["general_vqa"].judge_status_counts == {
        "not_requested": 1,
        "succeeded": 1,
    }
    vqa_metrics = by_task["general_vqa"].metrics["general_vqa"]
    assert vqa_metrics["metric"] == "exact_match_accuracy"
    assert vqa_metrics["score"] == 1.0
    counting_metrics = by_task["counting"].metrics["counting"]
    assert counting_metrics["exact_match_accuracy"] == 1.0
    samples = {item.sample_id: item for item in report.samples}
    assert samples["s1"].judge_status == "succeeded"
    assert samples["s1"].prediction == "yes"
    assert samples["s1"].execution_agent == "general_vqa_agent"
    assert samples["s1"].inference_seconds == 0.42
    assert samples["s3"].error_code == "RuntimeError"
    assert samples["s4"].judge_status == "not_requested"


def test_build_report_last_row_wins(tmp_path: Path) -> None:
    run_dir = _create_run(tmp_path)
    _write_sample(
        run_dir,
        run_task="general_vqa",
        sample=_sample("s1"),
        status=_status("s1", "general_vqa", "succeeded"),
        trace=_trace(resolved_task="general_vqa", execution_agent="general_vqa_agent"),
        evaluation=_vqa_record("s1", exact=True),
    )
    _write_sample(
        run_dir,
        run_task="general_vqa",
        sample=_sample("s1"),
        status=_status("s1", "general_vqa", "failed", error_code="RuntimeError"),
    )
    report = build_report(run_dir)
    assert report.total == 1
    assert report.failed == 1
    assert report.samples[0].state == "failed"
    assert report.samples[0].error_code == "RuntimeError"


def test_build_report_corrupt_optional_artifacts(tmp_path: Path) -> None:
    run_dir = _create_run(tmp_path)
    _write_sample(
        run_dir,
        run_task="general_vqa",
        sample=_sample("s1"),
        status=_status("s1", "general_vqa", "succeeded"),
        trace=_trace(resolved_task="general_vqa", execution_agent="general_vqa_agent"),
        evaluation=_vqa_record("s1", exact=True),
        corrupt_trace=True,
        corrupt_evaluation=True,
        corrupt_status=True,
    )
    report = build_report(run_dir)
    assert report.total == 1
    sample = report.samples[0]
    assert sample.state == "succeeded"  # the index row remains authoritative
    assert sample.error_code is None
    assert sample.evaluation is None
    assert sample.execution_agent is None
    assert sample.fallback_used is False


def test_build_report_deterministic(tmp_path: Path) -> None:
    run_dir = _build_mixed_run(tmp_path)
    first = build_report(run_dir).model_dump(mode="json")
    second = build_report(run_dir).model_dump(mode="json")
    assert first == second


def test_build_report_no_machine_paths_or_secrets(tmp_path: Path) -> None:
    run_dir = _build_mixed_run(tmp_path)
    serialized = json.dumps(build_report(run_dir).model_dump(mode="json"))
    assert str(tmp_path) not in serialized
    assert "sk-" not in serialized
    assert "Bearer " not in serialized
    # Sample ids and run-relative paths only. / 只有样本 id 与 run 相对路径。
    assert "tasks/general_vqa/samples/" in serialized


def test_exporters_json_and_csv(tmp_path: Path) -> None:
    report = build_report(_build_mixed_run(tmp_path))
    json_path = write_json(report, tmp_path / "report.json")
    csv_path = write_csv(report, tmp_path / "samples.csv")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["total"] == 4
    assert len(payload["samples"]) == 4
    csv_text = csv_path.read_text(encoding="utf-8-sig")
    assert csv_text.startswith("run_task,sample_id,task,state")
    assert "general_vqa,s1,general_vqa,succeeded" in csv_text
    assert "RuntimeError" in csv_text
    # Deterministic exporters: re-writing yields identical bytes.
    # 确定性导出：重复写出字节一致。
    write_json(report, json_path)
    write_csv(report, csv_path)
    assert json_path.read_bytes() == json_path.read_bytes()
    assert csv_path.read_bytes() == csv_path.read_bytes()


# ── terminal samples without result paths (Fix B) / 无结果路径的终态样本 ────


def _terminal_sample(
    run_dir: Path,
    *,
    run_task: str,
    sample_id: str,
    task: str,
    state: str,
    error_code: str,
    trace: dict | None = None,
    with_sample_json: bool = True,
) -> None:
    writer = ArtifactWriter()
    sample_dir = run_dir / "tasks" / run_task / "samples" / _storage_key(sample_id)
    if with_sample_json:
        # A pre-task failure has no materialized sample at all.
        # 预 task 失败根本没有物化样本。
        writer.write_sample(sample_dir, _sample(sample_id, task=task))
    writer.write_final_status(
        sample_dir,
        _status(sample_id, task, state, error_code=error_code, result_path=None),
    )
    if trace is not None:
        writer.write_trace(sample_dir, trace)
    writer.append_prediction(
        run_dir,
        sample_id=sample_id,
        run_task=run_task,
        task=task,
        status=_status(sample_id, task, state, error_code=error_code, result_path=None),
        result_path=None,
    )


def test_report_reads_terminal_samples_without_result_path(tmp_path: Path) -> None:
    """failed/skipped samples with result_path=null must still surface
    status.json error codes through the identity-based sample directory.
    result_path=null 的 failed/skipped 样本仍必须经身份样本目录读出
    status.json 的错误码。"""
    run_dir = _create_run(tmp_path)
    _write_probe(run_dir, "auto")
    _terminal_sample(
        run_dir,
        run_task="auto",
        sample_id="cancelled-1",
        task="caption",
        state="skipped",
        error_code="FAIL_FAST_CANCELLED",
    )
    _terminal_sample(
        run_dir,
        run_task="auto",
        sample_id="notstarted-1",
        task="caption",
        state="skipped",
        error_code="FAIL_FAST_NOT_STARTED",
    )
    _terminal_sample(
        run_dir,
        run_task="auto",
        sample_id="pre-task-1",
        task="unknown",
        state="failed",
        error_code="EMPTY_UNRESOLVABLE_REQUEST",
        with_sample_json=False,
    )
    report = build_report(run_dir)
    samples = {item.sample_id: item for item in report.samples}
    assert samples["cancelled-1"].state == "skipped"
    assert samples["cancelled-1"].error_code == "FAIL_FAST_CANCELLED"
    assert samples["notstarted-1"].error_code == "FAIL_FAST_NOT_STARTED"
    assert samples["pre-task-1"].state == "failed"
    assert samples["pre-task-1"].task == "unknown"
    assert samples["pre-task-1"].error_code == "EMPTY_UNRESOLVABLE_REQUEST"
    assert report.skipped == 2
    assert report.failed == 1


# ── path-escape regression (Fix A) / 路径逃逸回归 ───────────────────────────


def test_report_never_reads_outside_run_and_result_path_fails_closed(
    tmp_path: Path,
) -> None:
    """A malicious result_path must never steer reporting outside the run
    directory: the sample directory comes from (run_task, sample_id), the
    display path degrades to None, and sentinel content outside the run never
    reaches the serialized report. 恶意 result_path 绝不引导 reporting 读取
    run 目录之外：样本目录来自 (run_task, sample_id)，展示路径降级为 None，
    run 外的哨兵内容绝不进入序列化报告。"""
    run_dir = _create_run(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = "DO_NOT_READ_OUTSIDE_RUN"
    for filename in ("status.json", "sample.json", "agent_result.json"):
        (outside / filename).write_text(
            json.dumps({"sentinel": sentinel}), encoding="utf-8"
        )
    _write_probe(run_dir, "general_vqa")
    writer = ArtifactWriter()
    sample = _sample("s1")
    sample_dir = run_dir / "tasks" / "general_vqa" / "samples" / _storage_key("s1")
    writer.write_sample(sample_dir, sample)
    status = _status("s1", "general_vqa", "succeeded")
    writer.write_final_status(sample_dir, status)
    writer.write_trace(
        sample_dir,
        _trace(resolved_task="general_vqa", execution_agent="general_vqa_agent"),
    )
    for malicious in (
        "../outside/agent_result.json",
        "C:/outside/agent_result.json",
        r"\\server\share\agent_result.json",
        "foo/../../outside/result.json",
    ):
        writer.append_prediction(
            run_dir,
            sample_id="s1",
            run_task="general_vqa",
            task="general_vqa",
            status=status,
            result_path=malicious,
        )
    report = build_report(run_dir)
    assert report.total == 1
    sample_row = report.samples[0]
    assert sample_row.result_path is None  # corrupt index degrades / 降级
    assert sample_row.execution_agent == "general_vqa_agent"  # identity dir used
    serialized = json.dumps(report.model_dump(mode="json"))
    assert sentinel not in serialized
    assert "outside" not in serialized


def test_report_unsafe_run_task_ignored(tmp_path: Path) -> None:
    """Unsafe run_task namespaces must never be joined into paths.
    不安全的 run_task 命名空间绝不拼接进路径。"""
    run_dir = _create_run(tmp_path)
    writer = ArtifactWriter()
    status = _status("s1", "general_vqa", "succeeded")
    for malicious_run_task in ("../x", "a\\b", "C:drive", "a/b"):
        writer.append_prediction(
            run_dir,
            sample_id="s1",
            run_task=malicious_run_task,
            task="general_vqa",
            status=status,
            result_path="tasks/a/samples/k/agent_result.json",
        )
    report = build_report(run_dir)
    assert report.total == 4
    assert all(item.execution_agent is None for item in report.samples)
