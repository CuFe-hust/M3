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
from agents.counting.schema import (
    CountingBackendAttemptAudit,
    CountingExecutionAudit,
    CountingResult,
    GlobalPointObservation,
    IssueRecord,
    PointProvenance,
)
from agents.schema import AgentResult
from data.schema import GroundTruth, ImageRef, UnifiedSample
from evaluation.records import (
    CaptionDeterministicMetrics,
    CountDeterministicMetrics,
    EvaluationRecord,
    GroundingDeterministicMetrics,
    VQADeterministicMetrics,
)
from reporting.adapters import (
    load_counting_attempts,
    load_evaluation,
    load_model_calls,
    load_payload,
)
from reporting.builder import _execution_path, build_report
from reporting.exporters import write_csv, write_json
from workflows.artifact_writer import ArtifactWriter
from workflows.run_store import RunStore
from workflows.schema import SampleRunStatus


def test_reporting_recognizes_historical_and_v5_planner_traces(
    tmp_path: Path,
) -> None:
    for mode in (
        "visual-task-plan-v2", "visual-task-plan-v3", "visual-task-plan-v4",
        "visual-task-plan-v5",
    ):
        path = _execution_path(
            tmp_path,
            run_task="general_vqa",
            task="general_vqa",
            trace={"planning_mode": mode},
            model_calls=[],
            structured_artifacts=[],
            evaluation=None,
        )
        assert "workflows.visual_planner.VisualTaskPlanner" in path


def _v2_point(
    point_id: str, x: int, *, accepted: bool, source: str
) -> GlobalPointObservation:
    return GlobalPointObservation(
        global_id=point_id,
        target="small-vehicle",
        source_tile_id="tile-0",
        local_id=point_id,
        local_x_norm=x,
        local_y_norm=x,
        local_radius_norm=2,
        global_x_px=x,
        global_y_px=x,
        global_x_norm=x,
        global_y_norm=x,
        radius_px=3,
        confidence=0.9,
        ownership_valid=True,
        near_core_boundary=False,
        accepted=accepted,
        rejection_reason=None if accepted else "LOW_CONFIDENCE",
        short_evidence="persisted point",
        provenance=PointProvenance(source=source),  # type: ignore[arg-type]
    )


def test_load_payload_uses_canonical_fine_grained_counting_family(
    tmp_path: Path,
) -> None:
    payload = CountingResult(
        sample_id="fg1",
        target="car",
        question="How many cars?",
        source_width=10,
        source_height=10,
        tile_count=0,
        final_count=0,
        status="completed",
    )
    (tmp_path / "counting_result.json").write_text(
        payload.model_dump_json(), encoding="utf-8"
    )
    assert load_payload(tmp_path, "fine_grained_counting") == payload


def test_load_payload_does_not_treat_unknown_task_as_vqa(tmp_path: Path) -> None:
    (tmp_path / "agent_result.json").write_text(
        AgentResult(agent_name="general_vqa_agent", answer="x").model_dump_json(),
        encoding="utf-8",
    )
    assert load_payload(tmp_path, "unknown") is None


def test_reporting_loads_e2_family_artifacts(tmp_path: Path) -> None:
    vqa = EvaluationRecord(
        sample_id="e2-vqa",
        task="general_vqa",
        deterministic_metrics=VQADeterministicMetrics(exact_match=True),
        judge_status="not_requested",
    )
    (tmp_path / "vqa_evaluation.json").write_text(
        vqa.model_dump_json(), encoding="utf-8"
    )
    payload = AgentResult(agent_name="change_agent", answer="road added")
    (tmp_path / "agent_result.json").write_text(
        payload.model_dump_json(), encoding="utf-8"
    )
    for task in ("change_qa", "spatial_relation"):
        assert load_evaluation(tmp_path, task) == vqa
        assert load_payload(tmp_path, task) == payload

    caption = EvaluationRecord(
        sample_id="e2-caption",
        task="caption",
        deterministic_metrics=CaptionDeterministicMetrics(
            candidate="road added", references=["road added"]
        ),
        judge_status="not_requested",
    )
    (tmp_path / "caption_evaluation.json").write_text(
        caption.model_dump_json(), encoding="utf-8"
    )
    assert load_evaluation(tmp_path, "change_caption") == caption
    assert load_payload(tmp_path, "change_caption") == payload


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


def _vqa_record(
    sample_id: str,
    exact: bool,
    judge: str = "not_requested",
    *,
    judge_parsed: object | None = None,
    judge_error: str | None = None,
) -> EvaluationRecord:
    return EvaluationRecord(
        sample_id=sample_id,
        task="general_vqa",
        deterministic_metrics=VQADeterministicMetrics(exact_match=exact),
        judge_status=judge,  # type: ignore[arg-type]
        judge_parsed=judge_parsed,
        judge_error=judge_error,
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


def _caption_record(sample_id: str, candidate: str = "a road") -> EvaluationRecord:
    return EvaluationRecord(
        sample_id=sample_id,
        task="caption",
        deterministic_metrics=CaptionDeterministicMetrics(
            candidate=candidate,
            references=[candidate],
        ),
        judge_status="not_requested",
    )


def _grounding_record(sample_id: str, iou: float) -> EvaluationRecord:
    return EvaluationRecord(
        sample_id=sample_id,
        task="grounding",
        deterministic_metrics=GroundingDeterministicMetrics(
            iou=iou,
            iou_at_0_5=iou >= 0.5,
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
            "grounding": "grounding_evaluation.json",
            "caption": "caption_evaluation.json",
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


def test_report_separates_deterministic_and_semantic_judge_metrics(
    tmp_path: Path,
) -> None:
    run_dir = _create_run(tmp_path)
    records = [
        _vqa_record("exact", exact=True),
        _vqa_record(
            "equivalent",
            exact=False,
            judge="succeeded",
            judge_parsed={"score": 1, "concise_rationale": "same meaning"},
        ),
        _vqa_record("unresolved", exact=False),
    ]
    for record in records:
        _write_sample(
            run_dir,
            run_task="general_vqa",
            sample=_sample(record.sample_id),
            status=_status(record.sample_id, "general_vqa", "succeeded"),
            evaluation=record,
        )
    task = build_report(run_dir).tasks[0]
    assert task.metrics["general_vqa"] == {
        "metric": "exact_match_accuracy",
        "correct": 1,
        "total": 3,
        "score": 1 / 3,
    }
    assert "coverage" not in task.metrics["general_vqa"]
    semantic = task.judge_metrics["vqa_semantic_equivalence"]
    assert semantic["deterministic_exact_correct"] == 1
    assert semantic["eligible_mismatches"] == 2
    assert semantic["semantic_equivalent_mismatches"] == 1
    assert semantic["coverage"] == 0.5
    assert semantic["corrected_correct"] == 2
    assert semantic["lower_bound_score"] == 2 / 3
    assert semantic["complete"] is False
    assert semantic["score"] is None


def test_no_vqa_records_have_empty_judge_metrics_and_counting_judge_is_excluded(
    tmp_path: Path,
) -> None:
    run_dir = _create_run(tmp_path)
    counting = _counting_record("count-1", exact=1).model_copy(
        update={
            "judge_status": "succeeded",
            "judge_parsed": {"verdict": "correct"},
        }
    )
    _write_sample(
        run_dir,
        run_task="counting",
        sample=_sample("count-1", task="counting", question="How many?"),
        status=_status("count-1", "counting", "succeeded"),
        evaluation=counting,
    )
    report = build_report(run_dir)
    assert report.tasks[0].metrics["counting"]["exact_match_accuracy"] == 1.0
    assert report.tasks[0].judge_metrics == {}


def test_report_aggregates_caption_and_change_caption_as_one_family(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Both runtime tasks contribute to the canonical caption corpus.
    两个 runtime task 都进入同一个 canonical caption 语料。"""
    run_dir = _create_run(tmp_path)
    captured_ids: list[str] = []

    def _fake_aggregate(records):
        captured_ids.extend(record.sample_id for record in records)
        return {
            "total": len(records),
            "BLEU_1": 0.5,
            "BLEU_2": 0.4,
            "BLEU_3": 0.3,
            "BLEU_4": 0.2,
            "METEOR": 0.42,
            "ROUGE_L": 0.43,
            "CIDEr": 0.44,
        }

    monkeypatch.setattr("reporting.builder.aggregate_caption", _fake_aggregate)
    for sample_id, runtime_task in (
        ("caption-1", "caption"),
        ("change-caption-1", "change_caption"),
    ):
        sample = _sample(sample_id, task="caption", question="")
        if runtime_task == "change_caption":
            sample = UnifiedSample(
                sample_id=sample_id,
                dataset="parity",
                split="test",
                task="change_caption",
                images=[
                    ImageRef(image_id="i0", path="t1.png", role="t1"),
                    ImageRef(image_id="i1", path="t2.png", role="t2"),
                ],
                question="",
                ground_truth=GroundTruth(answers=["a road"]),
            )
        _write_sample(
            run_dir,
            run_task="caption-corpus",
            sample=sample,
            status=_status(sample_id, runtime_task, "succeeded"),
            evaluation=_caption_record(sample_id),
        )

    metrics = build_report(run_dir).tasks[0].metrics["caption"]
    assert captured_ids == ["caption-1", "change-caption-1"]
    assert metrics == {
        "metric_status": "ok",
        "total": 2,
        "BLEU_1": 0.5,
        "BLEU_2": 0.4,
        "BLEU_3": 0.3,
        "BLEU_4": 0.2,
        "METEOR": 0.42,
        "ROUGE_L": 0.43,
        "CIDEr": 0.44,
    }


def test_report_caption_dependency_missing_is_nonfatal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import builtins

    run_dir = _create_run(tmp_path)
    _write_sample(
        run_dir,
        run_task="caption",
        sample=_sample("caption-1", task="caption", question=""),
        status=_status("caption-1", "caption", "succeeded"),
        evaluation=_caption_record("caption-1"),
    )
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "pycocoevalcap" or name.startswith("pycocoevalcap."):
            raise ImportError("raw environment detail")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    report = build_report(run_dir)
    assert report.tasks[0].metrics["caption"] == {
        "metric_status": "dependency_missing",
        "record_count": 1,
        "dependency": "pycocoevalcap",
    }
    assert "raw environment detail" not in report.model_dump_json()


def test_report_without_caption_never_imports_optional_dependency(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import builtins

    run_dir = _create_run(tmp_path)
    _write_sample(
        run_dir,
        run_task="general_vqa",
        sample=_sample("vqa-only"),
        status=_status("vqa-only", "general_vqa", "succeeded"),
        evaluation=_vqa_record("vqa-only", exact=True),
    )
    real_import = builtins.__import__
    attempts: list[str] = []

    def _guarded_import(name, *args, **kwargs):
        if name == "pycocoevalcap" or name.startswith("pycocoevalcap."):
            attempts.append(name)
            raise AssertionError("optional caption dependency imported")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    report = build_report(run_dir)
    assert report.tasks[0].metrics["general_vqa"]["score"] == 1.0
    assert attempts == []


def test_incompatible_grounding_is_excluded_from_metric_denominator(
    tmp_path: Path,
) -> None:
    run_dir = _create_run(tmp_path)
    _write_sample(
        run_dir,
        run_task="grounding",
        sample=_sample("compatible", task="grounding"),
        status=_status("compatible", "grounding", "succeeded"),
        evaluation=_grounding_record("compatible", 1.0),
    )
    _write_sample(
        run_dir,
        run_task="grounding",
        sample=_sample("incompatible", task="grounding"),
        status=_status("incompatible", "grounding", "succeeded"),
        evaluation=None,
    )
    task = build_report(run_dir).tasks[0]
    assert task.total == 2
    assert task.metrics["grounding"]["total"] == 1
    assert task.metrics["grounding"]["accuracy"] == 1.0


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
    assert all("judge_metrics" in task for task in payload["tasks"])
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

# ── benchmark/audit exporters (Task 11G) / 基准与审计导出 ───────────────────


def _export_report(tmp_path: Path, *, judged: bool = False) -> Report:
    """One minimal report with two samples; one optionally judged.
    含两个样本的最小报告；其中一个可选已 judge。"""
    from evaluation.judges.base import VQAAnswerJudgeResult
    from reporting.schema import Report, ReportSample

    samples = [
        ReportSample(
            sample_id="a1",
            run_task="general_vqa",
            task="general_vqa",
            state="succeeded",
            question="Is there a road?",
            prediction="yes",
            updated_at="2026-08-09T00:00:00Z",
        ),
        ReportSample(
            sample_id="a2",
            run_task="general_vqa",
            task="general_vqa",
            state="succeeded",
            question="How many cars?",
            prediction="3",
            updated_at="2026-08-09T00:00:01Z",
            evaluation=(
                EvaluationRecord(
                    sample_id="a2",
                    task="general_vqa",
                    deterministic_metrics=VQADeterministicMetrics(exact_match=True),
                    judge_status="succeeded",
                    judge_parsed=VQAAnswerJudgeResult(
                        score=1, concise_rationale="matches"
                    ),
                )
                if judged
                else None
            ),
        ),
    ]
    return Report(
        run_id="export-run",
        dataset="demo",
        total=2,
        succeeded=2,
        partial=0,
        failed=0,
        skipped=0,
        samples=samples,
    )


def test_exporters_samples_jsonl_deterministic_utf8(tmp_path: Path) -> None:
    from reporting.exporters import write_samples_jsonl

    report = _export_report(tmp_path)
    path = write_samples_jsonl(report, tmp_path / "samples.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert rows[0]["sample_id"] == "a1"
    assert rows[1]["sample_id"] == "a2"
    assert rows[1]["evaluation"] is None  # no evaluation attached
    # deterministic: same report yields the identical bytes / 确定性：相同报告
    # 产生相同字节。
    again = tmp_path / "again.jsonl"
    write_samples_jsonl(report, again)
    assert path.read_bytes() == again.read_bytes()


def test_exporters_deepseek_audit_stable_metadata_no_secret(tmp_path: Path) -> None:
    from reporting.exporters import write_deepseek_audit

    report = _export_report(tmp_path, judged=True)
    path = write_deepseek_audit(report, tmp_path / "deepseek_audit.jsonl")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1  # only the judged sample / 仅已 judge 样本
    row = json.loads(lines[0])
    assert row["sample_id"] == "a2"
    # without run_dir there is no persisted RequestMeta: identity stays null,
    # never a synthesized value. 无 run_dir 时没有持久化 RequestMeta：身份
    # 保持 null，绝不合成值。
    assert row["request_id"] is None
    assert row["request_hash"] is None
    assert row["prompt_version"] is None
    assert row["judge_status"] == "succeeded"
    assert row["judge_parsed"]["score"] == 1
    text = path.read_text(encoding="utf-8")
    assert "api_key" not in text.lower()
    assert "authorization" not in text.lower()
    assert "sk-" not in text
    assert "raw_response" not in text


def test_exporters_metadata_no_host_absolute_paths(tmp_path: Path) -> None:
    from reporting.exporters import REPORT_SCHEMA_VERSION, write_metadata_json

    report = _export_report(tmp_path)
    run_dir = tmp_path / "runs" / "export-run"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "export-run",
                "created_at": "2026-08-09T00:00:00Z",
                "git_commit": None,
                "git_dirty": None,
                "config_hash": "hash",
                "prompt_hashes": {},
                "model_ids": {"qwen": "fake-qwen", "deepseek": "fake-deepseek"},
                "dataset": "demo",
                "split": "test",
                "sample_filter": None,
            }
        ),
        encoding="utf-8",
    )
    path = write_metadata_json(report, tmp_path / "metadata.json", run_dir=run_dir)
    metadata = json.loads(path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == REPORT_SCHEMA_VERSION
    assert metadata["run_id"] == "export-run"
    assert metadata["dataset"] == "demo"
    assert metadata["split"] == "test"
    assert metadata["model_ids"]["qwen"] == "fake-qwen"
    assert metadata["created_at"] == "2026-08-09T00:00:00Z"
    assert metadata["counts"]["succeeded"] == 2
    assert metadata["sample_count"] == 2
    text = path.read_text(encoding="utf-8")
    assert tmp_path.as_posix() not in text  # no host absolute path


def test_exporters_external_standard_namespace(tmp_path: Path) -> None:
    from reporting.exporters import write_external_standard_report

    standard = {"primary_metric": "open_vqa_accuracy", "score": 75.0}
    path = write_external_standard_report(standard, tmp_path / "standard.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["external_standard"]["score"] == 75.0
    # never merged into deterministic metric names / 绝不并入确定性指标名
    assert "primary_metric" not in payload
    assert "score" not in payload


def test_exporters_mme_official_export_read_only(tmp_path: Path) -> None:
    from reporting.exporters import write_mme_official_export

    source = tmp_path / "MME_RealWorld.json"
    original = [
        {
            "Question_id": "q1",
            "Text": "Which is true?",
            "Answer choices": ["A", "B"],
            "Ground truth": "A",
        },
        {
            "Question_id": "q2",
            "Text": "Which is false?",
            "Answer choices": ["A", "B"],
            "Ground truth": "B",
        },
    ]
    source.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
    source_before = source.read_bytes()
    output = tmp_path / "mme_real_rs.official.json"
    write_mme_official_export(
        source, {"q1": "A", "q2": "B"}, output
    )
    rows = json.loads(output.read_text(encoding="utf-8"))
    assert rows[0]["Output"] == "A"
    assert rows[1]["Output"] == "B"
    # unrelated fields preserved exactly / 未关联字段原样保留
    assert rows[0]["Text"] == "Which is true?"
    assert rows[0]["Ground truth"] == "A"
    assert rows[1]["Answer choices"] == ["A", "B"]
    # source untouched byte-for-byte / 源文件逐字节不变
    assert source.read_bytes() == source_before
    # missing prediction leaves an empty Output / 缺失预测保留空 Output
    output2 = tmp_path / "partial.official.json"
    write_mme_official_export(source, {"q1": "A"}, output2)
    rows2 = json.loads(output2.read_text(encoding="utf-8"))
    assert rows2[0]["Output"] == "A"
    assert rows2[1]["Output"] == ""


def test_report_v2_projects_manifest_counting_routing_and_aggregates(
    tmp_path: Path,
) -> None:
    run_dir = _create_run(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "dataset": "audit-set",
        "split": "test",
        "git_commit": "abc123",
        "git_dirty": True,
        "config_hash": "cfg123",
        "model_ids": {"counter": "logical-counter"},
        "prompt_hashes": {"count": "prompt123"},
    })
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    sample = UnifiedSample(
        sample_id="count-v2",
        dataset="audit-set",
        split="test",
        task="counting",
        images=[ImageRef(image_id="image-0", path="image.png", role="image")],
        question="How many small vehicles?",
        ground_truth=GroundTruth(count=3),
    )
    evaluation = EvaluationRecord(
        sample_id="count-v2",
        task="counting",
        deterministic_metrics=CountDeterministicMetrics(
            predicted_count=2,
            gold_count=3,
            exact_match=0,
            absolute_error=1,
            relative_error=1 / 3,
            smooth_error_score=0.5,
        ),
        judge_status="not_requested",
    )
    sample_dir = _write_sample(
        run_dir,
        run_task="counting",
        sample=sample,
        status=_status("count-v2", "counting", "partial", result_path="counting_result.json"),
        trace={
            "resolved_task": "counting",
            "execution_agent": "counting_agent",
            "inference_seconds": 1.25,
            "candidate_backends": ["detector_obb_csl_001", "segmenter_mitb2_001", "quantity_proposal", "qwen_point"],
            "attempted_backends": ["detector_obb_csl_001", "segmenter_mitb2_001"],
            "primary_backend": "detector_obb_csl_001",
            "primary_backend_kind": "yolo_obb",
            "final_backend": "segmenter_mitb2_001",
            "final_backend_kind": "semantic_segmentation",
            "fallback_triggered": True,
            "fallback_history": [{
                "backend": "detector_obb_csl_001",
                "kind": "yolo_obb",
                "reason_code": "BACKEND_UNAVAILABLE",
                "error_type": "DetectorWeightsMissingError",
            }],
            "selection_reason": ["fixed_kind_rank_then_priority_then_name"],
            "backend_trace": {"counting_mode": "connected_components"},
            "status": "partial",
        },
        evaluation=evaluation,
    )
    result = CountingResult(
        sample_id="count-v2",
        target="small-vehicle",
        question=sample.question,
        source_width=100,
        source_height=100,
        tile_count=2,
        initial_tile_count=1,
        leaf_tile_count=2,
        succeeded_tiles=["tile-0"],
        failed_tiles=["tile-1"],
        global_points=[
            _v2_point("p1", 10, accepted=True, source="yolo_obb_center"),
            _v2_point("p2", 20, accepted=True, source="semantic_component_centroid"),
            _v2_point("p3", 30, accepted=False, source="qwen_point"),
        ],
        merged_groups=[["p1", "p2"]],
        unresolved_conflicts=["p3"],
        warnings=[IssueRecord(code="SEMANTIC_TILE_INFERENCE_FAILED", message="safe")],
        final_count=2,
        status="partial",
    )
    (sample_dir / "counting_result.json").write_text(result.model_dump_json(), encoding="utf-8")

    report = build_report(run_dir)
    row = report.samples[0]
    assert report.metadata is not None
    assert report.metadata.model_dump(mode="json") == {
        "run_id": "report-run", "dataset": "audit-set", "split": "test",
        "git_commit": "abc123", "git_dirty": True, "config_hash": "cfg123",
        "model_ids": {"counter": "logical-counter"},
        "prompt_hashes": {"count": "prompt123"},
        "created_at": manifest["created_at"], "sample_filter": None,
    }
    assert row.routing.primary_backend == "detector_obb_csl_001"
    assert row.routing.final_backend == "segmenter_mitb2_001"
    assert row.routing.fallback_used is True
    assert row.routing.fallback_history[0].reason_code == "BACKEND_UNAVAILABLE"
    assert row.task_detail is not None and row.task_detail.kind == "counting"
    detail = row.task_detail
    assert detail.predicted_count == 2 and detail.gold_count == 3
    assert detail.absolute_error == 1 and detail.exact_match is False
    assert detail.accepted_point_count == 2 and detail.rejected_point_count == 1
    assert detail.provenance_usage == {
        "qwen_point": 1,
        "semantic_component_centroid": 1,
        "yolo_obb_center": 1,
    }
    assert report.routing_summary.primary_backend_usage == {"detector_obb_csl_001": 1}
    assert report.routing_summary.final_backend_usage == {"segmenter_mitb2_001": 1}
    assert report.failure_summary.warning_codes == {"SEMANTIC_TILE_INFERENCE_FAILED": 1}
    assert report.counting_target_summary[0].evaluated_count == 1
    assert report.counting_target_summary[0].mae == 1


def test_report_v2_missing_trace_and_private_run_request_stay_safe(tmp_path: Path) -> None:
    run_dir = _create_run(tmp_path)
    (run_dir / "run_request.json").write_text(json.dumps({
        "dataset": "private", "dataset_root": "C:/private/dataset", "split": "test",
        "task_mode": "explicit", "tasks": ["general_vqa"], "auto_task": False,
        "sample_ids": None, "limit": None, "start_index": 0, "shard_index": 0,
        "shard_count": 1, "sample_concurrency": 1, "evaluate": False,
        "judge_policy": "none", "judge_sample_rate": None, "render_errors": False,
        "fail_fast": False,
    }), encoding="utf-8")
    _write_sample(
        run_dir,
        run_task="general_vqa",
        sample=_sample("safe-v2"),
        status=_status("safe-v2", "general_vqa", "succeeded"),
        trace={
            "raw_exception": "/home/user/model.safetensors sk-test-secret",
            "selection_reason": ["C:/private/checkpoint.bin"],
        },
        payload=AgentResult(agent_name="general_vqa_agent", answer="yes"),
    )
    report = build_report(run_dir)
    serialized = json.dumps(report.model_dump(mode="json"), ensure_ascii=False)
    assert "dataset_root" not in serialized
    assert "C:/private" not in serialized
    assert "/home/user" not in serialized
    assert "sk-test-secret" not in serialized
    assert report.samples[0].routing.selection_reason is None


def test_exporters_mme_missing_source_fails_stably(tmp_path: Path) -> None:
    import pytest

    from reporting.exporters import write_mme_official_export

    with pytest.raises(FileNotFoundError):
        write_mme_official_export(
            tmp_path / "missing.json", {}, tmp_path / "out.json"
        )


def test_exporters_mme_container_shape_and_no_model_calls(tmp_path: Path) -> None:
    from reporting.exporters import write_mme_official_export

    source = tmp_path / "annotations.json"
    source.write_text(
        json.dumps({"annotations": [{"question_id": "x1", "field": 1}]}),
        encoding="utf-8",
    )
    output = tmp_path / "out.json"
    write_mme_official_export(source, {"x1": "yes"}, output)
    rows = json.loads(output.read_text(encoding="utf-8"))
    assert rows[0]["Output"] == "yes"
    assert rows[0]["field"] == 1


def test_v21_projects_ground_truth_and_persisted_backend_attempt_order(tmp_path: Path) -> None:
    run_dir = _create_run(tmp_path)
    sample = UnifiedSample(
        sample_id="audit-count", dataset="audit-set", split="test", task="counting",
        images=[ImageRef(image_id="image-0", path="image.png", role="image")],
        question="How many vehicles?",
        ground_truth=GroundTruth(count=1, boxes=[[1, 2, 3, 4]], labels=["vehicle"]),
    )
    sample_dir = _write_sample(
        run_dir, run_task="counting", sample=sample,
        status=_status("audit-count", "counting", "succeeded", result_path="counting_result.json"),
        trace=_trace(resolved_task="counting", execution_agent="counting_agent"),
    )
    zero = CountingResult(
        sample_id="audit-count", target="vehicle", question=sample.question,
        source_width=100, source_height=100, tile_count=1, final_count=0, status="completed",
    )
    positive = CountingResult(
        sample_id="audit-count", target="vehicle", question=sample.question,
        source_width=100, source_height=100, tile_count=1,
        global_points=[_v2_point("p1", 20, accepted=True, source="semantic_component_centroid")],
        final_count=1, status="completed",
    )
    (sample_dir / "counting_result.json").write_text(positive.model_dump_json(), encoding="utf-8")
    audit = CountingExecutionAudit(
        sample_id="audit-count", target="vehicle",
        attempts=[
            CountingBackendAttemptAudit(
                backend_name="detector", backend_kind="yolo_obb", phase="primary",
                status="succeeded", counting=zero,
                backend_trace={
                    "raw_detections": 0,
                    "classes": ["vehicle"],
                    "model_id": "YOLO11s:iSAID:epoch111",
                    "weights_file": "isaid-yolo11s-best.pt",
                    "weights_sha256": "a" * 64,
                    "source_dataset": "iSAID",
                },
            ),
            CountingBackendAttemptAudit(
                backend_name="segmenter", backend_kind="semantic_segmentation",
                phase="zero_review", status="succeeded", counting=positive,
                error_type="ZeroReviewRecovery",
                backend_trace={
                    "raw_components": 2,
                    "logical_model_id": "SegFormer-MiT-B2:iSAID:local",
                    "weights_sha256": "b" * 64,
                    "model_revision": "rev-2",
                    "nested_not_public": {"mask": [1, 2, 3]},
                    "checkpoint": "C:/private/model.bin",
                },
            ),
        ],
    )
    (sample_dir / "counting_attempts.json").write_text(audit.model_dump_json(), encoding="utf-8")

    row = build_report(run_dir).samples[0]
    assert row.ground_truth is not None and row.ground_truth.count == 1
    assert row.ground_truth.boxes == [[1.0, 2.0, 3.0, 4.0]]
    assert [(stage.order, stage.backend_name, stage.phase) for stage in row.backend_stages] == [
        (1, "detector", "primary"), (2, "segmenter", "zero_review")]
    assert [stage.predicted_count for stage in row.backend_stages] == [0, 1]
    assert row.backend_stages[1].accepted_count == 1
    assert row.backend_stages[1].error_type == "ZeroReviewRecovery"
    backend_steps = [step for step in row.execution_steps if step.phase == "backend"]
    assert [(step.backend_name, step.operation) for step in backend_steps] == [
        ("detector", "primary"), ("segmenter", "zero_review")]
    assert backend_steps[1].reason_code == "ZeroReviewRecovery"
    assert backend_steps[0].summary_fields["weights_file"] == "isaid-yolo11s-best.pt"
    assert backend_steps[1].summary_fields["logical_model_id"] == "SegFormer-MiT-B2:iSAID:local"
    assert row.task_routing.resolved_task == "counting"
    assert row.task_routing.executed_agent == "counting_agent"
    process = build_report(run_dir).process_report
    assert process.sample_process_count == 1
    assert len(process.workflow_sequences) == 1
    assert [item.family for item in process.model_weights] == ["segmentation", "yolo"]
    yolo = next(item for item in process.model_weights if item.family == "yolo")
    assert yolo.logical_model_id == "YOLO11s:iSAID:epoch111"
    assert yolo.weights_file == "isaid-yolo11s-best.pt"
    assert yolo.weights_sha256 == "a" * 64
    segmenter = next(item for item in process.model_weights if item.family == "segmentation")
    assert segmenter.logical_model_id == "SegFormer-MiT-B2:iSAID:local"
    assert segmenter.weights_sha256 == "b" * 64
    serialized = json.dumps(row.model_dump(mode="json"))
    assert "nested_not_public" not in serialized and "C:/private" not in serialized


def test_process_report_includes_vqa_evidence_yolo_and_segformer_calls(tmp_path: Path) -> None:
    run_dir = _create_run(tmp_path)
    sample = _sample("evidence-a")
    sample_dir = _write_sample(
        run_dir,
        run_task="general_vqa",
        sample=sample,
        status=_status("evidence-a", "general_vqa", "succeeded"),
        trace=_trace(resolved_task="general_vqa", execution_agent="general_vqa_agent"),
        payload=AgentResult(agent_name="general_vqa_agent", answer="yes"),
    )
    (sample_dir / "vqa_evidence.json").write_text(json.dumps({
        "workflow": "object_evidence_vqa",
        "call_audit": [
            {
                "layer": "yolo",
                "roi_id": "roi-1",
                "input_size": [640, 640],
                "logical_model_id": "YOLO11s:iSAID:epoch111",
                "weights_sha256": "c" * 64,
                "status": "succeeded",
                "error_code": None,
            },
            {
                "layer": "segformer",
                "roi_id": "roi-1",
                "input_size": [768, 768],
                "logical_model_id": "SegFormer-MiT-B2:iSAID:local",
                "weights_sha256": "d" * 64,
                "status": "succeeded",
                "error_code": None,
            },
        ],
    }), encoding="utf-8")

    report = build_report(run_dir)
    evidence_steps = [
        step for step in report.samples[0].execution_steps
        if step.phase == "evidence_model"
    ]
    assert [step.backend_name for step in evidence_steps] == ["yolo", "segformer"]
    assert evidence_steps[0].summary_fields["logical_model_id"] == "YOLO11s:iSAID:epoch111"
    assert evidence_steps[1].summary_fields["weights_sha256"] == "d" * 64
    assert [(item.family, item.logical_model_id) for item in report.process_report.model_weights] == [
        ("segmentation", "SegFormer-MiT-B2:iSAID:local"),
        ("yolo", "YOLO11s:iSAID:epoch111"),
    ]
    sequence = report.process_report.workflow_sequences[0]
    assert [step.backend_name for step in sequence.steps if step.phase == "evidence_model"] == [
        "yolo", "segformer",
    ]


def test_v21_model_call_loader_is_bounded_sanitized_and_best_effort(tmp_path: Path) -> None:
    call_dir = tmp_path / "sample" / "calls" / "01"
    call_dir.mkdir(parents=True)
    (call_dir / "request_meta.json").write_text(json.dumps({
        "request_id": "sample:qwen", "request_hash": "a" * 64,
        "prompt_version": "v1", "sample_id": "sample",
        "artifact_dir": "/home/private/model/output",
    }), encoding="utf-8")
    (call_dir / "request.json").write_text(json.dumps({
        "messages": [{"role": "user", "content": "safe question"}],
        "dataset_root": "/home/private/dataset", "image": "data:image/png;base64,AAAA",
    }), encoding="utf-8")
    (call_dir / "raw_response.txt").write_text(
        "<script>alert(1)</script>" + "x" * 9000, encoding="utf-8")
    (call_dir / "parsed.json").write_text(json.dumps({
        "answer": "yes", "artifact_dir": "C:/private/model/output",
    }), encoding="utf-8")
    (call_dir / "validation.json").write_text(json.dumps({
        "cache_hit": False, "valid": True,
        "response_metadata": {"latency_seconds": 0.25, "repair_used": True,
                              "token_usage": {"prompt_tokens": 4, "completion_tokens": 2}},
    }), encoding="utf-8")
    corrupt = tmp_path / "sample" / "calls" / "02"
    corrupt.mkdir()
    (corrupt / "request_meta.json").write_text("{bad", encoding="utf-8")

    calls = load_model_calls(tmp_path / "sample")
    assert len(calls) == 1
    call = calls[0]
    assert call.request_id == "sample:qwen" and call.prompt_version == "v1"
    assert call.raw_response_truncated is True
    assert call.raw_response is not None and len(call.raw_response) <= 8000
    assert call.raw_response.endswith("[truncated]")
    assert call.latency_seconds == 0.25
    assert call.token_usage == {"prompt_tokens": 4, "completion_tokens": 2}
    serialized = json.dumps(call.model_dump(mode="json"))
    assert "/home/private" not in serialized
    assert "C:/private" not in serialized
    assert "base64" not in serialized

    (tmp_path / "sample" / "counting_attempts.json").write_text("{corrupt", encoding="utf-8")
    assert load_counting_attempts(tmp_path / "sample") is None
