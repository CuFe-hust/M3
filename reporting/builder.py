"""Build the Report V2 presentation model from persisted artifacts only."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from agents.counting.schema import (
    CountingExecutionAudit,
    CountingResult,
    GlobalPointObservation,
)
from agents.schema import AgentResult
from evaluation.metrics.aggregate import aggregate_counting, aggregate_grounding, aggregate_vqa
from evaluation.metrics.caption import CaptionMetricDependencyError, aggregate_caption
from evaluation.metrics.vqa import aggregate_vqa_semantic_judge
from evaluation.records import (
    CaptionDeterministicMetrics,
    CountDeterministicMetrics,
    EvaluationRecord,
    GroundingDeterministicMetrics,
    VQADeterministicMetrics,
)
from reporting.adapters import (
    iter_current_predictions,
    load_counting_attempts,
    load_evaluation,
    load_model_calls,
    load_payload,
    load_routing_decision,
    load_run_manifest,
    load_sample,
    load_status,
    load_structured_artifacts,
    load_trace,
    prediction_text,
    read_json,
    safe_result_path,
    sample_dir_for_row,
)
from reporting.schema import (
    CaptionReportDetail,
    BackendStageView,
    ChangeReportDetail,
    CountingReportDetail,
    CountingTargetSummary,
    ExecutionStepView,
    FailureSummary,
    FallbackTransitionSummary,
    FallbackTransitionView,
    GeneralVQAReportDetail,
    GroundTruthView,
    GroundingReportDetail,
    LatencySummary,
    ModelWeightView,
    PointPreview,
    ProcessReport,
    Report,
    ReportSample,
    RoutingAttemptView,
    RoutingSummary,
    RoutingView,
    RunMetadata,
    SpatialReportDetail,
    TaskCandidateView,
    TaskRoutingView,
    TaskSummary,
    VisualAssetView,
    WorkflowSequenceView,
    WorkflowStepView,
)

_AGGREGATORS = {
    "general_vqa": aggregate_vqa,
    "counting": aggregate_counting,
    "grounding": aggregate_grounding,
}


def build_report(run_dir: Path) -> Report:
    """Create a deterministic, path-safe report without loading any image."""

    samples = [_build_sample(run_dir, row) for row in iter_current_predictions(run_dir)]
    samples.sort(key=lambda item: (item.run_task, item.sample_id))
    tasks = [
        _build_task_summary(name, [sample for sample in samples if sample.run_task == name])
        for name in sorted({sample.run_task for sample in samples})
    ]
    task_by_name = {task.run_task: task for task in tasks}
    for sample in samples:
        if isinstance(sample.task_detail, CaptionReportDetail):
            caption_metrics = task_by_name[sample.run_task].metrics.get("caption", {})
            if isinstance(caption_metrics, dict):
                sample.task_detail.metric_status = _value_str(
                    caption_metrics.get("metric_status")
                ) or "not_available"
    metadata = _run_metadata(run_dir)
    dataset = metadata.dataset if metadata and metadata.dataset else _find_dataset(run_dir)
    return Report(
        run_id=run_dir.name,
        dataset=dataset,
        metadata=metadata,
        total=len(samples),
        succeeded=sum(sample.state == "succeeded" for sample in samples),
        partial=sum(sample.state == "partial" for sample in samples),
        failed=sum(sample.state == "failed" for sample in samples),
        skipped=sum(sample.state == "skipped" for sample in samples),
        samples=samples,
        tasks=tasks,
        latency=_latency(samples),
        routing_summary=_routing_summary(samples),
        failure_summary=_failure_summary(samples),
        counting_target_summary=_counting_targets(samples),
        process_report=_process_report(samples),
        visual_total=sum(len(sample.visuals) for sample in samples),
        visual_materialized_count=sum(
            visual.status == "available" for sample in samples for visual in sample.visuals
        ),
    )


def _build_sample(run_dir: Path, row: dict[str, Any]) -> ReportSample:
    sample_dir = sample_dir_for_row(run_dir, row)
    status = load_status(sample_dir) if sample_dir is not None else None
    sample = load_sample(sample_dir) if sample_dir is not None else None
    trace = load_trace(sample_dir) if sample_dir is not None else None
    task = str(row.get("task", ""))
    evaluation = load_evaluation(sample_dir, task) if sample_dir is not None else None
    payload = load_payload(sample_dir, task) if sample_dir is not None else None
    counting_audit = load_counting_attempts(sample_dir) if sample_dir is not None else None
    if counting_audit is not None and counting_audit.sample_id != str(row.get("sample_id", "")):
        counting_audit = None
    routing_decision = load_routing_decision(sample_dir) if sample_dir is not None else None
    model_calls = load_model_calls(sample_dir) if sample_dir is not None else []
    structured_artifacts = (
        load_structured_artifacts(sample_dir) if sample_dir is not None else []
    )
    evaluation = _safe_evaluation(evaluation)
    routing = _routing(trace)
    judge_status = _judge_status(evaluation, trace)
    prediction = prediction_text(payload)
    references = (
        list(sample.ground_truth.answers)
        if sample is not None and sample.ground_truth is not None
        else []
    )
    ground_truth = _ground_truth_view(sample)
    warnings = _warning_codes(payload, trace)
    task_detail = _task_detail(task, sample, payload, evaluation, trace, judge_status)
    backend_stages = _backend_stages(counting_audit)
    task_routing = _task_routing(
        run_task=str(row.get("run_task", "")),
        task=task,
        trace=trace,
        routing_decision=routing_decision,
    )
    execution_steps = _execution_steps(
        run_dir,
        task=task,
        state=str(row.get("status", "")),
        trace=trace,
        task_routing=task_routing,
        backend_stages=backend_stages,
        model_calls=model_calls,
        structured_artifacts=structured_artifacts,
        evaluation=evaluation,
    )
    return ReportSample(
        sample_id=str(row.get("sample_id", "")),
        run_task=str(row.get("run_task", "")),
        task=task,
        state=str(row.get("status", "")),
        error_code=status.error_code if status is not None else None,
        result_path=safe_result_path(run_dir, row.get("result_path")),
        updated_at=row.get("updated_at") if isinstance(row.get("updated_at"), str) else None,
        question=sample.question if sample is not None else None,
        reference_answers=references,
        ground_truth=ground_truth,
        prediction=prediction,
        resolved_task=routing.resolved_task,
        execution_agent=routing.execution_agent,
        fallback_used=routing.fallback_used,
        judge_status=judge_status,
        inference_seconds=_trace_float(trace, "inference_seconds"),
        evaluation=evaluation,
        result_quality=_result_quality(evaluation, task_detail),
        routing=routing,
        task_routing=task_routing,
        execution_steps=execution_steps,
        execution_path=_execution_path(
            run_dir,
            run_task=str(row.get("run_task", "")),
            task=task,
            trace=trace,
            model_calls=model_calls,
            structured_artifacts=structured_artifacts,
            evaluation=evaluation,
        ),
        routing_decision=routing_decision,
        backend_stages=backend_stages,
        model_calls=model_calls,
        structured_artifacts=structured_artifacts,
        warnings=warnings,
        visuals=[
            VisualAssetView(
                image_id=image.image_id,
                role=image.role,
                width=image.width,
                height=image.height,
            )
            for image in (sample.images if sample is not None else [])
        ],
        task_detail=task_detail,
    )


def _task_routing(
    *,
    run_task: str,
    task: str,
    trace: dict[str, Any] | None,
    routing_decision: dict[str, Any] | None,
) -> TaskRoutingView:
    current = trace or {}
    decision = routing_decision or {}
    resolved_task = _value_str(current.get("resolved_task")) or _value_str(
        decision.get("resolved_task")
    ) or _value_str(decision.get("task")) or task
    executed_task = _value_str(current.get("execution_task")) or resolved_task
    executed_agent = _value_str(current.get("execution_agent"))
    primary_agent = _value_str(decision.get("primary_agent"))
    fallback_agents = _string_list(current.get("fallback_agents")) or _string_list(
        decision.get("fallback_agents")
    )
    candidate_tasks = _string_list(current.get("candidate_tasks"))
    attempt_agents = current.get("attempt_agents")
    skipped = current.get("skipped_candidates")
    skipped_by_task: dict[str, str | None] = {}
    skipped_text: list[str] = []
    if isinstance(skipped, list):
        for item in skipped:
            if isinstance(item, dict):
                skipped_task = _value_str(item.get("task"))
                reason = _value_str(item.get("reason_code")) or _value_str(item.get("reason"))
                if skipped_task is not None:
                    skipped_by_task[skipped_task] = reason
                    skipped_text.append(
                        f"{skipped_task}: {reason}" if reason is not None else skipped_task
                    )
    candidates: list[TaskCandidateView] = []
    for index, candidate_task in enumerate(candidate_tasks, start=1):
        agents: list[str] = []
        if isinstance(attempt_agents, list) and index <= len(attempt_agents):
            agents = _string_list(attempt_agents[index - 1])
        reason = skipped_by_task.get(candidate_task)
        executed = candidate_task == executed_task
        selected = candidate_task == resolved_task
        status = "executed" if executed else "selected" if selected else "skipped" if candidate_task in skipped_by_task else "considered"
        candidates.append(TaskCandidateView(
            order=index,
            task=candidate_task,
            agent_names=agents,
            status=status,
            reason_code=reason,
            selected=selected,
            executed=executed,
        ))
    return TaskRoutingView(
        source_task=run_task or task,
        resolved_task=resolved_task,
        executed_task=executed_task,
        planning_mode=_value_str(current.get("planning_mode")),
        resolution_source=_value_str(current.get("resolution_source")),
        candidate_tasks=candidates,
        primary_agent=primary_agent,
        fallback_agents=fallback_agents,
        executed_agent=executed_agent,
        execution_mode=_value_str(current.get("execution_mode")) or _value_str(
            decision.get("execution_mode")
        ),
        primary_reason=_value_str(current.get("primary_reason")),
        fallback_from_task=_value_str(current.get("fallback_from_task")),
        skipped_candidates=skipped_text,
        reason_codes=_string_list(decision.get("reason_codes")),
    )


def _execution_steps(
    run_dir: Path,
    *,
    task: str,
    state: str,
    trace: dict[str, Any] | None,
    task_routing: TaskRoutingView,
    backend_stages: list[BackendStageView],
    model_calls: list[Any],
    structured_artifacts: list[Any],
    evaluation: EvaluationRecord | None,
) -> list[ExecutionStepView]:
    current = trace or {}
    rows: list[dict[str, Any]] = []

    def add(phase: str, component: str, operation: str, **values: Any) -> None:
        rows.append({
            "phase": phase,
            "component": component,
            "operation": operation,
            **values,
        })

    manifest = load_run_manifest(run_dir)
    adapter = (
        "data.adapters.vrsbench.adapter.VRSBenchAdapter"
        if manifest is not None and manifest.dataset == "VRSBench"
        else "data.adapters.base.DatasetAdapter"
    )
    add("input", adapter, "iter_samples", status="recorded", task=task)

    planning_mode = task_routing.planning_mode
    if planning_mode in {"visual-task-plan-v2", "visual-task-plan-v3", "visual-task-plan-v4"}:
        add("planning", "workflows.visual_planner.VisualTaskPlanner", "plan", status="recorded", task=task_routing.resolved_task)
    elif current.get("joint_plan") is True:
        add("planning", "workflows.visual_planner.JointVisualPlanner", "plan", status="recorded", task=task_routing.resolved_task)
    elif task_routing.resolution_source == "model":
        add("planning", "workflows.task_resolver.TaskResolver", "resolve", status="recorded", task=task_routing.resolved_task)

    add(
        "routing",
        "routing.router.TaskRouter",
        "route",
        status="selected" if task_routing.resolved_task else "not_recorded",
        task=task_routing.resolved_task,
        agent_name=task_routing.primary_agent,
        reason_code=task_routing.primary_reason or (
            ", ".join(task_routing.reason_codes) or None
        ),
    )
    agent_component = _value_str(current.get("agent_class")) or "agents.registry.AgentRegistry"
    add(
        "agent",
        agent_component,
        "run",
        status=state or "not_recorded",
        task=task_routing.executed_task,
        agent_name=task_routing.executed_agent,
        reason_code=_value_str(current.get("failure_code")),
    )

    artifact_by_phase = {
        "visual_task_plan.json": "planning",
        "visual_plan.json": "planning",
        "joint_visual_plan.json": "planning",
        "vqa_evidence.json": "evidence",
        "grounding_evidence.json": "evidence",
    }
    for artifact in structured_artifacts:
        filename = _value_str(getattr(artifact, "filename", None))
        if filename is None:
            continue
        add(
            artifact_by_phase.get(filename, "artifact"),
            "reporting.adapters.StructuredArtifactView",
            "load",
            status="recorded",
            task=task_routing.executed_task,
            artifact_names=[filename],
        )
        for audit in _artifact_call_audits(artifact):
            layer = _value_str(audit.get("layer"))
            if layer is None:
                continue
            backend_kind = {
                "yolo": "yolo",
                "segformer": "segformer",
                "final_qwen": "qwen",
            }.get(layer, layer)
            summary = {
                "backend_kind": backend_kind,
                "logical_model_id": _value_str(audit.get("logical_model_id")),
                "weights_sha256": _summary_digest(audit, "weights_sha256"),
            }
            input_size = audit.get("input_size")
            if isinstance(input_size, list) and all(
                isinstance(value, int) and not isinstance(value, bool) for value in input_size
            ):
                summary["input_size"] = input_size
            add(
                "evidence_model",
                {
                    "yolo": "models.base.ObjectDetectionClient",
                    "segformer": "models.base.SemanticSegmentationClient",
                    "final_qwen": "models.base.StructuredModelClient",
                }.get(layer, "models.base.ModelClient"),
                "infer",
                status=_value_str(audit.get("status")) or "recorded",
                task=task_routing.executed_task,
                agent_name=task_routing.executed_agent,
                backend_name=layer,
                reason_code=_value_str(audit.get("error_code")),
                summary_fields={key: value for key, value in summary.items() if value is not None},
                artifact_names=[filename],
            )

    for stage in backend_stages:
        summary: dict[str, Any] = {
            "phase": stage.phase,
            "backend_kind": stage.backend_kind,
        }
        for key in (
            "model_id", "logical_model_id", "weights_file", "weights_sha256",
            "source_dataset", "model_revision", "runtime",
        ):
            if key in stage.summary_fields:
                summary[key] = stage.summary_fields[key]
        if stage.predicted_count is not None:
            summary["predicted_count"] = stage.predicted_count
        if stage.accepted_count is not None:
            summary["accepted_count"] = stage.accepted_count
        if stage.rejected_count is not None:
            summary["rejected_count"] = stage.rejected_count
        add(
            "backend",
            "agents.counting.CountingAgent",
            stage.phase,
            status=stage.status,
            task=task_routing.executed_task,
            agent_name=task_routing.executed_agent,
            backend_name=stage.backend_name,
            reason_code=stage.reason_code or stage.error_type,
            summary_fields=summary,
        )

    for call in model_calls:
        summary = {
            "prompt_version": call.prompt_version,
            "valid": call.valid,
            "cache_hit": call.cache_hit,
            "repair_used": call.repair_used,
        }
        if call.latency_seconds is not None:
            summary["latency_seconds"] = call.latency_seconds
        add(
            "model_call",
            "models.structured_client",
            "complete_json",
            status="succeeded" if call.valid is True else "failed" if call.valid is False else "recorded",
            task=task_routing.executed_task,
            agent_name=task_routing.executed_agent,
            request_id=call.request_id,
            summary_fields=summary,
        )

    if evaluation is not None:
        add("evaluation", "evaluation.records.EvaluationRecord", "evaluate", status="recorded", task=task_routing.executed_task)
    add("reporting", "reporting.builder", "build_report", status="recorded", task=task_routing.executed_task)
    return [ExecutionStepView(order=index, **row) for index, row in enumerate(rows, start=1)]


def _execution_path(
    run_dir: Path,
    *,
    run_task: str,
    task: str,
    trace: dict[str, Any] | None,
    model_calls: list[Any],
    structured_artifacts: list[Any],
    evaluation: EvaluationRecord | None,
) -> list[str]:
    """Project persisted runtime facts into a concise top-level module path.
    将已持久化运行事实投影为简洁的顶层模块路径。

    This is a read-only presentation projection: it never infers a different
    task or claims that an unrecorded backend ran. The HTML path makes the
    sample-level hand-off auditable without exposing prompt bodies or paths.
    这是只读展示投影：绝不推断其他 task，也不声称未记录的后端曾运行。
    HTML 中的路径用于审计样本级交接，同时不暴露 prompt 正文或物理路径。
    """

    current = trace or {}
    path = [
        "application.commands.run_dataset",
        "data.registry.DatasetRegistry",
    ]
    metadata = load_run_manifest(run_dir)
    if metadata is not None and metadata.dataset == "VRSBench":
        path.append("data.adapters.vrsbench.adapter.VRSBenchAdapter.iter_samples")
    else:
        path.append("data.adapters.base.DatasetAdapter.iter_samples")
    path.extend([
        "workflows.dataset_runner.DatasetRunner",
        "workflows.sample_runner.SampleRunner",
    ])
    resolution_source = _value_str(current.get("resolution_source"))
    if current.get("planning_mode") in {
        "visual-task-plan-v2",
        "visual-task-plan-v3",
        "visual-task-plan-v4",
        "visual-task-plan-v5",
    }:
        path.append("workflows.visual_planner.VisualTaskPlanner")
    elif current.get("joint_plan") is True:
        path.append("workflows.visual_planner.JointVisualPlanner")
    elif resolution_source == "model":
        path.append("workflows.task_resolver.TaskResolver")
    path.append("routing.router.TaskRouter.route")

    agent_class = _value_str(current.get("agent_class"))
    if agent_class is not None:
        path.append(agent_class)
    else:
        agent_by_task = {
            "caption": "agents.caption.agent.CaptionAgent",
            "grounding": "agents.grounding.agent.GroundingAgent",
            "general_vqa": "agents.general_vqa.agent.GeneralVQAAgent",
        }
        path.append(agent_by_task.get(task, f"agents.registry.AgentRegistry[{task}]"))

    filenames = {item.filename for item in structured_artifacts}
    if "vqa_evidence.json" in filenames:
        path.append("agents.general_vqa.evidence.executor.ObjectEvidenceExecutor")
    if "grounding_evidence.json" in filenames:
        path.append("agents.grounding.evidence.GroundingEvidenceExecutor")
    if model_calls:
        path.append("models.qwen_transformers.QwenTransformersClient.complete_json")
    if evaluation is not None:
        path.append("evaluation.records.EvaluationRecord")
    path.append("reporting.builder.build_report")
    return list(dict.fromkeys(path))


def _safe_evaluation(record: EvaluationRecord | None) -> EvaluationRecord | None:
    if record is None:
        return None
    data = record.model_dump(mode="python")
    data["judge_raw"] = None
    data["judge_error"] = None
    data["judge_parsed"] = _safe_view_value(data.get("judge_parsed"))
    return EvaluationRecord.model_validate(data)


def _ground_truth_view(sample: Any) -> GroundTruthView | None:
    ground_truth = getattr(sample, "ground_truth", None)
    if ground_truth is None:
        return None
    return GroundTruthView(
        answers=list(ground_truth.answers),
        count=ground_truth.count,
        boxes=[list(box) for box in ground_truth.boxes],
        points=[list(point) for point in ground_truth.points],
        labels=list(ground_truth.labels),
        coordinate_frame=ground_truth.coordinate_frame,
    )


def _backend_stages(audit: CountingExecutionAudit | None) -> list[BackendStageView]:
    if audit is None:
        return []
    stages: list[BackendStageView] = []
    for order, attempt in enumerate(audit.attempts, start=1):
        counting = attempt.counting
        points = list(counting.global_points) if counting is not None else []
        summary = _stage_summary(attempt.backend_trace, attempt.agent_result)
        stages.append(BackendStageView(
            order=order,
            backend_name=attempt.backend_name,
            backend_kind=attempt.backend_kind,
            phase=attempt.phase,
            status=attempt.status,
            reason_code=attempt.reason_code,
            error_type=attempt.error_type,
            predicted_count=counting.final_count if counting is not None else None,
            counting_status=counting.status if counting is not None else None,
            accepted_count=(sum(point.accepted for point in points) if counting is not None else None),
            rejected_count=(sum(not point.accepted for point in points) if counting is not None else None),
            warning_codes=(sorted({warning.code for warning in counting.warnings}) if counting is not None else []),
            summary_fields=summary,
            overlay_asset=_safe_overlay_asset(attempt.backend_trace.get("overlay_asset")),
        ))
    return stages


def _stage_summary(
    backend_trace: dict[str, Any],
    agent_result: dict[str, Any] | None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    sources: list[dict[str, Any]] = [backend_trace]
    if isinstance(agent_result, dict):
        sources.append(agent_result)
        geometry = agent_result.get("geometry")
        if isinstance(geometry, dict):
            sources.append(geometry)
    for source in sources:
        for key, value in source.items():
            normalized_key = str(key)
            if normalized_key.casefold() in _UNSAFE_VIEW_KEYS or normalized_key == "overlay_asset":
                continue
            safe = _summary_value(value)
            if safe is not None or value is None:
                summary.setdefault(normalized_key, safe)
    return summary


def _summary_value(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value if math.isfinite(float(value)) else None
    if isinstance(value, str):
        return _value_str(value)
    if isinstance(value, list) and len(value) <= 50:
        if any(isinstance(item, (list, dict)) for item in value):
            return None
        items = [_summary_value(item) for item in value]
        if all(item is not None or original is None for item, original in zip(items, value)):
            return items
    return None


def _safe_overlay_asset(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("\\", "/")
    if normalized.startswith("/") or ".." in normalized.split("/"):
        return None
    if len(normalized) >= 2 and normalized[0].isalpha() and normalized[1] == ":":
        return None
    return _value_str(normalized)


def _routing(trace: dict[str, Any] | None) -> RoutingView:
    if trace is None:
        return RoutingView()
    candidates = _backend_names(trace.get("candidate_backends"))
    attempted_names = _string_list(trace.get("attempted_backends"))
    primary = _value_str(trace.get("primary_backend"))
    final = _value_str(trace.get("final_backend"))
    primary_kind = _value_str(trace.get("primary_backend_kind"))
    final_kind = _value_str(trace.get("final_backend_kind"))
    raw_history = trace.get("fallback_history")
    failures: dict[str, tuple[str | None, str | None]] = {}
    history_rows: list[dict[str, str | None]] = []
    if isinstance(raw_history, list):
        for entry in raw_history:
            if not isinstance(entry, dict):
                continue
            backend = _value_str(entry.get("backend")) or _value_str(entry.get("from_backend"))
            if backend is None:
                continue
            reason = _value_str(entry.get("reason_code"))
            kind = _value_str(entry.get("kind"))
            failures[backend] = (reason, kind)
            history_rows.append({
                "backend": backend,
                "reason": reason,
                "to": _value_str(entry.get("to_backend")),
            })
    attempts: list[RoutingAttemptView] = []
    final_status = _stable_final_status(trace)
    review_backend = _value_str(trace.get("review_backend"))
    for name in attempted_names:
        if name in failures:
            reason, kind = failures[name]
            attempt_status = "unavailable" if reason == "BACKEND_UNAVAILABLE" else "failed"
        elif name == review_backend and name != final:
            reason, kind, attempt_status = None, None, "zero_review"
        elif name == final:
            reason, kind, attempt_status = None, final_kind, final_status
        else:
            reason, kind, attempt_status = None, primary_kind if name == primary else None, "selected"
        attempts.append(RoutingAttemptView(
            backend_name=name, backend_kind=kind, status=attempt_status, reason_code=reason
        ))
    transitions: list[FallbackTransitionView] = []
    for row in history_rows:
        failed_name = row["backend"]
        next_name = row.get("to")
        if next_name is None:
            try:
                index = attempted_names.index(str(failed_name))
            except ValueError:
                next_name = None
            else:
                next_name = attempted_names[index + 1] if index + 1 < len(attempted_names) else None
        transitions.append(FallbackTransitionView(
            from_backend=failed_name, to_backend=next_name, reason_code=row["reason"]
        ))
    reason_value = trace.get("selection_reason")
    selection_reason = (
        " · ".join(_string_list(reason_value)) or None
        if isinstance(reason_value, list)
        else _value_str(reason_value)
    )
    return RoutingView(
        resolved_task=_value_str(trace.get("resolved_task")),
        execution_agent=_value_str(trace.get("execution_agent")),
        candidate_backends=candidates,
        attempted_backends=attempts,
        primary_backend=primary,
        primary_backend_kind=primary_kind,
        final_backend=final,
        final_backend_kind=final_kind,
        fallback_used=bool(trace.get("fallback_used") or trace.get("fallback_triggered") or transitions),
        fallback_history=transitions,
        review_backend=review_backend,
        selection_reason=selection_reason,
    )


def _stable_final_status(trace: dict[str, Any]) -> str:
    status = _value_str(trace.get("status"))
    if status in {"partial", "failed"}:
        return status
    return "succeeded"


def _task_detail(
    task: str, sample: Any, payload: object | None, evaluation: EvaluationRecord | None,
    trace: dict[str, Any] | None, judge_status: str,
) -> Any:
    metrics = evaluation.deterministic_metrics if evaluation is not None else None
    if isinstance(payload, CountingResult):
        count_metrics = metrics if isinstance(metrics, CountDeterministicMetrics) else None
        accepted = sorted((point for point in payload.global_points if point.accepted), key=_point_key)
        rejected = sorted((point for point in payload.global_points if not point.accepted), key=_point_key)
        usage = Counter(
            point.provenance.source for point in payload.global_points if point.provenance is not None
        )
        backend_trace = trace.get("backend_trace") if isinstance(trace, dict) else None
        mode = _value_str(backend_trace.get("counting_mode")) if isinstance(backend_trace, dict) else None
        return CountingReportDetail(
            target=payload.target,
            predicted_count=payload.final_count,
            gold_count=count_metrics.gold_count if count_metrics else None,
            absolute_error=count_metrics.absolute_error if count_metrics else None,
            exact_match=bool(count_metrics.exact_match) if count_metrics else None,
            counting_status=payload.status,
            tile_count=payload.tile_count,
            initial_tile_count=payload.initial_tile_count,
            leaf_tile_count=payload.leaf_tile_count,
            succeeded_tile_count=len(payload.succeeded_tiles),
            failed_tile_count=len(payload.failed_tiles),
            accepted_point_count=len(accepted),
            rejected_point_count=len(rejected),
            merged_group_count=len(payload.merged_groups),
            unresolved_conflict_count=len(payload.unresolved_conflicts),
            warning_codes=sorted({warning.code for warning in payload.warnings}),
            counting_mode=mode,
            provenance_usage=dict(sorted(usage.items())),
            accepted_preview=[_point_preview(point) for point in accepted[:50]],
            rejected_preview=[_point_preview(point) for point in rejected[:100]],
        )
    if not isinstance(payload, AgentResult):
        return None
    references = (
        list(sample.ground_truth.answers)
        if sample is not None and sample.ground_truth is not None
        else []
    )
    severity = _value_str(payload.geometry.get("repair_severity"))
    if task == "grounding":
        grounding = metrics if isinstance(metrics, GroundingDeterministicMetrics) else None
        gt_boxes = (
            [list(box) for box in sample.ground_truth.boxes]
            if sample is not None and sample.ground_truth is not None
            and len(sample.images) == 1
            else []
        )
        return GroundingReportDetail(
            prediction=payload.answer,
            reference=references,
            predicted_boxes=[list(box) for box in payload.boxes],
            ground_truth_boxes=gt_boxes,
            ground_truth_coordinate_frame=(
                sample.ground_truth.coordinate_frame
                if sample is not None and sample.ground_truth is not None
                else None
            ),
            iou=grounding.iou if grounding else None,
            iou_at_0_5=grounding.iou_at_0_5 if grounding else None,
            geometry_repair_severity=severity,
        )
    if task == "spatial_relation":
        return SpatialReportDetail(
            question=sample.question if sample is not None else None,
            prediction=payload.answer,
            reference_answers=references,
            evidence_text=list(payload.evidence),
            evidence_item_count=len(payload.evidence_items),
            geometry_repair_severity=severity,
        )
    if task in {"change_qa", "change_caption"}:
        return ChangeReportDetail(
            question=sample.question if sample is not None else None,
            prediction=payload.answer,
            reference_answers=references,
            evidence_text=list(payload.evidence),
            evidence_item_count=len(payload.evidence_items),
            geometry_summary=_geometry_summary(payload.geometry),
            geometry_repair_severity=severity,
        )
    if task == "caption":
        caption = metrics if isinstance(metrics, CaptionDeterministicMetrics) else None
        return CaptionReportDetail(
            generated_caption=payload.answer,
            reference_captions=list(caption.references) if caption else references,
            metric_status="per_run_only",
        )
    vqa = metrics if isinstance(metrics, VQADeterministicMetrics) else None
    judge = evaluation.judge_parsed if evaluation is not None else None
    judge_data = judge.model_dump() if hasattr(judge, "model_dump") else judge
    return GeneralVQAReportDetail(
        question=sample.question if sample is not None else None,
        reference_answers=references,
        prediction=payload.answer,
        exact_match=vqa.exact_match if vqa else None,
        judge_status=judge_status,
        judge_score=_number(judge_data.get("score")) if isinstance(judge_data, dict) else None,
        judge_concise_rationale=_value_str(judge_data.get("concise_rationale")) if isinstance(judge_data, dict) else None,
        visual_evidence_count=len(payload.evidence_items),
        geometry_repair_severity=severity,
    )


def _point_key(point: GlobalPointObservation) -> tuple[Any, ...]:
    return (point.global_y_px, point.global_x_px, point.global_id)


def _point_preview(point: GlobalPointObservation) -> PointPreview:
    provenance = point.provenance
    return PointPreview(
        point_id=point.global_id,
        x=point.global_x_px,
        y=point.global_y_px,
        confidence=point.confidence,
        accepted=point.accepted,
        rejection_reason=point.rejection_reason,
        source=provenance.source if provenance else None,
        backend_name=provenance.backend_name if provenance else None,
        model_id=provenance.model_id if provenance else None,
        source_class=provenance.source_class if provenance else None,
        source_dataset=provenance.detector_source_dataset if provenance else None,
        weights_sha256=provenance.weights_sha256 if provenance else None,
    )


def _geometry_summary(geometry: dict[str, Any]) -> str | None:
    keys = sorted(key for key in geometry if key not in {"raw", "path", "source_path"})
    return ", ".join(keys) if keys else None


def _warning_codes(payload: object | None, trace: dict[str, Any] | None) -> list[str]:
    codes: set[str] = set()
    if isinstance(payload, CountingResult):
        codes.update(warning.code for warning in payload.warnings)
    if trace is not None:
        for key in ("failure_code",):
            value = _value_str(trace.get(key))
            if value:
                codes.add(value)
    return sorted(codes)


def _result_quality(evaluation: EvaluationRecord | None, task_detail: object | None = None) -> str:
    metrics = evaluation.deterministic_metrics if evaluation is not None else None
    exact = getattr(task_detail, "exact_match", None)
    if exact is not None:
        return "correct" if exact else "incorrect"
    iou_match = getattr(task_detail, "iou_at_0_5", None)
    if iou_match is not None:
        return "correct" if iou_match else "incorrect"
    if metrics is None:
        return "unknown"
    if isinstance(metrics, CountDeterministicMetrics):
        return "correct" if metrics.exact_match else "incorrect"
    if isinstance(metrics, VQADeterministicMetrics):
        return "correct" if metrics.exact_match else "incorrect"
    if isinstance(metrics, GroundingDeterministicMetrics):
        return "correct" if metrics.iou_at_0_5 else "incorrect"
    return "not_applicable"


def _judge_status(evaluation: EvaluationRecord | None, trace: dict[str, Any] | None) -> str:
    if evaluation is not None and evaluation.judge_status:
        return evaluation.judge_status
    return (_value_str(trace.get("judge_status")) if trace else None) or "not_requested"


def _build_task_summary(run_task: str, samples: list[ReportSample]) -> TaskSummary:
    total = len(samples)
    fallback_count = sum(sample.fallback_used for sample in samples)
    return TaskSummary(
        run_task=run_task,
        total=total,
        succeeded=sum(sample.state == "succeeded" for sample in samples),
        partial=sum(sample.state == "partial" for sample in samples),
        failed=sum(sample.state == "failed" for sample in samples),
        skipped=sum(sample.state == "skipped" for sample in samples),
        fallback_count=fallback_count,
        fallback_rate=fallback_count / total if total else 0.0,
        agent_usage=_usage(sample.execution_agent for sample in samples),
        judge_status_counts=_usage(sample.judge_status for sample in samples),
        metrics=_aggregate_metrics(samples),
        judge_metrics=_aggregate_judge_metrics(samples),
        correct=sum(sample.result_quality == "correct" for sample in samples),
        incorrect=sum(sample.result_quality == "incorrect" for sample in samples),
        unknown_quality=sum(sample.result_quality in {"unknown", "not_applicable"} for sample in samples),
        latency=_latency(samples),
        primary_backend_usage=_usage(sample.routing.primary_backend for sample in samples),
        final_backend_usage=_usage(sample.routing.final_backend for sample in samples),
        warning_count=sum(len(sample.warnings) for sample in samples),
    )


def _aggregate_metrics(samples: list[ReportSample]) -> dict[str, Any]:
    records = [
        sample.evaluation for sample in samples
        if sample.evaluation is not None and sample.evaluation.deterministic_metrics is not None
    ]
    buckets: dict[str, list[EvaluationRecord]] = {}
    for record in records:
        buckets.setdefault(record.task, []).append(record)
    result: dict[str, Any] = {}
    for family, bucket in buckets.items():
        if family == "caption":
            try:
                result[family] = {"metric_status": "ok", **aggregate_caption(bucket)}
            except CaptionMetricDependencyError:
                result[family] = {
                    "metric_status": "dependency_missing", "record_count": len(bucket),
                    "dependency": "pycocoevalcap",
                }
        elif family in _AGGREGATORS:
            result[family] = _AGGREGATORS[family](bucket)
    return result


def _aggregate_judge_metrics(samples: list[ReportSample]) -> dict[str, Any]:
    records = [
        sample.evaluation for sample in samples
        if sample.evaluation is not None and sample.evaluation.task == "general_vqa"
        and isinstance(sample.evaluation.deterministic_metrics, VQADeterministicMetrics)
    ]
    return {"vqa_semantic_equivalence": aggregate_vqa_semantic_judge(records)} if records else {}


def _latency(samples: list[ReportSample]) -> LatencySummary:
    values = sorted(
        float(sample.inference_seconds) for sample in samples
        if sample.inference_seconds is not None and math.isfinite(float(sample.inference_seconds))
    )
    if not values:
        return LatencySummary()
    return LatencySummary(
        count=len(values), mean_seconds=sum(values) / len(values),
        p50_seconds=_percentile(values, 0.50), p95_seconds=_percentile(values, 0.95),
    )


def _percentile(values: list[float], quantile: float) -> float:
    """Linear interpolation on zero-based rank (stdlib-only, deterministic)."""
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * quantile
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (rank - lower)


def _routing_summary(samples: list[ReportSample]) -> RoutingSummary:
    transitions = Counter(
        (item.from_backend, item.to_backend, item.reason_code)
        for sample in samples for item in sample.routing.fallback_history
    )
    rows = [
        FallbackTransitionSummary(
            from_backend=key[0], to_backend=key[1], reason_code=key[2], count=count
        )
        for key, count in transitions.items()
    ]
    rows.sort(key=lambda row: (-row.count, row.from_backend or "", row.to_backend or "", row.reason_code or ""))
    fallback_count = sum(sample.fallback_used for sample in samples)
    reasons = Counter(
        item.reason_code for sample in samples for item in sample.routing.fallback_history
        if item.reason_code
    )
    return RoutingSummary(
        primary_backend_usage=_usage(sample.routing.primary_backend for sample in samples),
        final_backend_usage=_usage(sample.routing.final_backend for sample in samples),
        fallback_count=fallback_count,
        fallback_rate=fallback_count / len(samples) if samples else 0.0,
        fallback_transitions=rows,
        fallback_reason_counts=dict(sorted(reasons.items())),
    )


def _failure_summary(samples: list[ReportSample]) -> FailureSummary:
    return FailureSummary(
        sample_error_codes=_usage(sample.error_code for sample in samples),
        warning_codes=dict(sorted(Counter(code for sample in samples for code in sample.warnings).items())),
        backend_failure_counts=dict(sorted(Counter(
            item.reason_code for sample in samples for item in sample.routing.fallback_history
            if item.reason_code
        ).items())),
    )


def _counting_targets(samples: list[ReportSample]) -> list[CountingTargetSummary]:
    grouped: dict[str, list[tuple[CountingReportDetail, bool]]] = {}
    for sample in samples:
        if isinstance(sample.task_detail, CountingReportDetail) and sample.task_detail.target:
            grouped.setdefault(sample.task_detail.target, []).append(
                (sample.task_detail, sample.fallback_used)
            )
    rows: list[CountingTargetSummary] = []
    for target in sorted(grouped):
        pairs = grouped[target]
        details = [detail for detail, _ in pairs]
        evaluated = [detail for detail in details if detail.gold_count is not None]
        exact = sum(detail.exact_match is True for detail in evaluated)
        errors = [detail.absolute_error for detail in evaluated if detail.absolute_error is not None]
        fallback_count = sum(fallback for _, fallback in pairs)
        rows.append(CountingTargetSummary(
            target=target, sample_count=len(details), evaluated_count=len(evaluated), exact_count=exact,
            accuracy=exact / len(evaluated) if evaluated else None,
            mae=sum(errors) / len(errors) if errors else None,
            fallback_count=fallback_count,
            fallback_rate=fallback_count / len(details) if details else 0.0,
        ))
    return rows


def _process_report(samples: list[ReportSample]) -> ProcessReport:
    """Aggregate concrete, persisted execution order and weight identities.

    Only observed backend attempts are included. Physical checkpoint paths are
    never consulted; identities come from the backend's sanitized audit trace.
    """

    sequence_counts: Counter[tuple[str, tuple[tuple[str, str, str, str, int], ...]]] = Counter()
    for sample in samples:
        compressed: list[tuple[str, str, str, str, int]] = []
        for step in sample.execution_steps:
            item = (
                step.phase,
                step.component,
                step.operation or "",
                step.backend_name or "",
            )
            if compressed and compressed[-1][:4] == item:
                previous = compressed[-1]
                compressed[-1] = (*previous[:4], previous[4] + 1)
            else:
                compressed.append((*item, 1))
        if compressed:
            sequence_counts[(sample.task, tuple(compressed))] += 1

    workflow_sequences: list[WorkflowSequenceView] = []
    for (task, steps), count in sorted(
        sequence_counts.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    ):
        workflow_sequences.append(WorkflowSequenceView(
            task=task,
            sample_count=count,
            steps=[
                WorkflowStepView(
                    order=order,
                    phase=phase,
                    component=component,
                    operation=operation or None,
                    backend_name=backend_name or None,
                    repeat_count=repeat_count,
                )
                for order, (phase, component, operation, backend_name, repeat_count)
                in enumerate(steps, start=1)
            ],
        ))

    weight_rows: dict[tuple[str, ...], dict[str, Any]] = {}
    for sample in samples:
        for step in sample.execution_steps:
            backend_kind = _summary_str(step.summary_fields, "backend_kind")
            family = _model_weight_family(backend_kind or step.backend_name or "")
            if family is None:
                continue
            logical_model_id = (
                _summary_str(step.summary_fields, "logical_model_id")
                or _summary_str(step.summary_fields, "model_id")
            )
            weights_file = _summary_basename(step.summary_fields, "weights_file")
            weights_sha256 = _summary_digest(step.summary_fields, "weights_sha256")
            source_dataset = _summary_str(step.summary_fields, "source_dataset")
            model_revision = _summary_str(step.summary_fields, "model_revision")
            backend_name = step.backend_name or backend_kind or family
            resolved_kind = backend_kind or family
            key = (
                family, backend_name, resolved_kind,
                logical_model_id or "", weights_file or "", weights_sha256 or "",
                source_dataset or "", model_revision or "",
            )
            row = weight_rows.setdefault(key, {
                "family": family,
                "backend_name": backend_name,
                "backend_kind": resolved_kind,
                "logical_model_id": logical_model_id,
                "weights_file": weights_file,
                "weights_sha256": weights_sha256,
                "source_dataset": source_dataset,
                "model_revision": model_revision,
                "use_count": 0,
                "phases": set(),
                "statuses": set(),
            })
            row["use_count"] += 1
            row["phases"].add(step.phase)
            row["statuses"].add(step.status)

    model_weights = [
        ModelWeightView(
            **{key: value for key, value in row.items() if key not in {"phases", "statuses"}},
            phases=sorted(row["phases"]),
            statuses=sorted(row["statuses"]),
        )
        for _, row in sorted(weight_rows.items())
    ]
    return ProcessReport(
        sample_process_count=sum(bool(sample.execution_steps) for sample in samples),
        workflow_sequences=workflow_sequences,
        model_weights=model_weights,
    )


def _model_weight_family(backend_kind: str) -> str | None:
    normalized = backend_kind.casefold()
    if normalized.startswith("yolo"):
        return "yolo"
    if normalized in {"semantic_segmentation", "segmentation", "segformer"}:
        return "segmentation"
    return None


def _summary_str(summary: dict[str, Any], key: str) -> str | None:
    return _value_str(summary.get(key))


def _summary_basename(summary: dict[str, Any], key: str) -> str | None:
    value = _summary_str(summary, key)
    if value is None or "/" in value or "\\" in value:
        return None
    return value


def _summary_digest(summary: dict[str, Any], key: str) -> str | None:
    value = _summary_str(summary, key)
    if value is None:
        return None
    normalized = value.casefold()
    return normalized if re.fullmatch(r"[0-9a-f]{64}", normalized) else None


def _artifact_call_audits(artifact: Any) -> list[dict[str, Any]]:
    payload = getattr(artifact, "payload", None)
    audits = payload.get("call_audit") if isinstance(payload, dict) else None
    return [item for item in audits if isinstance(item, dict)] if isinstance(audits, list) else []


def _run_metadata(run_dir: Path) -> RunMetadata | None:
    manifest = load_run_manifest(run_dir)
    if manifest is None:
        return None
    return manifest.model_copy(deep=True)


def _usage(values: Any) -> dict[str, int]:
    return dict(sorted(Counter(value for value in values if isinstance(value, str) and value).items()))


def _find_dataset(run_dir: Path) -> str | None:
    tasks_dir = run_dir / "tasks"
    if not tasks_dir.is_dir():
        return None
    try:
        task_dirs = sorted(path for path in tasks_dir.iterdir() if path.is_dir())
    except OSError:
        return None
    for task_dir in task_dirs:
        for filename in ("dataset_probe.json", "dataset_summary.json"):
            raw = read_json(task_dir / filename)
            if isinstance(raw, dict) and isinstance(raw.get("dataset"), str) and raw["dataset"]:
                return raw["dataset"]
    return None


def _trace_float(trace: dict[str, Any] | None, key: str) -> float | None:
    value = trace.get(key) if trace is not None else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _string_list(value: Any) -> list[str]:
    return [safe for item in value if (safe := _value_str(item)) is not None] if isinstance(value, list) else []


def _backend_names(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    names: list[str] = []
    for item in value:
        name = _value_str(item) if isinstance(item, str) else _value_str(item.get("name")) if isinstance(item, dict) else None
        if name:
            names.append(name)
    return names


def _value_str(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if _UNSAFE_PUBLIC_RE.search(value):
        return None
    return value


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) else None


_UNSAFE_PUBLIC_RE = re.compile(
    r"(?i)(?:sk-[a-z0-9_-]{6,}|bearer\s+|https?://|[a-z]:[\\/]|(?:^|[^a-z0-9])/(?:tmp|home|users)/)"
)
_UNSAFE_VIEW_KEYS = {
    "dataset_root", "api_key", "authorization", "auth_header", "checkpoint_path",
    "checkpoint", "weights", "weights_path", "artifact_dir", "source_path",
    "cache_path", "cache_dir", "hostname", "username",
}


def _safe_view_value(value: Any) -> Any:
    if isinstance(value, str):
        return None if _UNSAFE_PUBLIC_RE.search(value) else value
    if isinstance(value, list):
        return [_safe_view_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _safe_view_value(item)
            for key, item in value.items()
            if str(key).casefold() not in _UNSAFE_VIEW_KEYS
        }
    return value
