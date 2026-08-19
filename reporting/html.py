"""Pure, deterministic rendering of the offline Report V2 dashboard."""

from __future__ import annotations

import html
import json
import re
from pathlib import PurePosixPath
from typing import Any

from reporting.schema import (
    CaptionReportDetail,
    ChangeReportDetail,
    CountingReportDetail,
    GeneralVQAReportDetail,
    GroundingReportDetail,
    Report,
    ReportSample,
    SpatialReportDetail,
    TaskSummary,
)


def build_html(report: Report) -> str:
    """Render a self-contained document; all images remain relative assets."""

    return "\n".join([
        "<!DOCTYPE html>",
        '<html lang="zh-CN"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>M3 Sample Audit Report · {_esc(report.run_id)}</title>",
        f"<style>{_CSS}</style></head><body><main>",
        _header(report),
        '<nav><a href="#samples">样本审计</a><a href="#overview">总体统计</a>'
        '<a href="#routing">Expert Routing</a><a href="#failures">Failures</a>'
        '<a href="#runtime">Runtime</a></nav>',
        _samples_section(report),
        _overview(report),
        '<details class="aggregate-panel"><summary>总体 Task / Target 统计</summary>'
        + _counting_target_table(report)
        + '<section id="tasks"><h2>Tasks</h2>' + _task_overview_table(report.tasks)
        + "".join(_task_section(task) for task in report.tasks) + "</section></details>",
        '<details class="aggregate-panel"><summary>Expert Routing</summary>'
        + _routing_section(report) + "</details>",
        '<details class="aggregate-panel"><summary>Failures</summary>'
        + _failures_section(report) + "</details>",
        '<details class="aggregate-panel"><summary>Runtime</summary>'
        + _runtime_section(report) + "</details>",
        '<section><h2>Visual legend</h2><div class="legend">'
        '<span class="green">● Prediction / accepted</span><span class="red">● Rejected</span>'
        '<span class="cyan">● GT / ground truth</span><span class="amber">● Unresolved</span>'
        '<span class="purple">● Reviewer</span></div></section>',
        '<div id="image-modal" class="image-modal" hidden role="dialog" aria-modal="true" '
        'aria-label="Image preview"><button type="button" class="image-modal-close" '
        'aria-label="关闭图片预览">×</button><img id="image-modal-img" alt=""></div>',
        f"<script>{_JS}</script>",
        "</main></body></html>",
    ])


def _header(report: Report) -> str:
    meta = report.metadata
    return (
        '<header><div><p class="eyebrow">M3 SAMPLE-CENTRIC AUDIT</p>'
        '<h1>M3 Sample Audit Report</h1><div class="run-meta">'
        + _meta_item("Dataset", report.dataset or "—")
        + _meta_item("Split", meta.split if meta else "—")
        + _meta_item("Run", report.run_id, mono=True)
        + _meta_item("Commit", meta.git_commit if meta else "—", mono=True)
        + '</div></div><div class="schema">REPORT V2.1 · OFFLINE</div></header>'
    )


def _overview(report: Report) -> str:
    latency = report.latency
    accuracy = _overall_accuracy(report)
    cards = [
        ("Samples", report.total), ("Succeeded", report.succeeded),
        ("Failed", report.failed), ("Fallback", report.routing_summary.fallback_count),
        ("Mean latency", _seconds(latency.mean_seconds)),
    ]
    if accuracy is not None:
        cards.append(("Accuracy", f"{accuracy:.1%}"))
    return '<section id="overview"><h2>Overview</h2><div class="cards">' + "".join(
        f'<article class="card"><span>{_esc(label)}</span><strong>{_esc(value)}</strong></article>'
        for label, value in cards
    ) + "</div></section>"


def _task_overview_table(tasks: list[TaskSummary]) -> str:
    rows = "".join(
        f'<tr>{_cells(task.run_task, task.total, task.succeeded, task.partial, task.failed, task.correct, task.incorrect, task.fallback_count, _seconds(task.latency.mean_seconds))}</tr>'
        for task in tasks
    )
    return '<div class="table-scroll"><table><thead><tr><th>Task</th><th>Samples</th><th>Succeeded</th><th>Partial</th><th>Failed</th><th>Correct</th><th>Incorrect</th><th>Fallback</th><th>Mean latency</th></tr></thead><tbody>' + (rows or '<tr><td colspan="9">No task rows</td></tr>') + '</tbody></table></div>'


def _counting_target_table(report: Report) -> str:
    if not report.counting_target_summary:
        return ""
    rows = "".join(
        f'<tr>{_cells(item.target, item.sample_count, item.evaluated_count, item.exact_count, _format_number(item.accuracy), _format_number(item.mae), item.fallback_count, f"{item.fallback_rate:.2%}")}</tr>'
        for item in report.counting_target_summary
    )
    return '<section id="counting-targets"><h2>Counting targets</h2><div class="table-scroll"><table><thead><tr><th>Target</th><th>Samples</th><th>With gold</th><th>Exact</th><th>Accuracy</th><th>MAE</th><th>Fallback</th><th>Fallback rate</th></tr></thead><tbody>' + rows + '</tbody></table></div></section>'


def _task_section(task: TaskSummary) -> str:
    summary = (
        f'<article class="paneltask"><h3>{_esc(task.run_task)}</h3>'
        f'<p>samples={task.total} succeeded={task.succeeded} partial={task.partial} '
        f'failed={task.failed} skipped={task.skipped} fallback={task.fallback_count} ({task.fallback_rate:.2%})</p>'
        f'<p>quality: correct={task.correct} incorrect={task.incorrect} unknown={task.unknown_quality}; '
        f'warnings={task.warning_count}; p95={_esc(_seconds(task.latency.p95_seconds))}</p>'
    )
    return summary + _usage_table("Agent usage", task.agent_usage) + _usage_table(
        "Final backend usage", task.final_backend_usage
    ) + _metrics_html(task.metrics) + _judge_metrics_html(task.judge_metrics) + "</article>"


def _routing_section(report: Report) -> str:
    routing = report.routing_summary
    transitions = "".join(
        "<tr>" + _cells(
            row.from_backend or "unknown", row.reason_code or "unknown",
            row.to_backend or "terminal", row.count,
        ) + "</tr>"
        for row in routing.fallback_transitions
    ) or '<tr><td colspan="4">No fallback transitions</td></tr>'
    return (
        '<section id="routing"><h2>Expert Routing</h2><div class="two-col">'
        + _usage_bars("Primary backend usage", routing.primary_backend_usage)
        + _usage_bars("Final backend usage", routing.final_backend_usage)
        + '</div><h3>Fallback transitions</h3><div class="table-scroll"><table><thead><tr><th>From</th><th>Reason</th><th>To</th><th>Count</th>'
        f'</tr></thead><tbody>{transitions}</tbody></table></div></section>'
    )


def _samples_section(report: Report) -> str:
    options = lambda values: "".join(f'<option value="{_attr(value)}">{_esc(value)}</option>' for value in sorted(values))
    tasks = {sample.task for sample in report.samples if sample.task}
    states = {sample.state for sample in report.samples if sample.state}
    backends = {sample.routing.final_backend for sample in report.samples if sample.routing.final_backend}
    filters = (
        '<div class="filters"><label>Search<input id="search" type="search" placeholder="sample, question, prediction"></label>'
        f'<label>Task<select id="task"><option value="">All</option>{options(tasks)}</select></label>'
        f'<label>State<select id="state"><option value="">All</option>{options(states)}</select></label>'
        '<label>Quality<select id="quality"><option value="">All</option><option>correct</option><option>incorrect</option>'
        '<option>unknown</option><option>not_applicable</option></select></label>'
        f'<label>Final backend<select id="backend"><option value="">All</option>{options(backends)}</select></label>'
        '<label>Fallback<select id="fallback"><option value="">All</option><option value="yes">yes</option><option value="no">no</option></select></label>'
        '<label>Warnings<select id="warning"><option value="">All</option><option value="yes">yes</option><option value="no">no</option></select></label></div>'
    )
    return f'<section id="samples"><h2>Samples</h2>{filters}<div id="sample-list">' + "".join(
        _sample_card(sample) for sample in report.samples
    ) + "</div></section>"


def _sample_card(sample: ReportSample) -> str:
    quality = _effective_quality(sample)
    judge_kind, judge_label, judge_score = _judge_view(sample)
    target = sample.task_detail.target if isinstance(sample.task_detail, CountingReportDetail) else None
    ground_truth = _ground_truth_text(sample)
    search = " ".join(filter(None, [
        sample.sample_id, sample.question, target, sample.prediction,
        ground_truth, sample.routing.final_backend,
    ])).casefold()
    attrs = (
        f'data-search="{_attr(search)}" data-task="{_attr(sample.task)}" '
        f'data-state="{_attr(sample.state)}" data-quality="{_attr(quality)}" '
        f'data-judge="{_attr(judge_kind)}" '
        f'data-backend="{_attr(sample.routing.final_backend or "")}" '
        f'data-error="{_attr(sample.error_code or "")}" '
        f'data-fallback="{"yes" if sample.fallback_used else "no"}" '
        f'data-warning="{"yes" if sample.warnings else "no"}"'
    )
    badges = (
        f'<span class="badge state-{_attr(sample.state)}">{_esc(_state_label(sample.state))}</span>'
        f'<span class="badge judge-{_attr(judge_kind)}">{_esc(judge_label)}</span>'
        + ('<span class="badge fallback">fallback</span>' if sample.fallback_used else "")
    )
    return (
        f'<article class="sample result-{_attr(quality)}" {attrs}><details><summary class="sample-preview">'
        + _summary_visuals(sample)
        + '<span class="sample-summary-main"><span class="sample-summary-top">'
        + f'<span class="sample-id">{_esc(sample.sample_id)}</span><span>{_esc(sample.task)}</span>{badges}'
        + '</span><span class="sample-question">'
        + _esc(sample.question or "—") + '</span><span class="answer-grid">'
        + _answer_cell("模型答案 / Prediction", sample.prediction)
        + _answer_cell("标准答案 / Ground Truth", ground_truth)
        + '</span><span class="sample-result-row">'
        + f'<strong>{_esc(judge_label)}{f" (score={judge_score})" if judge_score is not None else ""}</strong>'
        + f'<span>执行状态: {_esc(_state_label(sample.state))}</span>'
        + f'<span>Final backend: {_esc(sample.routing.final_backend or "—")}</span>'
        + f'<span>Fallback: {"Yes" if sample.fallback_used else "No"}</span>'
        + f'<span>Latency: {_esc(_seconds(sample.inference_seconds))}</span>'
        + '</span></span></summary><div class="sample-body">'
        + _sample_hero(sample) + _task_routing_html(sample) + _execution_process(sample) + _model_calls(sample)
        + _execution_path_html(sample)
        + _technical_details(sample) + "</div></details></article>"
    )


def _common_detail(sample: ReportSample) -> str:
    quality = _effective_quality(sample)
    judge_kind, judge_label, judge_score = _judge_view(sample)
    rationale = _judge_rationale(sample)
    return (
        '<div class="detail-block answer-panel"><h4>样本信息 / Sample</h4><dl>'
        + _dl("题目 / Question", sample.question)
        + _dl("模型答案 / Prediction", sample.prediction)
        + _dl("标准答案 / Ground Truth", _ground_truth_text(sample))
        + _dl("执行状态 / Execution", _state_label(sample.state))
        + _dl("结果 / Result", judge_label)
        + _dl("Agent", sample.execution_agent)
        + _dl("Final backend", sample.routing.final_backend)
        + _dl("Latency", _seconds(sample.inference_seconds))
        + _dl("Run task", sample.run_task) + _dl("Resolved task", sample.resolved_task)
        + _dl("DeepSeek 核对 / Judge", judge_label)
        + _dl("Judge score", judge_score)
        + _dl("Judge rationale", rationale)
        + _dl("Persisted metric", _sample_metric_text(sample))
        + _dl("Error code", sample.error_code) + _dl("Warnings", ", ".join(sample.warnings))
        + "</dl></div>"
    )


def _meta_item(label: str, value: Any, *, mono: bool = False) -> str:
    class_name = "run-meta-value mono" if mono else "run-meta-value"
    return f'<div class="run-meta-item"><span>{_esc(label)}</span><strong class="{class_name}">{_esc(value)}</strong></div>'


def _overall_accuracy(report: Report) -> float | None:
    qualities = [_effective_quality(sample) for sample in report.samples]
    correct = sum(value == "correct" for value in qualities)
    eligible = sum(value in {"correct", "incorrect"} for value in qualities)
    return correct / eligible if eligible else None


def _quality_label(value: str) -> str:
    return {
        "correct": "✓ 正确 / Correct",
        "incorrect": "✕ 错误 / Incorrect",
        "unknown": "? 未知 / Unknown",
        "not_applicable": "N/A",
    }.get(value, value)


def _state_label(value: str | None) -> str:
    return {
        "succeeded": "执行成功 / Executed",
        "partial": "部分完成 / Partial",
        "failed": "执行失败 / Failed",
        "skipped": "已跳过 / Skipped",
    }.get(value or "", value or "—")


def _judge_view(sample: ReportSample) -> tuple[str, str, int | None]:
    """Return a stable visual outcome for the sample row.

    A DeepSeek score is authoritative when present. Samples without a judge
    result remain amber so an absent judge is not confused with correctness.
    For legacy non-caption reports, deterministic quality is retained when no
    judge was requested.
    """

    evaluation = sample.evaluation
    score = _sample_judge_score(evaluation)
    if score == 1:
        return "correct", "DeepSeek: 正确 / Correct", score
    if score == 0:
        return "incorrect", "DeepSeek: 错误 / Incorrect", score
    if sample.judge_status not in {"not_requested", ""} or getattr(evaluation, "judge_status", "not_requested") not in {"not_requested", ""}:
        return "partial", "DeepSeek: 核对失败 / Judge failed", None
    if sample.task not in {"caption", "change_caption"}:
        quality = _effective_quality(sample)
        if quality == "correct":
            return "correct", _quality_label("correct"), None
        if quality == "incorrect":
            return "incorrect", _quality_label("incorrect"), None
    return "partial", "待核对 / Not checked", None


def _judge_rationale(sample: ReportSample) -> str | None:
    parsed = getattr(sample.evaluation, "judge_parsed", None)
    if isinstance(parsed, dict):
        return parsed.get("concise_rationale") or parsed.get("rationale")
    return getattr(parsed, "concise_rationale", None) or getattr(parsed, "rationale", None)


def _ground_truth_text(sample: ReportSample) -> str:
    gt = sample.ground_truth
    if gt is None:
        return "; ".join(sample.reference_answers) or "—"
    if gt.count is not None:
        return str(gt.count)
    if gt.answers:
        return "; ".join(gt.answers)
    if gt.boxes:
        return f"{len(gt.boxes)} boxes"
    if gt.points:
        return f"{len(gt.points)} points"
    return "—"


def _sample_thumbnail(sample: ReportSample) -> str:
    visual = sample.visuals[0] if sample.visuals else None
    asset = _safe_asset(visual.original_asset if visual else None)
    if asset:
        return f'<img class="sample-thumb" loading="lazy" src="{_attr(asset)}" alt="sample {_attr(sample.sample_id)}">'
    return '<span class="sample-thumb sample-thumb-empty" aria-label="image unavailable">No image</span>'


def _summary_visuals(sample: ReportSample) -> str:
    """Show both temporal thumbnails before the sample is expanded."""
    if not sample.visuals:
        return _sample_thumbnail(sample)
    items: list[str] = []
    for visual in sample.visuals[:2]:
        safe = _safe_asset(visual.original_asset)
        if safe:
            role = (visual.role or visual.image_id or "image").split("·", 1)[0].strip()
            items.append(
                f'<span class="summary-visual"><img class="sample-thumb" loading="lazy" '
                f'src="{_attr(safe)}" alt="{_attr(role)} {_attr(sample.sample_id)}">'
                f'<small>{_esc(role)}</small></span>'
            )
    return '<span class="sample-thumbnails">' + "".join(items) + '</span>' if items else _sample_thumbnail(sample)


def _effective_quality(sample: ReportSample) -> str:
    """Recover quality from persisted metrics in legacy report bundles."""
    if sample.result_quality != "unknown":
        return sample.result_quality
    detail = sample.task_detail
    exact = getattr(detail, "exact_match", None)
    if exact is not None:
        return "correct" if exact else "incorrect"
    iou_match = getattr(detail, "iou_at_0_5", None)
    if iou_match is not None:
        return "correct" if iou_match else "incorrect"
    metrics = getattr(sample.evaluation, "deterministic_metrics", None)
    exact = getattr(metrics, "exact_match", None)
    if exact is not None:
        return "correct" if exact else "incorrect"
    return sample.result_quality


def _answer_cell(label: str, value: Any) -> str:
    shown = "—" if value is None or value == "" else value
    return f'<span class="answer-cell"><small>{_esc(label)}</small><strong>{_esc(shown)}</strong></span>'


def _sample_hero(sample: ReportSample) -> str:
    return '<section class="sample-hero"><div class="hero-visual">' + _visuals(sample) + '</div><div class="hero-answer">' + _common_detail(sample) + '</div></section>'


def _task_routing_html(sample: ReportSample) -> str:
    route = sample.task_routing
    candidates = "".join(
        f'<li class="route-candidate"><strong>{_esc(item.order)}. {_esc(item.task)}</strong>'
        f' · {_esc(item.status)}'
        f'{(" · " + _esc(", ".join(item.agent_names))) if item.agent_names else ""}'
        f'{(" · " + _esc(item.reason_code)) if item.reason_code else ""}'
        f'{" · selected" if item.selected else ""}{" · executed" if item.executed else ""}</li>'
        for item in route.candidate_tasks
    ) or '<li>not recorded</li>'
    skipped = ", ".join(route.skipped_candidates) or "—"
    return (
        '<section class="detail-block routing-flow"><h4>Task Routing / 任务路由</h4>'
        '<div class="route-chain">'
        f'<span>{_esc(route.source_task or "not recorded")}</span><b>→</b>'
        f'<span>{_esc(route.resolved_task or "not recorded")}</span><b>→</b>'
        f'<span>{_esc(route.executed_agent or route.primary_agent or "not recorded")}</span><b>→</b>'
        f'<span>{_esc(sample.routing.final_backend or "backend not recorded")}</span>'
        '</div><dl>'
        + _dl("Planning mode", route.planning_mode)
        + _dl("Resolution source", route.resolution_source)
        + _dl("Executed task", route.executed_task)
        + _dl("Primary agent", route.primary_agent)
        + _dl("Fallback agents", ", ".join(route.fallback_agents))
        + _dl("Execution mode", route.execution_mode)
        + _dl("Primary reason", route.primary_reason)
        + _dl("Fallback from task", route.fallback_from_task)
        + _dl("Skipped candidates", skipped)
        + '</dl><h5>Task candidates</h5><ol>' + candidates + '</ol></section>'
    )


def _execution_process(sample: ReportSample) -> str:
    if not sample.execution_steps and not sample.backend_stages:
        return '<section class="detail-block execution-process"><h4>Execution Process / 执行过程</h4><p>not recorded</p></section>'
    steps = []
    for step in sample.execution_steps:
        summary = "".join(_dl(key, value) for key, value in sorted(step.summary_fields.items()))
        links = []
        if step.request_id:
            links.append(f"request={_esc(step.request_id)}")
        if step.artifact_names:
            links.append(f"artifacts={_esc(', '.join(step.artifact_names))}")
        steps.append(
            f'<article class="stage timeline-step"><h4>{step.order}. {_esc(step.phase)} · {_esc(step.component)}</h4><dl>'
            + _dl("Operation", step.operation) + _dl("Status", step.status)
            + _dl("Task", step.task) + _dl("Agent", step.agent_name)
            + _dl("Backend", step.backend_name) + _dl("Reason", step.reason_code)
            + ("<p class=\"stage-note\">" + _esc(" · ".join(links)) + "</p>" if links else "")
            + summary + '</dl></article>'
        )
    if not steps:
        for stage in sample.backend_stages:
            steps.append(
                f'<article class="stage"><h4>{stage.order}. {_esc(stage.backend_name)} · {_esc(stage.backend_kind)}</h4><dl>'
                + _dl("Phase", stage.phase) + _dl("Status", stage.status)
                + _dl("Reason", stage.reason_code or stage.error_type)
                + _dl("Prediction", stage.predicted_count) + _dl("Accepted", stage.accepted_count)
                + _dl("Rejected", stage.rejected_count) + '</dl></article>'
            )
    return '<section class="detail-block execution-process"><h4>Execution Process / 执行过程</h4>' + "".join(steps) + '</section>'


def _model_calls(sample: ReportSample) -> str:
    if not sample.model_calls and not sample.structured_artifacts:
        return ""
    calls = []
    for index, call in enumerate(sample.model_calls, start=1):
        raw = _esc(call.raw_response or "—")
        parsed = _esc(call.parsed_response or "—")
        request = _esc(call.request_summary or "—")
        calls.append(
            f'<article class="model-call"><h4>{index}. Model Call · {_esc(call.request_id)}</h4><dl>'
            + _dl("Prompt version", call.prompt_version)
            + _dl("Latency", _seconds(call.latency_seconds))
            + _dl("Valid", call.valid) + _dl("Cache hit", call.cache_hit)
            + _dl("Repair used", call.repair_used) + _dl("Tokens", call.token_usage)
            + '</dl><details><summary>模型输入 / Request</summary><pre>' + request
            + '</pre></details><details><summary>Raw response</summary><pre>' + raw
            + '</pre></details><details><summary>Parsed model output</summary><pre>' + parsed
            + '</pre></details></article>'
        )
    structured = []
    for artifact in sample.structured_artifacts:
        structured.append(
            '<article class="model-call"><h4>Structured submodel output · '
            + _esc(artifact.filename)
            + '</h4><pre>'
            + _esc(_json_text(artifact.payload))
            + '</pre></article>'
        )
    return (
        '<section class="detail-block model-calls"><h4>All model/submodel outputs / '
        '全部模型/子模型输出</h4>'
        + "".join(calls)
        + "".join(structured)
        + '</section>'
    )


def _execution_path_html(sample: ReportSample) -> str:
    """Render the persisted top-level module hand-off path.
    渲染已持久化的顶层模块交接路径。"""

    if not sample.execution_path:
        return (
            '<section class="detail-block execution-path"><h4>Top-level '
            'execution path / 顶层执行路径</h4><p>not recorded</p></section>'
        )
    items = "".join(f"<li><code>{_esc(item)}</code></li>" for item in sample.execution_path)
    return (
        '<section class="detail-block execution-path"><h4>Top-level execution '
        'path / 顶层执行路径</h4><ol>'
        + items
        + '</ol></section>'
    )


def _technical_details(sample: ReportSample) -> str:
    return '<details class="detail-block technical-details"><summary>Technical Details / 技术详情</summary>' + _routing_detail(sample) + _task_detail(sample) + '</details>'


def _routing_detail(sample: ReportSample) -> str:
    route = sample.routing
    chain = " → ".join(route.candidate_backends) or "not recorded"
    attempts = "".join(
        f'<li><strong>{_esc(item.backend_name)}</strong> · {_esc(item.backend_kind or "kind not recorded")} · '
        f'{_esc(item.status)}{(" · " + _esc(item.reason_code)) if item.reason_code else ""}</li>'
        for item in route.attempted_backends
    ) or "<li>not recorded</li>"
    transitions = "".join(
        f'<li>{_esc(item.from_backend or "unknown")} → {_esc(item.to_backend or "terminal")} · {_esc(item.reason_code or "unknown")}</li>'
        for item in route.fallback_history
    )
    decision = ""
    if sample.routing_decision:
        decision = '<h5>Routing Decision</h5><dl>' + "".join(
            _dl(key, value) for key, value in sorted(sample.routing_decision.items())
        ) + '</dl>'
    return (
        '<div class="detail-block"><h4>Routing</h4><dl>' + _dl("Candidate Chain", chain)
        + _dl("Primary", _backend(route.primary_backend, route.primary_backend_kind))
        + _dl("Final", _backend(route.final_backend, route.final_backend_kind))
        + _dl("Fallback", "YES" if route.fallback_used else "NO")
        + _dl("Selection reason", route.selection_reason) + _dl("Review backend", route.review_backend)
        + f'</dl><h5>Attempts</h5><ol>{attempts}</ol>'
        + (f'<h5>Fallback history</h5><ol>{transitions}</ol>' if transitions else "")
        + decision + "</div>"
    )


def _task_detail(sample: ReportSample) -> str:
    detail = sample.task_detail
    if detail is None:
        metric = _sample_metric_text(sample)
        return f'<div class="detail-block"><h4>Task detail</h4><p>{_esc(metric or "not available")}</p></div>'
    if isinstance(detail, CountingReportDetail):
        provenance = _usage_table("Point provenance", detail.provenance_usage)
        return '<div class="detail-block"><h4>Counting detail</h4><dl>' + "".join([
            _dl("Target", detail.target), _dl("Prediction", detail.predicted_count), _dl("Gold", detail.gold_count),
            _dl("Absolute Error", detail.absolute_error), _dl("Exact Match", detail.exact_match),
            _dl("Counting status / mode", f"{detail.counting_status or '—'} / {detail.counting_mode or '—'}"),
            _dl("Tiles", f"total={detail.tile_count} initial={detail.initial_tile_count} leaf={detail.leaf_tile_count} succeeded={detail.succeeded_tile_count} failed={detail.failed_tile_count}"),
            _dl("Accepted", detail.accepted_point_count), _dl("Rejected", detail.rejected_point_count),
            _dl("Merged", detail.merged_group_count), _dl("Unresolved", detail.unresolved_conflict_count),
        ]) + f'</dl>{provenance}{_point_table("Accepted preview", detail.accepted_preview)}{_point_table("Rejected preview", detail.rejected_preview)}</div>'
    if isinstance(detail, GeneralVQAReportDetail):
        return '<div class="detail-block"><h4>VQA detail</h4><dl>' + "".join([
            _dl("Question", detail.question), _dl("Reference Answers", "; ".join(detail.reference_answers)),
            _dl("Prediction", detail.prediction), _dl("Exact Match", detail.exact_match),
            _dl("Judge Status", detail.judge_status), _dl("Judge Score", detail.judge_score),
            _dl("Judge Concise Rationale", detail.judge_concise_rationale),
            _dl("Visual Evidence Count", detail.visual_evidence_count),
            _dl("Geometry Repair Severity", detail.geometry_repair_severity),
        ]) + "</dl></div>"
    if isinstance(detail, GroundingReportDetail):
        return '<div class="detail-block"><h4>Grounding detail</h4><dl>' + "".join([
            _dl("Prediction", detail.prediction), _dl("Reference", "; ".join(detail.reference)),
            _dl("Predicted evidence boxes", detail.predicted_boxes), _dl("Ground-truth boxes", detail.ground_truth_boxes),
            _dl("Ground-truth coordinate frame", detail.ground_truth_coordinate_frame),
            _dl("IoU", detail.iou), _dl("IoU@0.5", detail.iou_at_0_5),
            _dl("Geometry Repair Severity", detail.geometry_repair_severity),
        ]) + "</dl></div>"
    if isinstance(detail, SpatialReportDetail):
        return _text_detail("Spatial detail", detail.question, detail.prediction, detail.reference_answers, detail.evidence_text, detail.evidence_item_count, detail.geometry_repair_severity)
    if isinstance(detail, ChangeReportDetail):
        return _text_detail("Change detail · T1 / T2", detail.question, detail.prediction, detail.reference_answers, detail.evidence_text, detail.evidence_item_count, detail.geometry_repair_severity)[:-6] + _dl("Geometry summary", detail.geometry_summary) + "</dl></div>"
    if isinstance(detail, CaptionReportDetail):
        return '<div class="detail-block"><h4>Caption detail</h4><dl>' + _dl("Generated Caption", detail.generated_caption) + _dl("Reference Captions", "; ".join(detail.reference_captions)) + _dl("Per-run caption metric status", detail.metric_status) + "</dl></div>"
    return ""


def _text_detail(title: str, question: Any, prediction: Any, references: list[str], evidence: list[str], count: int, severity: Any) -> str:
    return f'<div class="detail-block"><h4>{_esc(title)}</h4><dl>' + _dl("Question", question) + _dl("Prediction", prediction) + _dl("Reference answers", "; ".join(references)) + _dl("Evidence text", "; ".join(evidence)) + _dl("Evidence items", count) + _dl("Geometry Repair Severity", severity) + "</dl></div>"


def _visuals(sample: ReportSample) -> str:
    if not sample.visuals:
        return '<div class="detail-block visuals"><h4>Visuals</h4><p>No visual references.</p></div>'
    figures = []
    for visual in sample.visuals:
        images = []
        original = _safe_asset(visual.original_asset)
        overlay = _safe_asset(visual.overlay_asset)
        if original:
            images.append(_image_button(original, f"{visual.role} {visual.image_id}"))
        if overlay:
            images.append(_image_button(overlay, f"overlay {visual.image_id}"))
        status_note = '<p>Visual asset omitted by report budget.</p>' if visual.status == "omitted_by_budget" else ""
        figures.append(f'<article class="visual"><h5>{_esc(visual.role)} · {_esc(visual.image_id)} · {_esc(visual.status)}</h5>{status_note}<div class="image-grid">{"".join(images)}</div></article>')
    return '<div class="detail-block visuals"><h4>Visuals</h4>' + "".join(figures) + "</div>"


def _image_button(asset: str, alt: str) -> str:
    return (
        f'<button type="button" class="image-button" data-lightbox-src="{_attr(asset)}" '
        f'data-lightbox-alt="{_attr(alt)}" aria-label="点击放大 {_attr(alt)}">'
        f'<img class="detail-thumb" loading="lazy" src="{_attr(asset)}" alt="{_attr(alt)}"></button>'
    )


def _failures_section(report: Report) -> str:
    failure = report.failure_summary
    return '<section id="failures"><h2>Failures</h2><div class="three-col">' + _error_buttons(failure.sample_error_codes) + _usage_table("Warning codes", failure.warning_codes) + _usage_table("Backend failure codes", failure.backend_failure_counts) + "</div></section>"


def _runtime_section(report: Report) -> str:
    meta = report.metadata
    if meta is None:
        return '<section id="runtime"><h2>Runtime</h2><p>Manifest unavailable.</p></section>'
    return '<section id="runtime"><h2>Runtime</h2><dl class="runtime">' + "".join([
        _dl("Created", meta.created_at), _dl("Git commit", meta.git_commit),
        _dl("Git dirty", meta.git_dirty), _dl("Config hash", meta.config_hash),
        _dl("Split", meta.split), _dl("Sample filter", meta.sample_filter),
        _dl("Model IDs", ", ".join(f"{key}={value}" for key, value in sorted(meta.model_ids.items()))),
        _dl("Prompt hashes", ", ".join(f"{key}={value}" for key, value in sorted(meta.prompt_hashes.items()))),
    ]) + "</dl></section>"


def _metrics_html(metrics: dict[str, Any]) -> str:
    blocks = []
    for family in sorted(metrics):
        payload = metrics[family]
        if not isinstance(payload, dict):
            continue
        exact = f'<p>Exact-match accuracy: {_esc(_format_number(payload["score"]))}</p>' if family == "general_vqa" and "score" in payload else ""
        note = (
            '<p>CHAIR2: not configured; no approved scorer is persisted for this run.</p>'
            if family == "caption" else ""
        )
        blocks.append(f'<h4>Metrics: {_esc(family)}</h4>{exact}{note}<table><tbody>' + "".join(
            f'<tr>{_cells(key, _format_number(value))}</tr>' for key, value in sorted(payload.items())
        ) + "</tbody></table>")
    return "".join(blocks)


def _judge_metrics_html(judge_metrics: dict[str, Any]) -> str:
    payload = judge_metrics.get("vqa_semantic_equivalence")
    if not isinstance(payload, dict):
        return ""
    coverage = payload.get("coverage")
    coverage_text = f"{coverage:.2%}" if isinstance(coverage, (int, float)) and not isinstance(coverage, bool) else "unknown"
    complete = payload.get("complete") is True
    rows = [
        f'<p>Semantic judge coverage: {_esc(coverage_text)}</p>',
        f'<p>Semantic equivalent mismatches: {_esc(_format_number(payload.get("semantic_equivalent_mismatches")))}</p>',
        f'<p>Judge failures: {_esc(_format_number(payload.get("judge_failures")))}</p>',
        f'<p>Unresolved mismatches: {_esc(_format_number(payload.get("unresolved_mismatches")))}</p>',
        f'<p>Complete: {"true" if complete else "false"}</p>',
    ]
    if complete:
        rows.append(f'<p>Judge-assisted semantic accuracy: {_esc(_format_number(payload.get("score")))}</p>')
    else:
        rows.extend(['<p>Judge-assisted semantic accuracy: incomplete</p>', f'<p>Confirmed lower bound: {_esc(_format_number(payload.get("lower_bound_score")))}</p>'])
    return "<h4>VQA semantic judge</h4>" + "".join(rows)


def _sample_metric_text(sample: ReportSample) -> str:
    evaluation = sample.evaluation
    if evaluation is None:
        return ""
    if evaluation.deterministic_metrics is None:
        score = _sample_judge_score(evaluation)
        return f"judge_score={score}" if score is not None else ""
    metrics = evaluation.deterministic_metrics
    if evaluation.task == "general_vqa":
        exact = f"exact_match={getattr(metrics, 'exact_match', None)}"
        score = _sample_judge_score(evaluation)
        return f"{exact} judge_score={score}" if score is not None else exact
    if evaluation.task == "counting":
        return f"predicted={getattr(metrics, 'predicted_count', None)} gold={getattr(metrics, 'gold_count', None)} exact_match={getattr(metrics, 'exact_match', None)}"
    if evaluation.task == "grounding":
        return f"iou={getattr(metrics, 'iou', None)} iou_at_0_5={getattr(metrics, 'iou_at_0_5', None)}"
    if evaluation.task == "caption":
        score = _sample_judge_score(evaluation)
        return f"caption record judge_score={score}" if score is not None else f"caption record judge_status={evaluation.judge_status}"
    return ""


def _sample_judge_score(evaluation: Any) -> int | None:
    if getattr(evaluation, "judge_status", None) != "succeeded":
        return None
    parsed = getattr(evaluation, "judge_parsed", None)
    score = parsed.get("score") if isinstance(parsed, dict) else getattr(parsed, "score", None)
    return score if type(score) is int and score in (0, 1) else None


def _point_table(title: str, points: list[Any]) -> str:
    if not points:
        return ""
    return f'<details><summary>{_esc(title)} ({len(points)})</summary><div class="table-scroll"><table><thead><tr><th>ID</th><th>x</th><th>y</th><th>confidence</th><th>source</th><th>backend</th><th>reason</th></tr></thead><tbody>' + "".join(
        f'<tr>{_cells(point.point_id or "", point.x, point.y, _format_number(point.confidence), point.source or "", point.backend_name or "", point.rejection_reason or "")}</tr>' for point in points
    ) + "</tbody></table></div></details>"


def _usage_table(title: str, usage: dict[str, int]) -> str:
    rows = "".join(f'<tr>{_cells(name, count)}</tr>' for name, count in sorted(usage.items()))
    body = rows or '<tr><td colspan="2">none</td></tr>'
    return f'<div><h4>{_esc(title)}</h4><div class="table-scroll"><table><thead><tr><th>Key</th><th>Count</th></tr></thead><tbody>{body}</tbody></table></div></div>'


def _usage_bars(title: str, usage: dict[str, int]) -> str:
    maximum = max(usage.values(), default=1)
    rows = "".join(
        f'<div class="bar-row"><span>{_esc(name)}</span><i><b style="width:{count / maximum * 100:.2f}%"></b></i><strong>{count}</strong></div>'
        for name, count in sorted(usage.items(), key=lambda item: (-item[1], item[0]))
    )
    return f'<div><h3>{_esc(title)}</h3>{rows or "<p>none</p>"}</div>'


def _error_buttons(usage: dict[str, int]) -> str:
    buttons = "".join(
        f'<button type="button" data-filter-error="{_attr(code)}">{_esc(code)} · {count}</button>'
        for code, count in sorted(usage.items())
    )
    return f'<div><h4>Sample error codes</h4><div class="code-buttons">{buttons or "none"}</div></div>'


def _dl(label: str, value: Any) -> str:
    shown = "—" if value is None or value == "" else str(value)
    return f'<dt>{_esc(label)}</dt><dd>{_esc(shown)}</dd>'


def _json_text(value: Any) -> str:
    """Render a bounded JSON-safe value for a preformatted audit block.
    将有界 JSON-safe 值渲染为预格式化审计块。"""

    try:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _cells(*values: Any) -> str:
    return "".join(f'<td>{_esc(value)}</td>' for value in values)


def _backend(name: str | None, kind: str | None) -> str:
    return f"{name or 'not recorded'} ({kind or 'kind not recorded'})"


def _seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}s"


def _format_number(value: Any) -> str:
    if value is None:
        return "—"
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def _safe_asset(value: str | None) -> str | None:
    if not value or "\\" in value or ":" in value or value.startswith("/"):
        return None
    path = PurePosixPath(value)
    if not path.parts or path.parts[0] != "assets" or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return value


def _esc(value: Any) -> str:
    return html.escape(_public_text(value), quote=True)


def _attr(value: Any) -> str:
    return html.escape(_public_text(value), quote=True)


_SECRET_RE = re.compile(r"(?i)(?<![a-z0-9_-])(?:sk-[a-z0-9_-]{6,}|bearer\s+[a-z0-9._~+/-]{6,})")
_WIN_PATH_RE = re.compile(r"(?i)[a-z]:[\\/][^\s\"'<>]+")
_POSIX_PATH_RE = re.compile(r"(?:(?:/tmp|/home|/users)/[^\s\"'<>]+)", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)


def _public_text(value: Any) -> str:
    text = str(value)
    text = _SECRET_RE.sub("[redacted-secret]", text)
    text = _WIN_PATH_RE.sub("[redacted-path]", text)
    text = _POSIX_PATH_RE.sub("[redacted-path]", text)
    return _URL_RE.sub("[redacted-url]", text)


_CSS = """
:root{color-scheme:light;--ink:#172033;--muted:#64748b;--line:#dbe3ef;--panel:#fff;--bg:#f3f6fa;--blue:#2563eb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1600px;margin:auto;padding:28px}
header{display:flex;justify-content:space-between;align-items:flex-start;padding:18px 22px;background:#0f172a;color:#fff;border-radius:12px}h1{margin:.1rem 0;font-size:1.8rem}h2{font-size:1.25rem;margin-top:0}h3{font-size:1.05rem}.eyebrow,.schema{letter-spacing:0;font-size:.72rem;color:#93c5fd}.schema{border:1px solid #475569;padding:8px;border-radius:6px}
nav{display:flex;gap:18px;position:sticky;top:0;z-index:2;background:rgba(243,246,250,.96);padding:14px 4px}nav a{color:#334155;text-decoration:none;font-weight:650}section{background:var(--panel);border:1px solid var(--line);border-radius:10px;margin:14px 0;padding:20px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(135px,1fr));gap:10px}.card{border-left:3px solid var(--blue);background:#f8fafc;padding:13px}.card span{display:block;color:var(--muted);font-size:.78rem}.card strong{font-size:1.35rem}
table{width:100%;border-collapse:collapse;margin:.5rem 0 1rem}th,td{border-bottom:1px solid var(--line);padding:7px;text-align:left;vertical-align:top}th{font-size:.75rem;text-transform:uppercase;color:var(--muted)}.two-col,.three-col{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}.paneltask{border-top:1px solid var(--line);padding:14px 0}.filters{display:grid;grid-template-columns:2fr repeat(6,1fr);gap:8px;margin-bottom:12px}.filters label{font-size:.72rem;color:var(--muted)}input,select{width:100%;padding:8px;border:1px solid var(--line);border-radius:6px;background:#fff}
.sample{border:1px solid var(--line);border-radius:8px;margin:7px 0;background:#fff}.sample summary{display:grid;grid-template-columns:minmax(160px,1fr) 140px repeat(3,max-content) minmax(160px,1fr);gap:10px;align-items:center;cursor:pointer;padding:11px}.sample summary.sample-preview{display:grid;grid-template-columns:236px minmax(0,1fr);gap:14px;align-items:start;padding:10px}.sample-id{font-family:ui-monospace,monospace;font-weight:700}.backend{text-align:right;color:var(--muted)}.badge{font-size:.7rem;border-radius:999px;padding:3px 7px;background:#e2e8f0}.failed,.judge-incorrect{background:#fee2e2;color:#991b1b}.partial,.fallback,.judge-partial{background:#fef3c7;color:#92400e}.succeeded,.judge-correct{background:#dcfce7;color:#166534}.sample.result-correct{border-left:5px solid #16a34a;background:#f0fdf4}.sample.result-incorrect{border-left:5px solid #dc2626;background:#fff1f2}.sample.result-unknown{border-left:5px solid #d97706;background:#fffbeb}.sample[data-judge="correct"]{border-left:5px solid #16a34a;background:#f0fdf4}.sample[data-judge="incorrect"]{border-left:5px solid #dc2626;background:#fff1f2}.sample[data-judge="partial"]{border-left:5px solid #d97706;background:#fffbeb}.sample[data-judge="correct"] .sample-result-row{color:#166534}.sample[data-judge="incorrect"] .sample-result-row{color:#991b1b}.sample[data-judge="partial"] .sample-result-row{color:#92400e}.sample-result-row strong{font-size:1rem}.sample-body{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;border-top:1px solid var(--line);padding:12px}.detail-block{background:#f8fafc;border-radius:7px;padding:12px;overflow:auto}.visuals{grid-column:1/-1}.sample-thumbnails{display:flex;gap:6px;align-items:flex-start}.summary-visual{display:grid;gap:2px;justify-items:center;width:112px}.summary-visual small{font-size:.65rem;color:var(--muted);text-transform:uppercase}.sample-thumbnails .sample-thumb{flex:0 0 112px}dl{display:grid;grid-template-columns:minmax(130px,.4fr) 1fr;gap:4px 12px;margin:.4rem 0}dt{color:var(--muted)}dd{margin:0;white-space:pre-wrap;overflow-wrap:anywhere}.image-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.image-button{display:block;width:100%;padding:0;border:1px solid var(--line);border-radius:6px;background:#fff;cursor:zoom-in;overflow:hidden}.image-button:hover{border-color:#2563eb;box-shadow:0 0 0 2px #bfdbfe}.detail-thumb{display:block;width:100%;height:240px;object-fit:contain;background:#e2e8f0}figure{margin:0}img{display:block;width:100%;max-height:620px;object-fit:contain;background:#e2e8f0}figcaption{text-align:center;color:var(--muted)}.legend{display:flex;gap:22px}.green{color:#22c55e}.red{color:#ef4444}.cyan{color:#38bdf8}.amber{color:#f59e0b}.purple{color:#a855f7}
.bar-row{display:grid;grid-template-columns:minmax(140px,1fr) 3fr 45px;gap:8px;align-items:center;margin:7px 0}.bar-row i{display:block;height:8px;background:#e2e8f0;border-radius:99px;overflow:hidden}.bar-row b{display:block;height:100%;background:#2563eb}
.code-buttons{display:flex;flex-wrap:wrap;gap:6px}.code-buttons button{border:1px solid var(--line);background:#fff;border-radius:6px;padding:6px 9px;cursor:pointer}
.run-meta{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin-top:14px;max-width:720px}.run-meta-item{min-width:0}.run-meta-item span{display:block;color:#cbd5e1;font-size:.72rem}.run-meta-value{display:block;overflow-wrap:anywhere;word-break:break-word}.mono{font-family:ui-monospace,monospace;overflow-wrap:anywhere}.aggregate-panel{background:transparent;border:0;margin:14px 0;padding:0}.aggregate-panel>summary{cursor:pointer;font-weight:700;padding:12px;background:#fff;border:1px solid var(--line);border-radius:8px}.sample-preview{display:grid;grid-template-columns:112px minmax(0,1fr);gap:14px;align-items:start;min-width:0;cursor:pointer;padding:12px}.sample-thumb{width:112px;height:112px;object-fit:cover;background:#e2e8f0;border-radius:4px}.sample-thumb-empty{display:grid;place-items:center;color:var(--muted);font-size:.75rem;text-align:center}.sample-summary-main{display:grid;gap:8px;min-width:0}.sample-summary-top{display:flex;gap:8px;align-items:center;flex-wrap:wrap;min-width:0}.sample-question{font-size:.95rem;overflow-wrap:anywhere}.answer-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;min-width:0}.answer-cell{display:grid;gap:3px;min-width:0;padding:8px;background:#fff;border:1px solid var(--line);border-radius:4px}.answer-cell small{color:var(--muted)}.answer-cell strong{overflow-wrap:anywhere;word-break:break-word}.sample-result-row{display:flex;gap:14px;align-items:center;flex-wrap:wrap;color:var(--muted);overflow-wrap:anywhere}.sample-result-row strong{color:var(--ink)}.sample-body{display:block;border-top:1px solid var(--line);padding:12px}.sample-hero{display:grid;grid-template-columns:minmax(360px,.95fr) minmax(0,1.05fr);gap:12px;min-width:0}.sample-hero>*{min-width:0}.hero-visual,.hero-answer{min-width:0}.routing-flow,.execution-process,.model-calls{margin-top:12px}.route-chain{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:.5rem 0 1rem}.route-chain span{padding:6px 9px;border:1px solid var(--line);border-radius:4px;background:#fff;font-family:ui-monospace,monospace;overflow-wrap:anywhere}.route-chain b{color:var(--muted)}.route-candidate{margin:.25rem 0;overflow-wrap:anywhere}.stage,.model-call{border-top:1px solid var(--line);padding:10px 0;min-width:0}.timeline-step{border-left:3px solid #94a3b8;padding-left:12px}.timeline-step h4{font-size:.92rem}.stage h4,.model-call h4{margin:.1rem 0 .5rem;overflow-wrap:anywhere}.technical-details{margin-top:12px}.technical-details>summary{cursor:pointer;font-weight:650}.table-scroll{width:100%;overflow-x:auto}.table-scroll table{min-width:640px}
@media(max-width:900px){main{padding:10px}.filters{grid-template-columns:1fr 1fr}.sample summary.sample-preview{grid-template-columns:160px minmax(0,1fr)}.sample-thumb{width:76px;height:76px}.answer-grid{grid-template-columns:1fr}.sample-hero{grid-template-columns:1fr}.sample-body{grid-template-columns:1fr}.visuals{grid-column:auto}}
@media(max-width:560px){header{display:block}.schema{display:inline-block;margin-top:12px}.filters{grid-template-columns:1fr}.sample summary.sample-preview{grid-template-columns:1fr}.sample-thumbnails{width:100%}.summary-visual{width:calc(50% - 3px)}.summary-visual .sample-thumb{width:100%;height:auto;aspect-ratio:1}.sample-result-row{gap:8px}}
.execution-path{margin-top:12px}.execution-path ol{margin:.5rem 0 0;padding-left:1.5rem}.execution-path code{font-size:.85rem;overflow-wrap:anywhere;word-break:break-word}.image-modal{position:fixed;inset:0;z-index:20;display:grid;place-items:center;padding:5vh 5vw;background:rgba(15,23,42,.86)}.image-modal[hidden]{display:none}.image-modal img{max-width:94vw;max-height:88vh;width:auto;height:auto;object-fit:contain;background:#fff;box-shadow:0 12px 50px rgba(0,0,0,.45)}.image-modal-close{position:fixed;top:18px;right:24px;width:40px;height:40px;border:0;border-radius:999px;background:#fff;color:#0f172a;font-size:28px;line-height:1;cursor:pointer}.modal-open{overflow:hidden}
"""


_JS = """
(()=>{const ids=['search','task','state','quality','backend','fallback','warning'];const controls=ids.map(id=>document.getElementById(id)).filter(Boolean);const cards=[...document.querySelectorAll('.sample')];let error='';function apply(){const v=Object.fromEntries(ids.map(id=>[id,(document.getElementById(id)?.value||'').toLowerCase()]));for(const card of cards){const ok=(!v.search||card.dataset.search.includes(v.search))&&(!v.task||card.dataset.task===v.task)&&(!v.state||card.dataset.state===v.state)&&(!v.quality||card.dataset.quality===v.quality)&&(!v.backend||card.dataset.backend===v.backend)&&(!v.fallback||card.dataset.fallback===v.fallback)&&(!v.warning||card.dataset.warning===v.warning)&&(!error||card.dataset.error===error);card.hidden=!ok}}controls.forEach(c=>c.addEventListener(c.type==='search'?'input':'change',apply));document.querySelectorAll('[data-filter-error]').forEach(button=>button.addEventListener('click',()=>{error=button.dataset.filterError;location.hash='samples';apply()}));const modal=document.getElementById('image-modal');const modalImage=document.getElementById('image-modal-img');const close=()=>{if(!modal)return;modal.hidden=true;modalImage.removeAttribute('src');document.body.classList.remove('modal-open')};document.querySelectorAll('.image-button').forEach(button=>button.addEventListener('click',()=>{if(!modal)return;modalImage.src=button.dataset.lightboxSrc||'';modalImage.alt=button.dataset.lightboxAlt||'';modal.hidden=false;document.body.classList.add('modal-open')}));modal?.addEventListener('click',event=>{if(event.target===modal)close()});document.querySelector('.image-modal-close')?.addEventListener('click',close);document.addEventListener('keydown',event=>{if(event.key==='Escape')close()});})();
"""
