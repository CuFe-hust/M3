"""Pure, deterministic rendering of the offline Report V2 dashboard."""

from __future__ import annotations

import html
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
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>Report V2 · {_esc(report.run_id)}</title>",
        f"<style>{_CSS}</style></head><body><main>",
        _header(report),
        '<nav><a href="#overview">Overview</a><a href="#tasks">Tasks</a>'
        '<a href="#routing">Expert Routing</a><a href="#samples">Samples</a>'
        '<a href="#failures">Failures</a><a href="#runtime">Runtime</a></nav>',
        _overview(report),
        _counting_target_table(report),
        '<section id="tasks"><h2>Tasks</h2>' + _task_overview_table(report.tasks) + "".join(_task_section(task) for task in report.tasks) + "</section>",
        _routing_section(report),
        _samples_section(report),
        _failures_section(report),
        _runtime_section(report),
        '<section><h2>Visual legend</h2><div class="legend">'
        '<span class="green">● Prediction / accepted</span><span class="red">● Rejected</span>'
        '<span class="cyan">● GT / ground truth</span><span class="amber">● Unresolved</span>'
        '<span class="purple">● Reviewer</span></div></section>',
        f"<script>{_JS}</script>",
        "</main></body></html>",
    ])


def _header(report: Report) -> str:
    return (
        '<header><div><p class="eyebrow">M3 MULTI-AGENT AUDIT</p>'
        f'<h1>Run {_esc(report.run_id)}</h1><p>{_esc(report.dataset or "unknown dataset")}</p></div>'
        '<div class="schema">REPORT V2 · OFFLINE</div></header>'
    )


def _overview(report: Report) -> str:
    latency = report.latency
    meta = report.metadata
    cards = [
        ("Dataset", report.dataset or "—"), ("Split", meta.split if meta else "—"),
        ("Run ID", report.run_id), ("Git commit", meta.git_commit if meta else "—"),
        ("Samples", report.total), ("Succeeded", report.succeeded),
        ("Partial", report.partial), ("Failed", report.failed),
        ("Skipped", report.skipped),
        ("Fallback", f"{report.routing_summary.fallback_rate:.1%}"),
        ("Mean latency", _seconds(latency.mean_seconds)),
        ("p50 latency", _seconds(latency.p50_seconds)),
        ("p95 latency", _seconds(latency.p95_seconds)),
        ("Visuals", f"{report.visual_materialized_count}/{report.visual_total}"),
    ]
    return '<section id="overview"><h2>Overview</h2><div class="cards">' + "".join(
        f'<article class="card"><span>{_esc(label)}</span><strong>{_esc(value)}</strong></article>'
        for label, value in cards
    ) + "</div></section>"


def _task_overview_table(tasks: list[TaskSummary]) -> str:
    rows = "".join(
        f'<tr>{_cells(task.run_task, task.total, task.succeeded, task.partial, task.failed, task.correct, task.incorrect, task.fallback_count, _seconds(task.latency.mean_seconds))}</tr>'
        for task in tasks
    )
    return '<table><thead><tr><th>Task</th><th>Samples</th><th>Succeeded</th><th>Partial</th><th>Failed</th><th>Correct</th><th>Incorrect</th><th>Fallback</th><th>Mean latency</th></tr></thead><tbody>' + (rows or '<tr><td colspan="9">No task rows</td></tr>') + '</tbody></table>'


def _counting_target_table(report: Report) -> str:
    if not report.counting_target_summary:
        return ""
    rows = "".join(
        f'<tr>{_cells(item.target, item.sample_count, item.evaluated_count, item.exact_count, _format_number(item.accuracy), _format_number(item.mae), item.fallback_count, f"{item.fallback_rate:.2%}")}</tr>'
        for item in report.counting_target_summary
    )
    return '<section id="counting-targets"><h2>Counting targets</h2><table><thead><tr><th>Target</th><th>Samples</th><th>With gold</th><th>Exact</th><th>Accuracy</th><th>MAE</th><th>Fallback</th><th>Fallback rate</th></tr></thead><tbody>' + rows + '</tbody></table></section>'


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
        + '</div><h3>Fallback transitions</h3><table><thead><tr><th>From</th><th>Reason</th><th>To</th><th>Count</th>'
        f'</tr></thead><tbody>{transitions}</tbody></table></section>'
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
    target = sample.task_detail.target if isinstance(sample.task_detail, CountingReportDetail) else None
    search = " ".join(
        filter(None, [sample.sample_id, sample.question, target, sample.prediction])
    ).casefold()
    attrs = (
        f'data-search="{_attr(search)}" data-task="{_attr(sample.task)}" '
        f'data-state="{_attr(sample.state)}" data-quality="{_attr(sample.result_quality)}" '
        f'data-backend="{_attr(sample.routing.final_backend or "")}" '
        f'data-error="{_attr(sample.error_code or "")}" '
        f'data-fallback="{"yes" if sample.fallback_used else "no"}" '
        f'data-warning="{"yes" if sample.warnings else "no"}"'
    )
    badges = (
        f'<span class="badge {_attr(sample.state)}">{_esc(sample.state)}</span>'
        f'<span class="badge quality-{_attr(sample.result_quality)}">{_esc(sample.result_quality)}</span>'
        + ('<span class="badge fallback">fallback</span>' if sample.fallback_used else "")
    )
    return (
        f'<article class="sample" {attrs}><details><summary><span class="sample-id">{_esc(sample.sample_id)}</span>'
        f'<span>{_esc(sample.task)}</span>{badges}<span class="backend">{_esc(sample.routing.final_backend or "—")}</span></summary>'
        '<div class="sample-body">'
        + _common_detail(sample) + _routing_detail(sample) + _task_detail(sample) + _visuals(sample)
        + "</div></details></article>"
    )


def _common_detail(sample: ReportSample) -> str:
    return (
        '<div class="detail-block"><h4>Common</h4><dl>'
        + _dl("Run task", sample.run_task) + _dl("Resolved task", sample.resolved_task)
        + _dl("Agent", sample.execution_agent) + _dl("Question", sample.question)
        + _dl("References", "; ".join(sample.reference_answers))
        + _dl("Prediction", sample.prediction) + _dl("Judge", sample.judge_status)
        + _dl("Persisted metric", _sample_metric_text(sample))
        + _dl("Inference time", _seconds(sample.inference_seconds))
        + _dl("Error code", sample.error_code) + _dl("Warnings", ", ".join(sample.warnings))
        + "</dl></div>"
    )


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
    return (
        '<div class="detail-block"><h4>Routing</h4><dl>' + _dl("Candidate Chain", chain)
        + _dl("Primary", _backend(route.primary_backend, route.primary_backend_kind))
        + _dl("Final", _backend(route.final_backend, route.final_backend_kind))
        + _dl("Fallback", "YES" if route.fallback_used else "NO")
        + _dl("Selection reason", route.selection_reason) + _dl("Review backend", route.review_backend)
        + f'</dl><h5>Attempts</h5><ol>{attempts}</ol>'
        + (f'<h5>Fallback history</h5><ol>{transitions}</ol>' if transitions else "") + "</div>"
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
            _dl("Predicted evidence boxes", len(detail.predicted_boxes)), _dl("Ground-truth boxes", len(detail.ground_truth_boxes)),
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
            images.append(f'<figure><img loading="lazy" src="{_attr(original)}" alt="original {_attr(visual.image_id)}"><figcaption>Original</figcaption></figure>')
        if overlay:
            images.append(f'<figure><img loading="lazy" src="{_attr(overlay)}" alt="overlay {_attr(visual.image_id)}"><figcaption>Overlay</figcaption></figure>')
        status_note = '<p>Visual asset omitted by report budget.</p>' if visual.status == "omitted_by_budget" else ""
        figures.append(f'<article class="visual"><h5>{_esc(visual.role)} · {_esc(visual.image_id)} · {_esc(visual.status)}</h5>{status_note}<div class="image-grid">{"".join(images)}</div></article>')
    return '<div class="detail-block visuals"><h4>Visuals</h4>' + "".join(figures) + "</div>"


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
        blocks.append(f'<h4>Metrics: {_esc(family)}</h4>{exact}<table><tbody>' + "".join(
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
    if evaluation is None or evaluation.deterministic_metrics is None:
        return ""
    metrics = evaluation.deterministic_metrics
    if evaluation.task == "general_vqa":
        exact = f"exact_match={getattr(metrics, 'exact_match', None)}"
        score = _sample_judge_score(evaluation)
        return f"{exact} judge_score={score}" if score is not None else exact
    if evaluation.task == "counting":
        return f"predicted={getattr(metrics, 'predicted_count', None)} gold={getattr(metrics, 'gold_count', None)} exact_match={getattr(metrics, 'exact_match', None)}"
    if evaluation.task == "grounding":
        return f"iou={getattr(metrics, 'iou', None)} iou_at_0_5={getattr(metrics, 'iou_at_0_5', None)}"
    return "caption record" if evaluation.task == "caption" else ""


def _sample_judge_score(evaluation: Any) -> int | None:
    if getattr(evaluation, "judge_status", None) != "succeeded":
        return None
    parsed = getattr(evaluation, "judge_parsed", None)
    score = parsed.get("score") if isinstance(parsed, dict) else getattr(parsed, "score", None)
    return score if type(score) is int and score in (0, 1) else None


def _point_table(title: str, points: list[Any]) -> str:
    if not points:
        return ""
    return f'<h5>{_esc(title)}</h5><table><thead><tr><th>ID</th><th>x</th><th>y</th><th>confidence</th><th>source</th><th>backend</th><th>reason</th></tr></thead><tbody>' + "".join(
        f'<tr>{_cells(point.point_id or "", point.x, point.y, _format_number(point.confidence), point.source or "", point.backend_name or "", point.rejection_reason or "")}</tr>' for point in points
    ) + "</tbody></table>"


def _usage_table(title: str, usage: dict[str, int]) -> str:
    rows = "".join(f'<tr>{_cells(name, count)}</tr>' for name, count in sorted(usage.items()))
    return f'<div><h4>{_esc(title)}</h4><table><thead><tr><th>Key</th><th>Count</th></tr></thead><tbody>{rows or "<tr><td colspan=\"2\">none</td></tr>"}</tbody></table></div>'


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


_SECRET_RE = re.compile(r"(?i)(?:sk-[a-z0-9_-]{6,}|bearer\s+[a-z0-9._~+/-]{6,})")
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
header{display:flex;justify-content:space-between;align-items:flex-start;padding:18px 22px;background:#0f172a;color:#fff;border-radius:12px}h1{margin:.1rem 0;font-size:1.8rem}h2{font-size:1.25rem;margin-top:0}h3{font-size:1.05rem}.eyebrow,.schema{letter-spacing:.13em;font-size:.72rem;color:#93c5fd}.schema{border:1px solid #475569;padding:8px;border-radius:6px}
nav{display:flex;gap:18px;position:sticky;top:0;z-index:2;background:rgba(243,246,250,.96);padding:14px 4px}nav a{color:#334155;text-decoration:none;font-weight:650}section{background:var(--panel);border:1px solid var(--line);border-radius:10px;margin:14px 0;padding:20px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(135px,1fr));gap:10px}.card{border-left:3px solid var(--blue);background:#f8fafc;padding:13px}.card span{display:block;color:var(--muted);font-size:.78rem}.card strong{font-size:1.35rem}
table{width:100%;border-collapse:collapse;margin:.5rem 0 1rem}th,td{border-bottom:1px solid var(--line);padding:7px;text-align:left;vertical-align:top}th{font-size:.75rem;text-transform:uppercase;color:var(--muted)}.two-col,.three-col{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}.paneltask{border-top:1px solid var(--line);padding:14px 0}.filters{display:grid;grid-template-columns:2fr repeat(6,1fr);gap:8px;margin-bottom:12px}.filters label{font-size:.72rem;color:var(--muted)}input,select{width:100%;padding:8px;border:1px solid var(--line);border-radius:6px;background:#fff}
.sample{border:1px solid var(--line);border-radius:8px;margin:7px 0;background:#fff}.sample summary{display:grid;grid-template-columns:minmax(160px,1fr) 140px repeat(3,max-content) minmax(160px,1fr);gap:10px;align-items:center;cursor:pointer;padding:11px}.sample-id{font-family:ui-monospace,monospace;font-weight:700}.backend{text-align:right;color:var(--muted)}.badge{font-size:.7rem;border-radius:999px;padding:3px 7px;background:#e2e8f0}.failed,.quality-incorrect{background:#fee2e2;color:#991b1b}.partial,.fallback{background:#fef3c7;color:#92400e}.succeeded,.quality-correct{background:#dcfce7;color:#166534}.sample-body{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;border-top:1px solid var(--line);padding:12px}.detail-block{background:#f8fafc;border-radius:7px;padding:12px;overflow:auto}.visuals{grid-column:1/-1}dl{display:grid;grid-template-columns:minmax(130px,.4fr) 1fr;gap:4px 12px;margin:.4rem 0}dt{color:var(--muted)}dd{margin:0;white-space:pre-wrap;overflow-wrap:anywhere}.image-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:10px}figure{margin:0}img{display:block;width:100%;max-height:620px;object-fit:contain;background:#e2e8f0}figcaption{text-align:center;color:var(--muted)}.legend{display:flex;gap:22px}.green{color:#22c55e}.red{color:#ef4444}.cyan{color:#38bdf8}.amber{color:#f59e0b}.purple{color:#a855f7}
.bar-row{display:grid;grid-template-columns:minmax(140px,1fr) 3fr 45px;gap:8px;align-items:center;margin:7px 0}.bar-row i{display:block;height:8px;background:#e2e8f0;border-radius:99px;overflow:hidden}.bar-row b{display:block;height:100%;background:#2563eb}
.code-buttons{display:flex;flex-wrap:wrap;gap:6px}.code-buttons button{border:1px solid var(--line);background:#fff;border-radius:6px;padding:6px 9px;cursor:pointer}
@media(max-width:900px){main{padding:10px}.filters{grid-template-columns:1fr 1fr}.sample summary{grid-template-columns:1fr 1fr}.sample-body{grid-template-columns:1fr}.visuals{grid-column:auto}}
"""


_JS = """
(()=>{const ids=['search','task','state','quality','backend','fallback','warning'];const controls=ids.map(id=>document.getElementById(id));const cards=[...document.querySelectorAll('.sample')];let error='';function apply(){const v=Object.fromEntries(ids.map((id,i)=>[id,controls[i].value.toLowerCase()]));for(const card of cards){const ok=(!v.search||card.dataset.search.includes(v.search))&&(!v.task||card.dataset.task===v.task)&&(!v.state||card.dataset.state===v.state)&&(!v.quality||card.dataset.quality===v.quality)&&(!v.backend||card.dataset.backend===v.backend)&&(!v.fallback||card.dataset.fallback===v.fallback)&&(!v.warning||card.dataset.warning===v.warning)&&(!error||card.dataset.error===error);card.hidden=!ok}}controls.forEach(c=>c.addEventListener(c.type==='search'?'input':'change',apply));document.querySelectorAll('[data-filter-error]').forEach(button=>button.addEventListener('click',()=>{error=button.dataset.filterError;location.hash='samples';apply()}));})();
"""
