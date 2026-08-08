"""Offline single-page HTML report rendering.

离线单页 HTML 报告渲染。完全离线：无 CDN、无外部资源、无 Base64 图像；
所有用户/模型文本经 html.escape；只输出稳定 code 与转义文本，绝不泄漏
API key、checkpoint 路径或原始异常。确定性输出（无时间戳）。
"""

from __future__ import annotations

import html
from typing import Any

from reporting.schema import Report, ReportSample, TaskSummary


def build_html(report: Report) -> str:
    """Render the report as one self-contained HTML document.
    将报告渲染为单个自包含 HTML 文档。"""

    sections = [
        "<!DOCTYPE html>",
        '<html lang="zh"><head><meta charset="utf-8">',
        "<title>M3 Run Report</title>",
        "<style>",
        _CSS,
        "</style></head><body>",
        f"<h1>Run Report: {_esc(report.run_id)}</h1>",
        f"<p>dataset: {_esc(report.dataset or 'unknown')}</p>",
        _totals_table(report),
        *[_task_section(task) for task in report.tasks],
        _samples_table(report),
        "</body></html>",
    ]
    return "\n".join(sections)


def _totals_table(report: Report) -> str:
    rows = [
        ("total", report.total),
        ("succeeded", report.succeeded),
        ("partial", report.partial),
        ("failed", report.failed),
        ("skipped", report.skipped),
    ]
    cells = "".join(f"<td>{_esc(str(value))}</td>" for _, value in rows)
    labels = "".join(f"<th>{_esc(label)}</th>" for label, _ in rows)
    return f"<h2>Totals</h2><table><tr>{labels}</tr><tr>{cells}</tr></table>"


def _task_section(task: TaskSummary) -> str:
    summary = (
        f"<h2>Task: {_esc(task.run_task)}</h2>"
        f"<p>samples={task.total} succeeded={task.succeeded} "
        f"partial={task.partial} failed={task.failed} skipped={task.skipped} "
        f"fallback={task.fallback_count} ({task.fallback_rate:.2%})</p>"
    )
    agent_cells = "".join(
        f"<td>{_esc(name)}</td><td>{count}</td>"
        for name, count in sorted(task.agent_usage.items())
    )
    agent_table = (
        "<h3>Agent usage</h3><table><tr><th>agent</th><th>samples</th></tr>"
        f"<tr>{agent_cells}</tr></table>"
        if agent_cells
        else "<h3>Agent usage</h3><p>none</p>"
    )
    judge_cells = "".join(
        f"<td>{_esc(status)}</td><td>{count}</td>"
        for status, count in sorted(task.judge_status_counts.items())
    )
    judge_table = (
        "<h3>Judge status</h3><table><tr><th>status</th><th>samples</th></tr>"
        f"<tr>{judge_cells}</tr></table>"
        if judge_cells
        else ""
    )
    metrics_html = _metrics_html(task.metrics)
    return summary + agent_table + judge_table + metrics_html


def _metrics_html(metrics: dict[str, Any]) -> str:
    if not metrics:
        return ""
    blocks = []
    for family in sorted(metrics):
        payload = metrics[family]
        if not isinstance(payload, dict):
            continue
        items = "".join(
            f"<td>{_esc(str(key))}</td><td>{_esc(_format_number(value))}</td>"
            for key, value in sorted(payload.items())
        )
        blocks.append(
            f"<h3>Metrics: {_esc(family)}</h3>"
            f"<table><tr>{items}</tr></table>"
        )
    return "".join(blocks)


def _samples_table(report: Report) -> str:
    header = (
        "<tr><th>run_task</th><th>sample_id</th><th>task</th><th>state</th>"
        "<th>fallback</th><th>judge</th><th>agent</th><th>question</th>"
        "<th>prediction</th><th>error_code</th></tr>"
    )
    body = "".join(_sample_row(sample) for sample in report.samples)
    return f"<h2>Samples</h2><table>{header}{body}</table>"


def _sample_row(sample: ReportSample) -> str:
    metric_text = _sample_metric_text(sample)
    return (
        "<tr>"
        f"<td>{_esc(sample.run_task)}</td>"
        f"<td>{_esc(sample.sample_id)}</td>"
        f"<td>{_esc(sample.task)}</td>"
        f"<td>{_esc(sample.state)}</td>"
        f"<td>{'yes' if sample.fallback_used else 'no'}</td>"
        f"<td>{_esc(sample.judge_status)}</td>"
        f"<td>{_esc(sample.execution_agent or '')}</td>"
        f"<td>{_esc(sample.question or '')}</td>"
        f"<td>{_esc(sample.prediction or '')}</td>"
        f"<td>{_esc(sample.error_code or '')}</td>"
        "</tr>"
    ) + (f"<tr><td colspan=\"10\">{_esc(metric_text)}</td></tr>" if metric_text else "")


def _sample_metric_text(sample: ReportSample) -> str:
    evaluation = sample.evaluation
    if evaluation is None or evaluation.deterministic_metrics is None:
        return ""
    metrics = evaluation.deterministic_metrics
    if evaluation.task == "general_vqa":
        return f"exact_match={getattr(metrics, 'exact_match', None)}"
    if evaluation.task == "counting":
        return (
            f"predicted={getattr(metrics, 'predicted_count', None)} "
            f"gold={getattr(metrics, 'gold_count', None)} "
            f"exact_match={getattr(metrics, 'exact_match', None)}"
        )
    if evaluation.task == "grounding":
        return (
            f"iou={getattr(metrics, 'iou', None)} "
            f"iou_at_0_5={getattr(metrics, 'iou_at_0_5', None)}"
        )
    if evaluation.task == "caption":
        return "caption record"
    return ""


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _esc(value: str) -> str:
    """HTML-escape user/model text; never trust raw model output.
    HTML 转义用户/模型文本；绝不信任原始模型输出。"""

    return html.escape(str(value), quote=True)


_CSS = """
body { font-family: system-ui, sans-serif; margin: 1.5rem; color: #1a1a1a; }
table { border-collapse: collapse; margin-bottom: 1.5rem; }
th, td { border: 1px solid #ccc; padding: 0.3rem 0.6rem; text-align: left; }
h1 { font-size: 1.4rem; } h2 { font-size: 1.1rem; } h3 { font-size: 0.95rem; }
"""
