"""Read-only reporting layer: report schema, adapters, builder, HTML, and
exporters. 只读报告层：报告 schema、适配器、构建器、HTML 与导出器。"""

from reporting.builder import build_report
from reporting.exporters import write_csv, write_json
from reporting.html import build_html
from reporting.schema import Report, ReportSample, TaskSummary
from reporting.visualization import render_counting_overlay

__all__ = [
    "Report",
    "ReportSample",
    "TaskSummary",
    "build_html",
    "build_report",
    "render_counting_overlay",
    "write_csv",
    "write_json",
]
