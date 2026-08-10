"""Cross-record metric aggregation for homogeneous evaluation records.

同质评估记录的跨记录指标汇总。纯本地计算，无网络副作用；仅对类型与
task 明确已知的记录做汇总，绝不混用不同任务的记录。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from evaluation.metrics.caption import aggregate_caption
from evaluation.metrics.counting import aggregate_counting
from evaluation.metrics.grounding import aggregate_grounding
from evaluation.metrics.vqa import aggregate_vqa
from evaluation.records import EvaluationRecord, VQAEvaluationRecord


def aggregate(records: Sequence[Any]) -> dict[str, Any]:
    """Route a homogeneous record set to its task-specific aggregate.
    将同质记录集路由到任务专属汇总。"""

    loaded = list(records)
    if not loaded:
        raise ValueError("aggregate requires at least one record")
    tasks = {_task_of(record) for record in loaded}
    if len(tasks) != 1:
        raise ValueError("One result file must contain one task type.")
    task = tasks.pop()
    if task == "counting":
        return aggregate_counting(loaded)
    if task == "general_vqa":
        return aggregate_vqa(loaded)
    if task == "grounding":
        return aggregate_grounding(loaded)
    if task == "caption":
        return aggregate_caption(loaded)
    raise ValueError(f"aggregation is not defined for task {task!r}")


def _task_of(record: Any) -> str:
    if isinstance(record, VQAEvaluationRecord):
        return "general_vqa"
    if isinstance(record, EvaluationRecord):
        return record.task
    raise ValueError(
        "records must be homogeneous EvaluationRecord or VQAEvaluationRecord"
    )
