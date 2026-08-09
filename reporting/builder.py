"""Report builder: turn a run directory into the read-only Report model.

报告构建器：把运行目录转换为只读 Report 模型。全部输入来自已持久化产物；
损坏的可选产物降级为空值。确定性：相同输入恒产生相同输出（无时间戳）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.metrics.aggregate import (
    aggregate_counting,
    aggregate_grounding,
    aggregate_vqa,
)
from evaluation.metrics.vqa import aggregate_vqa_semantic_judge
from evaluation.records import VQADeterministicMetrics
from reporting.adapters import (
    iter_current_predictions,
    load_evaluation,
    load_payload,
    load_sample,
    load_status,
    load_trace,
    prediction_text,
    read_json,
    safe_result_path,
    sample_dir_for_row,
)
from reporting.schema import Report, ReportSample, TaskSummary

# Metric families that aggregate purely offline; caption corpus metrics need
# the optional pycocoevalcap and stay out of the report layer.
# 可纯离线聚合的指标族；caption 语料级指标需要可选 pycocoevalcap，留在报告
# 层之外。
_AGGREGATORS = {
    "general_vqa": aggregate_vqa,
    "counting": aggregate_counting,
    "grounding": aggregate_grounding,
}


def build_report(run_dir: Path) -> Report:
    """Build the report from the execution index and best-effort sample
    artifacts. 从执行索引与尽力而为的样本产物构建报告。"""

    samples: list[ReportSample] = []
    for row in iter_current_predictions(run_dir):
        samples.append(_build_sample(run_dir, row))
    samples.sort(key=lambda item: (item.run_task, item.sample_id))
    tasks = [_build_task_summary(run_task, [s for s in samples if s.run_task == run_task])
             for run_task in sorted({s.run_task for s in samples})]
    return Report(
        run_id=run_dir.name,
        dataset=_find_dataset(run_dir),
        total=len(samples),
        succeeded=sum(1 for s in samples if s.state == "succeeded"),
        partial=sum(1 for s in samples if s.state == "partial"),
        failed=sum(1 for s in samples if s.state == "failed"),
        skipped=sum(1 for s in samples if s.state == "skipped"),
        samples=samples,
        tasks=tasks,
    )


def _build_sample(run_dir: Path, row: dict[str, Any]) -> ReportSample:
    """Enrich one execution-index row with best-effort sample artifacts.
    用尽力而为的样本产物增强一条执行索引行。"""

    sample_dir = sample_dir_for_row(run_dir, row)
    status = load_status(sample_dir) if sample_dir is not None else None
    sample = load_sample(sample_dir) if sample_dir is not None else None
    trace = load_trace(sample_dir) if sample_dir is not None else None
    task = str(row.get("task", ""))
    evaluation = load_evaluation(sample_dir, task) if sample_dir is not None else None
    payload = load_payload(sample_dir, task) if sample_dir is not None else None
    judge_status = _judge_status(evaluation, trace)
    return ReportSample(
        sample_id=str(row.get("sample_id", "")),
        run_task=str(row.get("run_task", "")),
        task=task,
        state=str(row.get("status", "")),
        error_code=status.error_code if status is not None else None,
        result_path=safe_result_path(run_dir, row.get("result_path")),
        updated_at=row.get("updated_at") if isinstance(row.get("updated_at"), str) else None,
        question=sample.question if sample is not None else None,
        prediction=prediction_text(payload),
        resolved_task=_trace_str(trace, "resolved_task"),
        execution_agent=_trace_str(trace, "execution_agent"),
        fallback_used=bool(trace.get("fallback_used")) if trace is not None else False,
        judge_status=judge_status,
        inference_seconds=_trace_float(trace, "inference_seconds"),
        evaluation=evaluation,
    )


def _judge_status(evaluation: Any, trace: dict[str, Any] | None) -> str:
    if evaluation is not None and evaluation.judge_status:
        return evaluation.judge_status
    if trace is not None:
        value = trace.get("judge_status")
        if isinstance(value, str) and value:
            return value
    return "not_requested"


def _trace_str(trace: dict[str, Any] | None, key: str) -> str | None:
    value = trace.get(key) if trace is not None else None
    return value if isinstance(value, str) and value else None


def _trace_float(trace: dict[str, Any] | None, key: str) -> float | None:
    value = trace.get(key) if trace is not None else None
    return value if isinstance(value, (int, float)) else None


def _build_task_summary(run_task: str, samples: list[ReportSample]) -> TaskSummary:
    """Aggregate one run-task namespace: state counts, fallback, agent usage,
    judge status, and offline deterministic metrics. 聚合一个 run-task 命名
    空间：状态计数、fallback、Agent 使用、judge 状态与离线确定性指标。"""

    total = len(samples)
    fallback_count = sum(1 for s in samples if s.fallback_used)
    agent_usage: dict[str, int] = {}
    judge_counts: dict[str, int] = {}
    for sample in samples:
        if sample.execution_agent:
            agent_usage[sample.execution_agent] = agent_usage.get(sample.execution_agent, 0) + 1
        judge_counts[sample.judge_status] = judge_counts.get(sample.judge_status, 0) + 1
    return TaskSummary(
        run_task=run_task,
        total=total,
        succeeded=sum(1 for s in samples if s.state == "succeeded"),
        partial=sum(1 for s in samples if s.state == "partial"),
        failed=sum(1 for s in samples if s.state == "failed"),
        skipped=sum(1 for s in samples if s.state == "skipped"),
        fallback_count=fallback_count,
        fallback_rate=fallback_count / total if total else 0.0,
        agent_usage=agent_usage,
        judge_status_counts=judge_counts,
        metrics=_aggregate_metrics(samples),
        judge_metrics=_aggregate_judge_metrics(samples),
    )


def _aggregate_metrics(samples: list[ReportSample]) -> dict[str, Any]:
    """Aggregate deterministic metrics per canonical family; records without
    deterministic metrics and caption corpus metrics are intentionally
    excluded. 按 canonical 族聚合确定性指标；无确定性指标的记录与 caption
    语料级指标有意排除。"""

    records = [
        sample.evaluation
        for sample in samples
        if sample.evaluation is not None and sample.evaluation.deterministic_metrics is not None
    ]
    aggregated: dict[str, Any] = {}
    caption_count = 0
    for record in records:
        family = record.task
        if family == "caption":
            caption_count += 1
            continue
        aggregator = _AGGREGATORS.get(family)
        if aggregator is None:
            continue
        bucket = aggregated.setdefault(family, [])
        bucket.append(record)
    result: dict[str, Any] = {}
    for family, bucket in aggregated.items():
        result[family] = _AGGREGATORS[family](bucket)
    if caption_count:
        result["caption"] = {"record_count": caption_count}
    return result


def _aggregate_judge_metrics(samples: list[ReportSample]) -> dict[str, Any]:
    """Aggregate persisted VQA semantic Judge quality without model calls;
    counting Judge remains represented only by status and audit outputs.
    仅从持久化记录聚合 VQA 语义 Judge 质量，不调用模型；counting Judge 仍只由
    状态数量与审计产物表示。"""

    records = [
        sample.evaluation
        for sample in samples
        if sample.evaluation is not None
        and sample.evaluation.task == "general_vqa"
        and isinstance(
            sample.evaluation.deterministic_metrics,
            VQADeterministicMetrics,
        )
    ]
    if not records:
        return {}
    return {
        "vqa_semantic_equivalence": aggregate_vqa_semantic_judge(records),
    }


def _find_dataset(run_dir: Path) -> str | None:
    """Best-effort dataset name from the first task probe or summary.
    从首个 task probe 或汇总中尽力获取数据集名。"""

    tasks_dir = run_dir / "tasks"
    if not tasks_dir.is_dir():
        return None
    try:
        task_dirs = sorted(
            path for path in tasks_dir.iterdir() if path.is_dir()
        )
    except OSError:
        return None
    for task_dir in task_dirs:
        probe = _read_probe(task_dir / "dataset_probe.json")
        if probe is not None:
            return probe
        summary = _read_probe(task_dir / "dataset_summary.json")
        if summary is not None:
            return summary
    return None


def _read_probe(path: Path) -> str | None:
    raw = read_json(path)
    if not isinstance(raw, dict):
        return None
    value = raw.get("dataset")
    return value if isinstance(value, str) and value else None
