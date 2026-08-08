"""Report records: per-sample rows, task summaries, and the run report.

报告记录：逐样本行、任务汇总与运行报告。纯类型契约：不读盘、不调用模型、
不重新计算结果。judge 状态只旁路记录，指标字段来自已持久化的
EvaluationRecord。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from evaluation.records import EvaluationRecord


class ReportSample(BaseModel):
    """One current-state sample row derived from the execution index plus
    best-effort artifact enrichment. 由执行索引与尽力而为的产物增强推导的
    单条当前状态样本行。"""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    run_task: str
    task: str  # execution task / 执行任务
    state: str
    error_code: str | None = None
    result_path: str | None = None
    updated_at: str | None = None
    question: str | None = None
    prediction: str | None = None
    resolved_task: str | None = None
    execution_agent: str | None = None
    fallback_used: bool = False
    judge_status: str = "not_requested"
    inference_seconds: float | None = None
    evaluation: EvaluationRecord | None = None


class TaskSummary(BaseModel):
    """Aggregate counts and quality signals for one run-task namespace.
    单个 run-task 命名空间的聚合计数与质量信号。"""

    model_config = ConfigDict(extra="forbid")

    run_task: str
    total: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    partial: int = Field(ge=0)
    failed: int = Field(ge=0)
    skipped: int = Field(ge=0)
    fallback_count: int = Field(ge=0)
    fallback_rate: float = Field(ge=0.0, le=1.0)
    agent_usage: dict[str, int] = Field(default_factory=dict)
    judge_status_counts: dict[str, int] = Field(default_factory=dict)
    # Deterministic metric aggregates keyed by canonical metric family; caption
    # intentionally carries record counts only (corpus metrics need the
    # optional pycocoevalcap and stay out of the offline report layer).
    # 按 canonical 指标族聚合的确定性指标；caption 只携带记录计数（语料级
    # 指标需要可选 pycocoevalcap，留在离线报告层之外）。
    metrics: dict[str, Any] = Field(default_factory=dict)


class Report(BaseModel):
    """The full read-only report of one run directory. 单个运行目录的完整
    只读报告。"""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    dataset: str | None = None
    total: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    partial: int = Field(ge=0)
    failed: int = Field(ge=0)
    skipped: int = Field(ge=0)
    samples: list[ReportSample] = Field(default_factory=list)
    tasks: list[TaskSummary] = Field(default_factory=list)
