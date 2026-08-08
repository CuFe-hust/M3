"""Run-state and reproducibility contracts for dataset workflows.

数据集工作流的运行状态与可复现契约。纯契约模块：不导入 application，
不构造 Agent，不持有完整 AppSettings。SampleRunStatus / DatasetRunSummary
可持久化；DatasetRunOptions / SampleRunOutcome 为运行时定型对象。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents.base import AgentExecution
from data.schema import TaskName
from routing.schema import RoutingDecision

SampleRunState = Literal[
    "pending", "running", "succeeded", "partial", "failed", "skipped"
]

# The only legal task labels on a sample status: a known task name, or the
# honest sentinel for pre-task draft failures — never a guessed task.
# 样本状态上唯一合法的任务标签：已知任务名，或预 task draft 失败的诚实哨兵
# 'unknown'——绝不猜测任务。
RunTaskName = TaskName | Literal["unknown"]


class SampleRunStatus(BaseModel):
    """Durable machine-readable state for one dataset sample. task is typed
    as RunTaskName: pre-task draft failures record the honest sentinel
    'unknown' instead of pretending to be a known task.
    单个数据集样本的可持久化机器可读状态。task 类型为 RunTaskName：预 task
    的 draft 失败记录诚实的哨兵 'unknown' 而非冒充已知任务。"""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    task: RunTaskName
    state: SampleRunState
    error_code: str | None = None
    error_message: str | None = None
    result_path: Path | None = None
    updated_at: str


class DatasetRunSummary(BaseModel):
    """Aggregate visible outcomes without hiding failed samples; the counts
    always close: total == succeeded + partial + failed + skipped.
    不隐藏失败样本的汇总结果；计数永远闭合：
    total == succeeded + partial + failed + skipped。"""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    dataset: str
    split: str
    task: str
    total: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    partial: int = Field(ge=0)
    failed: int = Field(ge=0)
    skipped: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_closed_accounting(self) -> "DatasetRunSummary":
        """Every selected sample must end in exactly one terminal bucket.
        每个选中样本必须恰好落入一个终态桶。"""
        accounted = self.succeeded + self.partial + self.failed + self.skipped
        if self.total != accounted:
            raise ValueError(
                "summary counts must be closed: "
                "total == succeeded + partial + failed + skipped"
            )
        return self


@dataclass(frozen=True)
class DatasetRunOptions:
    """Typed dataset run options for resume and fresh runs. None values do
    not participate in numeric comparisons. auto_task is the explicit switch
    for the auto-task draft path: auto_task=True requires empty tasks and
    auto_task=False requires at least one task.
    用于 resume 和新运行的定型数据集运行选项。None 值不参与数值比较。
    auto_task 是 auto-task draft 路径的显式开关：auto_task=True 要求 tasks
    为空，auto_task=False 要求 tasks 非空。"""

    dataset: str
    root: Path
    split: str
    tasks: tuple[str, ...]
    run_id: str | None = None
    resume: bool = False
    limit: int | None = None
    start_index: int = 0
    shard_index: int = 0
    shard_count: int = 1
    sample_concurrency: int = 1
    sample_ids: set[str] | None = None
    evaluate: bool = False
    judge_policy: str = "none"
    fail_fast: bool = False
    auto_task: bool = False

    def __post_init__(self) -> None:
        if self.auto_task and self.tasks:
            raise ValueError("auto_task=True requires tasks to be empty")
        if not self.auto_task and not self.tasks:
            raise ValueError("auto_task=False requires at least one task")


@dataclass(frozen=True)
class SampleRunOutcome:
    """All observable outputs from one SampleRunner invocation.
    一次 SampleRunner 调用产生的全部可观察输出。"""

    execution: AgentExecution | None
    status: SampleRunStatus
    routing: RoutingDecision | None
    evaluation: Any | None
    fallback_used: bool
