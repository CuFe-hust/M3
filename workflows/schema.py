"""Run-state and reproducibility contracts for dataset workflows.

数据集工作流的运行状态与可复现契约。纯契约模块：不导入 application，
不构造 Agent，不持有完整 AppSettings。SampleRunStatus / DatasetRunSummary
可持久化；DatasetRunOptions / SampleRunOutcome 为运行时定型对象。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agents.base import AgentExecution
from data.schema import TaskName
from routing.schema import RoutingDecision

SampleRunState = Literal[
    "pending", "running", "succeeded", "partial", "failed", "skipped"
]


class SampleRunStatus(BaseModel):
    """Durable machine-readable state for one dataset sample.
    单个数据集样本的可持久化机器可读状态。"""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    task: TaskName
    state: SampleRunState
    error_code: str | None = None
    error_message: str | None = None
    result_path: Path | None = None
    updated_at: str


class DatasetRunSummary(BaseModel):
    """Aggregate visible outcomes without hiding failed samples.
    不隐藏失败样本的汇总结果。"""

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


@dataclass(frozen=True)
class DatasetRunOptions:
    """Typed dataset run options for resume and fresh runs. None values do
    not participate in numeric comparisons.
    用于 resume 和新运行的定型数据集运行选项。None 值不参与数值比较。"""

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


@dataclass(frozen=True)
class SampleRunOutcome:
    """All observable outputs from one SampleRunner invocation.
    一次 SampleRunner 调用产生的全部可观察输出。"""

    execution: AgentExecution | None
    status: SampleRunStatus
    routing: RoutingDecision | None
    evaluation: Any | None
    fallback_used: bool
