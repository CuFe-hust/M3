"""Run-state and reproducibility contracts for dataset workflows.

数据集工作流的运行状态与可复现契约。纯契约模块：不导入 application，
不构造 Agent，不持有完整 AppSettings。SampleRunStatus / DatasetRunSummary
可持久化；DatasetRunOptions / SampleRunOutcome 为运行时定型对象。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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

    @field_validator("result_path", mode="before")
    @classmethod
    def validate_result_path(cls, value: Any) -> Any:
        """The persisted result path must be a plain basename: sample-relative,
        never absolute, drive, UNC, dot-dot, nested, or control-character
        laden. Legacy absolute paths fail validation and resume re-runs the
        sample instead of trusting them. 持久化结果路径必须是纯 basename：
        样本相对，绝不接受绝对、drive、UNC、dot-dot、嵌套或控制字符。旧版
        绝对路径校验失败，resume 将重新执行样本而非信任它。"""
        if value is None:
            return None
        if not isinstance(value, (str, Path)):
            raise ValueError("result_path must be a string or Path")
        text = str(value)
        if not text or text in {".", ".."}:
            raise ValueError("result_path must be a non-empty plain basename")
        if "/" in text or "\\" in text:
            raise ValueError("result_path must be a plain basename without separators")
        if any(ord(character) < 32 for character in text):
            raise ValueError("result_path must not contain control characters")
        if len(text) >= 2 and text[0].isalpha() and text[1] == ":":
            raise ValueError("result_path must not carry a drive prefix")
        return Path(text)


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
    judge_sample_rate: float | None = Field(default=None, ge=0.0, le=1.0)

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
    for the auto-task draft path: auto_task=True requires tasks to be empty.
    tasks=None is the adapter-default mode: run every adapter.supported_tasks
    with no TaskResolver call. judge_sample_rate (0..1) deterministically
    samples judge participation from the run/sample identity and is persisted
    in the summary so resume is identical.
    用于 resume 和新运行的定型数据集运行选项。None 值不参与数值比较。
    auto_task 是 auto-task draft 路径的显式开关：auto_task=True 要求 tasks
    为空。tasks=None 是 adapter 默认模式：运行全部 adapter.supported_tasks，
    不调用 TaskResolver。judge_sample_rate（0..1）按 run/sample 身份确定性
    抽样 judge 参与，并持久化在 summary 中使 resume 一致。"""

    dataset: str
    root: Path
    split: str
    tasks: tuple[str, ...] | None = None
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
    judge_sample_rate: float | None = None
    render_errors: bool = False
    fail_fast: bool = False
    auto_task: bool = False

    def __post_init__(self) -> None:
        if self.auto_task and self.tasks is None:
            raise ValueError("auto_task=True requires tasks=()")
        if self.auto_task and self.tasks:
            raise ValueError("auto_task=True requires tasks to be empty")
        if not self.auto_task and self.tasks == ():
            raise ValueError(
                "auto_task=False requires at least one task or tasks=None "
                "for adapter defaults"
            )
        if self.judge_sample_rate is not None and not (
            0.0 <= self.judge_sample_rate <= 1.0
        ):
            raise ValueError("judge_sample_rate must be within [0.0, 1.0]")


class RunRequest(BaseModel):
    """The concrete user/runtime invocation for one dataset run, persisted as
    ``runs/<run_id>/run_request.json``. This is not a replacement for the
    manifest: manifest.json carries run identity/reproducibility metadata,
    config.snapshot.json carries the application configuration snapshot, and
    run_request.json carries the actual invocation — including the real
    dataset root and the original judge policy/rate — so resume-run can
    reconstruct DatasetRunOptions without guessing.
    单个数据集运行的具体用户/运行时调用，持久化为
    ``runs/<run_id>/run_request.json``。它不是 manifest 的替代品：
    manifest.json 承载运行身份/可复现元数据，config.snapshot.json 承载应用
    配置快照，run_request.json 承载实际调用——包括真实数据集根与原始
    judge 策略/率——使 resume-run 无需猜测即可重建 DatasetRunOptions。

    dataset_root preserves the host path form (POSIX separators), consistent
    with the existing host-path-preserving snapshot decision; it is never
    claimed to be machine-independent. dataset_root 保留主机路径形式（正斜杠
    分隔），与既有 host-path-preserving 快照决策一致；绝不声称与机器无关。
    """

    model_config = ConfigDict(extra="forbid")

    dataset: str = Field(min_length=1)
    dataset_root: str = Field(min_length=1)
    split: str = Field(min_length=1)
    task_mode: Literal["explicit", "adapter_default", "auto"]
    tasks: list[str] = Field(default_factory=list)
    auto_task: bool = False
    sample_ids: list[str] | None = None
    limit: int | None = Field(default=None, ge=0)
    start_index: int = Field(default=0, ge=0)
    shard_index: int = Field(default=0, ge=0)
    shard_count: int = Field(default=1, ge=1)
    sample_concurrency: int = Field(default=1, ge=1)
    evaluate: bool = True
    judge_policy: str = "none"
    judge_sample_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    render_errors: bool = False
    fail_fast: bool = False
    # Single-image invocation identity (count-image only; dataset runs leave
    # these None). 单图调用身份（仅 count-image；数据集运行保持 None）。
    command: str | None = None
    image_identity: str | None = None
    question: str | None = None
    sample_id: str | None = None

    @model_validator(mode="after")
    def validate_invocation(self) -> "RunRequest":
        """Enforce the task-mode/tasks consistency and shard bounds; invalid
        persisted invocations must fail stably instead of being guessed.
        强制 task-mode/tasks 一致性与分片边界；非法持久化调用必须稳定失败
        而非被猜测。"""
        if self.shard_index >= self.shard_count:
            raise ValueError("shard_index must be within [0, shard_count)")
        if self.task_mode == "auto":
            if not self.auto_task or self.tasks:
                raise ValueError("auto task mode requires auto_task and empty tasks")
        elif self.task_mode == "explicit":
            if self.auto_task or not self.tasks:
                raise ValueError("explicit task mode requires tasks and no auto_task")
        else:  # adapter_default
            if self.auto_task or self.tasks:
                raise ValueError("adapter_default task mode requires no tasks or auto_task")
        return self


@dataclass(frozen=True)
class SampleRunOutcome:
    """All observable outputs from one SampleRunner invocation.
    一次 SampleRunner 调用产生的全部可观察输出。"""

    execution: AgentExecution | None
    status: SampleRunStatus
    routing: RoutingDecision | None
    evaluation: Any | None
    fallback_used: bool
