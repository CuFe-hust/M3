"""Routing contracts: sample capabilities, route policies, decisions, and the
pre-sample task-resolution contract.

路由契约：样本能力、路由策略、路由决策与样本前任务解析契约。本模块只依赖
data.schema 的 TaskName/JsonValue 与 agents.schema 的 AgentName；不导入
models。TaskResolution* 契约供 workflows.TaskResolver 使用——TaskRouter
绝不读取它们，也绝不见 question。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents.schema import AgentName
from data.schema import JsonValue, TaskName

ExecutionMode = Literal["single", "fallback"]

# Where a resolved task came from: an explicit field, a deterministic rule,
# or a model call. 解析出的任务来源：显式字段、确定性规则或模型调用。
ResolutionSource = Literal["explicit", "rule", "model"]


@dataclass(frozen=True)
class SampleCapabilities:
    """Input capabilities of one sample that routing may record. Routing stays
    fully deterministic: capabilities never change the selected policy, they
    only add reason codes (e.g. high_resolution).
    单条样本的输入能力，路由可记录它们。路由保持完全确定性：能力不改变
    所选策略，只增加 reason code（如 high_resolution）。"""

    high_resolution: bool = False


@dataclass(frozen=True)
class RoutePolicy:
    """Fixed mapping from a normalized task to primary/fallback agents and
    generic policy flags. requires_tiling is a policy field — it is never
    decided per dataset.
    规范化 task 到 primary/fallback Agent 与通用策略标志的固定映射。
    requires_tiling 是策略字段——绝不按数据集决定。"""

    task: TaskName
    primary_agent: AgentName
    fallback_agents: tuple[AgentName, ...] = ()
    requires_tiling: bool = False

    @property
    def execution_mode(self) -> ExecutionMode:
        """'fallback' when fallback agents exist, otherwise 'single'.
        存在 fallback Agent 时为 'fallback'，否则为 'single'。"""
        return "fallback" if self.fallback_agents else "single"


class RoutingDecision(BaseModel):
    """Auditable, fully deterministic routing decision.
    可审计、完全确定性的路由决策。"""

    model_config = ConfigDict(extra="forbid")

    task: TaskName
    primary_agent: AgentName
    fallback_agents: list[AgentName] = Field(default_factory=list, max_length=2)
    execution_mode: ExecutionMode = "single"
    requires_tiling: bool = False
    reason_codes: list[str] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_agents(self) -> "RoutingDecision":
        """Enforce consistency between primary, fallback, and execution mode.
        强制 primary、fallback 与 execution mode 的一致性。"""
        if self.primary_agent in self.fallback_agents:
            raise ValueError("primary_agent must not appear in fallback_agents")
        if len(self.fallback_agents) != len(set(self.fallback_agents)):
            raise ValueError("fallback_agents must not contain duplicates")
        if self.execution_mode == "single" and self.fallback_agents:
            raise ValueError("execution_mode='single' requires empty fallback_agents")
        if self.execution_mode == "fallback" and not self.fallback_agents:
            raise ValueError("execution_mode='fallback' requires at least one fallback_agent")
        return self


class TaskResolutionRequest(BaseModel):
    """Structured pre-sample input for task resolution; metadata_hints are
    JSON-safe only — never arbitrary runtime objects. UnifiedSample.task stays
    mandatory: this request exists only for samples that lack it.
    任务解析的结构化样本前输入；metadata_hints 仅限 JSON 安全值——绝非任意
    运行对象。UnifiedSample.task 保持必填：本请求只用于缺失 task 的样本。"""

    model_config = ConfigDict(extra="forbid")

    explicit_task: str | None = None
    question: str = ""
    image_count: int = Field(ge=1)
    metadata_hints: dict[str, JsonValue] = Field(default_factory=dict)


class TaskResolution(BaseModel):
    """Structured task-resolution outcome consumed before the deterministic
    TaskRouter. candidate_tasks[0] is always the selected task, deduplicated
    stably, with at most three candidates.
    确定性 TaskRouter 之前消费的结构化任务解析结果。candidate_tasks[0] 恒为
    选定任务，稳定去重，最多三个候选。"""

    model_config = ConfigDict(extra="forbid")

    task: TaskName
    confidence: float = Field(ge=0.0, le=1.0)
    candidate_tasks: list[TaskName] = Field(min_length=1, max_length=3)
    needs_candidate_fallback: bool = False
    source: ResolutionSource
    reason_codes: list[str] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_candidates(self) -> "TaskResolution":
        """Enforce candidate_tasks[0] == task with no duplicates.
        强制 candidate_tasks[0] == task 且无重复。"""
        if self.candidate_tasks[0] != self.task:
            raise ValueError("candidate_tasks[0] must equal task")
        if len(self.candidate_tasks) != len(set(self.candidate_tasks)):
            raise ValueError("candidate_tasks must not contain duplicates")
        return self
