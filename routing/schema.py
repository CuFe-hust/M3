"""Routing contracts: sample capabilities, route policies, decisions.

路由契约：样本能力、路由策略与路由决策。本模块只依赖 data.schema 的
TaskName 与 agents.schema 的 AgentName；不导入 models，不接收 question。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents.schema import AgentName
from data.schema import TaskName

ExecutionMode = Literal["single", "fallback"]


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
