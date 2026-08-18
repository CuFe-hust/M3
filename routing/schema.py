"""Deterministic routing contracts.

确定性路由契约。本模块只依赖 data.schema 的 TaskName 与 agents.schema 的
AgentName；视觉规划器已经决定 task，TaskRouter 只负责选择 Agent。
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
    """Input capabilities that routing may record without changing policy.
    路由可记录的样本输入能力；能力不会改变确定性策略。"""

    high_resolution: bool = False


@dataclass(frozen=True)
class RoutePolicy:
    """Fixed task-to-agent mapping and generic execution flags.
    task 到 Agent 的固定映射与通用执行标志。"""

    task: TaskName
    primary_agent: AgentName
    fallback_agents: tuple[AgentName, ...] = ()
    requires_tiling: bool = False

    @property
    def execution_mode(self) -> ExecutionMode:
        """Return fallback mode only when fallback agents exist.
        仅在存在 fallback Agent 时返回 fallback 模式。"""

        return "fallback" if self.fallback_agents else "single"


class RoutingDecision(BaseModel):
    """Auditable, fully deterministic routing decision.
    可审计且完全确定性的路由决策。"""

    model_config = ConfigDict(extra="forbid")

    task: TaskName
    primary_agent: AgentName
    fallback_agents: list[AgentName] = Field(default_factory=list, max_length=2)
    execution_mode: ExecutionMode = "single"
    requires_tiling: bool = False
    reason_codes: list[str] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_agents(self) -> "RoutingDecision":
        """Enforce consistency between primary, fallback, and mode.
        强制 primary、fallback 与执行模式保持一致。"""

        if self.primary_agent in self.fallback_agents:
            raise ValueError("primary_agent must not appear in fallback_agents")
        if len(self.fallback_agents) != len(set(self.fallback_agents)):
            raise ValueError("fallback_agents must not contain duplicates")
        if self.execution_mode == "single" and self.fallback_agents:
            raise ValueError("execution_mode='single' requires empty fallback_agents")
        if self.execution_mode == "fallback" and not self.fallback_agents:
            raise ValueError("execution_mode='fallback' requires at least one fallback_agent")
        return self
