"""Routing decision schemas — new primary/fallback model + legacy compatibility.
路由决策 Schema — 新 primary/fallback 模型 + 旧格式兼容。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spacers_agent.schemas import AgentName

# ── types / 类型 ──────────────────────────────────────────────────────────

RoutableTask = Literal[
    "counting",
    "fine_grained_counting",
    "change_caption",
    "change_qa",
    "grounding",
    "spatial_relation",
    "scene_classification",
    "general_vqa",
    "caption",
    "multiple_choice_vqa",
]

RouterSource = Literal[
    "dataset_task",
    "vrsbench_semantic_rule",
    "router_agent",
    "rule_fallback",
]

ExecutionMode = Literal["single", "fallback"]

# ── routing decision / 路由决策 ────────────────────────────────────────────


class RoutingDecision(BaseModel):
    """Auditable routing decision with explicit primary/fallback agents.
    具有显式 primary/fallback Agent 的可审计路由决策。
    """

    model_config = ConfigDict(extra="forbid")

    task: RoutableTask
    primary_agent: AgentName
    fallback_agents: list[AgentName] = Field(default_factory=list, max_length=2)
    execution_mode: ExecutionMode = "single"
    requires_tiling: bool
    reason_codes: list[str] = Field(min_length=1, max_length=8)
    router_source: RouterSource

    @model_validator(mode="after")
    def validate_agents(self) -> "RoutingDecision":
        """Enforce consistency between primary, fallback, and execution mode.
        强制 primary、fallback 和 execution mode 的一致性。
        """
        # primary not in fallback / primary 不在 fallback 中
        if self.primary_agent in self.fallback_agents:
            raise ValueError("primary_agent must not appear in fallback_agents")

        # no duplicate fallbacks / fallback 不重复
        if len(self.fallback_agents) != len(set(self.fallback_agents)):
            raise ValueError("fallback_agents must not contain duplicates")

        # single mode has no fallbacks / single 模式无 fallback
        if self.execution_mode == "single" and self.fallback_agents:
            raise ValueError("execution_mode='single' requires empty fallback_agents")

        # fallback mode has at least one fallback / fallback 模式至少有一个 fallback
        if self.execution_mode == "fallback" and not self.fallback_agents:
            raise ValueError("execution_mode='fallback' requires at least one fallback_agent")

        return self
