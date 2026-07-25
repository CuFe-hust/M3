"""Routing decision schemas — new primary/fallback model + legacy compatibility.
路由决策 Schema — 新 primary/fallback 模型 + 旧格式兼容。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

ExpertName = Literal[
    "counting_expert",
    "change_expert",
    "grounding_expert",
    "spatial_expert",
    "general_vqa_expert",
    "caption_expert",
]

AgentName = Literal[
    "counting_agent",
    "change_agent",
    "grounding_agent",
    "spatial_agent",
    "general_vqa_agent",
    "caption_agent",
]

RouterSource = Literal[
    "dataset_task",
    "vrsbench_semantic_rule",
    "router_agent",
    "rule_fallback",
]

ExecutionMode = Literal["single", "fallback"]

# ── legacy name normalization / 旧名规范化 ─────────────────────────────────

EXPERT_TO_AGENT: dict[str, AgentName] = {
    "counting_expert": "counting_agent",
    "change_expert": "change_agent",
    "grounding_expert": "grounding_agent",
    "spatial_expert": "spatial_agent",
    "general_vqa_expert": "general_vqa_agent",
    "caption_expert": "caption_agent",
}

AGENT_TO_EXPERT: dict[AgentName, str] = {v: k for k, v in EXPERT_TO_AGENT.items()}


def normalize_agent_name(raw: str) -> AgentName:
    """Map expert → agent; pass through valid agent names. / expert → agent 映射；有效 agent 名原样通过。"""
    if raw in EXPERT_TO_AGENT:
        return EXPERT_TO_AGENT[raw]
    from typing import get_args
    if raw in get_args(AgentName):
        return raw  # type: ignore[return-value]
    raise ValueError(f"Unknown agent/expert name: {raw!r}")


# ── new RoutingDecision / 新 RoutingDecision ───────────────────────────────


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

    @model_validator(mode="before")
    @classmethod
    def _from_legacy_format(cls, data: Any) -> Any:
        """Convert legacy `experts` list to primary/fallback model.
        将旧 `experts` 列表转换为 primary/fallback 模型。
        """
        if not isinstance(data, dict):
            return data
        if "experts" in data and "primary_agent" not in data:
            experts = data.pop("experts", [])
            if experts:
                first = experts[0].get("name") if isinstance(experts[0], dict) else getattr(experts[0], "name", None)
                fallback = [
                    (e.get("name") if isinstance(e, dict) else getattr(e, "name", None))
                    for e in experts[1:]
                ]
                data["primary_agent"] = normalize_agent_name(str(first)) if first else "general_vqa_agent"
                data["fallback_agents"] = [normalize_agent_name(str(f)) for f in fallback if f]
                data["execution_mode"] = "fallback" if data["fallback_agents"] and len(experts) > 1 else "single"
            if "router_source" not in data:
                data["router_source"] = "rule_fallback"
        return data


# ── legacy ExpertAssignment for backward compat / 向后兼容的旧 ExpertAssignment ──

class ExpertAssignment(BaseModel):
    """Legacy expert assignment — kept for old serialization compatibility.
    旧版专家分配 — 保留用于旧序列化兼容。
    """

    model_config = ConfigDict(extra="forbid")

    name: ExpertName
    weight: float = Field(default=1.0, gt=0.0, le=1.0)
