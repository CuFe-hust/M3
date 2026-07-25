"""Multi-Agent wrappers for remote-sensing task execution.
遥感多任务 Agent 封装。
"""

from spacers_agent.agents.base import (
    AGENT_TO_EXPERT,
    EXPERT_TO_AGENT,
    LEGACY_AGENT_NAME_ALIASES,
    Agent,
    AgentContext,
    AgentExecution,
    AgentName,
    AgentPayload,
    normalize_agent_name,
    validate_agent_execution,
)
from spacers_agent.agents.registry import AgentRegistry

__all__ = [
    "AGENT_TO_EXPERT",
    "EXPERT_TO_AGENT",
    "LEGACY_AGENT_NAME_ALIASES",
    "Agent",
    "AgentContext",
    "AgentExecution",
    "AgentName",
    "AgentPayload",
    "AgentRegistry",
    "normalize_agent_name",
    "validate_agent_execution",
]
