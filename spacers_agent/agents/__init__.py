"""Multi-Agent wrappers for remote-sensing task execution.
遥感多任务 Agent 封装。
"""

from spacers_agent.agents.base import (
    Agent,
    AgentContext,
    AgentExecution,
    AgentName,
    AgentPayload,
    validate_agent_execution,
)
from spacers_agent.agents.registry import AgentRegistry

__all__ = [
    "Agent",
    "AgentContext",
    "AgentExecution",
    "AgentName",
    "AgentPayload",
    "AgentRegistry",
    "validate_agent_execution",
]
