"""Deprecated compatibility exports for the former expert module.
前专家模块的弃用兼容导出。

All business implementations have moved to:
真实业务实现已移至：
- ``spacers_agent.agents.*``
- ``spacers_agent.schemas``
- ``spacers_agent.agents.base``
"""

from __future__ import annotations

from spacers_agent.agents.base import Agent, AgentContext, AgentExecution, AgentPayload, AgentName
from spacers_agent.agents.registry import AgentRegistry
from spacers_agent.schemas import ExpertResult as _ExpertResult
from spacers_agent.schemas import VisualEvidence as _VisualEvidence


VisualEvidence = _VisualEvidence
ExpertResult = _ExpertResult

__all__ = [
    "Agent",
    "AgentContext",
    "AgentExecution",
    "AgentName",
    "AgentPayload",
    "AgentRegistry",
    "Expert",
    "ExpertContext",
    "ExpertResult",
    "VisualEvidence",
]


class Expert:
    """Deprecated: use ``Agent`` from ``spacers_agent.agents.base`` instead.
    已弃用：请改用 ``spacers_agent.agents.base`` 中的 ``Agent``。
    """


class ExpertContext:
    """Deprecated: use ``AgentContext`` from ``spacers_agent.agents.base`` instead.
    已弃用：请改用 ``spacers_agent.agents.base`` 中的 ``AgentContext``。
    """
