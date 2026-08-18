"""Common agent contracts: results, protocol, context, registry, errors.
Agent 通用契约：结果、协议、上下文、注册表与错误。

This package only re-exports stable symbols.
本包只重导出稳定符号。
"""

from agents.base import (
    Agent,
    AgentContext,
    AgentExecution,
    CallBudget,
    validate_agent_execution,
)
from agents.errors import (
    AgentExecutionError,
    AgentTaskMismatchError,
    CountingBackendUnavailableError,
    DetectorClassMapMismatchError,
    DetectorInferenceError,
    DetectorTaskMismatchError,
    DetectorWeightsHashMismatchError,
    DetectorWeightsMissingError,
    DuplicateAgentError,
    OptionalDependencyMissingError,
    UnsupportedAgentError,
)
from agents.registry import AgentRegistry
from agents.schema import (
    AgentName,
    AgentResult,
    MaterializedVisualView,
    RegionRequest,
    VisualEvidence,
    VisualTaskPlan,
)

__all__ = [
    "Agent",
    "AgentContext",
    "AgentExecution",
    "AgentExecutionError",
    "AgentName",
    "AgentRegistry",
    "AgentResult",
    "AgentTaskMismatchError",
    "CallBudget",
    "CountingBackendUnavailableError",
    "DetectorClassMapMismatchError",
    "DetectorInferenceError",
    "DetectorTaskMismatchError",
    "DetectorWeightsHashMismatchError",
    "DetectorWeightsMissingError",
    "DuplicateAgentError",
    "OptionalDependencyMissingError",
    "UnsupportedAgentError",
    "VisualEvidence",
    "VisualTaskPlan",
    "MaterializedVisualView",
    "RegionRequest",
    "validate_agent_execution",
]
