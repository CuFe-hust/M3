"""Rule-first sparse routing, bounded call budgets, and expert-facing results.
规则优先的稀疏路由、受限调用预算和面向专家的结果封装。
"""

from spacers_agent.routing.budget import CallBudget, CallBudgetExceeded, make_budget_guard
from spacers_agent.routing.router import CountingExpert, CountingExpertAnswer, TaskRouter, attach_qwen_budget
from spacers_agent.routing.routes import (
    AGENT_TO_EXPERT,
    EXPERT_TO_AGENT,
    AgentName,
    ExpertAssignment,
    ExpertName,
    ExecutionMode,
    ROUTES,
    RoutableTask,
    RouterSource,
    RoutingDecision,
    needs_tiling,
    normalize_agent_name,
)

__all__ = [
    "AGENT_TO_EXPERT",
    "EXPERT_TO_AGENT",
    "AgentName",
    "CallBudget",
    "CallBudgetExceeded",
    "CountingExpert",
    "CountingExpertAnswer",
    "ExecutionMode",
    "ExpertAssignment",
    "ExpertName",
    "ROUTES",
    "RoutableTask",
    "RouterSource",
    "RoutingDecision",
    "TaskRouter",
    "attach_qwen_budget",
    "make_budget_guard",
    "needs_tiling",
    "normalize_agent_name",
]
