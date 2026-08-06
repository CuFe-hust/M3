"""Rule-first sparse routing and bounded call budgets.
规则优先的稀疏路由与受限调用预算。
"""

from spacers_agent.routing.budget import CallBudget, CallBudgetExceeded, CallBudgetFactory, make_budget_guard
from spacers_agent.routing.policies import ROUTES, needs_tiling
from spacers_agent.routing.router import TaskRouter
from spacers_agent.routing.schemas import ExecutionMode, RoutableTask, RouterSource, RoutingDecision

__all__ = [
    "CallBudget",
    "CallBudgetExceeded",
    "CallBudgetFactory",
    "ExecutionMode",
    "ROUTES",
    "RoutableTask",
    "RouterSource",
    "RoutingDecision",
    "TaskRouter",
    "make_budget_guard",
    "needs_tiling",
]
