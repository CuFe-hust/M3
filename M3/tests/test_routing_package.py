"""Test routing package public exports. / 测试 routing 包公开导出。"""

from __future__ import annotations

import pytest

from spacers_agent.routing import (
    ROUTES,
    CallBudget,
    CallBudgetExceeded,
    RoutingDecision,
    TaskRouter,
)


def test_routes_is_importable():
    """ROUTES dict uses AgentName values now. / ROUTES 字典现在使用 AgentName 值。"""
    assert "counting" in ROUTES
    assert ROUTES["counting"] == ("counting_agent",)


def test_task_router_route_known():
    """route_known returns new RoutingDecision. / route_known 返回新 RoutingDecision。"""
    router = TaskRouter()
    decision = router.route_known("counting")
    assert decision.task == "counting"
    assert decision.primary_agent == "counting_agent"
    assert decision.fallback_agents == []
    assert decision.requires_tiling is True


def test_task_router_route_known_no_tiling():
    """scene_classification does NOT require tiling. / scene_classification 不需 tiling。"""
    router = TaskRouter()
    decision = router.route_known("scene_classification")
    assert decision.requires_tiling is False


def test_call_budget_reserve_qwen():
    """reserve_qwen increments counter. / reserve_qwen 递增计数器。"""
    budget = CallBudget(max_qwen_calls=3)
    budget.reserve_qwen()
    assert budget.qwen_calls_used == 1


def test_call_budget_exceeded():
    """Exceeding budget raises CallBudgetExceeded. / 超出预算抛出 CallBudgetExceeded。"""
    budget = CallBudget(max_qwen_calls=1)
    budget.reserve_qwen()
    with pytest.raises(CallBudgetExceeded):
        budget.reserve_qwen()


def test_routing_decision_validation():
    """RoutingDecision validates new fields. / RoutingDecision 校验新字段。"""
    decision = RoutingDecision(
        task="counting", primary_agent="counting_agent",
        execution_mode="single", requires_tiling=True,
        reason_codes=["task_counting"], router_source="dataset_task",
    )
    assert decision.task == "counting"
    assert decision.router_source == "dataset_task"
