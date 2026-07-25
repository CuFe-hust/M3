"""Test TaskRouter route_known and route_sample behaviour. / 测试 TaskRouter。"""

from __future__ import annotations

import pytest

from spacers_agent.routing import ROUTES, TaskRouter, CallBudget
from spacers_agent.routing.schemas import RoutingDecision


def test_route_known_counting_is_single():
    router = TaskRouter()
    d = router.route_known("counting")
    assert d.primary_agent == "counting_agent"
    assert d.execution_mode == "single"
    assert d.requires_tiling is True
    assert d.router_source == "dataset_task"


def test_route_known_change_qa_is_fallback():
    router = TaskRouter()
    d = router.route_known("change_qa")
    assert d.primary_agent == "change_agent"
    assert d.execution_mode == "fallback"
    assert "general_vqa_agent" in d.fallback_agents


def test_route_known_caption_exists():
    router = TaskRouter()
    d = router.route_known("caption")
    assert d.primary_agent == "caption_agent"
    assert d.execution_mode == "single"


def test_route_known_all_tasks_covered():
    """Every task in ROUTES produces a valid RoutingDecision."""
    router = TaskRouter()
    for task in ROUTES:
        d = router.route_known(task)
        assert isinstance(d, RoutingDecision)
        assert d.primary_agent
        assert d.reason_codes


def test_route_known_general_vqa_no_tiling():
    router = TaskRouter()
    d = router.route_known("general_vqa")
    assert not d.requires_tiling


def test_route_unknown_needs_client():
    router = TaskRouter()  # no client
    budget = CallBudget(max_qwen_calls=5)
    with pytest.raises(ValueError, match="injected router client"):
        import asyncio
        asyncio.run(router.route_unknown("test?", budget=budget, sample_id="s1"))


def test_rule_fallback():
    router = TaskRouter()
    d = router._rule_fallback(high_resolution=True)
    assert d.primary_agent == "general_vqa_agent"
    assert d.router_source == "rule_fallback"
    assert d.execution_mode == "single"
    assert "high_resolution" in d.reason_codes
