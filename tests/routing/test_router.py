"""Contract tests for the deterministic synchronous TaskRouter.

确定性同步 TaskRouter 契约测试：route 是同步方法、每个 task 输出正确决策、
capabilities 只影响 reason code、未知 task 显式失败、不接受 question。
"""

from __future__ import annotations

import inspect
from typing import get_args

import pytest

from data.schema import TaskName
from routing.router import TaskRouter
from routing.schema import RoutingDecision, SampleCapabilities

_ALL_TASKS = get_args(TaskName)


def test_route_is_synchronous() -> None:
    """TaskRouter.route must be a plain synchronous method.
    TaskRouter.route 必须是普通同步方法。"""
    assert not inspect.iscoroutinefunction(TaskRouter.route)


def test_route_accepts_no_question_parameter() -> None:
    """The router must never receive or inspect the question.
    路由器绝不接收或检查问题文本。"""
    signature = inspect.signature(TaskRouter.route)
    assert "question" not in signature.parameters


def test_route_returns_deterministic_decisions() -> None:
    router = TaskRouter()
    for task in _ALL_TASKS:
        first = router.route(task)
        second = router.route(task)
        assert isinstance(first, RoutingDecision)
        assert first == second  # fully deterministic / 完全确定
        assert first.task == task
        assert first.reason_codes[0] == f"task_{task}"


def test_route_matches_policy_table() -> None:
    from routing.policies import policy_for

    router = TaskRouter()
    for task in _ALL_TASKS:
        decision = router.route(task)
        policy = policy_for(task)
        assert decision.primary_agent == policy.primary_agent
        assert decision.fallback_agents == list(policy.fallback_agents)
        assert decision.execution_mode == policy.execution_mode
        assert decision.requires_tiling == policy.requires_tiling


def test_route_records_high_resolution_capability() -> None:
    router = TaskRouter()
    decision = router.route(
        "general_vqa",
        capabilities=SampleCapabilities(high_resolution=True),
    )
    assert "high_resolution" in decision.reason_codes
    assert decision.requires_tiling is False


def test_route_capabilities_never_change_the_policy() -> None:
    router = TaskRouter()
    baseline = router.route("counting")
    with_capabilities = router.route(
        "counting", capabilities=SampleCapabilities(high_resolution=True)
    )
    assert with_capabilities.primary_agent == baseline.primary_agent
    assert with_capabilities.requires_tiling == baseline.requires_tiling
    assert with_capabilities.reason_codes == ["task_counting", "high_resolution"]


def test_route_unknown_task_fails_explicitly() -> None:
    router = TaskRouter()
    with pytest.raises(KeyError, match="Unknown routable task"):
        router.route("no-such-task")


def test_sample_capabilities_is_frozen() -> None:
    import dataclasses

    capabilities = SampleCapabilities(high_resolution=True)
    assert dataclasses.is_dataclass(capabilities)
    assert capabilities.__dataclass_params__.frozen
    assert capabilities.high_resolution is True
