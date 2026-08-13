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


# ── joint plan -> resolution conversion (doc 15) / 联合计划转 resolution ─


def _joint_plan_dict(*, task: str = "general_vqa", confidence: float = 0.9) -> dict:
    from agents.schema import JointQwenVisualPlan

    return JointQwenVisualPlan.model_validate(
        {
            "version": "joint-qwen-plan-v1",
            "task": task,
            "visual_plan": {
                "version": "first-qwen-plan-v1",
                "execution_family": "direct_vqa",
                "confidence": confidence,
                "roi_plan": {"rois": []},
                "evidence_request": None,
                "reason_codes": ["plan"],
            },
        }
    )


def test_joint_plan_to_resolution_carries_model_task() -> None:
    """The conversion is pure and deterministic: model task authoritative,
    single candidate, model source, confidence and reason codes flow through.
    转换纯且确定：模型 task 权威、单一候选、model 来源，置信度与 reason
    codes 透传。"""
    from routing.schema import joint_plan_to_resolution

    plan = _joint_plan_dict(task="grounding", confidence=0.8)
    resolution = joint_plan_to_resolution(plan)
    assert resolution.task == "grounding"
    assert resolution.confidence == 0.8
    assert resolution.candidate_tasks == ["grounding"]
    assert not resolution.needs_candidate_fallback
    assert resolution.source == "model"
    assert resolution.reason_codes[0] == "plan"
    assert "joint_plan_model_task" in resolution.reason_codes


def test_joint_plan_to_resolution_dedupes_reason_codes() -> None:
    """The injected marker never duplicates an existing reason code.
    注入标记绝不与已有 reason code 重复。"""
    from agents.schema import JointQwenVisualPlan

    from routing.schema import joint_plan_to_resolution

    plan = JointQwenVisualPlan.model_validate(
        {
            "version": "joint-qwen-plan-v1",
            "task": "general_vqa",
            "visual_plan": {
                "version": "first-qwen-plan-v1",
                "execution_family": "direct_vqa",
                "confidence": 0.9,
                "roi_plan": {"rois": []},
                "evidence_request": None,
                "reason_codes": ["joint_plan_model_task"],
            },
        }
    )
    resolution = joint_plan_to_resolution(plan)
    assert resolution.reason_codes == ["joint_plan_model_task"]
