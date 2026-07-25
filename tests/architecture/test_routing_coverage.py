"""Freeze routing coverage — every TaskName must be routable.
冻结路由覆盖 — 每个 TaskName 必须可路由。
"""

from __future__ import annotations

import pytest

from spacers_agent.routing import ROUTES
from spacers_agent.schemas import TaskName
from spacers_agent.routing.schemas import RoutableTask, AgentName


def _all_task_names() -> set[str]:
    import typing
    return set(typing.get_args(TaskName))


def _routable_tasks() -> set[str]:
    return set(ROUTES.keys())


def test_every_task_name_is_in_routes():
    """Every TaskName must have a corresponding ROUTE entry. / 每个 TaskName 必须有对应 ROUTE。"""
    all_tasks = _all_task_names()
    routable = _routable_tasks()
    missing = all_tasks - routable
    assert not missing, f"TaskName values without ROUTE entry: {sorted(missing)}"


def test_caption_route_exists():
    """caption has a ROUTE entry — caption_agent. / caption 已有 ROUTE。"""
    assert "caption" in ROUTES
    assert ROUTES["caption"] == ("caption_agent",)


def test_routable_tasks_are_subset_of_task_names():
    """All ROUTE keys correspond to valid TaskName values. / 所有 ROUTE key 是有效 TaskName。"""
    routable = _routable_tasks()
    all_tasks = _all_task_names()
    extra = routable - all_tasks
    assert not extra, f"ROUTE keys not in TaskName: {sorted(extra)}"


def test_route_values_are_valid_agent_names():
    """All route values reference valid AgentName values. / 所有 route 值引用有效 AgentName。"""
    import typing
    valid_agents = set(typing.get_args(AgentName))
    for task, agents in ROUTES.items():
        for agent in agents:
            assert agent in valid_agents, f"ROUTES[{task!r}] references unknown agent {agent!r}"


def test_counting_needs_tiling():
    """Counting tasks should generally require tiling. / 计数任务需要 tiling。"""
    from spacers_agent.routing import TaskRouter
    router = TaskRouter()
    for task in ("counting", "fine_grained_counting"):
        decision = router.route_known(task)  # type: ignore[arg-type]
        assert decision.requires_tiling, f"{task} should require tiling"


def test_vqa_does_not_require_tiling():
    """VQA/classification tasks should not require tiling. / VQA 任务不需 tiling。"""
    from spacers_agent.routing import TaskRouter
    router = TaskRouter()
    for task in ("general_vqa", "scene_classification", "multiple_choice_vqa"):
        decision = router.route_known(task)  # type: ignore[arg-type]
        assert not decision.requires_tiling, f"{task} should not require tiling"


def test_single_expert_routes_have_no_fallback():
    """Single-agent routes have empty fallback_agents. / 单 Agent 路由 fallback_agents 为空。"""
    from spacers_agent.routing import TaskRouter
    router = TaskRouter()
    single = {"counting", "change_caption", "grounding", "spatial_relation",
              "scene_classification", "general_vqa", "multiple_choice_vqa", "caption"}
    for task in single:
        decision = router.route_known(task)  # type: ignore[arg-type]
        assert decision.fallback_agents == [], f"{task} should have no fallback"


def test_change_qa_has_fallback():
    """change_qa has fallback to general_vqa_agent. / change_qa 有 general_vqa_agent fallback。"""
    from spacers_agent.routing import TaskRouter
    router = TaskRouter()
    decision = router.route_known("change_qa")
    assert decision.execution_mode == "fallback"
    assert "general_vqa_agent" in decision.fallback_agents
