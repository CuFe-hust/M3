"""Contract tests for the routing layer as a whole.

路由层整体契约测试：RoutingDecision 严格校验与 JSON 序列化、routing 源码
不含禁用概念（VRSBench / question_type / router_client / route_unknown）、
每个 TaskName 都有受测 policy 且路由不导入 models。
"""

from __future__ import annotations

from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from data.schema import TaskName
from routing.policies import policy_for
from routing.router import TaskRouter
from routing.schema import RoutingDecision

REPO_ROOT = Path(__file__).resolve().parents[2]
_ALL_TASKS = get_args(TaskName)


def _decision(**overrides) -> RoutingDecision:
    values = dict(
        task="general_vqa",
        primary_agent="general_vqa_agent",
        fallback_agents=[],
        execution_mode="single",
        requires_tiling=False,
        reason_codes=["task_general_vqa"],
    )
    values.update(overrides)
    return RoutingDecision(**values)


def test_decision_accepts_valid_single_route() -> None:
    decision = _decision()
    assert decision.execution_mode == "single"
    assert decision.reason_codes == ["task_general_vqa"]


def test_decision_accepts_valid_fallback_route() -> None:
    decision = _decision(
        task="change_qa",
        primary_agent="change_agent",
        fallback_agents=["general_vqa_agent"],
        execution_mode="fallback",
        requires_tiling=True,
        reason_codes=["task_change_qa"],
    )
    assert decision.fallback_agents == ["general_vqa_agent"]


def test_decision_rejects_inconsistent_agent_lists() -> None:
    with pytest.raises(ValidationError, match="primary_agent must not appear"):
        _decision(primary_agent="general_vqa_agent",
                  fallback_agents=["general_vqa_agent"], execution_mode="fallback")
    with pytest.raises(ValidationError, match="duplicates"):
        _decision(primary_agent="change_agent",
                  fallback_agents=["general_vqa_agent", "general_vqa_agent"],
                  execution_mode="fallback")
    with pytest.raises(ValidationError, match="requires empty"):
        _decision(primary_agent="change_agent",
                  fallback_agents=["general_vqa_agent"], execution_mode="single")
    with pytest.raises(ValidationError, match="at least one"):
        _decision(execution_mode="fallback")


def test_decision_is_json_serializable() -> None:
    import json

    decision = _decision(
        task="grounding",
        primary_agent="grounding_agent",
        requires_tiling=True,
        reason_codes=["task_grounding"],
    )
    payload = json.loads(decision.model_dump_json())
    assert payload["task"] == "grounding"
    assert payload["requires_tiling"] is True


def test_routing_source_files_have_no_forbidden_concepts() -> None:
    """No VRSBench branches, question_type reads, router_client, or
    route_unknown anywhere in routing. routing 源码不含 VRSBench 分支、
    question_type 读取、router_client 或 route_unknown。"""
    forbidden = ("VRSBench", "question_type", "router_client", "route_unknown")
    for relative in ("routing/schema.py", "routing/policies.py", "routing/router.py",
                     "routing/__init__.py"):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{relative} still contains {token}"


def test_routing_does_not_import_models() -> None:
    """routing must never import models. routing 绝不导入 models。"""
    for relative in ("routing/schema.py", "routing/policies.py", "routing/router.py",
                     "routing/__init__.py"):
        source = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "import models" not in source
        assert "from models" not in source


def test_every_task_name_has_a_tested_policy() -> None:
    """Every TaskName resolves through the public router entry.
    每个 TaskName 都经公开路由入口解析。"""
    router = TaskRouter()
    for task in _ALL_TASKS:
        decision = router.route(task)
        assert decision.primary_agent
        assert decision.reason_codes
    # The policy table and the TaskName literal never drift apart.
    # 策略表与 TaskName 字面量绝不漂移。
    from routing.policies import POLICIES

    assert set(POLICIES) == set(_ALL_TASKS)


# ── TaskResolution 契约 / task-resolution contract ─────────────────────────


def _resolution(**overrides) -> "TaskResolution":
    from routing.schema import TaskResolution

    values = dict(
        task="counting",
        confidence=1.0,
        candidate_tasks=["counting"],
        needs_candidate_fallback=False,
        source="explicit",
        reason_codes=["explicit_task:counting"],
    )
    values.update(overrides)
    return TaskResolution(**values)


def test_resolution_accepts_valid_contract() -> None:
    from routing.schema import TaskResolution

    resolution = _resolution()
    assert isinstance(resolution, TaskResolution)
    assert resolution.candidate_tasks[0] == resolution.task


def test_resolution_candidates_must_start_with_task() -> None:
    with pytest.raises(ValidationError, match="candidate_tasks\\[0\\] must equal task"):
        _resolution(candidate_tasks=["general_vqa"])


def test_resolution_rejects_duplicate_candidates() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        _resolution(candidate_tasks=["counting", "counting"])


def test_resolution_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValidationError):
        _resolution(confidence=1.5)
    with pytest.raises(ValidationError):
        _resolution(confidence=-0.1)


def test_resolution_is_json_serializable() -> None:
    import json

    resolution = _resolution(
        task="counting",
        confidence=0.8,
        candidate_tasks=["counting", "general_vqa"],
        needs_candidate_fallback=True,
        source="model",
        reason_codes=["low_confidence"],
    )
    payload = json.loads(resolution.model_dump_json())
    assert payload["candidate_tasks"] == ["counting", "general_vqa"]
    assert payload["needs_candidate_fallback"] is True


def test_resolution_request_requires_positive_image_count() -> None:
    from routing.schema import TaskResolutionRequest

    with pytest.raises(ValidationError):
        TaskResolutionRequest(image_count=0)
    request = TaskResolutionRequest(
        question="Q", image_count=1, metadata_hints={"split": "test"}
    )
    assert request.metadata_hints == {"split": "test"}


def test_resolution_request_rejects_non_json_hints() -> None:
    from routing.schema import TaskResolutionRequest

    with pytest.raises(ValidationError):
        TaskResolutionRequest(image_count=1, metadata_hints={"bad": object()})
    with pytest.raises(ValidationError):
        TaskResolutionRequest(image_count=1, metadata_hints={1: "x"})
