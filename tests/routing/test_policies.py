"""Contract tests for fixed route policies.

固定路由策略契约测试：每个 TaskName 恰好有一条受测 policy、primary/
fallback 与 requires_tiling 字段正确、未知 task 显式失败、策略不可变。
"""

from __future__ import annotations

from typing import get_args

import pytest

from data.schema import TaskName
from routing.policies import POLICIES, policy_for
from routing.schema import RoutePolicy

_ALL_TASKS = get_args(TaskName)


def test_every_task_name_has_a_policy() -> None:
    """Every TaskName must resolve to a tested policy.
    每个 TaskName 都必须解析到一条受测 policy。"""
    assert set(POLICIES) == set(_ALL_TASKS)
    for task in _ALL_TASKS:
        policy = policy_for(task)
        assert isinstance(policy, RoutePolicy)
        assert policy.task == task


def test_tiling_policies_are_generic_policy_fields() -> None:
    """requires_tiling lives on the policy, never decided per dataset.
    requires_tiling 是策略字段，绝不按数据集决定。"""
    assert policy_for("counting").requires_tiling is True
    assert policy_for("fine_grained_counting").requires_tiling is True
    assert policy_for("change_caption").requires_tiling is True
    assert policy_for("change_qa").requires_tiling is True
    assert policy_for("grounding").requires_tiling is True
    assert policy_for("spatial_relation").requires_tiling is False
    assert policy_for("general_vqa").requires_tiling is False
    assert policy_for("caption").requires_tiling is False


def test_primary_and_fallback_agents() -> None:
    assert policy_for("counting").primary_agent == "counting_agent"
    assert policy_for("change_caption").primary_agent == "change_agent"
    assert policy_for("change_qa").primary_agent == "change_agent"
    assert policy_for("change_qa").fallback_agents == ("general_vqa_agent",)
    assert policy_for("change_qa").execution_mode == "fallback"
    assert policy_for("general_vqa").fallback_agents == ()
    assert policy_for("general_vqa").execution_mode == "single"
    assert policy_for("scene_classification").primary_agent == "general_vqa_agent"
    assert policy_for("multiple_choice_vqa").primary_agent == "general_vqa_agent"
    assert policy_for("spatial_relation").primary_agent == "general_vqa_agent"
    assert policy_for("spatial_relation").fallback_agents == ()
    assert policy_for("spatial_relation").execution_mode == "single"


def test_unknown_task_fails_explicitly() -> None:
    """Unknown tasks raise KeyError; there is no general_vqa guessing.
    未知 task 抛 KeyError；不做 general_vqa 猜测。"""
    with pytest.raises(KeyError, match="Unknown routable task"):
        policy_for("no-such-task")
    with pytest.raises(KeyError, match="Unknown routable task"):
        policy_for("")


def test_policies_are_immutable() -> None:
    import dataclasses

    policy = policy_for("counting")
    assert dataclasses.is_dataclass(policy)
    assert policy.__dataclass_params__.frozen
    with pytest.raises(Exception):
        policy.primary_agent = "other_agent"  # type: ignore[misc]


def test_no_dataset_or_question_branches_in_policies() -> None:
    source = (__import__("pathlib").Path(__file__).resolve().parents[2]
              / "routing" / "policies.py").read_text(encoding="utf-8")
    assert "VRSBench" not in source
    assert "dataset" not in source
    assert "question" not in source
