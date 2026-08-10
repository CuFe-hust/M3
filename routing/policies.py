"""Fixed deterministic route policies keyed by normalized task names.

按规范化 task 名索引的固定确定性路由策略。每个 TaskName 恰好对应一条
RoutePolicy；requires_tiling 是策略字段，绝不按数据集或问题内容决定。
本模块不读取问题、不调用模型。
"""

from __future__ import annotations

from routing.schema import RoutePolicy

# Fixed task → policy table migrated from the baseline ROUTES table. The
# router never guesses: unknown tasks fail explicitly.
# 从基线 ROUTES 表迁移的固定 task → 策略表。路由器绝不猜测：未知 task
# 显式失败。
POLICIES: dict[str, RoutePolicy] = {
    "counting": RoutePolicy(
        task="counting", primary_agent="counting_agent", requires_tiling=True
    ),
    "fine_grained_counting": RoutePolicy(
        task="fine_grained_counting", primary_agent="counting_agent", requires_tiling=True
    ),
    "change_caption": RoutePolicy(
        task="change_caption", primary_agent="change_agent", requires_tiling=True
    ),
    "change_qa": RoutePolicy(
        task="change_qa",
        primary_agent="change_agent",
        fallback_agents=("general_vqa_agent",),
        requires_tiling=True,
    ),
    "grounding": RoutePolicy(
        task="grounding", primary_agent="grounding_agent", requires_tiling=True
    ),
    "spatial_relation": RoutePolicy(
        task="spatial_relation", primary_agent="spatial_agent"
    ),
    "scene_classification": RoutePolicy(
        task="scene_classification", primary_agent="general_vqa_agent"
    ),
    "general_vqa": RoutePolicy(
        task="general_vqa", primary_agent="general_vqa_agent"
    ),
    "caption": RoutePolicy(
        task="caption", primary_agent="caption_agent"
    ),
    "multiple_choice_vqa": RoutePolicy(
        task="multiple_choice_vqa", primary_agent="general_vqa_agent"
    ),
}


def policy_for(task: str) -> RoutePolicy:
    """Look up the fixed policy; unknown tasks fail explicitly instead of
    guessing general_vqa. 查找固定策略；未知 task 显式失败而非猜测
    general_vqa。"""
    if task not in POLICIES:
        raise KeyError(f"Unknown routable task {task!r}; registered: {sorted(POLICIES)}")
    return POLICIES[task]
