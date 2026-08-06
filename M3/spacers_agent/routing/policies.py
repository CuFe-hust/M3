"""Fixed route policies and VRSBench semantic rules.
固定路由策略与 VRSBench 语义规则。
"""

from __future__ import annotations

from spacers_agent.routing.schemas import AgentName, RoutableTask

# ── fixed route table / 固定路由表 ────────────────────────────────────────

ROUTES: dict[RoutableTask, tuple[AgentName, ...]] = {
    "counting": ("counting_agent",),
    "fine_grained_counting": ("counting_agent",),
    "change_caption": ("change_agent",),
    "change_qa": ("change_agent", "general_vqa_agent"),
    "grounding": ("grounding_agent",),
    "spatial_relation": ("spatial_agent",),
    "scene_classification": ("general_vqa_agent",),
    "general_vqa": ("general_vqa_agent",),
    "caption": ("caption_agent",),
    "multiple_choice_vqa": ("general_vqa_agent",),
}

# Tasks that require tiling / 需要切片的 task
_TILING_TASKS: frozenset[str] = frozenset({
    "counting", "fine_grained_counting", "grounding", "change_caption", "change_qa",
})


def needs_tiling(task: str) -> bool:
    """Return whether a task requires image tiling. / 返回 task 是否需要图像切片。"""
    return task in _TILING_TASKS


# ── VRSBench semantic routing / VRSBench 语义路由 ─────────────────────────

# Tasks that VRSBench semantic questions map to / VRSBench 语义问题映射到的 task
_VRSBENCH_SEMANTIC_TASK: dict[str, RoutableTask] = {
    "counting": "counting",
    "extreme_category": "spatial_relation",
    "grid_position": "spatial_relation",
    "orientation": "spatial_relation",
    "arrangement": "spatial_relation",
}


def vrsbench_semantic_to_task(subtype: str) -> RoutableTask:
    """Map VRSBench question subtype to RoutableTask. / 将 VRSBench 问题子类型映射为 RoutableTask。"""
    return _VRSBENCH_SEMANTIC_TASK.get(subtype, "general_vqa")
