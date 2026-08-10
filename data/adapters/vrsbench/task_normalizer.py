"""VRSBench question-to-task normalization (data layer, before routing).

VRSBench 问题到标准任务的规范化（数据层，路由之前）。把问题语义从运行时
Router/geometry 前移到 Adapter 预处理；只输出标准任务与结构化提示字典，
不返回任何 Agent 或后端名称，不调用模型、不计算评测分数、不修改答案。
"""

from __future__ import annotations

import re
from typing import Literal

from data.adapters.vrsbench import ontology
from data.schema import TaskNormalization

QuestionSubtype = Literal[
    "counting",
    "existence",
    "extreme_existence",
    "extreme_category",
    "grid_position",
    "orientation",
    "arrangement",
    "proximity",
    "color",
    "category",
    "general",
]

NORMALIZER_NAME = "vrsbench_task_normalizer"
NORMALIZER_VERSION = "1"
SOURCE_TASK = "vrsbench_vqa"

# Subtypes that map to the spatial expert task. / 映射到空间任务的标准子类型。
SPATIAL_SUBTYPES = frozenset(
    {"extreme_category", "grid_position", "orientation", "arrangement"}
)
# Reason codes per subtype; "counting" keeps its dedicated audit code.
# 各子类型的 reason code；"counting" 保留专用审计代码。
_REASON_CODES = {
    "counting": "quantity_question",
    "existence": "existence_question",
    "extreme_existence": "extreme_existence_question",
    "extreme_category": "extreme_category_question",
    "grid_position": "grid_position_question",
    "orientation": "orientation_question",
    "arrangement": "arrangement_question",
    "proximity": "proximity_question",
    "color": "color_question",
    "category": "category_question",
    "general": "general_question",
}


def _is_yes_no_question(question: str) -> bool:
    """Recognize unambiguous polar questions without alternative-answer clauses.
    识别不含二选一答案分支的明确是非问题。"""
    return bool(
        question
        and " or " not in question
        and re.match(r"^(?:is|are|was|were|do|does|did|can|could|has|have|will|would)\b", question)
    )


def _asks_vehicle_category(question: str) -> bool:
    """Require both an explicit vehicle target and a class/category request.
    要求问题同时明确车辆目标以及类别询问。"""
    return bool(
        re.search(r"\b(?:vehicle|car|truck|bus|trailer|motorcycle)s?\b", question)
        and re.search(r"\b(?:class|category|type|kind)\b", question)
    )


def _asks_extreme_vehicle_category(question: str) -> bool:
    """Recognize top/bottom vehicle-class questions supported by box geometry.
    识别可由框几何支持的最上方或最下方车辆类别问题。"""
    return bool(
        re.search(r"\b(?:top|bottom)[ -]?most\b", question)
        and _asks_vehicle_category(question)
    )


def _asks_grid_position(question: str) -> bool:
    """Recognize direct singular image-location questions without relation clauses.
    识别不含关系或比较分支的直接单目标图像位置问题。"""
    if not question or _is_yes_no_question(question):
        return False
    if re.search(
        r"\b(?:relative|relation|closer|closest|touch(?:ing)?|next to|near|adjacent|open space)\b"
        r"|\b(?:top|bottom|left|right)[ -]?most\b|\b(?:upper|lower)most\b",
        question,
    ):
        return False
    return bool(
        re.match(
            r"^where is\b.+?(?:located|positioned)?(?:\s+(?:in|within)\s+(?:the\s+)?image)?\??$",
            question,
        )
        or re.match(r"^what is the (?:position|location) of\b", question)
    )


def classify_question_subtype(
    question: str,
    question_type: str | None = None,
) -> QuestionSubtype:
    """Classify question semantics independently from the coarse official type.
    独立于较粗的官方类型识别问题语义。"""
    lowered = " ".join(question.casefold().replace("_", " ").split())
    if re.search(r"\bhow many\b|\b(?:total )?number of\b|\bcount(?: of)?\b|\bquantity of\b", lowered):
        return "counting"
    if _asks_extreme_vehicle_category(lowered):
        return "extreme_category"
    if re.search(r"\b(?:what|which)\b.*\b(?:orientation|direction)\b", lowered):
        return "orientation"
    if re.search(r"\b(?:arrangement|arranged|layout)\b", lowered):
        return "arrangement"
    if _asks_grid_position(lowered):
        return "grid_position"
    if _is_yes_no_question(lowered):
        if re.search(r"\b(top|bottom)[ -]?most\b", lowered):
            return "extreme_existence"
        if re.search(r"\b(?:near|adjacent|close to|next to)\b", lowered):
            return "proximity"
        return "existence"
    if re.search(r"\bwhat\s+colou?r\b|\bcolou?r\s+(?:is|are)\b", lowered):
        return "color"
    if re.search(r"\b(?:area|scene|landscape|region|surroundings)\b", lowered):
        return "general"
    if re.search(r"\b(?:class|category|type|kind)\b", lowered):
        return "category"
    return "general"


# Audited closed vocabularies per semantic subtype. / 各语义子类型的审计封闭词表。
_CLOSED_VOCABULARIES: dict[str, list[str]] = {
    "existence": ["yes", "no"],
    "extreme_existence": ["yes", "no"],
    "proximity": ["yes", "no"],
    "extreme_category": ["small-vehicle", "large-vehicle"],
    "grid_position": [
        "top-left", "top-middle", "top-right",
        "middle-left", "middle-middle", "middle-right",
        "bottom-left", "bottom-middle", "bottom-right",
    ],
    "orientation": ["north-south", "east-west"],
    "arrangement": ["in rows", "clustered", "scattered"],
    "color": ["black", "blue", "brown", "gray", "green", "orange", "red", "white", "yellow"],
}


def _closed_vocabulary(subtype: str) -> dict[str, object] | None:
    values = _CLOSED_VOCABULARIES.get(subtype)
    if values is None:
        return None
    return {"type": "closed_vocabulary", "values": values, "closed": True}


def normalize_task(
    question: str,
    question_type: str | None = None,
) -> TaskNormalization:
    """Normalize one VRSBench question to a standard task with structured hints.
    将一条 VRSBench 问题规范化为标准任务与结构化提示。"""
    if not question.strip():
        return TaskNormalization(
            source_task=SOURCE_TASK,
            normalized_task="general_vqa",
            confidence=0.5,
            normalizer=NORMALIZER_NAME,
            version=NORMALIZER_VERSION,
            reason_codes=["empty_question_fallback"],
        )
    subtype = classify_question_subtype(question, question_type)
    if subtype == "counting":
        task = "counting"
    elif subtype in SPATIAL_SUBTYPES:
        task = "spatial_relation"
    else:
        task = "general_vqa"

    spatial_query = {"operation": subtype} if task == "spatial_relation" else None
    answer_constraints: dict = {}
    if subtype in {"existence", "extreme_existence", "proximity"}:
        if _is_yes_no_question(question.casefold()):
            answer_constraints = _closed_vocabulary(subtype) or {}
    elif subtype in _CLOSED_VOCABULARIES:
        answer_constraints = _closed_vocabulary(subtype) or {}
    count_target_hint = ontology.count_target_hint(question) if task == "counting" else None

    return TaskNormalization(
        source_task=SOURCE_TASK,
        normalized_task=task,  # type: ignore[arg-type]
        semantic_subtype=subtype,
        confidence=1.0,
        normalizer=NORMALIZER_NAME,
        version=NORMALIZER_VERSION,
        reason_codes=[_REASON_CODES[subtype]],
        spatial_query=spatial_query,
        answer_constraints=answer_constraints,
        count_target_hint=count_target_hint,
    )
