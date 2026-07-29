"""Deterministic VRSBench VQA routing profiles and evidence geometry.
VRSBench VQA 的确定性路由配置与证据几何计算。
"""

from __future__ import annotations

import re
from typing import Any, Literal

from spacers_agent.schemas import CountTargetSpec, ExpertResult, VisualEvidence


ExecutionTask = Literal["counting", "spatial_relation", "general_vqa"]
VRSBenchQuestionSubtype = Literal[
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

_STATUS_ANSWER_PLACEHOLDERS = {"completed", "failed", "partial"}

_SMALL_VEHICLE_ALIASES = (
    "small vehicle",
    "small-vehicle",
    "car",
    "automobile",
    "passenger car",
    "motorcycle",
)
_LARGE_VEHICLE_ALIASES = (
    "large vehicle",
    "large-vehicle",
    "truck",
    "bus",
    "trailer",
    "semi-truck",
)


def execution_task_for_vrsbench(question_type: str, question: str | None = None) -> ExecutionTask:
    """Select a conservative task from question semantics, with a general fallback.
    根据问题语义保守选择任务，并为所有未知情况提供通用回退。
    """

    if not question:
        return "general_vqa"
    subtype = vrsbench_question_subtype(question, question_type)
    if subtype == "counting":
        return "counting"
    if subtype in {"extreme_category", "grid_position", "orientation", "arrangement"}:
        return "spatial_relation"
    return "general_vqa"


def vrsbench_question_subtype(question: str, question_type: str) -> VRSBenchQuestionSubtype:
    """Classify question semantics independently from the coarse official type.
    独立于较粗的官方类型识别问题语义。
    """

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


def vrsbench_answer_vocabulary(
    subtype: VRSBenchQuestionSubtype,
    question: str = "",
) -> list[str]:
    """Return a closed vocabulary only when the question itself entails it.
    仅在问题文本本身能够推出封闭答案空间时返回词表。
    """

    lowered = " ".join(question.casefold().replace("_", " ").split())
    if subtype in {"existence", "extreme_existence", "proximity"} and _is_yes_no_question(lowered):
        return ["yes", "no"]
    if subtype == "extreme_category" and _asks_extreme_vehicle_category(lowered):
        return ["small-vehicle", "large-vehicle"]
    if subtype == "category" and _asks_vehicle_category(lowered):
        return ["small-vehicle", "large-vehicle"]
    if subtype == "grid_position" and _asks_grid_position(lowered):
        return [
            f"{vertical}-{horizontal}"
            for vertical in ("top", "middle", "bottom")
            for horizontal in ("left", "middle", "right")
        ]
    if subtype == "orientation" and re.search(r"\b(?:orientation|direction)\b", lowered):
        return ["north-south", "east-west"]
    if subtype == "arrangement" and re.search(r"\b(?:arrangement|arranged|layout)\b", lowered):
        return ["in rows", "clustered", "scattered"]
    if subtype == "color" and re.search(r"\bcolou?r\b", lowered):
        return ["black", "blue", "brown", "gray", "green", "orange", "red", "white", "yellow"]
    return []


def _is_yes_no_question(question: str) -> bool:
    """Recognize unambiguous polar questions without alternative-answer clauses.
    识别不含二选一答案分支的明确是非问题。
    """

    return bool(
        question
        and " or " not in question
        and re.match(r"^(?:is|are|was|were|do|does|did|can|could|has|have|will|would)\b", question)
    )


def _asks_vehicle_category(question: str) -> bool:
    """Require both an explicit vehicle target and a class/category request.
    要求问题同时明确车辆目标以及类别询问。
    """

    return bool(
        re.search(r"\b(?:vehicle|car|truck|bus|trailer|motorcycle)s?\b", question)
        and re.search(r"\b(?:class|category|type|kind)\b", question)
    )


def _asks_extreme_vehicle_category(question: str) -> bool:
    """Recognize top/bottom vehicle-class questions supported by box geometry.
    识别可由框几何支持的最上方或最下方车辆类别问题。
    """

    return bool(
        re.search(r"\b(?:top|bottom)[ -]?most\b", question)
        and _asks_vehicle_category(question)
    )


def vrsbench_count_target(question: str) -> CountTargetSpec:
    """Return the audited VRSBench vehicle ontology without an LLM parse call.
    不调用大模型，返回经审计的 VRSBench 车辆计数类别体系。
    """

    lowered = " ".join(question.casefold().replace("-", " ").split())
    if "small vehicle" in lowered:
        return CountTargetSpec(
            canonical_label="small-vehicle",
            aliases=list(_SMALL_VEHICLE_ALIASES),
            inclusion_rule="Count each visible car, passenger car, or motorcycle as one small vehicle.",
            exclusion_rule="Exclude trucks, buses, trailers, and non-vehicle objects.",
        )
    if "large vehicle" in lowered:
        return CountTargetSpec(
            canonical_label="large-vehicle",
            aliases=list(_LARGE_VEHICLE_ALIASES),
            inclusion_rule="Count each visible truck, bus, trailer, or semi-truck as one large vehicle.",
            exclusion_rule="Exclude cars, motorcycles, and non-vehicle objects.",
        )
    return CountTargetSpec(
        canonical_label="vehicle",
        aliases=["vehicle", *_SMALL_VEHICLE_ALIASES, *_LARGE_VEHICLE_ALIASES],
        inclusion_rule="Count each visible small or large vehicle as one vehicle.",
        exclusion_rule="Exclude non-vehicle objects and do not count one vehicle more than once.",
    )


def apply_vrsbench_geometry(
    question: str,
    question_type: str,
    result: ExpertResult,
) -> ExpertResult:
    """Apply only reproducible answer rules supported by labeled evidence geometry.
    仅应用由带标签证据几何支持且可复现的答案规则。
    """

    subtype = vrsbench_question_subtype(question, question_type)
    audit: dict[str, Any] = {
        "version": "vrsbench-evidence-geometry-v4",
        "coordinate_frame": "normalized_0_999_top_left",
        "question_type": question_type,
        "semantic_subtype": subtype,
        "answer_source": "qwen_visual_answer",
        "rule": "no_deterministic_override",
    }
    for key in (
        "input_normalizations",
        "candidate_review_used",
        "candidate_review_added",
        "candidate_review_replaced",
        "candidate_review_labeled_boxes",
        "candidate_review_error",
        "evidence_quality",
        "repair_severity",
    ):
        if key in result.geometry:
            audit[key] = result.geometry[key]
    boxed = [item for item in result.evidence_items if item.box is not None]
    lowered = question.casefold()
    audit["evidence_valid"] = bool(result.evidence_items)

    extreme = "top" if re.search(r"\btop[ -]?most\b", lowered) else (
        "bottom" if re.search(r"\bbottom[ -]?most\b", lowered) else None
    )
    vehicle_boxes = [item for item in boxed if vrsbench_vehicle_class(item.label) in {"small-vehicle", "large-vehicle"}]
    if subtype == "extreme_category" and extreme and len(vehicle_boxes) >= 2:
        selected = min(vehicle_boxes, key=_center_y) if extreme == "top" else max(vehicle_boxes, key=_center_y)
        canonical = vrsbench_vehicle_class(selected.label)
        audit.update(
            {
                "answer_source": "deterministic_geometry",
                "rule": f"{extreme}_most_box_center_y",
                "selected_label": selected.label,
                "selected_box": selected.box,
                "selected_center_y": _center_y(selected),
                "candidate_count": len(vehicle_boxes),
                "evidence_complete": True,
            }
        )
        return _finalize_vrsbench_answer(result, subtype, audit, canonical)

    if subtype == "extreme_category" and extreme:
        audit.update(
            {
                "rule": "insufficient_extreme_candidates",
                "candidate_count": len(vehicle_boxes),
                "evidence_complete": False,
            }
        )
        return _finalize_vrsbench_answer(result, subtype, audit, result.answer)

    if subtype == "extreme_existence":
        audit.update(
            {
                "rule": "extreme_existence_requires_candidate_enumeration",
                "candidate_count": len(vehicle_boxes),
            }
        )

    if subtype == "grid_position" and boxed:
        target = _select_position_target(lowered, boxed)
        if target is not None:
            answer = _grid_position(target)
            audit.update(
                {
                    "answer_source": "deterministic_geometry",
                    "rule": "three_by_three_box_center",
                    "selected_label": target.label,
                    "selected_box": target.box,
                    "grid_boundaries": [333, 666],
                    "candidate_count": len(_position_candidates(lowered, boxed)),
                    "evidence_complete": True,
                }
            )
            return _finalize_vrsbench_answer(result, subtype, audit, answer)
        audit.update(
            {
                "rule": "ambiguous_or_missing_position_target",
                "candidate_count": len(_position_candidates(lowered, boxed)),
                "evidence_complete": False,
            }
        )

    if subtype == "orientation":
        audit.update(
            {
                "rule": "cardinal_direction_requires_dataset_north_up_assumption",
                "north_metadata_available": False,
            }
        )
    elif subtype == "proximity" and len(boxed) >= 2:
        first, second = boxed[0], boxed[1]
        audit.update(
            {
                "rule": "box_gap_recorded_without_threshold_override",
                "nearest_box_gap": round(_box_gap(first, second), 3),
            }
        )
    elif subtype == "arrangement":
        audit.update(
            {
                "rule": "arrangement_requires_instance_set",
                "candidate_count": len(vehicle_boxes),
            }
        )
    return _finalize_vrsbench_answer(result, subtype, audit, result.answer)


def _center_y(item: VisualEvidence) -> float:
    box = item.box
    if box is None:
        raise ValueError("box evidence required")
    return (box[1] + box[3]) / 2


def _grid_position(item: VisualEvidence) -> str:
    box = item.box
    if box is None:
        raise ValueError("box evidence required")
    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    horizontal = "left" if center_x < 333 else ("middle" if center_x < 666 else "right")
    vertical = "top" if center_y < 333 else ("middle" if center_y < 666 else "bottom")
    return f"{vertical}-{horizontal}"


def _asks_grid_position(question: str) -> bool:
    """Recognize direct singular image-location questions without relation clauses.
    识别不含关系或比较分支的直接单目标图像位置问题。
    """

    if not question or _is_yes_no_question(question):
        return False
    if re.search(
        r"\b(?:relative|relation|closer|closest|touch(?:ing)?|next to|near|adjacent|open space)\b"
        r"|\b(?:top|bottom|left|right)[ -]?most\b|\b(?:upper|lower)most\b",
        question,
    ):
        return False
    return bool(
        re.match(r"^where is\b.+?(?:located|positioned)?(?:\s+(?:in|within)\s+(?:the\s+)?image)?\??$", question)
        or re.match(r"^what is the (?:position|location) of\b", question)
    )


def _select_position_target(question: str, boxed: list[VisualEvidence]) -> VisualEvidence | None:
    candidates = _position_candidates(question, boxed)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) < 3:
        return None
    isolation = [
        [
            min(_center_distance(candidate, other) for other in candidates if other is not candidate),
            candidate,
        ]
        for candidate in candidates
    ]
    isolation.sort(key=lambda value: value[0])
    best_distance, best = isolation[-1]
    second_distance = isolation[-2][0]
    return best if best_distance >= 100 and best_distance >= second_distance * 1.5 else None


def _position_candidates(question: str, boxed: list[VisualEvidence]) -> list[VisualEvidence]:
    """Return boxes matching the target class named by a grid-position question.
    返回与九宫格位置问题所指目标类别匹配的框。
    """

    desired = "large-vehicle" if "large vehicle" in question else (
        "small-vehicle" if "small vehicle" in question else None
    )
    return [item for item in boxed if desired is None or vrsbench_vehicle_class(item.label) == desired]


def _center_distance(first: VisualEvidence, second: VisualEvidence) -> float:
    """Return Euclidean distance between two trusted box centres.
    返回两个可信框中心之间的欧氏距离。
    """

    if first.box is None or second.box is None:
        raise ValueError("box evidence required")
    first_x = (first.box[0] + first.box[2]) / 2
    second_x = (second.box[0] + second.box[2]) / 2
    return ((first_x - second_x) ** 2 + (_center_y(first) - _center_y(second)) ** 2) ** 0.5


def normalize_vrsbench_answer(question_type: str, answer: str) -> str:
    """Normalize declared VRSBench answer vocabularies without using references.
    不读取参考答案，仅规范化已声明的 VRSBench 答案词表。
    """

    normalized_type = " ".join(question_type.casefold().replace("_", " ").split())
    lowered = " ".join(answer.casefold().replace("_", " ").split())
    if normalized_type in {"object existence", "existence", "extreme existence", "proximity"}:
        if re.match(r"^(yes|yeah|true)\b", lowered):
            return "yes"
        if re.match(r"^(no|false)\b", lowered):
            return "no"
    if normalized_type in {"object category", "category", "extreme category"}:
        category = vrsbench_vehicle_class(answer)
        if category in {"small-vehicle", "large-vehicle"}:
            return category
    if normalized_type in {"object position", "grid position"}:
        compact = re.sub(r"[^a-z]+", "-", lowered).strip("-")
        for vertical in ("top", "middle", "bottom"):
            for horizontal in ("left", "middle", "right"):
                candidate = f"{vertical}-{horizontal}"
                if candidate in compact:
                    return candidate
    if normalized_type in {"object color", "color"}:
        colors = [
            color
            for color in ("black", "blue", "brown", "gray", "green", "orange", "red", "white", "yellow")
            if re.search(rf"\b{color}\b", lowered)
        ]
        if len(colors) == 1:
            return colors[0]
    if normalized_type in {"object direction", "orientation"}:
        if re.search(r"\bnorth[ -]?south\b", lowered):
            return "north-south"
        if re.search(r"\beast[ -]?west\b", lowered):
            return "east-west"
    if normalized_type == "arrangement":
        if re.search(r"\b(row|rows|line|lines|parallel|side by side)\b", lowered):
            return "in rows"
        if re.search(r"\b(cluster|clustered|group|grouped|together)\b", lowered):
            return "clustered"
        if re.search(r"\b(scattered|random|dispersed|spread out)\b", lowered):
            return "scattered"
    return answer.strip()


def vrsbench_vehicle_class(label: str) -> str:
    """Map common vehicle names to the closed VRSBench vehicle classes.
    将常见车辆名称映射到封闭的 VRSBench 车辆类别。
    """

    normalized = re.sub(r"[_\s]+", "-", label.strip().casefold())
    if normalized in {value.replace(" ", "-") for value in _SMALL_VEHICLE_ALIASES}:
        return "small-vehicle"
    if normalized in {value.replace(" ", "-") for value in _LARGE_VEHICLE_ALIASES}:
        return "large-vehicle"
    for alias in sorted(_SMALL_VEHICLE_ALIASES, key=len, reverse=True):
        pattern = re.escape(alias).replace(r"\ ", r"[\s-]+")
        if re.search(rf"\b{pattern}\b", label.casefold()):
            return "small-vehicle"
    for alias in sorted(_LARGE_VEHICLE_ALIASES, key=len, reverse=True):
        pattern = re.escape(alias).replace(r"\ ", r"[\s-]+")
        if re.search(rf"\b{pattern}\b", label.casefold()):
            return "large-vehicle"
    return normalized


def _finalize_vrsbench_answer(
    result: ExpertResult,
    question_type: str,
    audit: dict[str, Any],
    answer: str,
) -> ExpertResult:
    raw_answer = result.answer
    normalized = normalize_vrsbench_answer(question_type, answer)
    placeholder_token = re.sub(r"[^a-z]+", "", normalized.casefold())
    placeholder_removed = placeholder_token in _STATUS_ANSWER_PLACEHOLDERS
    if placeholder_removed:
        normalized = ""
    audit.update(
        {
            "raw_answer": raw_answer,
            "normalized_answer": normalized,
            "answer_normalization_version": "vrsbench-answer-v2",
            "status_answer_placeholder_removed": placeholder_removed,
        }
    )
    if placeholder_removed or audit.get("evidence_complete") is False:
        status = "partial"
    elif audit.get("answer_source") == "deterministic_geometry" and audit.get("evidence_complete") is True:
        status = "completed"
    else:
        status = result.status
    audit["workflow_status"] = status
    return result.model_copy(update={"answer": normalized, "geometry": audit, "status": status})


def _box_gap(first: VisualEvidence, second: VisualEvidence) -> float:
    first_box, second_box = first.box, second.box
    if first_box is None or second_box is None:
        raise ValueError("box evidence required")
    dx = max(first_box[0] - second_box[2], second_box[0] - first_box[2], 0)
    dy = max(first_box[1] - second_box[3], second_box[1] - first_box[3], 0)
    return (dx * dx + dy * dy) ** 0.5
