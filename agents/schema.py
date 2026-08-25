"""Common agent result contracts: AgentName, VisualEvidence, AgentResult.

跨任务通用 Agent 输出契约：AgentName、VisualEvidence、AgentResult。
包含常见的 corner pair / flat box 防御性归一化并记录 repair severity；
本模块不定义 CountingResult（属于计数域 schema）。
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from data.schema import TaskName
from models.images import materialize_quantized_roi as _materialize_quantized_roi

AgentName = Literal[
    "counting_agent",
    "change_agent",
    "grounding_agent",
    "general_vqa_agent",
    "caption_agent",
]

_COORD_MAX = 999


class VisualEvidence(BaseModel):
    """One labeled visual observation in normalized whole-image coordinates.
    一条使用整图归一化坐标的带标签视觉证据。"""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    box: list[int] | None = None
    point: list[int] | None = None
    image_id: str | None = None
    coordinate_frame: Literal["normalized_0_999_top_left"] = "normalized_0_999_top_left"

    @model_validator(mode="before")
    @classmethod
    def drop_legacy_confidence(cls, value: Any) -> Any:
        """Read legacy artifacts without republishing their confidence field.
        兼容读取历史产物，但不再向外序列化其置信度字段。"""
        if not isinstance(value, dict):
            return value
        data = dict(value)
        data.pop("confidence", None)
        return data

    @model_validator(mode="after")
    def validate_geometry(self) -> "VisualEvidence":
        """Require exactly one valid box or point in the declared coordinate frame.
        要求在声明的坐标系中恰好提供一个合法框或点。"""
        if (self.box is None) == (self.point is None):
            raise ValueError("visual evidence requires exactly one of box or point")
        if self.box is not None:
            if len(self.box) != 4 or any(value < 0 or value > _COORD_MAX for value in self.box):
                raise ValueError("box must be [x1,y1,x2,y2] in 0..999")
            if self.box[0] >= self.box[2] or self.box[1] >= self.box[3]:
                raise ValueError("box corners must satisfy x1<x2 and y1<y2")
        if self.point is not None and (
            len(self.point) != 2 or any(value < 0 or value > _COORD_MAX for value in self.point)
        ):
            raise ValueError("point must be [x,y] in 0..999")
        return self


class AgentResult(BaseModel):
    """Uniform non-counting Agent result with verifiable evidence.
    含可验证证据的统一非计数 Agent 结果。"""

    model_config = ConfigDict(extra="forbid")

    agent_name: AgentName
    answer: str
    # Model-facing boxes use the same integer 0..999 xyxy JSON geometry as
    # Phase 2 SFT; runtime consumers still receive the canonical list form.
    # 模型侧框与 Phase 2 SFT 使用相同的 0..999 整数 xyxy JSON 几何；运行时
    # 消费者仍接收统一的列表形式。
    boxes: list[list[int]] = Field(default_factory=list)
    evidence_items: list[VisualEvidence] = Field(default_factory=list, max_length=200)
    geometry: dict[str, Any] = Field(default_factory=dict)
    status: Literal["completed", "partial", "failed"] = "completed"

    @model_validator(mode="before")
    @classmethod
    def normalize_corner_pair_geometry(cls, value: Any) -> Any:
        """Normalize common two-corner model output before strict evidence validation.
        在严格证据校验前归一化模型常见的双角点输出。"""
        if not isinstance(value, dict):
            return value
        data = dict(value)
        # Legacy persisted results may contain text evidence. Accept it only
        # as migration input; the current contract never republishes it.
        # 历史持久化结果可能含文本 evidence；仅作为迁移输入接受，当前契约
        # 永不再次输出该字段。
        data.pop("evidence", None)
        raw_boxes = data.get("boxes")
        normalized_boxes, normalizations = _normalize_model_boxes(raw_boxes)
        items = data.get("evidence_items")
        evidence_quality: list[str] = []
        if isinstance(items, list):
            normalized_items: list[Any] = []
            for index, raw_item in enumerate(items):
                if isinstance(raw_item, VisualEvidence):
                    normalized_items.append(raw_item)
                    evidence_quality.append("trusted_box" if raw_item.box is not None else "trusted_point")
                    continue
                if not isinstance(raw_item, dict):
                    normalized_items.append(raw_item)
                    evidence_quality.append("invalid")
                    continue
                item = dict(raw_item)
                item.pop("confidence", None)
                box, point = item.get("box"), item.get("point")
                if _is_coordinate_pair(box) and _is_coordinate_pair(point):
                    normalized_box = _normalize_box_geometry([*box, *point], normalizations)
                    if _box_is_degenerate(normalized_box):
                        item["box"] = None
                        item["point"] = _degenerate_box_point(normalized_box)
                        normalizations.append("degenerate_evidence_box_reclassified_as_point")
                        evidence_quality.append("repaired_point")
                    else:
                        item["box"] = normalized_box
                        item["point"] = None
                        normalizations.append("evidence_box_and_point_combined_as_corners")
                        evidence_quality.append("trusted_box")
                elif _is_coordinate_pair(box):
                    if index < len(normalized_boxes):
                        item["box"] = normalized_boxes[index]
                        normalizations.append("evidence_box_completed_from_top_level_corners")
                        evidence_quality.append("trusted_box")
                    else:
                        item["box"] = None
                        item["point"] = [int(box[0]), int(box[1])]
                        normalizations.append("two_value_evidence_box_reclassified_as_point")
                        evidence_quality.append("repaired_point")
                elif isinstance(box, list) and len(box) == 4:
                    normalized_box = _normalize_box_geometry(box, normalizations)
                    if _box_is_degenerate(normalized_box):
                        item["box"] = None
                        if _is_coordinate_pair(point):
                            item["point"] = [int(point[0]), int(point[1])]
                            normalizations.append("degenerate_evidence_box_dropped_in_favor_of_point")
                            evidence_quality.append("trusted_point")
                        else:
                            item["point"] = _degenerate_box_point(normalized_box)
                            normalizations.append("degenerate_evidence_box_reclassified_as_point")
                            evidence_quality.append("repaired_point")
                    else:
                        item["box"] = normalized_box
                        evidence_quality.append("trusted_box")
                    if point is not None and item.get("box") is not None:
                        item["point"] = None
                        normalizations.append("evidence_point_dropped_in_favor_of_box")
                elif _is_coordinate_pair(point) and box is not None:
                    item["box"] = None
                    normalizations.append("invalid_evidence_box_dropped_in_favor_of_point")
                    evidence_quality.append("trusted_point")
                elif _is_coordinate_pair(point):
                    evidence_quality.append("trusted_point")
                else:
                    evidence_quality.append("invalid")
                normalized_items.append(item)
            data["evidence_items"] = normalized_items
        data["boxes"] = normalized_boxes
        if normalizations or evidence_quality:
            geometry = dict(data.get("geometry") or {})
            if normalizations:
                geometry["input_normalizations"] = list(dict.fromkeys(normalizations))
            if evidence_quality:
                geometry["evidence_quality"] = evidence_quality
            geometry["repair_severity"] = _repair_severity(normalizations)
            data["geometry"] = geometry
        return data

    @model_validator(mode="after")
    def retain_evidence_boxes(self) -> "AgentResult":
        """Retain labeled evidence boxes in the canonical box list.
        将带标签证据框同步保留到统一框列表中。"""
        labeled_boxes = [list(item.box) for item in self.evidence_items if item.box is not None]
        if labeled_boxes:
            self.boxes = labeled_boxes
        return self


def _normalize_model_boxes(value: Any) -> tuple[list[list[int]], list[str]]:
    """Convert flat boxes or adjacent corner pairs to normalized box arrays.
    将扁平框或相邻角点对转换为规范框数组。"""
    normalizations: list[str] = []
    if not isinstance(value, list):
        return [], normalizations
    if all(isinstance(item, list) and len(item) == 4 for item in value):
        boxes = [_normalize_box_geometry(item, normalizations) for item in value]
        return _drop_degenerate_top_level_boxes(boxes, normalizations), normalizations
    if value and len(value) % 2 == 0 and all(_is_coordinate_pair(item) for item in value):
        boxes = [
            _normalize_box_geometry([*value[index], *value[index + 1]], normalizations)
            for index in range(0, len(value), 2)
        ]
        normalizations.append("top_level_corner_pairs_combined_as_boxes")
        return _drop_degenerate_top_level_boxes(boxes, normalizations), normalizations
    if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
        normalizations.append("flat_top_level_box_wrapped")
        boxes = [_normalize_box_geometry(value, normalizations)]
        return _drop_degenerate_top_level_boxes(boxes, normalizations), normalizations
    return [], normalizations


def _is_coordinate_pair(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(
        isinstance(item, (int, float)) for item in value
    )


def _normalize_box_geometry(value: list[Any], normalizations: list[str]) -> list[int]:
    """Canonicalize box order without inventing area for a line or point.
    规范化框的角点顺序，但不为线或点虚构面积。"""
    converted = [int(item) for item in value]
    clamped = [max(0, min(_COORD_MAX, item)) for item in converted]
    if clamped != converted:
        normalizations.append("box_coordinates_clamped_to_0_999")
    x1, y1, x2, y2 = clamped
    ordered = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    if ordered != clamped:
        normalizations.append("box_corners_reordered")
    return ordered


def _box_is_degenerate(box: list[int]) -> bool:
    """Return whether a normalized box has zero extent on either axis.
    返回归一化框是否在任一坐标轴上为零长度。"""
    return box[0] == box[2] or box[1] == box[3]


def _degenerate_box_point(box: list[int]) -> list[int]:
    """Preserve a degenerate observation as a midpoint, not a fabricated box.
    将退化观测保留为中点，而不是虚构检测框。"""
    return [round((box[0] + box[2]) / 2), round((box[1] + box[3]) / 2)]


def _drop_degenerate_top_level_boxes(
    boxes: list[list[int]], normalizations: list[str]
) -> list[list[int]]:
    """Drop unlabeled degenerate legacy boxes; labeled evidence keeps a point.
    丢弃无标签的退化旧框；带标签证据则保留为点。"""
    retained = [box for box in boxes if not _box_is_degenerate(box)]
    if len(retained) != len(boxes):
        normalizations.append("degenerate_top_level_box_dropped")
    return retained


def _repair_severity(normalizations: list[str]) -> str:
    """Summarize whether a geometry repair reduced spatial information.
    汇总几何修复是否降低了空间信息质量。"""
    if any("degenerate" in value for value in normalizations):
        return "high"
    if normalizations:
        return "low"
    return "none"


# ── Visual-only task planning contract (doc 20) ───────────────────────────
# The v5 planner receives only normalized image previews and the raw question.
# Its output contains task/assistance intent and an optional strict 0..999 ROI;
# it never carries an answer or implementation choice. v5 规划器只接收规范化
# 图像预览与原始问题；输出任务/辅助意图及可选严格 0..999 ROI，绝不携带答案
# 或实现选择。

VISUAL_TASK_PLAN_SCHEMA_VERSION = "visual-task-plan-v5"
COUNTING_TASKS = frozenset({"counting", "fine_grained_counting"})

# Tasks owned by the GeneralVQAAgent and its shared VQA evidence protocol. The
# set expresses capability ownership — which agent/evidence protocol answers
# these tasks — never that evidence must be enabled; the planner's
# needs_visual_assistance stays the only switch. Planner, composition root and
# the Agent reuse this single set so the four-task list cannot drift.
# GeneralVQAAgent 及其共享 VQA 证据协议拥有的 task 集合。该集合只表达能力归属
# ——由哪个 Agent/证据协议回答这些 task——绝不表达必须启用证据；planner 的
# needs_visual_assistance 仍是唯一开关。planner、组合根与 Agent 复用同一集合，
# 避免四处维护的四个 task 列表发生漂移。
GENERAL_VQA_AGENT_TASKS = frozenset(
    {
        "general_vqa",
        "scene_classification",
        "multiple_choice_vqa",
        "spatial_relation",
    }
)


class RegionRequest(BaseModel):
    """Optional explicit attention rectangle in 0..999 image space.
    使用 0..999 图像坐标表达的可选显式注意力矩形。"""

    model_config = ConfigDict(extra="forbid")

    explicit: bool = False
    image_index: int | None = Field(default=None, ge=0)
    roi_xyxy: tuple[StrictInt, StrictInt, StrictInt, StrictInt] | None = None

    @model_validator(mode="after")
    def validate_linkage(self) -> "RegionRequest":
        """Keep explicit ROI fields all-or-nothing and strictly integral.
        保持显式 ROI 字段要么全部存在、要么全部缺省，并严格要求整数。"""
        if not self.explicit:
            if self.image_index is not None or self.roi_xyxy is not None:
                raise ValueError("implicit region_request must not carry ROI fields")
            return self
        if self.image_index is None or self.roi_xyxy is None:
            raise ValueError("explicit region_request requires image_index and roi_xyxy")
        if any(isinstance(value, bool) for value in self.roi_xyxy):
            raise ValueError("roi_xyxy must contain strict integers")
        if not all(0 <= value <= 999 for value in self.roi_xyxy):
            raise ValueError("roi_xyxy must be within [0, 999]")
        x0, y0, x1, y1 = self.roi_xyxy
        if x0 >= x1 or y0 >= y1:
            raise ValueError("roi_xyxy must be non-degenerate with x0<x1 and y0<y1")
        return self


class VisualTaskPlan(BaseModel):
    """Strict output of the single visual-only planning call.
    单次纯视觉规划调用的严格输出。"""

    model_config = ConfigDict(extra="forbid")

    version: Literal[VISUAL_TASK_PLAN_SCHEMA_VERSION]  # type: ignore[valid-type]
    task: TaskName
    needs_visual_assistance: bool = False
    object_categories: list[str] = Field(default_factory=list, max_length=8)
    count_target: str | None = Field(default=None, max_length=80)
    region_request: RegionRequest = Field(default_factory=RegionRequest)
    reason_codes: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_assistance_linkage(self) -> "VisualTaskPlan":
        """Require executable object categories exactly when assistance is on.
        只有启用视觉辅助时才允许携带对象类别，并要求类别非空。"""
        if self.needs_visual_assistance and not self.object_categories:
            raise ValueError("visual assistance requires object_categories")
        if not self.needs_visual_assistance and self.object_categories:
            raise ValueError("object_categories require visual assistance")
        for category in self.object_categories:
            stripped = category.strip()
            if not stripped or any(ord(character) < 32 for character in category):
                raise ValueError(f"invalid object category: {category!r}")
            if "/" in category or "\\" in category:
                raise ValueError(f"object category must not be path-like: {category!r}")
        if len(set(self.object_categories)) != len(self.object_categories):
            raise ValueError("object_categories must not contain duplicates")
        if self.task in COUNTING_TASKS:
            if self.count_target is None or not self.count_target.strip():
                raise ValueError("counting task requires count_target")
        elif self.count_target is not None:
            raise ValueError("non-counting task must not carry count_target")
        if self.count_target is not None:
            target = self.count_target
            if target.strip() != target:
                raise ValueError("count_target must not have surrounding whitespace")
            if any(ord(character) < 32 or ord(character) == 127 for character in target):
                raise ValueError("count_target must not contain control characters")
            if "/" in target or "\\" in target or target in {".", ".."}:
                raise ValueError("count_target must not be path-like")
            compact_number = target.replace(",", "").replace(".", "", 1)
            if compact_number.isdigit():
                raise ValueError("count_target must describe a semantic target")
        for reason_code in self.reason_codes:
            if (
                not reason_code
                or len(reason_code) > 64
                or any(
                    character not in "abcdefghijklmnopqrstuvwxyz0123456789_.:-"
                    for character in reason_code
                )
            ):
                raise ValueError("reason_codes must contain stable code strings")
        return self


class MaterializedVisualView(BaseModel):
    """Immutable geometry record for the exact image sent to final agents.
    发送给最终 Agent 的精确图像视图的不可变几何记录。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    image_id: str = Field(min_length=1)
    view_mode: Literal["full_image", "quantized_roi"]
    source_size: tuple[int, int]
    crop_xyxy: tuple[int, int, int, int]
    crop_size: tuple[int, int]
    # These fields make quantized ROI materialization auditable without
    # persisting raw model output, image bytes, or paths. 这些字段用于审计量化
    # ROI 物化过程，不持久化原始模型响应、图像字节或路径。
    requested_roi_xyxy_0_999: tuple[StrictInt, StrictInt, StrictInt, StrictInt] | None = None
    requested_pixel_xyxy: tuple[int, int, int, int] | None = None
    roi_quantum: StrictInt | None = None
    quantized_side: StrictInt | None = None
    ideal_square_xyxy: tuple[int, int, int, int] | None = None
    was_clipped: bool | None = None

    @model_validator(mode="after")
    def validate_geometry(self) -> "MaterializedVisualView":
        """Validate exact half-open bounds and v5 ROI audit geometry.
        校验精确的半开区间边界与 v5 ROI 审计几何。"""
        width, height = self.source_size
        x0, y0, x1, y1 = self.crop_xyxy
        crop_width, crop_height = self.crop_size
        if width <= 0 or height <= 0:
            raise ValueError("source_size must be positive")
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("crop_xyxy must be inside source_size")
        if (x1 - x0, y1 - y0) != (crop_width, crop_height):
            raise ValueError("crop_size must match crop_xyxy")
        if self.view_mode == "full_image":
            if self.crop_xyxy != (0, 0, width, height):
                raise ValueError("full_image view must cover the source")
            if any(
                value is not None
                for value in (
                    self.requested_roi_xyxy_0_999,
                    self.requested_pixel_xyxy,
                    self.roi_quantum,
                    self.quantized_side,
                    self.ideal_square_xyxy,
                    self.was_clipped,
                )
            ):
                raise ValueError("full_image view must not carry ROI audit geometry")
            return self

        audit_fields = (
            self.requested_roi_xyxy_0_999,
            self.requested_pixel_xyxy,
            self.roi_quantum,
            self.quantized_side,
            self.ideal_square_xyxy,
            self.was_clipped,
        )
        if any(value is None for value in audit_fields):
            raise ValueError("quantized_roi view requires complete ROI audit geometry")
        assert self.requested_roi_xyxy_0_999 is not None
        assert self.requested_pixel_xyxy is not None
        assert self.roi_quantum is not None
        assert self.quantized_side is not None
        assert self.ideal_square_xyxy is not None
        assert self.was_clipped is not None
        if self.roi_quantum <= 0 or self.roi_quantum != 1024:
            raise ValueError("roi_quantum must equal 1024")
        rx0, ry0, rx1, ry1 = self.requested_roi_xyxy_0_999
        if not (0 <= rx0 < rx1 <= 999 and 0 <= ry0 < ry1 <= 999):
            raise ValueError("requested_roi_xyxy_0_999 is invalid")
        requested = self.requested_pixel_xyxy
        if not (
            0 <= requested[0] < requested[2] <= width
            and 0 <= requested[1] < requested[3] <= height
        ):
            raise ValueError("requested_pixel_xyxy must be inside source_size")
        if self.quantized_side < self.roi_quantum or self.quantized_side % self.roi_quantum:
            raise ValueError("quantized_side must be a positive quantum multiple")
        ideal = self.ideal_square_xyxy
        if (
            ideal[2] - ideal[0] != self.quantized_side
            or ideal[3] - ideal[1] != self.quantized_side
        ):
            raise ValueError("ideal_square_xyxy must match quantized_side")
        expected_crop = (
            max(0, ideal[0]),
            max(0, ideal[1]),
            min(width, ideal[2]),
            min(height, ideal[3]),
        )
        if self.crop_xyxy != expected_crop:
            raise ValueError("crop_xyxy must be the clipped ideal square")
        if self.was_clipped != (ideal != self.crop_xyxy):
            raise ValueError("was_clipped does not match ideal/crop geometry")
        try:
            expected = _materialize_quantized_roi(
                self.source_size,
                self.requested_roi_xyxy_0_999,
                roi_quantum=self.roi_quantum,
            )
        except ValueError as exc:
            raise ValueError("quantized ROI audit geometry is invalid") from exc
        if (
            self.requested_pixel_xyxy != expected.requested_pixel_xyxy
            or self.quantized_side != expected.quantized_side
            or self.ideal_square_xyxy != expected.ideal_square_xyxy
            or self.crop_xyxy != expected.crop_xyxy
            or self.crop_size != expected.crop_size
            or self.was_clipped != expected.was_clipped
        ):
            raise ValueError(
                "quantized ROI audit geometry does not match shared materialization"
            )
        return self
