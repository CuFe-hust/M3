"""Common agent result contracts: AgentName, VisualEvidence, AgentResult.

跨任务通用 Agent 输出契约：AgentName、VisualEvidence、AgentResult。
包含常见的 corner pair / flat box 防御性归一化并记录 repair severity；
本模块不定义 CountingResult（属于计数域 schema）。
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from data.schema import TaskName

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
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    image_id: str | None = None
    coordinate_frame: Literal["normalized_0_999_top_left"] = "normalized_0_999_top_left"

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
    boxes: list[list[float]] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list, max_length=12)
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


# ── First-Qwen visual plan contracts ──────────────────────────────────────
# Shared by the VisualPlanner and the VQA protocol owner. The shared schema
# deliberately holds no VQA masks, no Grounding box ids, and no
# CountingResult; backend/checkpoint/device and the final answer are never
# allowed here. 由 VisualPlanner 与 VQA 协议 owner 共用。共享 schema 刻意不
# 携带 VQA 掩膜、Grounding box_id 或 CountingResult；此处不允许出现
# backend/checkpoint/device 或最终答案。

# Frozen plan schema version; the model output must match it exactly.
# 冻结的计划 schema 版本；模型输出必须精确匹配。
PLAN_SCHEMA_VERSION = "first-qwen-plan-v1"

# Execution family selects the internal completion path; it never rewrites
# UnifiedSample.task. 内部完成路径选择；绝不改写 UnifiedSample.task。
ExecutionFamily = Literal["direct_vqa", "object_evidence_vqa"]

_MAX_PLAN_COMPOSITE_CATEGORIES = 3

_ROI_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"


class RoiRegion(BaseModel):
    """One semantic attention ROI in the frozen normalized [0,1] top-left
    xyxy frame. image_id must reference an existing UnifiedSample image id
    (enforced by the planner); the full image is expressed as [0,0,1,1].
    Geometric validity (range, degeneracy) is NOT rejected here: per 14B
    §6.2 the planner collapses an invalid ROI plan to the unique full-image
    ROI instead of rejecting it; non-finite coordinates stay a strict
    schema rejection (a value that is not even a number cannot be parsed
    into geometry). 一条使用冻结 [0,1] top-left xyxy 制式的语义注意 ROI。
    image_id 必须引用已存在 UnifiedSample 的图像 id（由 Planner 强制校验）；
    整图表示为 [0,0,1,1]。几何合法性（范围、退化）不在此拒绝：按 14B §6.2
    规划器将非法 ROI 计划折叠为唯一整图 ROI 而非拒绝；非有限坐标仍属 schema
    严格拒绝（连数字都算不上的值无法解析成几何）。"""

    model_config = ConfigDict(extra="forbid")

    roi_id: str = Field(min_length=1, pattern=_ROI_ID_PATTERN)
    image_id: str = Field(min_length=1)
    xyxy: tuple[float, float, float, float]

    @model_validator(mode="after")
    def validate_xyxy(self) -> "RoiRegion":
        """Require finite coordinates; the [0,1] range and degeneracy are
        decided by the planner per 14B §6.2. 要求坐标有限；[0,1] 范围与退化
        由规划器按 14B §6.2 决定。"""
        if not all(math.isfinite(value) for value in self.xyxy):
            raise ValueError("ROI xyxy must be finite")
        return self


class RoiPlan(BaseModel):
    """Validated ROI plan: zero ROIs mean "no reliable spatial constraint"
    (the geometry layer maps this to the unique full-image ROI), otherwise
    the attention ROIs. The count is not capped here: per 14B §6.2 an
    over-limit, out-of-range, or degenerate plan collapses to the unique
    full-image ROI in the planner — never truncated, never re-called.
    校验后的 ROI 计划：零 ROI 表示“无可靠空间约束”（几何层映射为唯一整图
    ROI），否则为注意 ROI。数量不在此封顶：按 14B §6.2，超限、越界或退化
    的计划在规划器中折叠为唯一整图 ROI——绝不截断、绝不重调。"""

    model_config = ConfigDict(extra="forbid")

    rois: list[RoiRegion] = Field(default_factory=list)


class ObjectEvidenceRequest(BaseModel):
    """Requested closed composite categories for object-evidence VQA. The
    planner validates every category against the same-version catalog and
    deduplicates them; the schema enforces only the count and string shape.
    对象证据 VQA 请求的封闭组合类别。Planner 使用同版本目录校验每个类别并
    去重；schema 只强制数量与字符串形状。"""

    model_config = ConfigDict(extra="forbid")

    composite_categories: list[str] = Field(
        min_length=1, max_length=_MAX_PLAN_COMPOSITE_CATEGORIES
    )

    @model_validator(mode="after")
    def validate_categories(self) -> "ObjectEvidenceRequest":
        """Reject blank, control-character, or path-like category names.
        拒绝空白、控制字符或类路径的类别名。"""
        for category in self.composite_categories:
            stripped = category.strip()
            if not stripped or any(ord(character) < 32 for character in category):
                raise ValueError(f"invalid composite category: {category!r}")
            if "/" in category or "\\" in category:
                raise ValueError(f"composite category must not be path-like: {category!r}")
        return self


class FirstQwenVisualPlan(BaseModel):
    """Strict first-Qwen planning output. version is frozen; execution_family
    selects the internal completion path without ever rewriting
    UnifiedSample.task; no backend/checkpoint/device or final answer is
    allowed here. object_evidence_vqa requires an evidence request while
    direct_vqa must not carry one; the plan never reads Ground Truth.
    严格的第一次 Qwen 规划输出。version 冻结；execution_family 选择内部完成
    路径，绝不改写 UnifiedSample.task；此处不允许 backend/checkpoint/device
    或最终答案。object_evidence_vqa 必须携带 evidence request，direct_vqa
    不得携带；本计划绝不读取 Ground Truth。"""

    model_config = ConfigDict(extra="forbid")

    version: Literal[PLAN_SCHEMA_VERSION]  # type: ignore[valid-type]
    execution_family: ExecutionFamily
    confidence: float = Field(ge=0.0, le=1.0)
    roi_plan: RoiPlan
    evidence_request: ObjectEvidenceRequest | None = None
    reason_codes: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def validate_family_linkage(self) -> "FirstQwenVisualPlan":
        """Enforce the required linkage between execution family and evidence
        request: object evidence implies categories, direct VQA forbids them.
        强制执行家族与证据请求的联动：对象证据必须携带类别，直接 VQA 禁止
        携带类别。"""
        if self.execution_family == "object_evidence_vqa" and self.evidence_request is None:
            raise ValueError("object_evidence_vqa requires an evidence_request")
        if self.execution_family == "direct_vqa" and self.evidence_request is not None:
            raise ValueError("direct_vqa must not carry an evidence_request")
        return self


# ── Joint task + visual planning contract (doc 15) ────────────────────────
# One schema-validated Qwen call returns the authoritative execution task and
# the reusable visual-plan substructure together. The task becomes the
# materialized/routed/executed task; a dataset-supplied source task is audit
# only and never overrides it. The substructure reuses the frozen
# FirstQwenVisualPlan contract (ROI, evidence request, execution family) — no
# parallel ROI or coordinate implementation is created here. 单次 schema 校验
# 的 Qwen 调用同时返回权威执行 task 与可复用的视觉计划子结构。该 task 成为
# 物化、路由与执行使用的 task；数据集提供的来源 task 只作审计保留，绝不
# 覆盖它。子结构复用冻结的 FirstQwenVisualPlan 契约（ROI、证据请求、
# execution family）——不创建平行的 ROI 或坐标实现。

# Frozen joint schema version; the model output must match it exactly.
# 冻结的联合 schema 版本；模型输出必须精确匹配。
JOINT_PLAN_SCHEMA_VERSION = "joint-qwen-plan-v1"


class JointQwenVisualPlan(BaseModel):
    """Versioned strict joint planning output: the authoritative execution
    task plus the reusable visual-plan substructure. task must belong to the
    closed data.schema.TaskName set; the visual_plan substructure is a
    validated FirstQwenVisualPlan (family/evidence linkage, ROI frame, and
    closed-category membership are enforced by that contract plus the
    planner). The plan never carries a final answer, backend, checkpoint,
    device, path, secret, or Ground Truth; extra="forbid" rejects undeclared
    fields. 版本化严格联合规划输出：权威执行 task 加可复用视觉计划子结构。
    task 必须属于封闭 data.schema.TaskName 集合；visual_plan 子结构是已校验
    的 FirstQwenVisualPlan（family/证据联动、ROI 制式与封闭类别归属由该契约
    与规划器共同强制）。计划绝不携带最终答案、backend、checkpoint、device、
    路径、secret 或 Ground Truth；extra="forbid" 拒绝未声明字段。"""

    model_config = ConfigDict(extra="forbid")

    version: Literal[JOINT_PLAN_SCHEMA_VERSION]  # type: ignore[valid-type]
    task: TaskName
    visual_plan: FirstQwenVisualPlan
