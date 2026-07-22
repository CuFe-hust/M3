"""Pydantic contracts for unified samples, tiles, points, and counting results.
统一样本、切片、点和计数结果的 Pydantic 契约。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


TaskName = Literal[
    "counting",
    "fine_grained_counting",
    "change_caption",
    "change_qa",
    "grounding",
    "spatial_relation",
    "scene_classification",
    "general_vqa",
    "caption",
    "multiple_choice_vqa",
]


class TargetParseResult(BaseModel):
    """Structured target extraction used before a counting run. / 计数运行前使用的结构化目标提取结果。"""

    model_config = ConfigDict(extra="forbid")

    target: CountTargetSpec
    short_rationale: str = Field(max_length=240)


class SampleRunStatus(BaseModel):
    """Durable machine-readable state for one dataset sample. / 单个数据集样本的可持久化机器可读状态。"""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    task: TaskName
    state: Literal["pending", "running", "succeeded", "partial", "failed", "skipped"]
    error_code: str | None = None
    error_message: str | None = None
    result_path: Path | None = None
    updated_at: str


class DatasetRunSummary(BaseModel):
    """Aggregate visible outcomes without hiding failed samples. / 不隐藏失败样本的汇总结果。"""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    dataset: str
    split: str
    task: str
    total: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    partial: int = Field(ge=0)
    failed: int = Field(ge=0)
    skipped: int = Field(ge=0)


class VisualEvidence(BaseModel):
    """One labeled visual observation in normalized whole-image coordinates.
    一条使用整图归一化坐标的带标签视觉证据。
    """

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
        要求在声明的坐标系中恰好提供一个合法框或点。
        """

        if (self.box is None) == (self.point is None):
            raise ValueError("visual evidence requires exactly one of box or point")
        if self.box is not None:
            if len(self.box) != 4 or any(value < 0 or value > 999 for value in self.box):
                raise ValueError("box must be [x1,y1,x2,y2] in 0..999")
            if self.box[0] >= self.box[2] or self.box[1] >= self.box[3]:
                raise ValueError("box corners must satisfy x1<x2 and y1<y2")
        if self.point is not None and (
            len(self.point) != 2 or any(value < 0 or value > 999 for value in self.point)
        ):
            raise ValueError("point must be [x,y] in 0..999")
        return self


class ExpertResult(BaseModel):
    """Uniform non-counting expert result with verifiable evidence. / 含可验证证据的统一非计数专家结果。"""

    model_config = ConfigDict(extra="forbid")

    expert: str
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
        在严格证据校验前归一化模型常见的双角点输出。
        """

        if not isinstance(value, dict):
            return value
        data = dict(value)
        raw_boxes = data.get("boxes")
        normalized_boxes, normalizations = _normalize_model_boxes(raw_boxes)
        items = data.get("evidence_items")
        if isinstance(items, list):
            normalized_items: list[Any] = []
            for index, raw_item in enumerate(items):
                if not isinstance(raw_item, dict):
                    normalized_items.append(raw_item)
                    continue
                item = dict(raw_item)
                box, point = item.get("box"), item.get("point")
                if _is_coordinate_pair(box) and _is_coordinate_pair(point):
                    item["box"] = _clamp_box([*box, *point], normalizations)
                    item["point"] = None
                    normalizations.append("evidence_box_and_point_combined_as_corners")
                elif _is_coordinate_pair(box):
                    if index < len(normalized_boxes):
                        item["box"] = normalized_boxes[index]
                        normalizations.append("evidence_box_completed_from_top_level_corners")
                    else:
                        item["box"] = None
                        item["point"] = [int(box[0]), int(box[1])]
                        normalizations.append("two_value_evidence_box_reclassified_as_point")
                elif isinstance(box, list) and len(box) == 4:
                    item["box"] = _clamp_box(box, normalizations)
                normalized_items.append(item)
            data["evidence_items"] = normalized_items
        data["boxes"] = normalized_boxes
        if normalizations:
            geometry = dict(data.get("geometry") or {})
            geometry["input_normalizations"] = list(dict.fromkeys(normalizations))
            data["geometry"] = geometry
        return data

    @model_validator(mode="after")
    def retain_evidence_boxes(self) -> "ExpertResult":
        """Retain labeled evidence boxes in the legacy canonical box list.
        将带标签证据框同步保留到旧版统一框列表中。
        """

        labeled_boxes = [list(item.box) for item in self.evidence_items if item.box is not None]
        if labeled_boxes:
            self.boxes = labeled_boxes
        return self


def _normalize_model_boxes(value: Any) -> tuple[list[list[int]], list[str]]:
    """Convert flat boxes or adjacent corner pairs to normalized box arrays.
    将扁平框或相邻角点对转换为规范框数组。
    """

    normalizations: list[str] = []
    if not isinstance(value, list):
        return [], normalizations
    if all(isinstance(item, list) and len(item) == 4 for item in value):
        return [_clamp_box(item, normalizations) for item in value], normalizations
    if value and len(value) % 2 == 0 and all(_is_coordinate_pair(item) for item in value):
        boxes = [
            _clamp_box([*value[index], *value[index + 1]], normalizations)
            for index in range(0, len(value), 2)
        ]
        normalizations.append("top_level_corner_pairs_combined_as_boxes")
        return boxes, normalizations
    if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
        normalizations.append("flat_top_level_box_wrapped")
        return [_clamp_box(value, normalizations)], normalizations
    return [], normalizations


def _is_coordinate_pair(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(isinstance(item, (int, float)) for item in value)


def _clamp_box(value: list[Any], normalizations: list[str]) -> list[int]:
    converted = [int(item) for item in value]
    clamped = [max(0, min(999, item)) for item in converted]
    if clamped != converted:
        normalizations.append("box_coordinates_clamped_to_0_999")
    return clamped


class ImageRef(BaseModel):
    """One immutable image reference in a unified sample.
    统一样本中一条不可变的图像引用。
    """

    model_config = ConfigDict(extra="forbid")

    image_id: str = Field(min_length=1)
    path: Path
    role: Literal["image", "t1", "t2", "context"]
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    sha256: str | None = None


class GroundTruth(BaseModel):
    """Preserved ground truth without changing source annotations.
    在不改变源标注的前提下保留真值。
    """

    model_config = ConfigDict(extra="forbid")

    answers: list[str] = Field(default_factory=list)
    count: int | None = Field(default=None, ge=0)
    boxes: list[list[float]] = Field(default_factory=list)
    points: list[list[float]] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class UnifiedSample(BaseModel):
    """Dataset-neutral sample consumed by new multi-Agent workflows.
    新多 Agent 工作流消费的与数据集无关的样本。
    """

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    split: str = Field(min_length=1)
    task: TaskName
    images: list[ImageRef] = Field(min_length=1)
    question: str
    ground_truth: GroundTruth | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_temporal_order(self) -> "UnifiedSample":
        """Require ordered temporal images for change tasks.
        要求变化任务的时相图像顺序正确。
        """

        if self.task in {"change_caption", "change_qa"}:
            roles = [image.role for image in self.images]
            if roles[:2] != ["t1", "t2"]:
                raise ValueError("change samples must place t1 before t2")
        return self


class PixelRect(BaseModel):
    """Half-open integer pixel rectangle used by all internal geometry.
    所有内部几何使用的半开整数像素矩形。
    """

    model_config = ConfigDict(extra="forbid")

    left: int
    top: int
    right: int
    bottom: int

    @model_validator(mode="after")
    def validate_rect(self) -> "PixelRect":
        """Reject empty or reversed half-open rectangles.
        拒绝为空或方向颠倒的半开矩形。
        """

        if not (self.left < self.right and self.top < self.bottom):
            raise ValueError("invalid half-open rectangle")
        return self

    @property
    def width(self) -> int:
        """Return rectangle width in pixels.
        返回矩形的像素宽度。
        """

        return self.right - self.left

    @property
    def height(self) -> int:
        """Return rectangle height in pixels.
        返回矩形的像素高度。
        """

        return self.bottom - self.top


class TileSpec(BaseModel):
    """One owner-core tile and its halo crop geometry.
    一块 owner core 切片及其 halo 裁剪几何。
    """

    model_config = ConfigDict(extra="forbid")

    tile_id: str = Field(min_length=1)
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    crop_global: PixelRect
    owner_core_global: PixelRect
    owner_core_local: PixelRect
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    model_input_width: int = Field(gt=0)
    model_input_height: int = Field(gt=0)
    recursive_depth: int = Field(default=0, ge=0)
    parent_tile_id: str | None = None

    @model_validator(mode="after")
    def validate_tile_geometry(self) -> "TileSpec":
        """Ensure crop, core, and local coordinates share one valid geometry.
        确保 crop、core 与局部坐标共享同一有效几何。
        """

        crop = self.crop_global
        core = self.owner_core_global
        if crop.right > self.source_width or crop.bottom > self.source_height:
            raise ValueError("crop exceeds source image")
        if not (crop.left <= core.left < core.right <= crop.right and crop.top <= core.top < core.bottom <= crop.bottom):
            raise ValueError("owner core must be inside crop")
        expected = PixelRect(
            left=core.left - crop.left,
            top=core.top - crop.top,
            right=core.right - crop.left,
            bottom=core.bottom - crop.top,
        )
        if self.owner_core_local != expected:
            raise ValueError("owner_core_local must match owner core relative to crop")
        return self


class LocalPointObservation(BaseModel):
    """One model point relative to the actual transmitted tile crop.
    相对于实际发送切片 crop 的一个模型点。
    """

    model_config = ConfigDict(extra="forbid")

    local_id: str = Field(min_length=1)
    x: int = Field(ge=0, le=999)
    y: int = Field(ge=0, le=999)
    confidence: float = Field(ge=0.0, le=1.0)
    radius: int = Field(default=0, ge=0, le=250)
    touches_crop_border: bool = False
    short_evidence: str = Field(max_length=120)


class CountTargetSpec(BaseModel):
    """Stable counting-target definition shared by every tile of one sample.
    单个样本全部切片共享的稳定计数目标定义。
    """

    model_config = ConfigDict(extra="forbid")

    canonical_label: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    required_attributes: list[str] = Field(default_factory=list)
    excluded_attributes: list[str] = Field(default_factory=list)
    spatial_constraints: list[str] = Field(default_factory=list)
    inclusion_rule: str = Field(min_length=1)
    exclusion_rule: str = Field(min_length=1)
    ambiguity: list[str] = Field(default_factory=list)


class TileCountResponse(BaseModel):
    """Validated point-counting response for one tile.
    一块切片的经校验点式计数响应。
    """

    model_config = ConfigDict(extra="forbid")

    target: str = Field(min_length=1)
    tile_id: str = Field(min_length=1)
    points: list[LocalPointObservation] = Field(default_factory=list)
    reported_count: int = Field(ge=0)
    needs_split: bool = False
    uncertainty: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def count_matches(self) -> "TileCountResponse":
        """Require one unique local point for each reported instance.
        要求每个报告实例对应一个唯一局部点。
        """

        if self.reported_count != len(self.points):
            raise ValueError("reported_count must equal len(points)")
        local_ids = [point.local_id for point in self.points]
        if len(local_ids) != len(set(local_ids)):
            raise ValueError("duplicate local_id")
        return self


class GlobalPointObservation(BaseModel):
    """One converted point with full coordinate provenance and acceptance status.
    具有完整坐标来源与接受状态的一条转换点。
    """

    model_config = ConfigDict(extra="forbid")

    global_id: str = Field(min_length=1)
    target: str = Field(min_length=1)
    source_tile_id: str = Field(min_length=1)
    local_id: str = Field(min_length=1)
    local_x_norm: int = Field(ge=0, le=999)
    local_y_norm: int = Field(ge=0, le=999)
    local_radius_norm: int = Field(ge=0, le=250)
    global_x_px: int = Field(ge=0)
    global_y_px: int = Field(ge=0)
    global_x_norm: int = Field(ge=0, le=999)
    global_y_norm: int = Field(ge=0, le=999)
    global_x_was_clamped: bool = False
    global_y_was_clamped: bool = False
    radius_px: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    ownership_valid: bool
    near_core_boundary: bool
    accepted: bool
    rejection_reason: str | None = None
    short_evidence: str = Field(max_length=120)


class IssueRecord(BaseModel):
    """Machine-readable counting warning or failure evidence. / 机器可读的计数告警或失败证据。"""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    tile_ids: list[str] = Field(default_factory=list)
    point_ids: list[str] = Field(default_factory=list)


class CountingDraft(BaseModel):
    """Collected tile evidence before seam and review finalization. / seam 与复核最终化前收集的 tile 证据。"""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    target: str
    question: str
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    initial_tile_count: int = Field(ge=0)
    succeeded_tiles: list[str] = Field(default_factory=list)
    failed_tiles: list[str] = Field(default_factory=list)
    raw_global_points: list[GlobalPointObservation] = Field(default_factory=list)
    processed_tiles: list[TileSpec] = Field(default_factory=list)
    boundary_conflicts: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[IssueRecord] = Field(default_factory=list)


class CountingResult(BaseModel):
    """Final point-derived count with explicit partial and failure state.
    具有明确部分与失败状态的最终点导出计数。
    """

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    target: str = Field(min_length=1)
    question: str
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    tile_count: int = Field(ge=0)
    initial_tile_count: int | None = Field(default=None, ge=0)
    leaf_tile_count: int | None = Field(default=None, ge=0)
    succeeded_tiles: list[str] = Field(default_factory=list)
    failed_tiles: list[str] = Field(default_factory=list)
    global_points: list[GlobalPointObservation] = Field(default_factory=list)
    merged_groups: list[list[str]] = Field(default_factory=list)
    unresolved_conflicts: list[str] = Field(default_factory=list)
    warnings: list[IssueRecord] = Field(default_factory=list)
    final_count: int = Field(ge=0)
    status: Literal["completed", "completed_with_warnings", "partial", "failed"]

    @model_validator(mode="after")
    def check_count(self) -> "CountingResult":
        """Enforce that final count equals accepted points and failures are visible.
        强制最终数量等于接受点数量且失败状态可见。
        """

        accepted = sum(point.accepted for point in self.global_points)
        if self.final_count != accepted:
            raise ValueError("final_count must equal accepted points")
        if self.failed_tiles and self.status not in {"partial", "failed"}:
            raise ValueError("failed tiles require partial or failed status")
        return self


def stable_sample_id(source_id: str | None, relative_image_path: Path, question: str, source_index: int) -> str:
    """Return source ID when present or a stable 20-character fallback digest.
    存在源 ID 时返回源 ID，否则返回稳定的 20 字符备用摘要。
    """

    if source_id:
        return source_id
    payload = f"{relative_image_path.as_posix()}\n{question}\n{source_index}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]
