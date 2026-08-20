"""Counting-domain contracts: geometry, observations, results.

计数域契约：几何、观测与结果。本模块是计数域 Schema 的唯一所有权所在，
不得放回 agents.schema 或 data.schema；不导入应用级配置层，不定义任何
后端选择或执行逻辑。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from data.schema import JsonValue


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
        拒绝为空或方向颠倒的半开矩形。"""
        if not (self.left < self.right and self.top < self.bottom):
            raise ValueError("invalid half-open rectangle")
        return self

    @property
    def width(self) -> int:
        """Return rectangle width in pixels. / 返回矩形的像素宽度。"""
        return self.right - self.left

    @property
    def height(self) -> int:
        """Return rectangle height in pixels. / 返回矩形的像素高度。"""
        return self.bottom - self.top


class TileSpec(BaseModel):
    """One owner-core tile and its halo crop geometry.
    一块 owner core 切片及其 halo 裁剪几何。"""

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
        确保 crop、core 与局部坐标共享同一有效几何。"""
        crop = self.crop_global
        core = self.owner_core_global
        if crop.right > self.source_width or crop.bottom > self.source_height:
            raise ValueError("crop exceeds source image")
        if not (
            crop.left <= core.left < core.right <= crop.right
            and crop.top <= core.top < core.bottom <= crop.bottom
        ):
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
    相对于实际发送切片 crop 的一个模型点。"""

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
    单个样本全部切片共享的稳定计数目标定义。"""

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
    一块切片的经校验点式计数响应。"""

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
        要求每个报告实例对应一个唯一局部点。"""
        if self.reported_count != len(self.points):
            raise ValueError("reported_count must equal len(points)")
        local_ids = [point.local_id for point in self.points]
        if len(local_ids) != len(set(local_ids)):
            raise ValueError("duplicate local_id")
        return self


class PointProvenance(BaseModel):
    """Traceable source of a detected point (backward-compatible).
    可追踪的检测点来源（向后兼容）。"""

    model_config = ConfigDict(extra="forbid")

    source: Literal[
        "qwen_point",
        "semantic_component_centroid",
        "yolo_box_center",
        "yolo_obb_center",
        "fused",
    ] = "qwen_point"
    backend_name: str | None = None
    model_id: str | None = None
    source_class: str | None = None
    detector_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    obb_polygon_local_px: list[list[float]] | None = None
    obb_polygon_global_px: list[list[float]] | None = None
    bbox_xyxy_local_px: list[float] | None = None
    bbox_xyxy_global_px: list[float] | None = None
    detector_task: Literal["obb", "detect"] | None = None
    detector_source_dataset: str | None = None
    weights_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class GlobalPointObservation(BaseModel):
    """One converted point with full coordinate provenance and acceptance status.
    具有完整坐标来源与接受状态的一条转换点。"""

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
    provenance: PointProvenance | None = Field(default=None)


class IssueRecord(BaseModel):
    """Machine-readable counting warning or failure evidence.
    机器可读的计数告警或失败证据。"""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    tile_ids: list[str] = Field(default_factory=list)
    point_ids: list[str] = Field(default_factory=list)


class SeamDecision(BaseModel):
    """Minimal visual judgment for one ambiguous local seam pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: Literal[
        "same_instance",
        "different_instances",
        "uncertain",
    ]


class CountingDraft(BaseModel):
    """Collected tile evidence before seam and review finalization.
    seam 与复核最终化前收集的 tile 证据。"""

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
    具有明确部分与失败状态的最终点导出计数。"""

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
        """Enforce that final count equals accepted points and failures are
        visible. 强制最终数量等于接受点数量且失败状态可见。"""
        accepted = sum(point.accepted for point in self.global_points)
        if self.final_count != accepted:
            raise ValueError("final_count must equal accepted points")
        if self.failed_tiles and self.status not in {"partial", "failed"}:
            raise ValueError("failed tiles require partial or failed status")
        return self


class CountingBackendAttemptAudit(BaseModel):
    """Persisted, path-free audit record for one backend execution attempt."""

    model_config = ConfigDict(extra="forbid")

    backend_name: str = Field(min_length=1)
    backend_kind: str = Field(min_length=1)
    phase: Literal["primary", "fallback", "zero_review"]
    status: Literal["succeeded", "partial", "failed", "unavailable"]
    reason_code: str | None = None
    error_type: str | None = None
    counting: CountingResult | None = None
    agent_result: dict[str, JsonValue] | None = None
    backend_trace: dict[str, JsonValue] = Field(default_factory=dict)


class CountingExecutionAudit(BaseModel):
    """Complete ordered audit for one sample's counting backend attempts."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    target: str = Field(min_length=1)
    attempts: list[CountingBackendAttemptAudit] = Field(default_factory=list)
