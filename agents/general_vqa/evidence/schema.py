"""Strict VQA evidence contracts for the object-evidence subworkflow.

object_evidence_vqa 子工作流的严格 VQA 证据契约。所有持久化字段严格
JSON-safe 且路径安全：不允许 tensor/PIL/bytes/NaN/Base64/raw exception/
物理路径。检测记录不向最终 Qwen 或公共 trace 暴露 confidence；SegFormer
只保留掩膜存在性证据，不转框、不计数；不存在 valid_empty——成功但筛选为空
属于 missing。掩膜与裁切图等内存结构绝不进入本模块定义的 JSON 模型。
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Stable per-leaf evidence states. missing/unsupported/unavailable/error may
# continue to the next approved fallback; a hit leaf is never re-run or
# overwritten. not_run marks a leaf that was already hit at an upper layer.
# 叶子类别的稳定证据状态。missing/unsupported/unavailable/error 可以进入下一
# 批准回退；已 hit 的叶子绝不重跑或被覆盖。not_run 标记已在上层命中的叶子。
EvidenceState = Literal[
    "hit",
    "missing",
    "unsupported",
    "unavailable",
    "error",
    "not_run",
]

# The three evidence layers of the frozen executor state machine.
# 冻结执行器状态机的三层证据层。
EvidenceLayer = Literal["yolo", "segformer", "final_visual"]

_BOX_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"


def _validate_xyxy_box(box: tuple[float, float, float, float], where: str) -> None:
    """Require finite, non-negative, non-degenerate pixel coordinates.
    要求有限、非负、非退化的像素坐标。"""
    if not all(math.isfinite(value) for value in box):
        raise ValueError(f"{where} must be finite")
    if not all(value >= 0.0 for value in box):
        raise ValueError(f"{where} must be non-negative")
    x1, y1, x2, y2 = box
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"{where} must be non-degenerate with x1<x2 and y1<y2")


def _validate_size(size: tuple[int, int], where: str) -> None:
    """Require positive pixel dimensions. 要求正的像素尺寸。"""
    if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"{where} must contain positive (width, height)")


class YoloDetectionRecord(BaseModel):
    """One retained YOLO detection in crop-local and whole-image pixel
    frames. Confidence never appears here: it is consumed internally for
    thresholding, dedup, and conflict resolution only. local_xyxy is in the
    crop-pixel frame the final Qwen sees (inverse letterbox applied), and
    global_xyxy is the whole-image pixel frame.
    一条保留的 YOLO 检测记录（crop 局部与整图像素坐标）。此处绝不出现
    confidence：它只在内部用于阈值、去重与冲突裁决。local_xyxy 是最终 Qwen
    看到的 crop 像素坐标（已应用 letterbox 逆变换），global_xyxy 是整图像素
    坐标。"""

    model_config = ConfigDict(extra="forbid")

    leaf_category: str = Field(min_length=1)
    roi_id: str = Field(min_length=1, pattern=_BOX_PATTERN)
    local_xyxy: tuple[float, float, float, float]
    local_roi_size: tuple[int, int]
    global_xyxy: tuple[float, float, float, float]
    global_image_size: tuple[int, int]

    @model_validator(mode="after")
    def validate_geometry(self) -> "YoloDetectionRecord":
        """Validate both coordinate frames and their size references.
        校验两个坐标系及各自尺寸引用。"""
        _validate_xyxy_box(self.local_xyxy, "local_xyxy")
        _validate_xyxy_box(self.global_xyxy, "global_xyxy")
        _validate_size(self.local_roi_size, "local_roi_size")
        _validate_size(self.global_image_size, "global_image_size")
        if self.local_xyxy[2] > self.local_roi_size[0] or self.local_xyxy[3] > self.local_roi_size[1]:
            raise ValueError("local_xyxy exceeds the ROI crop size")
        if self.global_xyxy[2] > self.global_image_size[0] or self.global_xyxy[3] > self.global_image_size[1]:
            raise ValueError("global_xyxy exceeds the whole-image size")
        return self


class SegFormerEvidenceRecord(BaseModel):
    """One retained SegFormer mask hit. The mask itself travels in memory and
    never enters persisted records; no box conversion and no instance count
    are produced. 一条保留的 SegFormer 掩膜命中记录。掩膜本身只在内存中传递，
    绝不进入持久化记录；不转框、不生成实例数量。"""

    model_config = ConfigDict(extra="forbid")

    leaf_category: str = Field(min_length=1)
    roi_id: str = Field(min_length=1, pattern=_BOX_PATTERN)


class RoiEvidenceRecord(BaseModel):
    """Persisted-safe ROI geometry for the final bundle; the crop image
    itself travels in memory. core_xyxy is the pre-halo pixel box on the
    source image, expanded_xyxy the halo-expanded box, and crop_size the
    final crop dimensions. 最终证据包中持久化安全的 ROI 几何；裁切图本身只在
    内存中传递。core_xyxy 是源图上的扩张前像素框，expanded_xyxy 是扩张后
    像素框，crop_size 是最终裁切尺寸。"""

    model_config = ConfigDict(extra="forbid")

    roi_id: str = Field(min_length=1, pattern=_BOX_PATTERN)
    image_id: str = Field(min_length=1)
    source_size: tuple[int, int]
    core_xyxy: tuple[int, int, int, int]
    expanded_xyxy: tuple[int, int, int, int]
    crop_size: tuple[int, int]

    @model_validator(mode="after")
    def validate_geometry(self) -> "RoiEvidenceRecord":
        """Require consistent nested pixel geometry: core inside expanded
        inside the source image. 要求嵌套像素几何一致：core 在 expanded 内，
        expanded 在源图内。"""
        _validate_size(self.source_size, "source_size")
        _validate_size(self.crop_size, "crop_size")
        for name, box in (("core_xyxy", self.core_xyxy), ("expanded_xyxy", self.expanded_xyxy)):
            if len(box) != 4 or any(not isinstance(value, int) for value in box):
                raise ValueError(f"{name} must contain four integers")
            if box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError(f"{name} must be non-degenerate")
            if box[0] < 0 or box[1] < 0 or box[2] > self.source_size[0] or box[3] > self.source_size[1]:
                raise ValueError(f"{name} exceeds the source image size")
        if not (
            self.core_xyxy[0] >= self.expanded_xyxy[0]
            and self.core_xyxy[1] >= self.expanded_xyxy[1]
            and self.core_xyxy[2] <= self.expanded_xyxy[2]
            and self.core_xyxy[3] <= self.expanded_xyxy[3]
        ):
            raise ValueError("core_xyxy must lie inside expanded_xyxy")
        if self.expanded_xyxy[2] - self.expanded_xyxy[0] != self.crop_size[0]:
            raise ValueError("crop_size width must match expanded_xyxy")
        if self.expanded_xyxy[3] - self.expanded_xyxy[1] != self.crop_size[1]:
            raise ValueError("crop_size height must match expanded_xyxy")
        return self


class LayerStateRecord(BaseModel):
    """One per-(leaf, layer) diagnostic state. The final leaf state is the
    deepest layer's state; hit leaves reach no further layer.
    一条逐（叶子，层）诊断状态。叶子最终状态取最深层的状态；已 hit 叶子不再
    进入后续层。"""

    model_config = ConfigDict(extra="forbid")

    leaf_category: str = Field(min_length=1)
    layer: EvidenceLayer
    state: EvidenceState

    @model_validator(mode="after")
    def validate_state_layer(self) -> "LayerStateRecord":
        """not_run is meaningful only below the layer where the leaf hit.
        not_run 只在叶子命中的上层之下有意义。"""
        if self.state == "not_run" and self.layer == "yolo":
            raise ValueError("not_run is invalid for the first (yolo) layer")
        return self


class ModelCallAudit(BaseModel):
    """Auditable metadata of one model call: stable identity and status only.
    Raw responses, tensors, credentials, and physical model paths never
    appear here. 一次模型调用的可审计元数据：仅稳定身份与状态。raw
    response、tensor、凭据与物理模型路径绝不出现。"""

    model_config = ConfigDict(extra="forbid")

    layer: Literal["yolo", "segformer"]
    roi_id: str = Field(min_length=1, pattern=_BOX_PATTERN)
    input_size: tuple[int, int]
    logical_model_id: str = Field(min_length=1)
    weights_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    status: Literal["succeeded", "failed"] = "succeeded"
    error_code: str | None = None

    @model_validator(mode="after")
    def validate_call(self) -> "ModelCallAudit":
        """A failed call requires a stable error code; a succeeded call must
        not carry one. 失败调用必须有稳定错误码；成功调用不得携带。"""
        _validate_size(self.input_size, "input_size")
        if self.status == "failed" and not self.error_code:
            raise ValueError("failed model calls require a stable error_code")
        if self.status == "succeeded" and self.error_code is not None:
            raise ValueError("succeeded model calls must not carry an error_code")
        return self


class VqaEvidenceBundle(BaseModel):
    """Final JSON-safe evidence bundle for the single final-Qwen call of the
    object-evidence workflow. Images travel in memory separately; every
    persisted field is JSON-safe and path-safe.
    供 object_evidence 工作流唯一最终 Qwen 调用使用的最终 JSON 安全证据包。
    图像另行在内存中传递；所有持久化字段 JSON 安全且路径安全。"""

    model_config = ConfigDict(extra="forbid")

    workflow: Literal["object_evidence_vqa"] = "object_evidence_vqa"
    # Catalog version pins the closed category/leaf mapping this evidence was
    # produced from (14A2 §5.1); audit and resume can verify provenance.
    # catalog version 固定产生本证据的封闭类别/叶子映射（14A2 §5.1）；审计与
    # resume 可据此核验来源。
    catalog_version: str = Field(min_length=1)
    rois: list[RoiEvidenceRecord] = Field(default_factory=list)
    detections: list[YoloDetectionRecord] = Field(default_factory=list)
    segments: list[SegFormerEvidenceRecord] = Field(default_factory=list)
    missing_leaves: list[str] = Field(default_factory=list)
    leaf_states: dict[str, EvidenceState] = Field(default_factory=dict)
    call_audit: list[ModelCallAudit] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_bundle(self) -> "VqaEvidenceBundle":
        """Enforce cross-record references: every detection/segment roi must
        exist, every missing leaf must be non-hit, and leaf_states must cover
        every missing leaf. 强制跨记录引用：每条检测/掩膜 roi 必须存在，每个
        缺失叶子必须未命中，leaf_states 必须覆盖全部缺失叶子。"""
        roi_ids = {record.roi_id for record in self.rois}
        for record in [*self.detections, *self.segments]:
            if record.roi_id not in roi_ids:
                raise ValueError(f"evidence references unknown roi_id {record.roi_id!r}")
        seen_leaves: list[str] = []
        for leaf in [*self.missing_leaves, *self.leaf_states]:
            if leaf not in seen_leaves:
                seen_leaves.append(leaf)
        hit_leaves = {leaf for leaf, state in self.leaf_states.items() if state == "hit"}
        for leaf in self.missing_leaves:
            if leaf in hit_leaves:
                raise ValueError(f"hit leaf {leaf!r} must not appear in missing_leaves")
        for leaf in self.missing_leaves:
            if leaf not in self.leaf_states:
                raise ValueError(f"missing leaf {leaf!r} is absent from leaf_states")
        return self
