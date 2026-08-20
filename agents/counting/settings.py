"""Counting-domain settings: deterministic defaults and YOLO declarations.

计数域配置：确定性默认值与 YOLO 声明。配置只做结构校验，不访问权重文件、
不定义任何后端选择或执行逻辑；不导入应用级配置层。
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CountingSettings(BaseModel):
    """Deterministic defaults shared by future point-counting components.
    未来点式计数组件共用的确定性默认配置。"""

    model_config = ConfigDict(extra="forbid")

    tile_core_size: int = Field(default=896, gt=0)
    halo_size: int = Field(default=128, ge=0)
    model_max_side: int = Field(default=1280, gt=0)
    max_pixels_without_tiling: int = Field(default=1_600_000, gt=0)
    boundary_band_px: int = Field(default=32, ge=0)
    min_confidence: float = Field(default=0.2, ge=0.0, le=1.0)
    max_points_per_tile: int = Field(default=200, gt=0)
    sequential: bool = True
    concurrency: int = Field(default=1, ge=1)
    seam_verify: bool = True
    seam_review_enabled: bool = True
    seam_auto_merge_distance_factor: float = Field(default=0.35, gt=0.0, lt=1.0)
    seam_review_max_distance_factor: float = Field(default=0.75, gt=0.0, le=1.0)
    seam_conflict_min_distance_px: float = Field(default=6.0, gt=0.0)
    seam_conflict_max_distance_px: float = Field(default=64.0, gt=0.0)
    seam_conflict_core_ratio: float = Field(default=0.01, gt=0.0, le=1.0)
    recursive_split_enabled: bool = True
    max_recursive_depth: int = Field(default=2, ge=0)
    min_core_size: int = Field(default=224, gt=0)
    seam_crop_margin_px: int = Field(default=128, ge=0)
    unresolved_conflict_policy: Literal["flag_for_review"] = "flag_for_review"
    prompt_version: str = "count-point-v4"
    small_object_min_scan_depth: int = Field(default=0, ge=0)
    verify_empty_tiles: bool = False
    small_object_upscale_max_side: int | None = Field(default=None, gt=0)
    fallback_on_backend_unavailable: bool = True
    fallback_on_backend_error: bool = True
    verify_empty_detection: bool = True
    verify_empty_semantic: bool = False
    trust_empty_detection: bool = False
    multi_detector_enabled: bool = True
    max_selected_detector_experts: int = Field(default=5, ge=1, le=5)
    min_successful_detector_experts: int = Field(default=1, ge=1)
    ensemble_iou_threshold: float = Field(default=0.45, gt=0.0, le=1.0)
    ensemble_center_distance_ratio: float = Field(default=0.60, gt=0.0, le=2.0)
    ensemble_singleton_high_confidence: float = Field(default=0.65, ge=0.0, le=1.0)
    max_disagreement_regions: int = Field(default=12, ge=1, le=12)
    disagreement_context_padding_ratio: float = Field(default=0.35, ge=0.0, le=2.0)
    unresolved_ensemble_policy: Literal[
        "retain_high_confidence", "reject_unresolved"
    ] = "retain_high_confidence"

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_small_object_keys(cls, data: Any) -> Any:
        """Accept legacy configuration keys only at the settings boundary.

        Business code sees the dataset-neutral names exclusively. Supplying
        both names is rejected instead of silently choosing one.
        """

        if not isinstance(data, Mapping):
            return data
        migrated = dict(data)
        aliases = {
            "vrsbench_min_scan_depth": "small_object_min_scan_depth",
            "vrsbench_zero_review": "verify_empty_tiles",
            "vrsbench_tile_upscale_max_side": "small_object_upscale_max_side",
        }
        for legacy, current in aliases.items():
            if legacy not in migrated:
                continue
            if current in migrated:
                raise ValueError(
                    f"cannot configure both legacy key {legacy!r} and {current!r}"
                )
            migrated[current] = migrated.pop(legacy)
        return migrated

    def model_post_init(self, __context: Any) -> None:
        if self.sequential and self.concurrency != 1:
            raise ValueError("sequential counting requires concurrency=1")
        if self.small_object_min_scan_depth > self.max_recursive_depth:
            raise ValueError(
                "small_object_min_scan_depth cannot exceed max_recursive_depth"
            )
        if (
            self.seam_auto_merge_distance_factor
            >= self.seam_review_max_distance_factor
        ):
            raise ValueError(
                "seam_auto_merge_distance_factor must be smaller than "
                "seam_review_max_distance_factor"
            )
        if self.seam_conflict_min_distance_px > self.seam_conflict_max_distance_px:
            raise ValueError(
                "seam_conflict_min_distance_px cannot exceed "
                "seam_conflict_max_distance_px"
            )
        if self.min_successful_detector_experts > self.max_selected_detector_experts:
            raise ValueError(
                "min_successful_detector_experts cannot exceed "
                "max_selected_detector_experts"
            )


class CountingTargetStrategy(BaseModel):
    """Immutable JSON-safe behavior derived only from catalog target hints."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    small_object: bool = False
    dense_instances: bool = False
    verify_empty: bool = False

    @classmethod
    def from_hint_names(cls, hints: object) -> "CountingTargetStrategy":
        if not isinstance(hints, (list, tuple, frozenset, set)):
            return cls()
        names = frozenset(
            value.strip().casefold()
            for value in hints
            if isinstance(value, str) and value.strip()
        )
        return cls(
            small_object="small_object" in names,
            dense_instances="dense_instances" in names,
            verify_empty="verify_empty" in names,
        )


class AgentCountingSettings(BaseModel):
    """Counting agent configuration. / 计数 Agent 配置。"""

    model_config = ConfigDict(extra="forbid")

    default_backend: Literal["auto", "qwen_point", "yolo_obb", "yolo_detect"] = "auto"


class YoloDetectorSettings(BaseModel):
    """One YOLO detector with its class mapping and priority. Validation is
    purely structural — weight files are never touched.
    一个 YOLO 检测器及其类别映射与优先级。校验纯结构性的——绝不访问权重文件。
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    enabled: bool = False
    weights: Path
    runtime: Literal["ultralytics", "onnx_yolov5_obb"] = "ultralytics"
    task: Literal["obb", "detect"] = "obb"
    model_id: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_dataset: str = Field(default="DOTAv1", min_length=1)
    priority: int = Field(default=100, ge=0)
    classes: list[str] = Field(min_length=1)
    aliases: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=0.20, ge=0.0, le=1.0)
    iou: float = Field(default=0.50, ge=0.0, le=1.0)
    image_size: int = Field(default=1024, gt=0)
    device: str = "0"
    max_detections: int = Field(default=1000, gt=0)
    # ONNX runtime provider policy: GPU by default, CPU only when explicitly
    # allowed. ONNX 运行时 provider 策略：默认 GPU，仅显式允许时使用 CPU。
    require_cuda: bool = True
    allow_cpu_fallback: bool = False
    boundary_duplicate_iou: float = Field(default=0.50, ge=0.0, le=1.0)
    boundary_duplicate_center_px: float = Field(default=16.0, gt=0.0)

    @model_validator(mode="after")
    def validate_provider_device_contract(self) -> "YoloDetectorSettings":
        """require_cuda=True demands a non-negative integer device; CPU mode
        demands device='cpu' so the session never requests CUDA silently.
        require_cuda=True 要求非负整数 device；CPU 模式要求 device='cpu'，
        使会话绝不静默请求 CUDA。"""
        if self.require_cuda:
            if not self.device.isdigit():
                raise ValueError(
                    "require_cuda=True requires device to be a non-negative "
                    f"integer, got {self.device!r}"
                )
        elif self.device != "cpu":
            raise ValueError(
                "require_cuda=False requires device='cpu', "
                f"got {self.device!r}"
            )
        if not self.require_cuda and self.allow_cpu_fallback:
            raise ValueError(
                "CPU-only mode must not enable CPU fallback"
            )
        return self

    @model_validator(mode="after")
    def validate_detector_contract(self) -> "YoloDetectorSettings":
        """Normalize and validate detector class declarations without I/O.
        在不访问文件系统的前提下规范并校验检测器类别声明。"""
        self.sha256 = self.sha256.casefold()
        normalized = [value.strip() for value in self.classes]
        folded = [value.casefold() for value in normalized]
        if len(folded) != len(set(folded)):
            raise ValueError("YOLO detector classes must be unique after normalization")
        known = set(folded)
        for alias, target in self.aliases.items():
            if target.strip().casefold() not in known:
                raise ValueError(f"YOLO alias {alias!r} targets unknown class {target!r}")
        if self.name in {"qwen_point", "vrsbench_qwen_count"}:
            raise ValueError(f"YOLO detector name {self.name!r} is reserved")
        self.classes = normalized
        return self


class YoloCountingSettings(BaseModel):
    """YOLO deployment schema and generic defaults.

    Concrete detector inventory belongs to runtime configuration, not Python.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    detectors: list[YoloDetectorSettings] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_enabled_detectors(self) -> "YoloCountingSettings":
        """Require an enabled detector only when YOLO execution is enabled.
        仅在启用 YOLO 执行时要求至少一个已启用检测器。"""
        names = [detector.name for detector in self.detectors]
        if len(names) != len(set(names)):
            raise ValueError("YOLO detector names must be unique")
        if self.enabled and not any(detector.enabled for detector in self.detectors):
            raise ValueError("enabled YOLO requires at least one enabled detector")
        return self
