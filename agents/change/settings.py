"""Change-domain local settings: conservative harmonization limits.

变化域局部设置：保守的一致化限制。纯声明配置，不读取环境、不访问权重。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChangeHarmonizationSettings(BaseModel):
    """Conservative PIF/LAB harmonization limits. / 保守的 PIF/LAB 一致化限制。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    version: str = "pif_lab_midpoint_v1"
    retain_raw_images: bool = True
    save_artifacts: bool = True
    calibration_file: Path | None = None
    reject_when_pif_mad_worse: bool = True
    max_pif_mad_degradation_ratio: float = Field(default=1.05, ge=1.0)
    match_sharpness: bool = True
    max_blur_sigma: float = Field(default=1.5, gt=0.0)
    min_retained_lapvar_ratio: float = Field(default=0.65, gt=0.0, le=1.0)
    sharpness_tolerance_ratio: float = Field(default=1.15, ge=1.0)
    min_pif_ratio: float = Field(default=0.02, ge=0.0, le=1.0)
    min_pif_pixels: int = Field(default=64, ge=1)
    pif_blur_ksize: int = Field(default=21, ge=3)
    pif_diff_k: float = Field(default=1.5, ge=0.0)
    pif_grad_k: float = Field(default=1.5, ge=0.0)
    max_abs_gain: float = Field(default=4.0, gt=0.0)
    max_abs_offset: float = Field(default=160.0, gt=0.0)
    max_clipped_pixel_ratio: float = Field(default=0.15, ge=0.0, le=1.0)


class ChangeProposalSettings(BaseModel):
    """Explainable difference proposal configuration. / 可解释差异候选配置。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    rgb_weight: float = Field(default=0.50, ge=0.0)
    edge_weight: float = Field(default=0.25, ge=0.0)
    structure_weight: float = Field(default=0.25, ge=0.0)
    threshold_quantile: float = Field(default=0.90, gt=0.0, lt=1.0)
    min_component_area_ratio: float = Field(default=0.0005, gt=0.0, lt=1.0)
    max_component_area_ratio: float = Field(default=0.50, gt=0.0, le=1.0)
    max_proposals: int = Field(default=6, ge=1, le=12)
    # These weights fuse the three major V2 branches. They are distinct from
    # rgb/edge/structure, which only compose the low-level branch.
    fusion_low_level_weight: float = Field(default=0.25, ge=0.0)
    fusion_feature_weight: float = Field(default=0.50, ge=0.0)
    fusion_semantic_weight: float = Field(default=0.25, ge=0.0)
    threshold_mode: Literal["pif_robust"] = "pif_robust"
    pif_threshold_k: float = Field(default=4.5, ge=0.0)
    threshold_floor: float = Field(default=0.10, ge=0.0, le=1.0)
    pif_fallback_quantile: float = Field(default=0.90, gt=0.0, lt=1.0)
    mask_close_kernel: int = Field(default=5, ge=1)
    proposal_padding_ratio: float = Field(default=0.08, ge=0.0, le=1.0)

    def model_post_init(self, __context: Any) -> None:
        if self.rgb_weight + self.edge_weight + self.structure_weight <= 0:
            raise ValueError("change proposal weights must have a positive sum")
        if (
            self.fusion_low_level_weight
            + self.fusion_feature_weight
            + self.fusion_semantic_weight
            <= 0
        ):
            raise ValueError("change proposal fusion weights must have a positive sum")
        if self.mask_close_kernel % 2 == 0:
            raise ValueError("mask_close_kernel must be odd")


class ChangeSemanticSettings(BaseModel):
    """Optional Change V2 dense-semantic strategy; disabled by default."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    feature_stage: int = 1
    tile_size: int = Field(default=768, ge=128)
    tile_overlap: int = Field(default=64, ge=0)
    local_match_radius: int = Field(default=1, ge=0, le=3)
    min_pif_feature_cells: int = Field(default=32, ge=1)
    feature_scale_epsilon: float = Field(default=1e-3, gt=0.0)
    semantic_confidence_floor: float = Field(default=0.45, ge=0.0, le=1.0)
    js_epsilon: float = Field(default=1e-6, gt=0.0)
    failure_policy: Literal["fallback_legacy", "fail"] = "fallback_legacy"

    def model_post_init(self, __context: Any) -> None:
        if self.tile_overlap >= self.tile_size:
            raise ValueError("tile_overlap must be smaller than tile_size")


class ChangeReviewSettings(BaseModel):
    """Rule reviewer configuration. / 规则复核配置。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    require_proposal_evidence: bool = True


class AgentChangeSettings(BaseModel):
    """Change-agent local configuration group. / 变化 Agent 局部配置组。"""

    model_config = ConfigDict(extra="forbid")

    harmonization: ChangeHarmonizationSettings = Field(default_factory=ChangeHarmonizationSettings)
    proposals: ChangeProposalSettings = Field(default_factory=ChangeProposalSettings)
    semantic: ChangeSemanticSettings = Field(default_factory=ChangeSemanticSettings)
    review: ChangeReviewSettings = Field(default_factory=ChangeReviewSettings)
