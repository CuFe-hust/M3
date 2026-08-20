"""Change-domain local settings: conservative harmonization limits.

变化域局部设置：保守的一致化限制。纯声明配置，不读取环境、不访问权重。
"""

from __future__ import annotations

import math
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


class ChangeRegistrationSettings(BaseModel):
    """Conservative geometric-registration contract.

    The fields are intentionally limited to first-generation global models.
    The implementation is added in the dedicated registration stage; this
    settings object only provides a validated, YAML-overridable boundary.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    version: str = "global_registration_v1"
    prefer_metadata_alignment: bool = True
    matcher: Literal["opencv"] = "opencv"
    feature_detector: Literal["sift", "orb"] = "sift"
    max_features: int = Field(default=2000, ge=64)
    ratio_test: float = Field(default=0.75, gt=0.0, lt=1.0)
    min_matches: int = Field(default=8, ge=1)
    min_inliers: int = Field(default=6, ge=1)
    min_inlier_ratio: float = Field(default=0.35, ge=0.0, le=1.0)
    max_median_reprojection_error: float = Field(default=3.0, gt=0.0)
    min_overlap_ratio: float = Field(default=0.60, ge=0.0, le=1.0)
    allow_similarity: bool = True
    allow_affine: bool = True
    allow_homography: bool = True
    max_scale_ratio: float = Field(default=1.35, ge=1.0)
    max_rotation_deg: float = Field(default=15.0, ge=0.0, le=180.0)
    max_translation_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    max_perspective_magnitude: float = Field(default=0.002, ge=0.0)
    quality_policy: Literal["fallback_raw", "fail"] = "fallback_raw"
    save_artifacts: bool = True

    def model_post_init(self, __context: Any) -> None:
        if self.min_inliers > self.min_matches:
            raise ValueError("min_inliers cannot exceed min_matches")


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
    """Default Change dense-semantic strategy with audited legacy fallback."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    feature_stage: int = 1
    # Full perception defaults to the V3 feature pyramid.  An empty tuple
    # still migrates the legacy ``feature_stage`` setting.
    feature_stages: tuple[int, ...] = (1, 2, 3)
    feature_stage_weights: dict[int, float] = Field(
        default_factory=lambda: {1: 1.0, 2: 1.0, 3: 1.0}
    )
    tile_size: int = Field(default=768, ge=128)
    tile_overlap: int = Field(default=64, ge=0)
    local_match_radius: int = Field(default=1, ge=0, le=3)
    min_pif_feature_cells: int = Field(default=32, ge=1)
    feature_scale_epsilon: float = Field(default=1e-3, gt=0.0)
    semantic_confidence_floor: float = Field(default=0.45, ge=0.0, le=1.0)
    js_epsilon: float = Field(default=1e-6, gt=0.0)
    failure_policy: Literal["fallback_legacy", "fail"] = "fallback_legacy"
    multi_expert_enabled: bool = True
    max_experts: int = Field(default=4, ge=1, le=5)
    min_successful_experts: int = Field(default=1, ge=1)

    def model_post_init(self, __context: Any) -> None:
        if self.tile_overlap >= self.tile_size:
            raise ValueError("tile_overlap must be smaller than tile_size")
        if (
            isinstance(self.feature_stage, bool)
            or not isinstance(self.feature_stage, int)
            or self.feature_stage < 0
            or self.feature_stage > 4
        ):
            raise ValueError("feature_stage must be an integer in the range 0..4")
        stages = tuple(dict.fromkeys(self.feature_stages))
        # A legacy YAML file may only contain feature_stage.  The default
        # one-element tuple is treated as the migration sentinel in that case.
        if stages == (1,) and self.feature_stage != 1:
            stages = (self.feature_stage,)
        if not stages:
            stages = (self.feature_stage,)
        if any(
            isinstance(stage, bool) or not isinstance(stage, int) or stage < 0 or stage > 4
            for stage in stages
        ):
            raise ValueError("feature_stages must contain integers in the range 0..4")
        weights = dict(self.feature_stage_weights)
        if any(
            isinstance(stage, bool) or not isinstance(stage, int) or stage < 0 or stage > 4
            for stage in weights
        ):
            raise ValueError("feature_stage_weights contains an invalid stage")
        for stage in stages:
            weights.setdefault(stage, 1.0)
        selected_weights = {stage: float(weights[stage]) for stage in stages}
        if any(value <= 0.0 or not math.isfinite(value) for value in selected_weights.values()):
            raise ValueError("feature_stage_weights must be finite and positive")
        total = sum(selected_weights.values())
        object.__setattr__(self, "feature_stages", stages)
        object.__setattr__(
            self,
            "feature_stage_weights",
            {stage: value / total for stage, value in selected_weights.items()},
        )
        if self.min_successful_experts > self.max_experts:
            raise ValueError("min_successful_experts cannot exceed max_experts")


class ChangeReliabilitySettings(BaseModel):
    """Deterministic reliability clamps used to modulate proposal branches."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    registration_error_scale: float = Field(default=3.0, gt=0.0)
    min_branch_reliability: float = Field(default=0.05, ge=0.0, le=1.0)
    semantic_confidence_floor: float = Field(default=0.45, ge=0.0, le=1.0)
    feature_residual_scale: float = Field(default=1.0, gt=0.0)


class ChangeLearnedChangeSettings(BaseModel):
    """Optional inference hook for one future learned change head.

    This declaration contains no checkpoint path or training parameter.  A
    concrete client is supplied by the application composition root.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    fusion_weight: float = Field(default=0.0, ge=0.0)
    failure_policy: Literal["fallback_rule", "fail"] = "fallback_rule"


class ChangeReviewSettings(BaseModel):
    """Rule reviewer configuration. / 规则复核配置。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    require_proposal_evidence: bool = True
    no_change_conflict_min_score: float = Field(default=0.18, ge=0.0, le=1.0)
    no_change_conflict_min_proposals: int = Field(default=2, ge=1)
    no_change_conflict_min_total_area_ratio: float = Field(
        default=0.01, ge=0.0, le=1.0
    )
    require_temporal_pair_evidence: bool = True
    adjudication_enabled: bool = True
    negative_strong_score: float = Field(default=0.35, ge=0.0, le=1.0)
    negative_moderate_score: float = Field(default=0.24, ge=0.0, le=1.0)
    negative_min_reliable_components: int = Field(default=2, ge=1)
    negative_large_total_area_ratio: float = Field(default=0.08, ge=0.0, le=1.0)
    negative_edge_score: float = Field(default=0.20, ge=0.0, le=1.0)
    negative_edge_margin_ratio: float = Field(default=0.06, ge=0.0, le=0.5)


class ChangeEvidenceSettings(BaseModel):
    """Bounded, role-labelled visual evidence sent to the VLM."""

    model_config = ConfigDict(extra="forbid")

    initial_max_proposals: int = Field(default=3, ge=0, le=6)
    adjudication_max_proposals: int = Field(default=3, ge=1, le=6)
    attach_proposal_overlay: bool = True
    attach_registered_global: bool = False
    attach_harmonized_global: bool = False
    edge_margin_ratio: float = Field(default=0.06, ge=0.0, le=0.5)


class AgentChangeSettings(BaseModel):
    """Change-agent local configuration group. / 变化 Agent 局部配置组。"""

    model_config = ConfigDict(extra="forbid")

    harmonization: ChangeHarmonizationSettings = Field(default_factory=ChangeHarmonizationSettings)
    registration: ChangeRegistrationSettings = Field(default_factory=ChangeRegistrationSettings)
    proposals: ChangeProposalSettings = Field(default_factory=ChangeProposalSettings)
    semantic: ChangeSemanticSettings = Field(default_factory=ChangeSemanticSettings)
    reliability: ChangeReliabilitySettings = Field(default_factory=ChangeReliabilitySettings)
    learned_change: ChangeLearnedChangeSettings = Field(
        default_factory=ChangeLearnedChangeSettings
    )
    evidence: ChangeEvidenceSettings = Field(default_factory=ChangeEvidenceSettings)
    review: ChangeReviewSettings = Field(default_factory=ChangeReviewSettings)
