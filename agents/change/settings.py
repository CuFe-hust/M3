"""Change-domain local settings: conservative harmonization limits.

变化域局部设置：保守的一致化限制。纯声明配置，不读取环境、不访问权重。
"""

from __future__ import annotations

from pathlib import Path

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
