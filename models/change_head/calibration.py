"""Validated calibration parameters persisted beside a ChangeHead."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChangeHeadCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    temperature: float = Field(gt=0.0)
    rescue_probability_threshold: float = Field(ge=0.0, le=1.0)
    rescue_min_component_area_ratio: float = Field(gt=0.0, lt=1.0)
    validation_reliability: float = Field(ge=0.0, le=1.0)
    optional_expert_missing_reliability_factor: float = Field(
        default=0.90, gt=0.0, le=1.0
    )

