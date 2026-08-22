"""Versioned, torch-free ChangeHead checkpoint manifest schema."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from models.base import (
    LEARNED_CHANGE_INPUT_CONTRACT_VERSION,
    LEARNED_CHANGE_OUTPUT_CONTRACT_VERSION,
    LearnedChangeExpertRequirement,
    LearnedChangeInputSpec,
    validate_logical_model_id,
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ChangeHeadManifestError(ValueError):
    """Stable public error for invalid learned checkpoint metadata."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def hash_class_names(class_names: Sequence[str]) -> str:
    """Hash the canonical ordered class-name list shared by train/runtime."""

    payload = json.dumps(
        list(class_names), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ChangeHeadExpertManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expert_id: str
    logical_model_id: str
    weights_sha256: str = Field(pattern=_SHA256_PATTERN)
    class_names_sha256: str = Field(pattern=_SHA256_PATTERN)
    feature_stages: tuple[int, ...]
    feature_channels_by_stage: dict[int, int]
    required: bool = True
    use_semantic_probabilities: bool = True
    missing_policy: Literal["error", "zero_with_presence_mask"] = "error"

    @field_validator("expert_id")
    @classmethod
    def validate_expert_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("expert_id must not be empty")
        return value

    @field_validator("logical_model_id")
    @classmethod
    def validate_logical_id(cls, value: str) -> str:
        return validate_logical_model_id(value, where="logical_model_id")

    @field_validator("feature_stages")
    @classmethod
    def validate_stages(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or len(set(value)) != len(value):
            raise ValueError("feature_stages must be non-empty and unique")
        if any(isinstance(stage, bool) or stage < 0 for stage in value):
            raise ValueError("feature_stages must contain non-negative integers")
        return value

    @field_validator("feature_channels_by_stage")
    @classmethod
    def validate_channels(cls, value: dict[int, int]) -> dict[int, int]:
        if not value or any(stage < 0 or channels <= 0 for stage, channels in value.items()):
            raise ValueError("feature_channels_by_stage must contain positive channels")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> "ChangeHeadExpertManifest":
        if set(self.feature_channels_by_stage) != set(self.feature_stages):
            raise ValueError("feature channel stages must match feature_stages")
        if self.required and self.missing_policy != "error":
            raise ValueError("required expert must use missing_policy=error")
        return self

    def to_requirement(self) -> LearnedChangeExpertRequirement:
        return LearnedChangeExpertRequirement(
            expert_id=self.expert_id,
            logical_model_id=self.logical_model_id,
            weights_sha256=self.weights_sha256,
            class_names_sha256=self.class_names_sha256,
            feature_stages=self.feature_stages,
            required=self.required,
            use_semantic_probabilities=self.use_semantic_probabilities,
            missing_policy=self.missing_policy,
        )


class ChangeHeadArchitectureManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["multi_expert_siamese_change_head_v1"]
    hidden_dim: int = Field(gt=0)
    semantic_dim: int = Field(gt=0)
    decoder_dim: int = Field(gt=0)
    optional_expert_dropout_supported: bool
    use_pif_mask: bool
    use_rgb_pair: bool


class ChangeHeadManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    input_contract_version: str
    output_contract_version: str
    architecture: ChangeHeadArchitectureManifest
    experts: tuple[ChangeHeadExpertManifest, ...]
    pipeline_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    model_weights_sha256: str = Field(pattern=_SHA256_PATTERN)
    created_from_git_commit: str
    training_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    logits_semantics: Literal["binary_change_logits"] = "binary_change_logits"

    @model_validator(mode="after")
    def validate_manifest(self) -> "ChangeHeadManifest":
        if self.input_contract_version != LEARNED_CHANGE_INPUT_CONTRACT_VERSION:
            raise ChangeHeadManifestError(
                "LEARNED_CHANGE_CONTRACT_MISMATCH",
                "input contract version is unsupported",
            )
        if self.output_contract_version != LEARNED_CHANGE_OUTPUT_CONTRACT_VERSION:
            raise ChangeHeadManifestError(
                "LEARNED_CHANGE_CONTRACT_MISMATCH",
                "output contract version is unsupported",
            )
        ids = [expert.expert_id for expert in self.experts]
        if len(ids) != len(set(ids)):
            raise ChangeHeadManifestError(
                "LEARNED_CHANGE_EXPERT_ID_MISMATCH",
                "expert ids must be unique",
            )
        if not any(expert.required for expert in self.experts):
            raise ValueError("manifest requires at least one required expert")
        if any(
            not expert.required
            and expert.missing_policy == "zero_with_presence_mask"
            and not self.architecture.optional_expert_dropout_supported
            for expert in self.experts
        ):
            raise ValueError(
                "optional zero-missing experts require expert dropout support"
            )
        return self

    def input_spec(self) -> LearnedChangeInputSpec:
        return LearnedChangeInputSpec(
            contract_version=self.input_contract_version,
            expert_requirements=tuple(expert.to_requirement() for expert in self.experts),
            use_pif_mask=self.architecture.use_pif_mask,
            use_rgb_pair=self.architecture.use_rgb_pair,
            optional_expert_dropout_supported=(
                self.architecture.optional_expert_dropout_supported
            ),
        )
