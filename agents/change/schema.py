"""Runtime schemas for auditable bi-temporal change preprocessing.

可审计双时相变化预处理的运行时 Schema。全部类型可序列化，不含任何语义
变化描述（不生成语义结论）。
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from agents.counting.schema import IssueRecord
from agents.schema import AgentResult, VisualEvidence

CANONICAL_NO_CHANGE = "No significant semantic change detected."


class ChangeInitialResult(AgentResult):
    """Change-only tolerant initial schema for canonical negative outputs."""

    @model_validator(mode="before")
    @classmethod
    def normalize_canonical_negative(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if (
            data.get("agent_name") == "change_agent"
            and str(data.get("answer", "")).strip() == CANONICAL_NO_CHANGE
        ):
            data["boxes"] = []
            data["evidence"] = []
            data["evidence_items"] = []
            geometry = dict(data.get("geometry") or {})
            normalizations = list(geometry.get("change_input_normalizations") or [])
            normalizations.append("canonical_no_change_cleared_model_evidence")
            geometry["change_input_normalizations"] = list(dict.fromkeys(normalizations))
            data["geometry"] = geometry
        return data


RegistrationModel = Literal["identity", "similarity", "affine", "homography", "none"]
RegistrationStatus = Literal["skipped", "applied", "rejected", "failed"]


class RegistrationMetrics(BaseModel):
    """JSON-safe quality measurements for one registration attempt.

    This is deliberately a data contract only.  It does not prescribe how
    matches or transforms are computed; the registration implementation owns
    those details in a later Change V3 stage.
    """

    model_config = ConfigDict(extra="forbid")

    match_count: int = Field(default=0, ge=0)
    inlier_count: int = Field(default=0, ge=0)
    inlier_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    median_reprojection_error: float = Field(default=0.0, ge=0.0)
    p95_reprojection_error: float = Field(default=0.0, ge=0.0)
    overlap_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    scale_x: float = Field(default=1.0, gt=0.0)
    scale_y: float = Field(default=1.0, gt=0.0)
    rotation_deg: float = 0.0
    translation_x: float = 0.0
    translation_y: float = 0.0
    perspective_magnitude: float = Field(default=0.0, ge=0.0)

    @field_validator(
        "inlier_ratio",
        "median_reprojection_error",
        "p95_reprojection_error",
        "overlap_ratio",
        "scale_x",
        "scale_y",
        "rotation_deg",
        "translation_x",
        "translation_y",
        "perspective_magnitude",
    )
    @classmethod
    def finite_numeric_values(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("registration metrics must be finite")
        return value

    @model_validator(mode="after")
    def validate_inliers(self) -> "RegistrationMetrics":
        if self.inlier_count > self.match_count:
            raise ValueError("inlier_count cannot exceed match_count")
        return self


class RegistrationDecision(BaseModel):
    """Stable decision contract for a registration quality gate."""

    model_config = ConfigDict(extra="forbid")

    version: str
    status: RegistrationStatus
    model: RegistrationModel
    reason_codes: list[str] = Field(default_factory=list)
    used_for_comparison: bool = False


class RegistrationReport(BaseModel):
    """Serializable registration audit record; never contains host paths."""

    model_config = ConfigDict(extra="forbid")

    decision: RegistrationDecision
    metrics: RegistrationMetrics | None = None
    transform_matrix: list[list[float]] | None = None
    source_size_t1: list[int] = Field(default_factory=list)
    source_size_t2: list[int] = Field(default_factory=list)
    output_size: list[int] = Field(default_factory=list)
    diagnostics: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("transform_matrix")
    @classmethod
    def validate_transform_matrix(
        cls, value: list[list[float]] | None
    ) -> list[list[float]] | None:
        if value is None:
            return None
        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise ValueError("transform_matrix must be a 3x3 matrix")
        if any(not math.isfinite(float(item)) for row in value for item in row):
            raise ValueError("transform_matrix must contain finite values")
        return [[float(item) for item in row] for row in value]

    @field_validator("source_size_t1", "source_size_t2", "output_size")
    @classmethod
    def validate_sizes(cls, value: list[int]) -> list[int]:
        if value and (len(value) != 2 or any(item <= 0 for item in value)):
            raise ValueError("registration sizes must be positive [width, height]")
        return value


class PairValidationReport(BaseModel):
    """Validation facts recorded before any pixel transform.
    像素变换前记录的校验事实。"""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    temporal_roles_valid: bool
    same_size: bool
    alignment_status: Literal[
        "assumed_dataset_aligned",
        "metadata_aligned",
        "weakly_aligned",
        "same_size_unverified",
        "registration_required",
        "unreliable",
    ]
    original_sizes: list[list[int]] = Field(default_factory=list)
    warnings: list[IssueRecord] = Field(default_factory=list)
    # A decoded, role-valid pair may proceed to a future registration stage
    # even when the legacy ``valid`` flag remains false for compatibility.
    registration_eligible: bool = False
    strong_alignment_evidence: bool = False
    registration_required: bool = False


class HarmonizationMetrics(BaseModel):
    """Pair metrics; every ratio is represented on the inclusive 0..1 scale.
    图对指标；所有比例统一使用闭区间 0..1。"""

    model_config = ConfigDict(extra="forbid")

    pif_ratio: float = Field(ge=0.0, le=1.0)
    mad_full_before: float
    mad_full_after: float
    mad_pif_before: float
    mad_pif_after: float
    corr_full_before: float | None = None
    corr_full_after: float | None = None
    corr_pif_before: float | None = None
    corr_pif_after: float | None = None
    pct_diff_gt20_before: float = Field(ge=0.0, le=1.0)
    pct_diff_gt20_after: float = Field(ge=0.0, le=1.0)
    lapvar_t1_before: float
    lapvar_t2_before: float
    lapvar_t1_after: float
    lapvar_t2_after: float


class HarmonizationDecision(BaseModel):
    """Stable quality-gate decision. / 稳定的质量门控决策。"""

    model_config = ConfigDict(extra="forbid")

    version: str
    status: Literal["applied", "skipped", "rejected", "failed"]
    reason_codes: list[str]
    metrics: HarmonizationMetrics | None
    used_for_proposal: bool
    raw_retained: bool = True


class SemanticTransition(BaseModel):
    """Auxiliary T1-to-T2 semantic candidate, never a ground-truth claim."""

    model_config = ConfigDict(extra="forbid")

    from_class: str
    from_confidence: float = Field(ge=0.0, le=1.0)
    to_class: str
    to_confidence: float = Field(ge=0.0, le=1.0)
    changed_class: str | None = None
    support_ratio: float = Field(ge=0.0, le=1.0)
    transition_confidence: float = Field(ge=0.0, le=1.0)

    @property
    def t1_top_class(self) -> str:
        return self.from_class

    @property
    def t1_confidence(self) -> float:
        return self.from_confidence

    @property
    def t2_top_class(self) -> str:
        return self.to_class

    @property
    def t2_confidence(self) -> float:
        return self.to_confidence


class ChangeProposal(BaseModel):
    """Explainable attention proposal, not a semantic change claim.
    可解释关注候选，不代表语义变化结论。"""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    box: list[int]
    pixel_box: list[int]
    score: float = Field(ge=0.0, le=1.0)
    area_ratio: float = Field(ge=0.0, le=1.0)
    source: Literal["difference_map_v1", "fused_change_v2"] = "difference_map_v1"
    evidence_filenames: list[str] = Field(default_factory=list)
    component_scores: dict[str, float] = Field(default_factory=dict)
    mask_filename: str | None = None
    semantic_transition: SemanticTransition | None = None
    semantic_transitions: list[dict[str, JsonValue]] = Field(default_factory=list)
    semantic_consensus: dict[str, JsonValue] = Field(default_factory=dict)
    effective_weights: dict[str, float] = Field(default_factory=dict)
    reliability: dict[str, float] = Field(default_factory=dict)
    registration_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class StructuralRescueCandidate(BaseModel):
    """Building-only rescue evidence kept separate from core proposals."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    expert_id: str
    label: Literal["building"] = "building"
    direction: Literal["added", "removed"]
    box: tuple[int, int, int, int]
    normalized_box: tuple[int, int, int, int]
    score: float = Field(ge=0.0, le=1.0)
    target_mean_probability: float = Field(ge=0.0, le=1.0)
    target_p10_probability: float = Field(ge=0.0, le=1.0)
    target_p50_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    source_mean_probability: float = Field(ge=0.0, le=1.0)
    source_p90_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    source_p95_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    source_max_probability: float = Field(ge=0.0, le=1.0)
    area_px: int = Field(ge=1)
    area_ratio: float = Field(gt=0.0, le=1.0)
    edge_flags: tuple[str, ...] = ()
    registration_tolerance_px: int = Field(ge=0)
    supporting_core_signals: dict[str, float] = Field(default_factory=dict)
    artifact_files: tuple[str, ...] = ()


RescueVerdict = Literal[
    "confirmed_added_building",
    "confirmed_removed_building",
    "reject",
    "insufficient",
]


class BuildingRescueCandidateReview(BaseModel):
    """One narrow review for one building-only rescue candidate."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    verdict: RescueVerdict
    visible_building_count: int | None = Field(default=None, ge=0)
    reason: str = Field(min_length=1)


class BuildingRescueReview(BaseModel):
    """Bounded second-pass review; direction checks use supplied candidates."""

    model_config = ConfigDict(extra="forbid")

    reviews: tuple[BuildingRescueCandidateReview, ...]
    final_answer: str | None = None

    @model_validator(mode="after")
    def validate_review_answer(self) -> "BuildingRescueReview":
        candidate_ids = [item.candidate_id for item in self.reviews]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("building rescue review contains duplicate candidate IDs")
        confirmed = any(item.verdict.startswith("confirmed_") for item in self.reviews)
        if not confirmed and self.final_answer is not None:
            raise ValueError("building rescue final_answer requires a confirmation")
        return self


class ChangePreprocessResult(BaseModel):
    """Serializable preprocessing result referenced by trace and artifacts.
    由 trace 与产物引用的可序列化预处理结果。"""

    model_config = ConfigDict(extra="forbid")

    validation: PairValidationReport
    decision: HarmonizationDecision
    proposals: list[ChangeProposal]
    artifact_files: dict[str, str]
    transform_summary: dict[str, object] = Field(default_factory=dict)
    diagnostics: dict[str, JsonValue] = Field(default_factory=dict)
    registration: RegistrationReport | None = None
    rescue_candidates: list[StructuralRescueCandidate] = Field(default_factory=list)


CandidateVerdict = Literal[
    "persistent_change", "appearance_only", "registration_artifact", "transient",
    "insufficient_visual_evidence",
]
GlobalVerdict = Literal[
    "persistent_change", "no_persistent_change", "appearance_only",
    "registration_artifact", "insufficient_visual_evidence",
]
PersistentChangeCategory = Literal[
    "building_structure", "road_network", "vegetation_extent", "land_use_conversion",
    "water_geometry", "other_persistent_infrastructure",
]


class ChangeCandidateReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    proposal_id: str
    verdict: CandidateVerdict
    t1_state: str
    t2_state: str
    reason: str
    change_category: PersistentChangeCategory | None = None
    persistent_geometry_changed: bool | None = None
    geometry_change_description: str | None = None
    normalization_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_review_category(cls, value: object) -> object:
        """Drop irrelevant categories before Literal validation.

        Models sometimes attach a persistent-change category to a clearly
        non-persistent candidate.  The category is not authoritative in that
        case and must not turn an otherwise usable adjudication into a schema
        failure.  An explicit ``persistent_change + transient`` combination is
        a structured contradiction, so it is deterministically downgraded.
        """

        if not isinstance(value, dict):
            return value
        data = dict(value)
        verdict = data.get("verdict")
        category = data.get("change_category")
        reasons = list(data.get("normalization_reasons") or [])
        pseudo_verdicts = {
            "transient": "transient",
            "appearance_only": "appearance_only",
            "registration_artifact": "registration_artifact",
            "insufficient_visual_evidence": "insufficient_visual_evidence",
        }
        if verdict == "persistent_change" and category in pseudo_verdicts:
            data["verdict"] = pseudo_verdicts[category]
            data["change_category"] = None
            data["persistent_geometry_changed"] = False
            reasons.append(
                "ADJUDICATION_TRANSIENT_CATEGORY_DOWNGRADED"
                if category == "transient"
                else "ADJUDICATION_NONPERSISTENT_CATEGORY_DOWNGRADED"
            )
        elif verdict != "persistent_change" and category is not None:
            data["change_category"] = None
            reasons.append("ADJUDICATION_NONPERSISTENT_CATEGORY_CLEARED")
        data["normalization_reasons"] = list(dict.fromkeys(reasons))
        return data

    @model_validator(mode="after")
    def validate_category(self) -> "ChangeCandidateReview":
        if (self.verdict == "persistent_change") != (self.change_category is not None):
            raise ValueError("persistent candidate verdict requires category only")
        return self


class ChangeGlobalReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    verdict: GlobalVerdict
    t1_state: str
    t2_state: str
    reason: str
    change_category: PersistentChangeCategory | None = None
    persistent_geometry_changed: bool | None = None
    geometry_change_description: str | None = None
    normalization_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_review_category(cls, value: object) -> object:
        """Normalize contradictory global review fields before literals."""

        if not isinstance(value, dict):
            return value
        data = dict(value)
        verdict = data.get("verdict")
        category = data.get("change_category")
        reasons = list(data.get("normalization_reasons") or [])
        pseudo_verdicts = {
            "transient": "no_persistent_change",
            "appearance_only": "no_persistent_change",
            "registration_artifact": "no_persistent_change",
            "insufficient_visual_evidence": "no_persistent_change",
        }
        if verdict == "persistent_change" and category in pseudo_verdicts:
            data["verdict"] = pseudo_verdicts[category]
            data["change_category"] = None
            data["persistent_geometry_changed"] = False
            reasons.append(
                "ADJUDICATION_TRANSIENT_CATEGORY_DOWNGRADED"
                if category == "transient"
                else "ADJUDICATION_NONPERSISTENT_CATEGORY_DOWNGRADED"
            )
        elif verdict != "persistent_change" and category is not None:
            data["change_category"] = None
            reasons.append("ADJUDICATION_NONPERSISTENT_CATEGORY_CLEARED")
        data["normalization_reasons"] = list(dict.fromkeys(reasons))
        return data

    @model_validator(mode="after")
    def validate_category(self) -> "ChangeGlobalReview":
        if (self.verdict == "persistent_change") != (self.change_category is not None):
            raise ValueError("persistent global verdict requires category only")
        return self


class ChangeAdjudicationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_name: Literal["change_agent"]
    global_review: ChangeGlobalReview
    candidate_reviews: list[ChangeCandidateReview]
    answer: str
    boxes: list[list[int]] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    evidence_items: list[VisualEvidence] = Field(default_factory=list)
    geometry: dict[str, JsonValue] = Field(default_factory=dict)
    status: Literal["completed", "partial"] = "completed"

    @model_validator(mode="before")
    @classmethod
    def normalize_canonical_negative(cls, value: object) -> object:
        """Keep canonical no-change output free of rejected model evidence."""

        if not isinstance(value, dict):
            return value
        data = dict(value)
        data.pop("$schema", None)
        if (
            data.get("agent_name") == "change_agent"
            and str(data.get("answer", "")).strip() == CANONICAL_NO_CHANGE
        ):
            data["boxes"] = []
            data["evidence"] = []
            data["evidence_items"] = []
            geometry = dict(data.get("geometry") or {})
            normalizations = list(geometry.get("change_input_normalizations") or [])
            normalizations.append("canonical_no_change_cleared_model_evidence")
            geometry["change_input_normalizations"] = list(dict.fromkeys(normalizations))
            data["geometry"] = geometry
        return data
