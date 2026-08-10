"""Stable, JSON-safe presentation models for the offline Report V2 bundle."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from evaluation.records import EvaluationRecord


class _ViewModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunMetadata(_ViewModel):
    run_id: str
    dataset: str | None = None
    split: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    config_hash: str | None = None
    model_ids: dict[str, str] = Field(default_factory=dict)
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    created_at: str | None = None
    sample_filter: str | None = None


class LatencySummary(_ViewModel):
    count: int = Field(default=0, ge=0)
    mean_seconds: float | None = None
    p50_seconds: float | None = None
    p95_seconds: float | None = None


RoutingAttemptStatus = Literal[
    "selected", "succeeded", "partial", "failed", "unavailable",
    "zero_review", "skipped",
]


class RoutingAttemptView(_ViewModel):
    backend_name: str
    backend_kind: str | None = None
    status: RoutingAttemptStatus
    reason_code: str | None = None


class FallbackTransitionView(_ViewModel):
    from_backend: str | None = None
    to_backend: str | None = None
    reason_code: str | None = None


class RoutingView(_ViewModel):
    resolved_task: str | None = None
    execution_agent: str | None = None
    candidate_backends: list[str] = Field(default_factory=list)
    attempted_backends: list[RoutingAttemptView] = Field(default_factory=list)
    primary_backend: str | None = None
    primary_backend_kind: str | None = None
    final_backend: str | None = None
    final_backend_kind: str | None = None
    fallback_used: bool = False
    fallback_history: list[FallbackTransitionView] = Field(default_factory=list)
    review_backend: str | None = None
    selection_reason: str | None = None


VisualStatus = Literal[
    "available", "not_materialized", "omitted_by_budget", "missing_source",
    "invalid_source", "dimension_mismatch", "unsupported_geometry",
]


class VisualAssetView(_ViewModel):
    image_id: str
    role: str
    original_asset: str | None = None
    overlay_asset: str | None = None
    width: int | None = None
    height: int | None = None
    status: VisualStatus = "not_materialized"


class PointPreview(_ViewModel):
    point_id: str | None = None
    x: float
    y: float
    confidence: float | None = None
    accepted: bool
    rejection_reason: str | None = None
    source: str | None = None
    backend_name: str | None = None
    model_id: str | None = None
    source_class: str | None = None
    source_dataset: str | None = None
    weights_sha256: str | None = None


class CountingReportDetail(_ViewModel):
    kind: Literal["counting"] = "counting"
    target: str | None = None
    predicted_count: int | None = None
    gold_count: int | None = None
    absolute_error: int | None = None
    exact_match: bool | None = None
    counting_status: str | None = None
    tile_count: int | None = None
    initial_tile_count: int | None = None
    leaf_tile_count: int | None = None
    succeeded_tile_count: int = 0
    failed_tile_count: int = 0
    accepted_point_count: int = 0
    rejected_point_count: int = 0
    merged_group_count: int = 0
    unresolved_conflict_count: int = 0
    warning_codes: list[str] = Field(default_factory=list)
    counting_mode: str | None = None
    provenance_usage: dict[str, int] = Field(default_factory=dict)
    accepted_preview: list[PointPreview] = Field(default_factory=list, max_length=50)
    rejected_preview: list[PointPreview] = Field(default_factory=list, max_length=100)


class GeneralVQAReportDetail(_ViewModel):
    kind: Literal["general_vqa"] = "general_vqa"
    question: str | None = None
    reference_answers: list[str] = Field(default_factory=list)
    prediction: str | None = None
    exact_match: bool | None = None
    judge_status: str = "not_requested"
    judge_score: float | None = None
    judge_concise_rationale: str | None = None
    visual_evidence_count: int = 0
    geometry_repair_severity: str | None = None


class GroundingReportDetail(_ViewModel):
    kind: Literal["grounding"] = "grounding"
    prediction: str | None = None
    reference: list[str] = Field(default_factory=list)
    predicted_boxes: list[list[float]] = Field(default_factory=list)
    ground_truth_boxes: list[list[float]] = Field(default_factory=list)
    iou: float | None = None
    iou_at_0_5: bool | None = None
    geometry_repair_severity: str | None = None


class SpatialReportDetail(_ViewModel):
    kind: Literal["spatial"] = "spatial"
    question: str | None = None
    prediction: str | None = None
    reference_answers: list[str] = Field(default_factory=list)
    evidence_text: list[str] = Field(default_factory=list)
    evidence_item_count: int = 0
    geometry_repair_severity: str | None = None


class ChangeReportDetail(_ViewModel):
    kind: Literal["change"] = "change"
    question: str | None = None
    prediction: str | None = None
    reference_answers: list[str] = Field(default_factory=list)
    evidence_text: list[str] = Field(default_factory=list)
    evidence_item_count: int = 0
    geometry_summary: str | None = None
    geometry_repair_severity: str | None = None


class CaptionReportDetail(_ViewModel):
    kind: Literal["caption"] = "caption"
    generated_caption: str | None = None
    reference_captions: list[str] = Field(default_factory=list)
    metric_status: str | None = None


TaskDetail = Annotated[
    CountingReportDetail | GeneralVQAReportDetail | GroundingReportDetail
    | SpatialReportDetail | ChangeReportDetail | CaptionReportDetail,
    Field(discriminator="kind"),
]


class ReportSample(_ViewModel):
    sample_id: str
    run_task: str
    task: str
    state: str
    error_code: str | None = None
    result_path: str | None = None
    updated_at: str | None = None
    question: str | None = None
    reference_answers: list[str] = Field(default_factory=list)
    prediction: str | None = None
    resolved_task: str | None = None
    execution_agent: str | None = None
    fallback_used: bool = False
    judge_status: str = "not_requested"
    inference_seconds: float | None = None
    evaluation: EvaluationRecord | None = None
    result_quality: Literal["correct", "incorrect", "unknown", "not_applicable"] = "unknown"
    routing: RoutingView = Field(default_factory=RoutingView)
    warnings: list[str] = Field(default_factory=list)
    visuals: list[VisualAssetView] = Field(default_factory=list)
    task_detail: TaskDetail | None = None


class TaskSummary(_ViewModel):
    run_task: str
    total: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    partial: int = Field(ge=0)
    failed: int = Field(ge=0)
    skipped: int = Field(ge=0)
    fallback_count: int = Field(ge=0)
    fallback_rate: float = Field(ge=0.0, le=1.0)
    agent_usage: dict[str, int] = Field(default_factory=dict)
    judge_status_counts: dict[str, int] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    judge_metrics: dict[str, Any] = Field(default_factory=dict)
    correct: int = 0
    incorrect: int = 0
    unknown_quality: int = 0
    latency: LatencySummary = Field(default_factory=LatencySummary)
    primary_backend_usage: dict[str, int] = Field(default_factory=dict)
    final_backend_usage: dict[str, int] = Field(default_factory=dict)
    warning_count: int = 0


class FallbackTransitionSummary(FallbackTransitionView):
    count: int = Field(ge=1)


class RoutingSummary(_ViewModel):
    primary_backend_usage: dict[str, int] = Field(default_factory=dict)
    final_backend_usage: dict[str, int] = Field(default_factory=dict)
    fallback_count: int = 0
    fallback_rate: float = 0.0
    fallback_transitions: list[FallbackTransitionSummary] = Field(default_factory=list)
    fallback_reason_counts: dict[str, int] = Field(default_factory=dict)


class FailureSummary(_ViewModel):
    sample_error_codes: dict[str, int] = Field(default_factory=dict)
    warning_codes: dict[str, int] = Field(default_factory=dict)
    backend_failure_counts: dict[str, int] = Field(default_factory=dict)


class CountingTargetSummary(_ViewModel):
    target: str
    sample_count: int = 0
    evaluated_count: int = 0
    exact_count: int = 0
    accuracy: float | None = None
    mae: float | None = None
    fallback_count: int = 0
    fallback_rate: float = 0.0


class Report(_ViewModel):
    run_id: str
    dataset: str | None = None
    total: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    partial: int = Field(ge=0)
    failed: int = Field(ge=0)
    skipped: int = Field(ge=0)
    samples: list[ReportSample] = Field(default_factory=list)
    tasks: list[TaskSummary] = Field(default_factory=list)
    metadata: RunMetadata | None = None
    latency: LatencySummary = Field(default_factory=LatencySummary)
    routing_summary: RoutingSummary = Field(default_factory=RoutingSummary)
    failure_summary: FailureSummary = Field(default_factory=FailureSummary)
    counting_target_summary: list[CountingTargetSummary] = Field(default_factory=list)
    visual_materialized_count: int = 0
    visual_total: int = 0
