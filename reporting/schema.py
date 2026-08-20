"""Stable, JSON-safe presentation models for the offline Report V2 bundle."""

from __future__ import annotations

import math
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from data.schema import JsonScalar, JsonValue
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


class TaskCandidateView(_ViewModel):
    order: int = Field(ge=1)
    task: str
    agent_names: list[str] = Field(default_factory=list)
    status: str = "not_recorded"
    reason_code: str | None = None
    selected: bool = False
    executed: bool = False


class TaskRoutingView(_ViewModel):
    source_task: str | None = None
    resolved_task: str | None = None
    executed_task: str | None = None
    planning_mode: str | None = None
    resolution_source: str | None = None
    candidate_tasks: list[TaskCandidateView] = Field(default_factory=list)
    primary_agent: str | None = None
    fallback_agents: list[str] = Field(default_factory=list)
    executed_agent: str | None = None
    execution_mode: str | None = None
    primary_reason: str | None = None
    fallback_from_task: str | None = None
    skipped_candidates: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class ExecutionStepView(_ViewModel):
    order: int = Field(ge=1)
    phase: str
    component: str
    operation: str | None = None
    status: str = "not_recorded"
    task: str | None = None
    agent_name: str | None = None
    backend_name: str | None = None
    reason_code: str | None = None
    request_id: str | None = None
    artifact_names: list[str] = Field(default_factory=list)
    summary_fields: dict[str, JsonScalar | list[JsonScalar]] = Field(default_factory=dict)


class WorkflowStepView(_ViewModel):
    """One normalized step in an observed run-level execution sequence."""

    order: int = Field(ge=1)
    phase: str
    component: str
    operation: str | None = None
    backend_name: str | None = None
    repeat_count: int = Field(default=1, ge=1)


class WorkflowSequenceView(_ViewModel):
    """A concrete execution sequence shared by one or more samples."""

    task: str
    sample_count: int = Field(ge=1)
    steps: list[WorkflowStepView] = Field(default_factory=list)


class ModelWeightView(_ViewModel):
    """Path-free identity of one YOLO or segmentation weight actually used."""

    family: Literal["yolo", "segmentation"]
    backend_name: str
    backend_kind: str
    logical_model_id: str | None = None
    weights_file: str | None = Field(default=None, pattern=r"^[^/\\:]+$")
    weights_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_dataset: str | None = None
    model_revision: str | None = None
    use_count: int = Field(ge=1)
    phases: list[str] = Field(default_factory=list)
    statuses: list[str] = Field(default_factory=list)


class ProcessReport(_ViewModel):
    """Observed workflow order and local visual-expert weight identities."""

    sample_process_count: int = Field(default=0, ge=0)
    workflow_sequences: list[WorkflowSequenceView] = Field(default_factory=list)
    model_weights: list[ModelWeightView] = Field(default_factory=list)


_UNSAFE_SUMMARY_RE = re.compile(
    r"(?i)(?:data:image/[^;]+;base64,|https?://|(?:^|[^a-z0-9])(?:[a-z]:[\\/]|/(?:home|tmp|users|private|var)/))"
)


def _safe_summary_scalar(value: JsonScalar) -> bool:
    if isinstance(value, float) and not math.isfinite(value):
        return False
    if not isinstance(value, str):
        return True
    return _UNSAFE_SUMMARY_RE.search(value) is None


class GroundTruthView(_ViewModel):
    """Task-neutral, read-only ground truth projection for sample audits."""

    answers: list[str] = Field(default_factory=list)
    count: int | None = None
    boxes: list[list[float]] = Field(default_factory=list)
    points: list[list[float]] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    coordinate_frame: str | None = None


class BackendStageView(_ViewModel):
    """One persisted counting backend attempt in execution order."""

    order: int = Field(ge=1)
    backend_name: str
    backend_kind: str
    phase: str
    status: str
    reason_code: str | None = None
    predicted_count: int | None = None
    counting_status: str | None = None
    accepted_count: int | None = None
    rejected_count: int | None = None
    warning_codes: list[str] = Field(default_factory=list)
    error_type: str | None = None
    summary_fields: dict[str, JsonScalar | list[JsonScalar]] = Field(default_factory=dict)
    overlay_asset: str | None = None

    @field_validator("summary_fields")
    @classmethod
    def validate_summary_fields(cls, value: dict[str, JsonScalar | list[JsonScalar]]):
        for key, item in value.items():
            if isinstance(item, list):
                if len(item) > 50 or any(not _safe_summary_scalar(entry) for entry in item):
                    raise ValueError(f"summary_fields[{key!r}] must contain only small scalar lists")
            elif not _safe_summary_scalar(item):
                raise ValueError(f"summary_fields[{key!r}] contains an unsafe value")
        return value


class ModelCallAuditView(_ViewModel):
    """Sanitized view of one persisted structured-model call."""

    request_id: str
    prompt_version: str
    request_hash: str | None = None
    sample_id: str | None = None
    tile_id: str | None = None
    image_sha256: str | None = None
    cache_hit: bool | None = None
    valid: bool | None = None
    repair_used: bool | None = None
    latency_seconds: float | None = None
    token_usage: dict[str, int] | None = None
    raw_response: str | None = None
    raw_response_truncated: bool = False
    parsed_response: str | None = None
    request_summary: str | None = None


class StructuredArtifactView(_ViewModel):
    """Safe view of one persisted structured submodel artifact.
    一份已持久化结构化子模型产物的安全视图。"""

    filename: str
    payload: JsonValue


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
    ground_truth_coordinate_frame: str | None = None
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
    ground_truth: GroundTruthView | None = None
    prediction: str | None = None
    resolved_task: str | None = None
    execution_agent: str | None = None
    fallback_used: bool = False
    judge_status: str = "not_requested"
    inference_seconds: float | None = None
    evaluation: EvaluationRecord | None = None
    result_quality: Literal["correct", "incorrect", "unknown", "not_applicable"] = "unknown"
    routing: RoutingView = Field(default_factory=RoutingView)
    task_routing: TaskRoutingView = Field(default_factory=TaskRoutingView)
    execution_steps: list[ExecutionStepView] = Field(default_factory=list)
    execution_path: list[str] = Field(default_factory=list)
    routing_decision: dict[str, JsonValue] | None = None
    backend_stages: list[BackendStageView] = Field(default_factory=list)
    model_calls: list[ModelCallAuditView] = Field(default_factory=list)
    structured_artifacts: list[StructuredArtifactView] = Field(default_factory=list)
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
    process_report: ProcessReport = Field(default_factory=ProcessReport)
    visual_materialized_count: int = 0
    visual_total: int = 0
