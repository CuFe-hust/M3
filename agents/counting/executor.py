"""Ordered execution of a capability-based counting backend plan.

Specialist failures advance through the declared chain; Qwen and invalid
contracts are terminal. Zero is always valid and only enters explicit review.
专家失败时按声明顺序继续；Qwen 与非法契约为终止错误。零始终是合法结果，
只能由显式复核策略触发额外调用。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Sequence

from agents.base import AgentContext
from agents.counting.backends.base import (
    KNOWN_BACKEND_KINDS,
    BackendKind,
    BackendPlan,
    CountingBackendOutcome,
    CountingRequest,
)
from agents.counting.backends.selector import BackendSelector
from agents.counting.point_pipeline import fuse_detector_observations
from agents.counting.schema import (
    CountingBackendAttemptAudit,
    DisagreementReview,
    IssueRecord,
)
from agents.errors import (
    AgentExecutionError,
    CountingBackendUnavailableError,
    DetectorClassMapMismatchError,
    DetectorTaskMismatchError,
    DetectorWeightsHashMismatchError,
    DetectorWeightsMissingError,
    OptionalDependencyMissingError,
)
from agents.schema import AgentName

_UNAVAILABLE_ERRORS = (
    DetectorWeightsMissingError,
    DetectorWeightsHashMismatchError,
    OptionalDependencyMissingError,
    DetectorTaskMismatchError,
    DetectorClassMapMismatchError,
)


@dataclass(frozen=True)
class CountingExecutionPolicy:
    """Explicit fallback and zero-review switches.
    显式 fallback 与 zero-review 开关。"""

    fallback_on_backend_unavailable: bool
    fallback_on_backend_error: bool
    verify_empty_detection: bool
    trust_empty_detection: bool
    verify_empty_semantic: bool = False
    min_successful_detector_experts: int = 1
    ensemble_iou_threshold: float = 0.45
    ensemble_center_distance_ratio: float = 0.60
    ensemble_singleton_high_confidence: float = 0.65
    unresolved_ensemble_policy: Literal[
        "retain_high_confidence", "reject_unresolved"
    ] = "retain_high_confidence"


@dataclass(frozen=True)
class BackendFailureRecord:
    """One path-free failed attempt suitable for public trace.
    一条不含路径的失败尝试，可安全写入公共 trace。"""

    backend: str
    kind: BackendKind
    reason_code: str
    error_type: str

    def to_trace(self) -> dict[str, str]:
        return {
            "backend": self.backend,
            "kind": self.kind,
            "reason_code": self.reason_code,
            "error_type": self.error_type,
        }


@dataclass(frozen=True)
class CountingExecutionResult:
    """Structured execution state used to build the public agent trace.
    用于构建公共 Agent trace 的结构化执行状态。"""

    outcome: CountingBackendOutcome
    primary_backend: str
    primary_kind: BackendKind
    final_backend: str
    final_kind: BackendKind
    candidate_backends: tuple[str, ...]
    attempted_backends: tuple[str, ...]
    review_backend: str | None
    review_error_type: str | None
    fallback_history: tuple[BackendFailureRecord, ...]
    fallback_triggered: bool
    fallback_kind: str | None
    fallback_reason_code: str | None
    fallback_error_type: str | None
    yolo_trace: dict[str, object] | None
    attempt_audits: tuple[CountingBackendAttemptAudit, ...]


class CountingPlanExecutor:
    """Execute every declared candidate in order until one succeeds.
    按声明顺序执行候选 backend，直到一个成功。"""

    def __init__(
        self,
        selector: BackendSelector,
        *,
        policy: CountingExecutionPolicy,
    ) -> None:
        self._selector = selector
        self._policy = policy

    async def execute(
        self,
        *,
        plan: BackendPlan,
        request: CountingRequest,
        context: AgentContext,
        agent_name: AgentName,
    ) -> CountingExecutionResult:
        if len(plan.selected_backend_names) > 1:
            return await self._execute_detector_ensemble(
                plan=plan,
                request=request,
                context=context,
                agent_name=agent_name,
            )
        candidates = (plan.primary_backend_name, *plan.fallback_backend_names)
        if len(candidates) != len(set(candidates)):
            raise AgentExecutionError(
                agent_name,
                request.sample.sample_id,
                cause="INVALID_BACKEND_PLAN",
            )

        primary = self._resolve_backend(
            plan.primary_backend_name,
            request=request,
            agent_name=agent_name,
            primary=True,
        )
        primary_kind = _backend_kind(
            primary,
            agent_name=agent_name,
            sample_id=request.sample.sample_id,
        )
        yolo_trace = _yolo_profile(primary, primary_kind)
        attempted: list[str] = []
        history: list[BackendFailureRecord] = []
        attempt_audits: list[CountingBackendAttemptAudit] = []
        outcome: CountingBackendOutcome | None = None
        final_backend = plan.primary_backend_name
        final_kind = primary_kind
        succeeded_index = -1

        for index, name in enumerate(candidates):
            backend = self._resolve_backend(
                name,
                request=request,
                agent_name=agent_name,
                primary=index == 0,
            )
            kind = _backend_kind(
                backend,
                agent_name=agent_name,
                sample_id=request.sample.sample_id,
            )
            _validate_backend_contract(
                backend,
                agent_name=agent_name,
                sample_id=request.sample.sample_id,
            )
            attempted.append(name)

            try:
                available = backend.is_available()
            except Exception as error:
                if self._record_or_raise(
                    history,
                    backend=name,
                    kind=kind,
                    reason_code="BACKEND_RUNTIME_ERROR",
                    error=error,
                    index=index,
                    candidate_count=len(candidates),
                    request=request,
                    agent_name=agent_name,
                    attempt_audits=attempt_audits,
                ):
                    continue
                raise AssertionError("unreachable")
            if not isinstance(available, bool):
                raise AgentExecutionError(
                    agent_name,
                    request.sample.sample_id,
                    cause="INVALID_BACKEND_CONTRACT",
                )
            if not available:
                if self._record_or_raise(
                    history,
                    backend=name,
                    kind=kind,
                    reason_code="BACKEND_UNAVAILABLE",
                    error=BackendUnavailable(),
                    index=index,
                    candidate_count=len(candidates),
                    request=request,
                    agent_name=agent_name,
                    attempt_audits=attempt_audits,
                ):
                    continue
                raise AssertionError("unreachable")

            try:
                candidate_outcome = await backend.count(request, context)
            except Exception as error:
                reason = (
                    "BACKEND_UNAVAILABLE"
                    if isinstance(error, _UNAVAILABLE_ERRORS)
                    else "BACKEND_RUNTIME_ERROR"
                )
                if self._record_or_raise(
                    history,
                    backend=name,
                    kind=kind,
                    reason_code=reason,
                    error=error,
                    index=index,
                    candidate_count=len(candidates),
                    request=request,
                    agent_name=agent_name,
                    attempt_audits=attempt_audits,
                ):
                    continue
                raise AssertionError("unreachable")

            if not isinstance(candidate_outcome, CountingBackendOutcome):
                raise AgentExecutionError(
                    agent_name,
                    request.sample.sample_id,
                    cause="INVALID_BACKEND_CONTRACT",
                )
            outcome = candidate_outcome
            final_backend = name
            final_kind = kind
            succeeded_index = index
            attempt_audits.append(
                _success_attempt_audit(
                    backend=name,
                    kind=kind,
                    phase="primary" if index == 0 else "fallback",
                    outcome=outcome,
                )
            )
            if kind in {"yolo_obb", "yolo_detect"}:
                yolo_trace = {**(yolo_trace or {}), **dict(outcome.trace or {})}
            break

        if outcome is None:
            raise AgentExecutionError(
                agent_name,
                request.sample.sample_id,
                cause="BACKEND_CHAIN_EXHAUSTED",
            )

        original_outcome = outcome
        review_backend: str | None = None
        review_error_type: str | None = None
        zero_overridden = False
        review_index = self._review_index(
            candidates,
            succeeded_index=succeeded_index,
            final_kind=final_kind,
        )
        if outcome.counting.final_count == 0 and review_index is not None:
            review_backend = candidates[review_index]
            attempted.append(review_backend)
            review_obj = self._resolve_backend(
                review_backend,
                request=request,
                agent_name=agent_name,
                primary=False,
            )
            review_kind = _backend_kind(
                review_obj,
                agent_name=agent_name,
                sample_id=request.sample.sample_id,
            )
            _validate_backend_contract(
                review_obj,
                agent_name=agent_name,
                sample_id=request.sample.sample_id,
            )
            if yolo_trace is not None and final_kind in {"yolo_obb", "yolo_detect"}:
                yolo_trace.update(
                    {
                        "zero_review_triggered": True,
                        "zero_review_backend": review_backend,
                    }
                )
            try:
                review_available = review_obj.is_available()
                if not isinstance(review_available, bool):
                    raise InvalidBackendContract()
                if not review_available:
                    raise BackendUnavailable()
                review = await review_obj.count(request, context)
                if not isinstance(review, CountingBackendOutcome):
                    raise InvalidBackendContract()
                if yolo_trace is not None and final_kind in {"yolo_obb", "yolo_detect"}:
                    yolo_trace.update(
                        {
                            "zero_review_status": review.counting.status,
                            "zero_review_result_count": review.counting.final_count,
                        }
                    )
                if review.counting.final_count > 0:
                    outcome = review
                    final_backend = review_backend
                    final_kind = review_kind
                    zero_overridden = True
                attempt_audits.append(
                    _success_attempt_audit(
                        backend=review_backend,
                        kind=review_kind,
                        phase="zero_review",
                        outcome=review,
                    )
                )
            except InvalidBackendContract as error:
                raise AgentExecutionError(
                    agent_name,
                    request.sample.sample_id,
                    cause="INVALID_BACKEND_CONTRACT",
                ) from error
            except Exception as error:
                review_error_type = type(error).__name__
                unavailable = isinstance(error, (BackendUnavailable, *_UNAVAILABLE_ERRORS))
                attempt_audits.append(
                    CountingBackendAttemptAudit(
                        backend_name=review_backend,
                        backend_kind=review_kind,
                        phase="zero_review",
                        status="unavailable" if unavailable else "failed",
                        reason_code=(
                            "BACKEND_UNAVAILABLE"
                            if unavailable
                            else "BACKEND_RUNTIME_ERROR"
                        ),
                        error_type=type(error).__name__,
                    )
                )
                outcome = _with_review_warning(original_outcome, source_kind=final_kind)
                if yolo_trace is not None and final_kind in {"yolo_obb", "yolo_detect"}:
                    yolo_trace.update(
                        {
                            "zero_review_status": "failed",
                            "zero_review_result_count": None,
                        }
                    )
            if yolo_trace is not None and primary_kind in {"yolo_obb", "yolo_detect"}:
                yolo_trace["zero_overridden"] = zero_overridden

        first_failure = history[0] if history else None
        fallback_kind = None
        fallback_reason_code = None
        fallback_error_type = None
        if first_failure is not None:
            fallback_kind = (
                "unavailable"
                if first_failure.reason_code == "BACKEND_UNAVAILABLE"
                else "runtime_error"
            )
            fallback_reason_code = (
                "PRIMARY_BACKEND_UNAVAILABLE"
                if first_failure.reason_code == "BACKEND_UNAVAILABLE"
                else "PRIMARY_BACKEND_FAILED"
            )
            fallback_error_type = first_failure.error_type
        elif zero_overridden:
            fallback_kind = "zero_review"
            fallback_reason_code = (
                "DETECTOR_ZERO_OVERRIDDEN_BY_REVIEW"
                if primary_kind in {"yolo_obb", "yolo_detect"}
                else "SEMANTIC_ZERO_OVERRIDDEN_BY_REVIEW"
            )

        return CountingExecutionResult(
            outcome=outcome,
            primary_backend=plan.primary_backend_name,
            primary_kind=primary_kind,
            final_backend=final_backend,
            final_kind=final_kind,
            candidate_backends=candidates,
            attempted_backends=tuple(attempted),
            review_backend=review_backend,
            review_error_type=review_error_type,
            fallback_history=tuple(history),
            fallback_triggered=bool(history) or zero_overridden,
            fallback_kind=fallback_kind,
            fallback_reason_code=fallback_reason_code,
            fallback_error_type=fallback_error_type,
            yolo_trace=yolo_trace,
            attempt_audits=tuple(attempt_audits),
        )

    async def _execute_detector_ensemble(
        self,
        *,
        plan: BackendPlan,
        request: CountingRequest,
        context: AgentContext,
        agent_name: AgentName,
    ) -> CountingExecutionResult:
        """Run every co-primary detector, then fall back as one group."""

        selected = plan.selected_backend_names
        candidates = (*selected, *plan.fallback_backend_names)
        if len(candidates) != len(set(candidates)):
            raise AgentExecutionError(
                agent_name, request.sample.sample_id, cause="INVALID_BACKEND_PLAN"
            )
        attempted: list[str] = []
        history: list[BackendFailureRecord] = []
        audits: list[CountingBackendAttemptAudit] = []
        successes: list[tuple[str, CountingBackendOutcome]] = []
        primary_kind: BackendKind | None = None
        yolo_trace: dict[str, object] | None = None
        for index, name in enumerate(selected):
            backend = self._resolve_backend(
                name,
                request=request,
                agent_name=agent_name,
                primary=index == 0,
            )
            kind = _backend_kind(
                backend, agent_name=agent_name, sample_id=request.sample.sample_id
            )
            if kind not in {"yolo_obb", "yolo_detect"}:
                raise AgentExecutionError(
                    agent_name, request.sample.sample_id, cause="INVALID_BACKEND_PLAN"
                )
            if primary_kind is None:
                primary_kind = kind
                yolo_trace = _yolo_profile(backend, kind)
            _validate_backend_contract(
                backend,
                agent_name=agent_name,
                sample_id=request.sample.sample_id,
            )
            attempted.append(name)
            try:
                available = backend.is_available()
                if not isinstance(available, bool):
                    raise InvalidBackendContract()
                if not available:
                    raise BackendUnavailable()
                outcome = await backend.count(request, context)
                if not isinstance(outcome, CountingBackendOutcome):
                    raise InvalidBackendContract()
                successes.append((name, outcome))
                audits.append(
                    _success_attempt_audit(
                        backend=name,
                        kind=kind,
                        phase="primary" if index == 0 else "ensemble",
                        outcome=outcome,
                    )
                )
                if yolo_trace is not None:
                    yolo_trace.update(dict(outcome.trace or {}))
            except InvalidBackendContract as error:
                raise AgentExecutionError(
                    agent_name, request.sample.sample_id, cause="INVALID_BACKEND_CONTRACT"
                ) from error
            except Exception as error:
                reason = "BACKEND_UNAVAILABLE" if isinstance(error, (BackendUnavailable, *_UNAVAILABLE_ERRORS)) else "BACKEND_RUNTIME_ERROR"
                history.append(
                    BackendFailureRecord(
                        backend=name,
                        kind=kind,
                        reason_code=reason,
                        error_type=type(error).__name__,
                    )
                )
                audits.append(
                    CountingBackendAttemptAudit(
                        backend_name=name,
                        backend_kind=kind,
                        phase="primary" if index == 0 else "ensemble",
                        status="unavailable" if reason == "BACKEND_UNAVAILABLE" else "failed",
                        reason_code=reason,
                        error_type=type(error).__name__,
                    )
                )

        if len(successes) >= self._policy.min_successful_detector_experts:
            fused = fuse_detector_observations(
                [
                    (name, outcome.counting.global_points)
                    for name, outcome in successes
                ],
                iou_threshold=self._policy.ensemble_iou_threshold,
                center_distance_ratio=self._policy.ensemble_center_distance_ratio,
                singleton_high_confidence=self._policy.ensemble_singleton_high_confidence,
            )
            first_name, first_outcome = successes[0]
            base = first_outcome.counting
            review_trace: dict[str, object] = {
                "disagreement_review_triggered": False,
                "review_backend": None,
                "review_request_hash": None,
                "requested_conflict_ids": [
                    str(item.get("conflict_id")) for item in fused.review_candidates
                ],
                "reviewed_conflict_ids": [],
                "truncated_conflict_ids": [],
                "review_decisions": [],
                "review_failure": None,
                "unresolved_ensemble_policy": self._policy.unresolved_ensemble_policy,
                "unresolved_policy_applied": False,
            }
            reviewed_points = fused.points
            reviewed_unresolved = fused.unresolved_conflicts
            review_warnings = list(fused.warnings)
            reviewer = next(
                (
                    self._selector.backend_by_name(name)
                    for name in plan.fallback_backend_names
                    if callable(getattr(self._selector.backend_by_name(name), "review_disagreements", None))
                ),
                None,
            ) if fused.review_candidates else None
            if reviewer is not None:
                review_trace["review_backend"] = getattr(reviewer, "name", None)
                try:
                    review = await reviewer.review_disagreements(
                        request=request,
                        conflicts=fused.review_candidates,
                        context=context,
                    )
                    review = DisagreementReview.model_validate(review)
                    reviewed_points, reviewed_unresolved, review_warnings = _apply_disagreement_review(
                        fused,
                        review,
                    )
                    review_trace.update(
                        {
                            "disagreement_review_triggered": True,
                            "review_backend": getattr(reviewer, "name", None),
                            "reviewed_conflict_ids": [item.conflict_id for item in review.decisions],
                            "review_decisions": [item.model_dump(mode="json") for item in review.decisions],
                        }
                    )
                    reviewer_trace = getattr(reviewer, "last_disagreement_review_trace", None)
                    if isinstance(reviewer_trace, dict):
                        review_trace.update(reviewer_trace)
                except Exception as error:
                    review_trace["review_failure"] = type(error).__name__
                    review_warnings.append(
                        IssueRecord(
                            code="DETECTOR_DISAGREEMENT_REVIEW_FAILED",
                            message=f"Disagreement review failed: {type(error).__name__}.",
                        )
                    )
            if reviewed_unresolved:
                unresolved_before_policy = list(reviewed_unresolved)
                reviewed_points, reviewed_unresolved, policy_warnings = resolve_unresolved_observations(
                    reviewed_points,
                    self._policy.unresolved_ensemble_policy,
                    self._policy.ensemble_singleton_high_confidence,
                    unresolved_conflict_ids=reviewed_unresolved,
                )
                review_warnings.extend(policy_warnings)
                review_trace["unresolved_policy_applied"] = {
                    "policy": self._policy.unresolved_ensemble_policy,
                    "singleton_min_confidence": self._policy.ensemble_singleton_high_confidence,
                    "conflict_ids": unresolved_before_policy,
                    "remaining_conflict_ids": reviewed_unresolved,
                }
            status = "completed_with_warnings" if (history or review_warnings or base.status == "completed_with_warnings") else base.status
            counting = base.model_copy(
                update={
                    "global_points": reviewed_points,
                    "merged_groups": fused.merged_groups,
                    "unresolved_conflicts": reviewed_unresolved,
                    "warnings": [*base.warnings, *review_warnings],
                    "final_count": sum(point.accepted for point in reviewed_points),
                    "status": status,
                }
            )
            trace = dict(first_outcome.trace or {})
            trace["ensemble"] = {
                "selected_experts": list(selected),
                "successful_experts": [name for name, _ in successes],
                "failed_experts": [entry.to_trace() for entry in history],
                "fused_instance_count": counting.final_count,
                "merged_groups": fused.merged_groups,
                "unresolved_conflicts": reviewed_unresolved,
                "review_required": bool(fused.unresolved_conflicts),
                "disagreement_review": review_trace,
            }
            outcome = CountingBackendOutcome(counting=counting, trace=trace)
            return CountingExecutionResult(
                outcome=outcome,
                primary_backend=plan.primary_backend_name,
                primary_kind=primary_kind or "yolo_obb",
                final_backend=first_name,
                final_kind=primary_kind or "yolo_obb",
                candidate_backends=candidates,
                attempted_backends=tuple(attempted),
                review_backend=(getattr(reviewer, "name", None) if reviewer is not None else None),
                review_error_type=(
                    str(review_trace["review_failure"])
                    if review_trace["review_failure"] is not None
                    else None
                ),
                fallback_history=tuple(history),
                fallback_triggered=bool(history),
                fallback_kind="ensemble_degraded" if history else None,
                fallback_reason_code="DETECTOR_ENSEMBLE_PARTIAL" if history else None,
                fallback_error_type=history[0].error_type if history else None,
                yolo_trace=yolo_trace,
                attempt_audits=tuple(audits),
            )

        if not plan.fallback_backend_names:
            raise CountingBackendUnavailableError(
                request.target.canonical_label,
                primary_backend=plan.primary_backend_name,
                reason_code="DETECTOR_ENSEMBLE_EXHAUSTED",
            )
        fallback_plan = BackendPlan(
            primary_backend_name=plan.fallback_backend_names[0],
            fallback_backend_names=plan.fallback_backend_names[1:],
        )
        fallback_result = await self.execute(
            plan=fallback_plan,
            request=request,
            context=context,
            agent_name=agent_name,
        )
        return replace(
            fallback_result,
            primary_backend=plan.primary_backend_name,
            primary_kind=primary_kind or fallback_result.primary_kind,
            candidate_backends=candidates,
            attempted_backends=tuple(attempted) + fallback_result.attempted_backends,
            fallback_history=tuple(history) + fallback_result.fallback_history,
            fallback_triggered=True,
            fallback_kind=fallback_result.fallback_kind or "ensemble_exhausted",
            fallback_reason_code=fallback_result.fallback_reason_code or "DETECTOR_ENSEMBLE_EXHAUSTED",
            fallback_error_type=fallback_result.fallback_error_type or (history[0].error_type if history else None),
            yolo_trace=yolo_trace or fallback_result.yolo_trace,
            attempt_audits=tuple(audits) + fallback_result.attempt_audits,
        )

    def _record_or_raise(
        self,
        history: list[BackendFailureRecord],
        *,
        backend: str,
        kind: BackendKind,
        reason_code: str,
        error: Exception,
        index: int,
        candidate_count: int,
        request: CountingRequest,
        agent_name: AgentName,
        attempt_audits: list[CountingBackendAttemptAudit],
    ) -> bool:
        unavailable = reason_code == "BACKEND_UNAVAILABLE"
        attempt_audits.append(
            CountingBackendAttemptAudit(
                backend_name=backend,
                backend_kind=kind,
                phase="primary" if index == 0 else "fallback",
                status="unavailable" if unavailable else "failed",
                reason_code=reason_code,
                error_type=type(error).__name__,
            )
        )
        history.append(
            BackendFailureRecord(
                backend=backend,
                kind=kind,
                reason_code=reason_code,
                error_type=type(error).__name__,
            )
        )
        policy_allows = (
            self._policy.fallback_on_backend_unavailable
            if unavailable
            else self._policy.fallback_on_backend_error
        )
        can_continue = (
            kind != "qwen_point"
            and index + 1 < candidate_count
            and policy_allows
        )
        if can_continue:
            return True
        if unavailable:
            raise CountingBackendUnavailableError(
                request.target.canonical_label,
                primary_backend=backend,
                reason_code=(
                    "PRIMARY_BACKEND_UNAVAILABLE"
                    if index == 0
                    else "FALLBACK_BACKEND_UNAVAILABLE"
                ),
            ) from error
        raise AgentExecutionError(
            agent_name,
            request.sample.sample_id,
            cause=("PRIMARY_BACKEND_FAILED" if index == 0 else "FALLBACK_BACKEND_FAILED"),
        ) from error

    def _resolve_backend(
        self,
        name: str,
        *,
        request: CountingRequest,
        agent_name: AgentName,
        primary: bool,
    ):
        try:
            return self._selector.backend_by_name(name)
        except KeyError as error:
            raise CountingBackendUnavailableError(
                request.target.canonical_label,
                primary_backend=name,
                reason_code=(
                    "PRIMARY_BACKEND_MISSING" if primary else "FALLBACK_BACKEND_MISSING"
                ),
            ) from error
        except CountingBackendUnavailableError:
            raise
        except Exception as error:
            raise AgentExecutionError(
                agent_name,
                request.sample.sample_id,
                cause="INVALID_BACKEND_CONTRACT",
            ) from error

    def _review_index(
        self,
        candidates: tuple[str, ...],
        *,
        succeeded_index: int,
        final_kind: BackendKind,
    ) -> int | None:
        if (
            final_kind in {"yolo_obb", "yolo_detect"}
            and not self._policy.trust_empty_detection
            and self._policy.verify_empty_detection
        ):
            return succeeded_index + 1 if succeeded_index + 1 < len(candidates) else None
        if final_kind != "semantic_segmentation":
            return None
        if not self._policy.verify_empty_semantic:
            return None
        for index in range(succeeded_index + 1, len(candidates)):
            backend = self._selector.backend_by_name(candidates[index])
            if getattr(backend, "kind", None) in {
                "quantity_proposal",
                "qwen_point",
            }:
                return index
        return None


class BackendUnavailable(RuntimeError):
    """Internal path-free availability marker. / 内部无路径可用性标记。"""


class InvalidBackendContract(RuntimeError):
    """Internal path-free contract marker. / 内部无路径契约标记。"""


def _success_attempt_audit(
    *,
    backend: str,
    kind: BackendKind,
    phase: Literal["primary", "ensemble", "fallback", "zero_review"],
    outcome: CountingBackendOutcome,
) -> CountingBackendAttemptAudit:
    counting_status = outcome.counting.status
    status = (
        "succeeded"
        if counting_status in {"completed", "completed_with_warnings"}
        else counting_status
    )
    return CountingBackendAttemptAudit(
        backend_name=backend,
        backend_kind=kind,
        phase=phase,
        status=status,
        counting=outcome.counting,
        agent_result=(
            outcome.agent_result.model_dump(mode="json")
            if outcome.agent_result is not None
            else None
        ),
        backend_trace=dict(outcome.trace or {}),
    )


def _backend_kind(backend: object, *, agent_name: str, sample_id: str) -> BackendKind:
    kind = getattr(backend, "kind", None)
    if kind not in KNOWN_BACKEND_KINDS:
        raise AgentExecutionError(agent_name, sample_id, cause="INVALID_BACKEND_KIND")
    return kind  # type: ignore[return-value]


def _validate_backend_contract(
    backend: object,
    *,
    agent_name: str,
    sample_id: str,
) -> None:
    if (
        not isinstance(getattr(backend, "name", None), str)
        or not isinstance(getattr(backend, "priority", None), int)
        or not callable(getattr(backend, "is_available", None))
        or not callable(getattr(backend, "count", None))
    ):
        raise AgentExecutionError(agent_name, sample_id, cause="INVALID_BACKEND_CONTRACT")


def _yolo_profile(backend: object, kind: BackendKind) -> dict[str, object] | None:
    profile = getattr(backend, "trace_profile", None)
    return dict(profile()) if kind in {"yolo_obb", "yolo_detect"} and callable(profile) else None


def _with_review_warning(
    outcome: CountingBackendOutcome,
    *,
    source_kind: BackendKind,
) -> CountingBackendOutcome:
    code = (
        "DETECTOR_ZERO_REVIEW_FAILED"
        if source_kind in {"yolo_obb", "yolo_detect"}
        else "SEMANTIC_ZERO_REVIEW_FAILED"
    )
    warning = IssueRecord(
        code=code,
        message="Zero review failed; original zero result retained.",
    )
    counting = outcome.counting
    return CountingBackendOutcome(
        counting=counting.model_copy(
            update={
                "status": (
                    "completed_with_warnings"
                    if counting.status == "completed"
                    else counting.status
                ),
                "warnings": [*counting.warnings, warning],
            }
        ),
        agent_result=outcome.agent_result,
        trace=outcome.trace,
    )


def resolve_unresolved_observations(
    observations: Sequence,
    policy: Literal["retain_high_confidence", "reject_unresolved"],
    singleton_min_confidence: float,
    *,
    unresolved_conflict_ids: Sequence[str] = (),
) -> tuple[list, list[str], list[IssueRecord]]:
    """Apply the deterministic policy to every unresolved detector candidate."""

    if policy not in {"retain_high_confidence", "reject_unresolved"}:
        raise ValueError("unknown unresolved ensemble policy")
    if not 0.0 <= singleton_min_confidence <= 1.0:
        raise ValueError("invalid singleton confidence threshold")
    points = list(observations)
    point_ids = {point.global_id for point in points}
    unresolved_points: set[str] = set()
    matched_conflicts: set[str] = set()
    remaining: list[str] = []
    for conflict_id in unresolved_conflict_ids:
        conflict_id = str(conflict_id)
        candidate_ids = (
            {conflict_id}
            if conflict_id in point_ids
            else set(conflict_id.split("|"))
        )
        present = candidate_ids & point_ids
        if not present:
            remaining.append(conflict_id)
            continue
        matched_conflicts.add(conflict_id)
        unresolved_points.update(present)

    retained: list[str] = []
    rejected: list[str] = []
    for index, point in enumerate(points):
        if point.global_id not in unresolved_points:
            continue
        keep = (
            policy == "retain_high_confidence"
            and point.confidence >= singleton_min_confidence
        )
        if keep:
            retained.append(point.global_id)
            provenance = point.provenance
            if provenance is not None:
                points[index] = point.model_copy(
                    update={
                        "provenance": provenance.model_copy(
                            update={"review_status": "retained_high_confidence_unresolved"}
                        )
                    }
                )
        else:
            rejected.append(point.global_id)
            points[index] = point.model_copy(
                update={
                    "accepted": False,
                    "rejection_reason": (
                        "DETECTOR_DISAGREEMENT_REJECT_UNRESOLVED"
                        if policy == "reject_unresolved"
                        else "DETECTOR_DISAGREEMENT_BELOW_SINGLETON_THRESHOLD"
                    ),
                }
            )
    if matched_conflicts:
        message = (
            "Unresolved detector candidates were retained only when they met the "
            "configured singleton confidence threshold."
            if policy == "retain_high_confidence"
            else "Unresolved detector candidates were rejected by policy."
        )
        warnings = [
            IssueRecord(
                code="DETECTOR_ENSEMBLE_UNRESOLVED_POLICY_APPLIED",
                message=message,
                point_ids=sorted({*retained, *rejected}),
            )
        ]
    else:
        warnings = []
    return points, sorted(remaining), warnings


def _apply_disagreement_review(
    fused: object,
    review: DisagreementReview,
) -> tuple[list, list[str], list[IssueRecord]]:
    """Apply only exact, reviewer-supplied candidate decisions.

    Consensus points are intentionally addressed by exact ``global_id`` only;
    a reviewer cannot replace, delete, or re-select a fused consensus point.
    """

    points = list(getattr(fused, "points"))
    requested = {
        str(item.get("conflict_id")): item
        for item in getattr(fused, "review_candidates")
    }
    unresolved = set(getattr(fused, "unresolved_conflicts"))
    warnings = list(getattr(fused, "warnings"))
    seen: set[str] = set()
    for decision in review.decisions:
        conflict = requested.get(decision.conflict_id)
        if conflict is None:
            raise ValueError("review returned an unknown conflict")
        candidate_id_list = [str(value) for value in conflict.get("candidate_ids", [])]
        candidate_ids = set(candidate_id_list)
        if len(candidate_id_list) != len(candidate_ids):
            raise ValueError("review request contains duplicate candidates")
        accepted_ids = [str(value) for value in decision.accepted_candidate_ids]
        if len(accepted_ids) != len(set(accepted_ids)):
            raise ValueError("review returned duplicate candidate ids")
        if not set(accepted_ids).issubset(candidate_ids):
            raise ValueError("review returned an unknown candidate")
        if decision.conflict_id in seen:
            raise ValueError("review returned a duplicate conflict")
        seen.add(decision.conflict_id)
        point_index_by_id = {point.global_id: index for index, point in enumerate(points)}
        candidate_indexes = {
            candidate_id: point_index_by_id[candidate_id]
            for candidate_id in candidate_ids
            if candidate_id in point_index_by_id
        }
        if set(candidate_indexes) != candidate_ids:
            raise ValueError("review candidate is not present in fused result")
        if decision.decision == "accept_one":
            if len(accepted_ids) != 1:
                raise ValueError("accept_one must select exactly one requested candidate")
            keep_id = accepted_ids[0]
            for candidate_id, index in candidate_indexes.items():
                if candidate_id != keep_id:
                    points[index] = points[index].model_copy(
                        update={
                            "accepted": False,
                            "rejection_reason": "DISAGREEMENT_REVIEW_ACCEPT_ONE",
                        }
                    )
        elif decision.decision == "accept_multiple":
            if len(accepted_ids) != decision.instance_count or len(accepted_ids) < 2:
                raise ValueError(
                    "accept_multiple instance_count must equal exact selected candidates"
                )
            for candidate_id, index in candidate_indexes.items():
                if candidate_id not in accepted_ids:
                    points[index] = points[index].model_copy(
                        update={
                            "accepted": False,
                            "rejection_reason": "DISAGREEMENT_REVIEW_ACCEPT_MULTIPLE",
                        }
                    )
        elif decision.decision == "reject_all":
            if accepted_ids:
                raise ValueError("reject_all cannot accept candidates")
            for index in candidate_indexes.values():
                points[index] = points[index].model_copy(
                    update={
                        "accepted": False,
                        "rejection_reason": "DISAGREEMENT_REVIEW_REJECTED",
                    }
                )
        elif decision.decision == "uncertain":
            if accepted_ids:
                raise ValueError("uncertain cannot accept candidates")
            warnings.append(
                IssueRecord(
                    code="DETECTOR_DISAGREEMENT_REVIEW_UNCERTAIN",
                    message="Detector disagreement remained uncertain; unresolved policy will decide.",
                    point_ids=sorted(candidate_ids),
                )
            )
        if decision.decision != "uncertain":
            unresolved.discard(decision.conflict_id)
    return points, sorted(unresolved), warnings
