"""Runtime execution of an approved counting backend plan.

执行已选定 BackendPlan 的运行时执行器。Executor 负责 primary backend 调用、
detector unavailable/runtime 回退、detector zero review 与结构化执行结果；
它不负责 target 解析、backend 规划或 AgentExecution 包装。所有公共错误为
稳定错误，结果对象绝不携带原始异常文本、绝对路径、密钥或 Base64。
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.base import AgentContext
from agents.counting.backends.base import (
    BackendKind,
    BackendPlan,
    CountingBackendOutcome,
    CountingRequest,
)
from agents.counting.backends.selector import BackendSelector
from agents.counting.schema import IssueRecord
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

# Detector-loading failures that make a YOLO primary backend unavailable.
# 使 YOLO 主后端不可用的检测器加载失败类型。
_UNAVAILABLE_ERRORS = (
    DetectorWeightsMissingError,
    DetectorWeightsHashMismatchError,
    OptionalDependencyMissingError,
    DetectorTaskMismatchError,
    DetectorClassMapMismatchError,
)

@dataclass(frozen=True)
class CountingExecutionPolicy:
    """Execution knobs deciding when a detector may fall back or be reviewed.
    决定检测器何时允许回退或复核的执行开关。"""

    fallback_to_qwen_on_unavailable: bool
    fallback_to_qwen_on_error: bool
    verify_empty_with_qwen: bool
    trust_empty_detection: bool


@dataclass(frozen=True)
class CountingExecutionResult:
    """Full structured state of one plan execution; every public trace field is
    derived from these attributes. 一次计划执行的完整结构化状态；所有公开
    trace 字段都可由这些属性导出。"""

    outcome: CountingBackendOutcome
    primary_backend: str
    primary_kind: BackendKind
    final_backend: str
    final_kind: BackendKind
    attempted_backends: tuple[str, ...]
    review_backend: str | None
    fallback_triggered: bool
    fallback_kind: str | None
    fallback_reason_code: str | None
    fallback_error_type: str | None
    yolo_trace: dict[str, object] | None


class CountingPlanExecutor:
    """Execute an approved BackendPlan with explicit fallback and zero review.
    执行已批准 BackendPlan：primary 调用、detector unavailable/runtime 回退
    与 detector zero review；绝不静默隐藏回退。"""

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
        """Run the primary backend, fall back when a detector cannot run, and
        review detector-zero results when enabled. 执行主后端；检测器无法运行
        时回退；启用时对检测器零结果进行复核。"""
        primary = self._selector.backend_by_name(plan.primary_backend_name)
        primary_kind = _backend_kind(
            primary, agent_name=agent_name, sample_id=request.sample.sample_id
        )
        attempted = [plan.primary_backend_name]
        final_backend = plan.primary_backend_name
        final_kind: BackendKind = primary_kind
        review_backend: str | None = None
        fallback_triggered = False
        fallback_kind: str | None = None
        fallback_reason_code: str | None = None
        fallback_error_type: str | None = None
        yolo_trace: dict[str, object] | None = (
            dict(primary.trace_profile())
            if primary_kind == "yolo_obb"
            and callable(getattr(primary, "trace_profile", None))
            else None
        )
        try:
            outcome = await primary.count(request, context)
        except _UNAVAILABLE_ERRORS as exc:
            if (
                primary_kind != "yolo_obb"
                or not self._policy.fallback_to_qwen_on_unavailable
                or not plan.fallback_backend_names
            ):
                raise CountingBackendUnavailableError(
                    request.target.canonical_label,
                    primary_backend=plan.primary_backend_name,
                    reason_code="PRIMARY_BACKEND_UNAVAILABLE",
                ) from exc
            return await self._fallback(
                plan=plan,
                request=request,
                context=context,
                attempted=attempted,
                primary_kind=primary_kind,
                yolo_trace=yolo_trace,
                kind="unavailable",
                error=exc,
                agent_name=agent_name,
            )
        except Exception as exc:
            if (
                primary_kind != "yolo_obb"
                or not self._policy.fallback_to_qwen_on_error
                or not plan.fallback_backend_names
            ):
                raise AgentExecutionError(
                    agent_name,
                    request.sample.sample_id,
                    cause="PRIMARY_BACKEND_FAILED",
                ) from exc
            return await self._fallback(
                plan=plan,
                request=request,
                context=context,
                attempted=attempted,
                primary_kind=primary_kind,
                yolo_trace=yolo_trace,
                kind="runtime_error",
                error=exc,
                agent_name=agent_name,
            )
        else:
            if primary_kind == "yolo_obb":
                yolo_trace = {**(yolo_trace or {}), **dict(outcome.trace or {})}
                if (
                    outcome.counting.final_count == 0
                    and self._policy.verify_empty_with_qwen
                    and not self._policy.trust_empty_detection
                ):
                    yolo_trace["zero_review_triggered"] = True
                    review_name = (
                        plan.fallback_backend_names[0]
                        if plan.fallback_backend_names
                        else "qwen_point"
                    )
                    review_backend = review_name
                    yolo_trace["zero_review_backend"] = review_name
                    try:
                        review_backend_obj = self._selector.backend_by_name(review_name)
                        attempted.append(review_name)
                        review = await review_backend_obj.count(request, context)
                        yolo_trace["zero_review_status"] = review.counting.status
                        yolo_trace["zero_review_result_count"] = review.counting.final_count
                        if review.counting.final_count > 0:
                            outcome = review
                            final_backend = review_name
                            final_kind = _backend_kind(
                                review_backend_obj,
                                agent_name=agent_name,
                                sample_id=request.sample.sample_id,
                            )
                            fallback_triggered = True
                            fallback_kind = "zero_review"
                            fallback_reason_code = "DETECTOR_ZERO_OVERRIDDEN_BY_REVIEW"
                            yolo_trace["zero_overridden"] = True
                        else:
                            yolo_trace["zero_overridden"] = False
                    except Exception:
                        yolo_trace.update(
                            {
                                "zero_review_status": "failed",
                                "zero_review_result_count": None,
                                "zero_overridden": False,
                            }
                        )
                        warning = IssueRecord(
                            code="DETECTOR_ZERO_REVIEW_FAILED",
                            message=(
                                "Qwen zero review failed; detector zero result "
                                "retained."
                            ),
                        )
                        outcome = CountingBackendOutcome(
                            counting=outcome.counting.model_copy(
                                update={
                                    "status": (
                                        "completed_with_warnings"
                                        if outcome.counting.status == "completed"
                                        else outcome.counting.status
                                    ),
                                    "warnings": [
                                        *outcome.counting.warnings,
                                        warning,
                                    ],
                                }
                            ),
                            agent_result=outcome.agent_result,
                            trace=outcome.trace,
                        )
            return CountingExecutionResult(
                outcome=outcome,
                primary_backend=plan.primary_backend_name,
                primary_kind=primary_kind,
                final_backend=final_backend,
                final_kind=final_kind,
                attempted_backends=tuple(attempted),
                review_backend=review_backend,
                fallback_triggered=fallback_triggered,
                fallback_kind=fallback_kind,
                fallback_reason_code=fallback_reason_code,
                fallback_error_type=fallback_error_type,
                yolo_trace=yolo_trace,
            )

    async def _fallback(
        self,
        *,
        plan: BackendPlan,
        request: CountingRequest,
        context: AgentContext,
        attempted: list[str],
        primary_kind: BackendKind,
        yolo_trace: dict[str, object] | None,
        kind: str,
        error: Exception,
        agent_name: AgentName,
    ) -> CountingExecutionResult:
        """Execute the first fallback backend and return the full result; a
        missing or failing fallback fails with a stable error.
        执行第一个回退后端并返回完整结果；回退后端缺失或失败时以稳定错误
        失败。"""
        name = plan.fallback_backend_names[0]
        attempted.append(name)
        try:
            backend = self._selector.backend_by_name(name)
        except KeyError as exc:
            raise CountingBackendUnavailableError(
                request.target.canonical_label,
                primary_backend=name,
                reason_code="FALLBACK_BACKEND_MISSING",
            ) from exc
        try:
            outcome = await backend.count(request, context)
        except Exception as exc:
            raise AgentExecutionError(
                agent_name,
                request.sample.sample_id,
                cause="FALLBACK_BACKEND_FAILED",
            ) from exc
        return CountingExecutionResult(
            outcome=outcome,
            primary_backend=plan.primary_backend_name,
            primary_kind=primary_kind,
            final_backend=name,
            final_kind=_backend_kind(
                backend, agent_name=agent_name, sample_id=request.sample.sample_id
            ),
            attempted_backends=tuple(attempted),
            review_backend=None,
            fallback_triggered=True,
            fallback_kind=kind,
            fallback_reason_code=(
                "PRIMARY_BACKEND_UNAVAILABLE"
                if kind == "unavailable"
                else "PRIMARY_BACKEND_FAILED"
            ),
            fallback_error_type=type(error).__name__,
            yolo_trace=yolo_trace,
        )


def _backend_kind(backend: object, *, agent_name: str, sample_id: str) -> BackendKind:
    """Resolve the explicit backend kind; unknown kinds fail with a stable
    error instead of being treated as detectors.
    解析显式后端 kind；未知 kind 以稳定错误失败而非当作检测器。"""
    kind = getattr(backend, "kind", None)
    if kind not in {"qwen_point", "quantity_proposal", "yolo_obb"}:
        raise AgentExecutionError(
            agent_name,
            sample_id,
            cause="INVALID_BACKEND_KIND",
        )
    return kind  # type: ignore[return-value]
