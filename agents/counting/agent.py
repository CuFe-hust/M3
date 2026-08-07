"""Counting agent with explicit backend plans and visible fallback.

具有显式后端计划与可见回退的计数 Agent。Agent 负责执行 primary、处理
unavailable/runtime 回退与可选 zero review；后端自身绝不切换。返回
CountingResult；需要通用 VQA answer 时（由中性 metadata 开关驱动）生成
AgentResult 作为 primary，并把 CountingResult 放入 additional_results——
不做任何数据集判断。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents.base import AgentContext, AgentExecution
from agents.counting.backends.base import (
    CountingBackendOutcome,
    CountingBackendUnavailableError,
    CountingRequest,
)
from agents.counting.backends.registry import BackendRegistry
from agents.counting.backends.selector import BackendSelector
from agents.counting.schema import CountTargetSpec, CountingResult, IssueRecord
from agents.counting.target_parser import CountTargetParser
from agents.errors import (
    AgentTaskMismatchError,
    DetectorClassMapMismatchError,
    DetectorTaskMismatchError,
    DetectorWeightsHashMismatchError,
    DetectorWeightsMissingError,
    OptionalDependencyMissingError,
)
from agents.schema import AgentName, AgentResult, VisualEvidence
from models.images import read_normalized_image

# Detector-loading failures that make the primary backend unavailable.
# 使主后端不可用的检测器加载失败类型。
_UNAVAILABLE_ERRORS = (
    DetectorWeightsMissingError,
    DetectorWeightsHashMismatchError,
    OptionalDependencyMissingError,
    DetectorTaskMismatchError,
    DetectorClassMapMismatchError,
)

_AGENT_ANSWER_METADATA_KEY = "answer_as_agent_result"


class CountingAgent:
    """Execute counting with a point-derived final answer and explicit plans.
    以点导出最终答案执行计数，并携带显式执行计划。"""

    name: AgentName = "counting_agent"
    supported_tasks: frozenset[str] = frozenset({"counting", "fine_grained_counting"})

    def __init__(
        self,
        client: Any,
        *,
        target_prompt: str,
        backend_registry: BackendRegistry,
        target_prompt_version: str = "target-parse-v1",
        default_backend: str = "auto",
        fallback_to_qwen_on_unavailable: bool = True,
        fallback_to_qwen_on_error: bool = True,
        verify_empty_with_qwen: bool = True,
        trust_empty_detection: bool = False,
    ) -> None:
        self._client = client
        self._target_prompt = target_prompt
        self._target_prompt_version = target_prompt_version
        self._selector = BackendSelector(
            backend_registry, default_backend=default_backend
        )
        self._fallback_to_qwen_on_unavailable = fallback_to_qwen_on_unavailable
        self._fallback_to_qwen_on_error = fallback_to_qwen_on_error
        self._verify_empty_with_qwen = verify_empty_with_qwen
        self._trust_empty_detection = trust_empty_detection

    async def run(
        self,
        sample: Any,
        context: AgentContext,
    ) -> AgentExecution:
        """Run the selected plan and never silently hide a fallback.
        执行选定计划，绝不静默隐藏回退。"""
        # Task gating happens before images, target parsing, budget, or any
        # backend call. task 门控发生在读图、解析 target、消费 budget 或调用
        # 任何后端之前。
        if sample.task not in self.supported_tasks:
            raise AgentTaskMismatchError(
                self.name,
                sample.task,
                supported=self.supported_tasks,
            )
        target = await self._target(sample, context)
        hints: dict[str, Any] = {"quantity_estimation": True}
        plan = self._selector.plan(target, task=sample.task, hints=hints)
        if plan is None:
            raise CountingBackendUnavailableError(
                f"No counting backend plan for task={sample.task!r}, "
                f"target={target.canonical_label!r}"
            )
        request = CountingRequest(
            sample=sample,
            image=_resolve_sample_image(sample, context),
            target=target,
            artifact_dir=context.artifact_dir,
        )
        attempted = [plan.primary_backend_name]
        primary = self._selector.backend_by_name(plan.primary_backend_name)
        final_backend = plan.primary_backend_name
        review_backend: str | None = None
        fallback_triggered = False
        fallback_kind: str | None = None
        fallback_reason: str | None = None
        yolo_trace: dict[str, object] | None = (
            dict(primary.trace_profile())
            if self._is_yolo(primary) and callable(getattr(primary, "trace_profile", None))
            else None
        )
        try:
            outcome = await primary.count(request, context)
        except _UNAVAILABLE_ERRORS as exc:
            if not self._fallback_to_qwen_on_unavailable or not plan.fallback_backend_names:
                raise CountingBackendUnavailableError(
                    f"backend {plan.primary_backend_name!r} unavailable and no fallback "
                    f"allowed: {type(exc).__name__}"
                ) from exc
            outcome, attempted, fallback_triggered, fallback_kind, fallback_reason = (
                await self._fallback(plan, request, context, attempted, "unavailable", exc)
            )
            final_backend = attempted[-1]
        except Exception as exc:
            if (
                not self._is_yolo(primary)
                or not self._fallback_to_qwen_on_error
                or not plan.fallback_backend_names
            ):
                raise
            outcome, attempted, fallback_triggered, fallback_kind, fallback_reason = (
                await self._fallback(plan, request, context, attempted, "runtime_error", exc)
            )
            final_backend = attempted[-1]
        else:
            if self._is_yolo(primary):
                yolo_trace = {**(yolo_trace or {}), **dict(outcome.trace or {})}
                if (
                    outcome.counting.final_count == 0
                    and self._verify_empty_with_qwen
                    and not self._trust_empty_detection
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
                            fallback_triggered = True
                            fallback_kind = "zero_review"
                            fallback_reason = "DETECTOR_ZERO_OVERRIDDEN_BY_REVIEW"
                            yolo_trace["zero_overridden"] = True
                        else:
                            yolo_trace["zero_overridden"] = False
                    except Exception as exc:
                        yolo_trace.update(
                            {
                                "zero_review_status": "failed",
                                "zero_review_result_count": None,
                                "zero_overridden": False,
                            }
                        )
                        warning = IssueRecord(
                            code="DETECTOR_ZERO_REVIEW_FAILED",
                            message=f"{type(exc).__name__}: {_safe_error_text(exc)}",
                        )
                        outcome = CountingBackendOutcome(
                            counting=outcome.counting.model_copy(
                                update={
                                    "status": (
                                        "completed_with_warnings"
                                        if outcome.counting.status == "completed"
                                        else outcome.counting.status
                                    ),
                                    "warnings": [*outcome.counting.warnings, warning],
                                }
                            ),
                            agent_result=outcome.agent_result,
                            trace=outcome.trace,
                        )
        trace: dict[str, object] = {
            "agent_class": "agents.counting.agent.CountingAgent",
            "entrypoint": "run",
            "route": "CountingAgent.run -> BackendSelector.plan -> " + " -> ".join(attempted),
            "requested_backend_mode": self._selector_default_backend(),
            "primary_backend": plan.primary_backend_name,
            "review_backend": review_backend,
            "final_backend": final_backend,
            "executed_backend": final_backend,
            "backend": final_backend,
            "attempted_backends": attempted,
            "selection_reason": list(plan.reason_codes),
            "target": target.canonical_label,
            "target_classes": list(plan.target_classes),
            "fallback_triggered": fallback_triggered,
            "fallback_kind": fallback_kind,
            "fallback_reason": fallback_reason,
            "status": outcome.counting.status,
        }
        # Backend traces live in their own namespace so a plugin can never
        # overwrite agent-level fields. 后端 trace 位于独立命名空间，插件无法
        # 覆盖 Agent 级字段。
        trace["backend_trace"] = dict(outcome.trace or {})
        if yolo_trace is not None:
            yolo_trace.update(
                {
                    "attempted": True,
                    "used_for_final": final_backend == plan.primary_backend_name,
                }
            )
            trace["yolo"] = yolo_trace
        else:
            trace["yolo"] = {"attempted": False, "used_for_final": False}

        # The primary payload is always the CountingResult with a fixed
        # filename; an AgentResult — whether produced by the backend or
        # requested via the neutral metadata switch — is always an additional
        # result, never the primary schema.
        # 主载荷永远是 CountingResult 且文件名固定；AgentResult——无论来自
        # 后端还是由中性 metadata 开关请求——始终是附加结果，绝不成为主
        # Schema。
        additional_results: dict[str, Any] = {}
        if outcome.agent_result is not None:
            additional_results["agent_result.json"] = outcome.agent_result.model_dump(
                mode="json"
            )
        elif sample.metadata.get(_AGENT_ANSWER_METADATA_KEY, False):
            additional_results["agent_result.json"] = _agent_result(
                outcome.counting, sample.images[0].image_id
            ).model_dump(mode="json")
        return AgentExecution(
            agent_name=self.name,
            payload=outcome.counting,
            result_filename="counting_result.json",
            additional_results=additional_results,
            trace=trace,
        )

    async def _target(
        self,
        sample: Any,
        context: AgentContext,
    ) -> CountTargetSpec:
        """Resolve the target: normalization hint first, legacy metadata
        second, frozen Qwen contract last. 解析目标：优先 normalization
        hint，其次 legacy metadata，最后冻结 Qwen 契约。"""
        normalization_hint = (
            sample.normalization.count_target_hint
            if sample.normalization is not None
            else None
        )
        parser = CountTargetParser(
            self._client,
            self._target_prompt,
            _identity_model(self._client),
            prompt_version=self._target_prompt_version,
        )
        return await parser.parse(
            sample.question,
            sample_id=sample.sample_id,
            artifact_dir=context.artifact_dir,
            count_target_hint=normalization_hint,
            legacy_metadata=sample.metadata,
            budget=getattr(context, "call_budget", None),
        )

    async def _fallback(
        self,
        plan: Any,
        request: CountingRequest,
        context: AgentContext,
        attempted: list[str],
        kind: str,
        error: Exception,
    ):
        name = plan.fallback_backend_names[0]
        attempted.append(name)
        try:
            backend = self._selector.backend_by_name(name)
        except KeyError as exc:
            raise CountingBackendUnavailableError(
                f"fallback backend {name!r} is not registered"
            ) from exc
        outcome = await backend.count(request, context)
        return (
            outcome,
            attempted,
            True,
            kind,
            f"{type(error).__name__}: {_safe_error_text(error)}",
        )

    def _selector_default_backend(self) -> str:
        return str(getattr(self._selector, "_default_backend", "auto"))

    @staticmethod
    def _is_yolo(backend: object) -> bool:
        return getattr(backend, "name", "") != "qwen_point"


def _agent_result(counting: CountingResult, image_id: str) -> AgentResult:
    """Adapt detector points to the canonical VQA result; dataset-neutral.
    将检测器点适配为标准 VQA 结果；与数据集无关。"""
    accepted = [point for point in counting.global_points if point.accepted]
    evidence = [
        VisualEvidence(
            label=(
                point.provenance.source_class
                if point.provenance and point.provenance.source_class
                else counting.target
            ),
            point=[point.global_x_norm, point.global_y_norm],
            confidence=point.confidence,
            image_id=image_id,
        )
        for point in accepted
    ]
    status = (
        "failed"
        if counting.status == "failed"
        else "partial"
        if counting.status == "partial"
        else "completed"
    )
    return AgentResult(
        agent_name="counting_agent",
        answer=str(counting.final_count),
        evidence=[point.short_evidence for point in accepted],
        evidence_items=evidence,
        geometry={
            "version": "accepted-detector-point-count-v1",
            "coordinate_frame": "normalized_0_999_top_left",
            "rule": "final_count_equals_accepted_points",
            "accepted_point_count": len(accepted),
            "final_count": counting.final_count,
            "counting_status": counting.status,
            "warnings": [item.model_dump(mode="json") for item in counting.warnings],
        },
        status=status,
    )


def _resolve_sample_image(sample: Any, context: AgentContext) -> Any:
    """Resolve the first sample image against context.data_root with escape
    protection; never reads relative to the working directory.
    按 context.data_root 解析样本首图并防逃逸；绝不相对工作目录读取。"""
    if context.data_root is None:
        raise RuntimeError("CountingAgent requires context.data_root for image resolution")
    root = context.data_root.resolve()
    candidate = (root / sample.images[0].path).resolve()
    if not candidate.is_relative_to(root):
        raise RuntimeError(f"image path escapes data root: {sample.images[0].path}")
    if not candidate.is_file():
        raise RuntimeError(f"image file does not exist: {sample.images[0].path}")
    return read_normalized_image(candidate)


def _identity_model(client: Any) -> str | None:
    identity = getattr(client, "cache_identity", None)
    return identity.model if identity is not None else None


def _safe_error_text(error: Exception) -> str:
    """Keep trace error text short and path-free.
    保持 trace 错误文本简短且不含路径。"""
    text = str(error).strip()
    if len(text) > 200:
        text = text[:200] + "..."
    return text
