"""Counting agent with explicit backend plans and visible fallback.
具有显式后端计划与可见回退的计数 Agent。
"""

from __future__ import annotations

from spacers_agent.agents.base import AgentContext, AgentExecution, AgentName
from spacers_agent.agents.counting.backends.base import CountingBackendOutcome, CountingRequest
from spacers_agent.agents.counting.backends.registry import BackendRegistry
from spacers_agent.agents.counting.backends.selector import BackendSelector, is_vrsbench_quantity
from spacers_agent.agents.counting.target_parser import CountTargetParser
from spacers_agent.agents.errors import (
    DetectorClassMapMismatchError,
    DetectorInferenceError,
    DetectorTaskMismatchError,
    DetectorWeightsHashMismatchError,
    DetectorWeightsMissingError,
    OptionalDependencyMissingError,
)
from spacers_agent.imaging import read_normalized_image
from spacers_agent.schemas import CountTargetSpec, CountingResult, ExpertResult, IssueRecord, VisualEvidence
from spacers_agent.vqa_geometry import vrsbench_count_target


class CountingAgent:
    """Execute native or VRSBench counting with a point-derived final answer.
    以点导出最终答案执行原生或 VRSBench 计数。
    """

    name: AgentName = "counting_agent"
    supported_tasks: frozenset[str] = frozenset({"counting", "fine_grained_counting"})

    def __init__(self, client, prompts: dict[str, str], model: str, backend_registry: BackendRegistry, *, settings) -> None:
        self._client = client
        self._target_prompt = prompts["target"]
        self._model = model
        self._settings = settings
        self._selector = BackendSelector(backend_registry, default_backend=settings.agents.counting.default_backend)

    async def run(self, sample, context: AgentContext) -> AgentExecution:
        """Run the selected plan and never silently hide a detector fallback.
        执行选定计划，绝不静默隐藏检测器回退。
        """
        target = await self._target(sample, context)
        plan = self._selector.plan(target, sample)
        if plan is None:
            raise RuntimeError(f"No counting backend plan for task={sample.task!r}, target={target.canonical_label!r}")
        request = CountingRequest(sample=sample, image=read_normalized_image(sample.images[0].path), target=target, artifact_dir=context.artifact_dir)
        attempted = [plan.primary_backend_name]
        primary = self._selector.backend_by_name(plan.primary_backend_name)
        fallback_triggered = False
        fallback_kind: str | None = None
        fallback_reason: str | None = None
        yolo_trace: dict[str, object] | None = (
            dict(primary.trace_profile()) if self._is_yolo(primary) and callable(getattr(primary, "trace_profile", None)) else None
        )
        try:
            outcome = await primary.count(request, context)
        except (DetectorWeightsMissingError, DetectorWeightsHashMismatchError, OptionalDependencyMissingError, DetectorTaskMismatchError, DetectorClassMapMismatchError) as exc:
            if not self._settings.backend.yolo.fallback_to_qwen_on_unavailable or not plan.fallback_backend_names:
                raise
            outcome, attempted, fallback_triggered, fallback_kind, fallback_reason = await self._fallback(plan, request, context, attempted, "unavailable", exc)
        except Exception as exc:
            if not self._is_yolo(primary) or not self._settings.backend.yolo.fallback_to_qwen_on_error or not plan.fallback_backend_names:
                raise
            outcome, attempted, fallback_triggered, fallback_kind, fallback_reason = await self._fallback(plan, request, context, attempted, "runtime_error", exc)
        else:
            if self._is_yolo(primary):
                yolo_trace = {**(yolo_trace or {}), **dict(outcome.trace or {})}
                if outcome.counting.final_count == 0 and self._settings.backend.yolo.verify_empty_with_qwen and not self._settings.backend.trust_empty_detection:
                    yolo_trace["zero_review_triggered"] = True
                    review_name = plan.fallback_backend_names[0] if plan.fallback_backend_names else "qwen_point"
                    yolo_trace["zero_review_backend"] = review_name
                    try:
                        review_backend = self._selector.backend_by_name(review_name)
                        attempted.append(review_name)
                        review = await review_backend.count(request, context)
                        yolo_trace["zero_review_status"] = review.counting.status
                        yolo_trace["zero_review_result_count"] = review.counting.final_count
                        if review.counting.final_count > 0:
                            outcome = review
                            fallback_triggered = True
                            fallback_kind = "zero_review"
                            fallback_reason = "YOLO_ZERO_OVERRIDDEN_BY_QWEN_REVIEW"
                            yolo_trace["zero_overridden"] = True
                        else:
                            yolo_trace["zero_overridden"] = False
                    except Exception as exc:
                        yolo_trace.update({"zero_review_status": "failed", "zero_review_result_count": None, "zero_overridden": False})
                        warning = IssueRecord(code="YOLO_ZERO_REVIEW_FAILED", message=f"{type(exc).__name__}: {exc}")
                        outcome = CountingBackendOutcome(
                            counting=outcome.counting.model_copy(update={
                                "status": "completed_with_warnings" if outcome.counting.status == "completed" else outcome.counting.status,
                                "warnings": [*outcome.counting.warnings, warning],
                            }),
                            expert_result=outcome.expert_result,
                            trace=outcome.trace,
                        )
        executed = attempted[-1]
        trace: dict[str, object] = {
            "agent_class": "spacers_agent.agents.counting.agent.CountingAgent",
            "entrypoint": "run",
            "route": "CountingAgent.run -> BackendSelector.plan -> " + " -> ".join(attempted),
            "requested_backend_mode": self._settings.agents.counting.default_backend,
            "primary_backend": plan.primary_backend_name,
            "executed_backend": executed,
            "backend": executed,
            "attempted_backends": attempted,
            "selection_reason": list(plan.reason_codes),
            "target": target.canonical_label,
            "target_classes": list(plan.target_classes),
            "fallback_triggered": fallback_triggered,
            "fallback_kind": fallback_kind,
            "fallback_reason": fallback_reason,
            "status": outcome.counting.status,
        }
        if outcome.trace:
            trace.update(outcome.trace)
        if yolo_trace is not None:
            yolo_trace.update({"attempted": True, "used_for_final": executed == plan.primary_backend_name})
            trace["yolo"] = yolo_trace
        else:
            trace["yolo"] = {"attempted": False, "used_for_final": False}
        if is_vrsbench_quantity(sample) and outcome.expert_result is None:
            outcome = CountingBackendOutcome(
                counting=outcome.counting,
                expert_result=self._vrsbench_expert_result(outcome.counting, sample.images[0].image_id),
                trace=outcome.trace,
            )
        if outcome.expert_result is not None:
            return AgentExecution(agent_name=self.name, payload=outcome.expert_result, result_filename="expert_result.json", additional_results={"counting_result.json": outcome.counting}, trace=trace)
        return AgentExecution(agent_name=self.name, payload=outcome.counting, result_filename="counting_result.json", trace=trace)

    async def _target(self, sample, context: AgentContext) -> CountTargetSpec:
        override = sample.metadata.get("count_target_spec")
        if override is not None:
            return CountTargetSpec.model_validate(override)
        if is_vrsbench_quantity(sample):
            return vrsbench_count_target(sample.question)
        context.call_budget.reserve_qwen()
        return await CountTargetParser(self._client, self._target_prompt, self._model).parse(sample.question, sample_id=sample.sample_id, artifact_dir=context.artifact_dir)

    async def _fallback(self, plan, request, context, attempted, kind: str, error: Exception):
        name = plan.fallback_backend_names[0]
        attempted.append(name)
        outcome = await self._selector.backend_by_name(name).count(request, context)
        return outcome, attempted, True, kind, f"{type(error).__name__}: {error}"

    @staticmethod
    def _is_yolo(backend: object) -> bool:
        return getattr(backend, "name", "") not in {"qwen_point", "vrsbench_qwen_count"}

    @staticmethod
    def _vrsbench_expert_result(counting: CountingResult, image_id: str) -> ExpertResult:
        """Adapt detector points to the canonical VQA result required by reports and Judge.
        将检测器点适配为报告和审评所需的标准 VQA 结果。
        """
        accepted = [point for point in counting.global_points if point.accepted]
        evidence = [
            VisualEvidence(
                label=(point.provenance.source_class if point.provenance and point.provenance.source_class else counting.target),
                point=[point.global_x_norm, point.global_y_norm],
                confidence=point.confidence,
                image_id=image_id,
            )
            for point in accepted
        ]
        status = "failed" if counting.status == "failed" else "partial" if counting.status == "partial" else "completed"
        return ExpertResult(
            expert="counting_expert",
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
