"""Counting agent with explicit backend plans and visible fallback.

具有显式后端计划与可见回退的计数 Agent。Agent 只负责 task 门控、target
解析、BackendSelector.plan、构造 CountingRequest、调用
CountingPlanExecutor 与打包 AgentExecution/公共 trace；primary 执行、
unavailable/runtime 回退与 zero review 全部由 Executor 承担。后端自身绝不
切换。主 payload 恒为 CountingResult；AgentResult 只作为附加结果。backend
类型通过显式 kind 识别，只有 yolo_obb 进入检测器专属流程。公共入口只抛
稳定错误，trace 不含原始异常文本、路径、密钥或 Base64。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from agents.base import AgentContext, AgentExecution
from agents.counting.backends.base import BackendPlan, CountingRequest
from agents.counting.backends.registry import BackendRegistry
from agents.counting.backends.selector import BackendSelector
from agents.counting.expert_catalog import ExpertCatalog
from agents.counting.executor import (
    CountingExecutionPolicy,
    CountingExecutionResult,
    CountingPlanExecutor,
)
from agents.counting.schema import (
    CountingExecutionAudit,
    CountingResult,
)
from agents.counting.target_parser import (
    CountTargetResolutionError,
    CountTargetResolver,
    ResolvedCountTarget,
)
from agents.errors import (
    AgentExecutionError,
    AgentTaskMismatchError,
    CountingBackendUnavailableError,
)
from agents.schema import AgentName, AgentResult, VisualEvidence
from data.schema import UnifiedSample
from models.base import VisionLanguageClient
from models.images import read_normalized_image

_AGENT_ANSWER_METADATA_KEY = "answer_as_agent_result"


class CountingAgent:
    """Execute counting with a point-derived final answer and explicit plans.
    以点导出最终答案执行计数，并携带显式执行计划。"""

    name: AgentName = "counting_agent"
    supported_tasks: frozenset[str] = frozenset({"counting", "fine_grained_counting"})

    def __init__(
        self,
        client: VisionLanguageClient,
        *,
        target_resolver: CountTargetResolver,
        backend_registry: BackendRegistry,
        default_backend: str = "auto",
        fallback_on_backend_unavailable: bool = True,
        fallback_on_backend_error: bool = True,
        verify_empty_detection: bool = True,
        trust_empty_detection: bool = False,
        verify_empty_semantic: bool = False,
        expert_catalog: ExpertCatalog | None = None,
    ) -> None:
        self._client = client
        self._target_resolver = target_resolver
        self._expert_catalog = expert_catalog
        self._selector = BackendSelector(
            backend_registry, default_backend=default_backend
        )
        self._executor = CountingPlanExecutor(
            self._selector,
            policy=CountingExecutionPolicy(
                fallback_on_backend_unavailable=fallback_on_backend_unavailable,
                fallback_on_backend_error=fallback_on_backend_error,
                verify_empty_detection=verify_empty_detection,
                trust_empty_detection=trust_empty_detection,
                verify_empty_semantic=verify_empty_semantic,
            ),
        )

    async def run(
        self,
        sample: UnifiedSample,
        context: AgentContext,
    ) -> AgentExecution:
        """Gate the task, plan, execute via the executor, and package the final
        AgentExecution; never silently hide a fallback.
        task 门控、计划、经 Executor 执行并打包最终 AgentExecution；绝不静默
        隐藏回退。"""
        # Task gating happens before images, target parsing, budget, or any
        # backend call. task 门控发生在读图、解析 target、消费 budget 或调用
        # 任何后端之前。
        if sample.task not in self.supported_tasks:
            raise AgentTaskMismatchError(
                self.name,
                sample.task,
                supported=self.supported_tasks,
            )
        try:
            resolution = self._target_resolution(sample, context)
        except CountTargetResolutionError:
            raise
        except Exception as exc:
            raise AgentExecutionError(
                self.name, sample.sample_id, cause="TARGET_PARSE_FAILED"
            ) from exc
        hints: dict[str, Any] = {"quantity_estimation": True}
        if self._expert_catalog is not None:
            hints.update(self._expert_catalog.target_hints(resolution.target))
        plan = self._selector.plan(
            resolution.target,
            task=sample.task,
            executable_leaf_categories=resolution.executable_leaf_categories,
            hints=hints,
        )
        if plan is None:
            raise CountingBackendUnavailableError(
                resolution.target.canonical_label, reason_code="NO_BACKEND_PLAN"
            )
        request = CountingRequest(
            sample=sample,
            image=_resolve_sample_image(sample, context),
            target=resolution.target,
            executable_leaf_categories=resolution.executable_leaf_categories,
            artifact_dir=context.artifact_dir,
        )
        execution_state = await self._executor.execute(
            plan=plan,
            request=request,
            context=context,
            agent_name=self.name,
        )
        trace = self._build_trace(plan, resolution, execution_state)

        # The primary payload is always the CountingResult with a fixed
        # filename; an AgentResult — whether produced by the backend or
        # requested via the neutral metadata switch — is always an additional
        # result, never the primary schema.
        # 主载荷永远是 CountingResult 且文件名固定；AgentResult——无论来自
        # 后端还是由中性 metadata 开关请求——始终是附加结果，绝不成为主
        # Schema。
        additional_results: dict[str, Any] = {}
        if execution_state.outcome.agent_result is not None:
            additional_results["agent_result.json"] = (
                execution_state.outcome.agent_result.model_dump(mode="json")
            )
        elif sample.metadata.get(_AGENT_ANSWER_METADATA_KEY, False):
            additional_results["agent_result.json"] = _agent_result(
                execution_state.outcome.counting, sample.images[0].image_id
            ).model_dump(mode="json")
        additional_results["counting_attempts.json"] = CountingExecutionAudit(
            sample_id=sample.sample_id,
            target=resolution.target.canonical_label,
            attempts=list(execution_state.attempt_audits),
        ).model_dump(mode="json")
        return AgentExecution(
            agent_name=self.name,
            payload=execution_state.outcome.counting,
            result_filename="counting_result.json",
            additional_results=additional_results,
            trace=trace,
        )

    def _build_trace(
        self,
        plan: BackendPlan,
        resolution: ResolvedCountTarget,
        state: CountingExecutionResult,
    ) -> dict[str, object]:
        """Build the public trace from the executor's structured state; every
        field comes from public attributes of the result.
        由 Executor 的结构化状态构建公共 trace；所有字段都来自结果的公开
        属性。"""
        attempted = list(state.attempted_backends)
        trace: dict[str, object] = {
            "agent_class": "agents.counting.agent.CountingAgent",
            "entrypoint": "run",
            "route": "CountingAgent.run -> BackendSelector.plan -> " + " -> ".join(attempted),
            "requested_backend_mode": self._selector.default_backend,
            "primary_backend": state.primary_backend,
            "primary": state.primary_backend,
            "primary_backend_kind": state.primary_kind,
            "review_backend": state.review_backend,
            "review_error_type": state.review_error_type,
            "final_backend": state.final_backend,
            "final": state.final_backend,
            "final_backend_kind": state.final_kind,
            "executed_backend": state.final_backend,
            "backend": state.final_backend,
            "candidate_backends": list(state.candidate_backends),
            "attempted_backends": attempted,
            "fallback_history": [
                entry.to_trace() for entry in state.fallback_history
            ],
            "selection_reason": list(plan.reason_codes),
            "target": resolution.target.canonical_label,
            "target_source": resolution.target_source,
            "planner_target": resolution.planner_target,
            "planner_object_categories": list(resolution.planner_object_categories),
            "executable_leaf_categories": list(
                resolution.executable_leaf_categories
            ),
            "target_validation": resolution.validation_status,
            "verifier_source": resolution.verifier_source,
            "target_classes": list(plan.target_classes),
            "fallback_triggered": state.fallback_triggered,
            "fallback_kind": state.fallback_kind,
            "fallback_reason_code": state.fallback_reason_code,
            "fallback_error_type": state.fallback_error_type,
            "status": state.outcome.counting.status,
        }
        # Backend traces live in their own namespace so a plugin can never
        # overwrite agent-level fields. 后端 trace 位于独立命名空间，插件无法
        # 覆盖 Agent 级字段。
        trace["backend_trace"] = dict(state.outcome.trace or {})
        if state.yolo_trace is not None:
            yolo_trace = dict(state.yolo_trace)
            yolo_trace.update(
                {
                    "attempted": True,
                    "used_for_final": state.final_backend == state.primary_backend,
                }
            )
            trace["yolo"] = yolo_trace
        else:
            trace["yolo"] = {"attempted": False, "used_for_final": False}
        return trace

    def _target_resolution(
        self,
        sample: UnifiedSample,
        context: AgentContext,
    ) -> ResolvedCountTarget:
        """Resolve the planner proposal and deterministic verifier hint."""
        normalization_hint = (
            sample.normalization.count_target_hint
            if sample.normalization is not None
            else None
        )
        plan = context.visual_task_plan
        return self._target_resolver.resolve(
            task=sample.task,
            question=sample.question,
            planner_target=plan.count_target if plan is not None else None,
            planner_object_categories=tuple(
                plan.object_categories if plan is not None else ()
            ),
            count_target_hint=normalization_hint,
            legacy_metadata=sample.metadata,
        )

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


def _resolve_sample_image(sample: UnifiedSample, context: AgentContext) -> Image.Image:
    """Resolve the first sample image against context.data_root with escape
    protection; failures use stable error codes and never leak absolute
    paths. 按 context.data_root 解析样本首图并防逃逸；失败使用稳定错误码，
    绝不泄漏绝对路径。"""
    agent_name = "counting_agent"
    if context.data_root is None:
        raise AgentExecutionError(agent_name, sample.sample_id, cause="DATA_ROOT_REQUIRED")
    root = context.data_root.resolve()
    candidate = (root / sample.images[0].path).resolve()
    if not candidate.is_relative_to(root):
        raise AgentExecutionError(agent_name, sample.sample_id, cause="IMAGE_PATH_ESCAPE")
    if not candidate.is_file():
        raise AgentExecutionError(agent_name, sample.sample_id, cause="IMAGE_NOT_FOUND")
    try:
        return read_normalized_image(candidate)
    except OSError as exc:
        raise AgentExecutionError(agent_name, sample.sample_id, cause="IMAGE_READ_FAILED") from exc
