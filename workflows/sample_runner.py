"""Single-sample execution kernel: routing, candidate fallback, agent
fallback, shared budget, evaluation, and optional judge.

单样本执行内核：路由、候选兜底、Agent 兜底、共享预算、评测与可选 judge。

SampleRunner 不拥有数据集迭代/resume/shard/模型创建/AppSettings/
PromptCatalog/reporting/CLI——所有依赖注入构造。低置信度 TaskResolution 按
candidate_tasks（最多 3 个）构建 attempt plan，路由后按 AgentName 稳定去重，
绝不跑所有 Agent；候选样本经 model_copy 重建（task 替换、normalization 清空、
图像角色重建），绝不原地修改 UnifiedSample；不兼容候选稳定跳过。失败只记录
稳定 code（错误类名或显式 code），绝不泄漏原始异常文本、路径或密钥。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agents.base import AgentContext, AgentExecution
from agents.counting.schema import CountingResult
from agents.registry import AgentRegistry
from data.schema import CHANGE_TASKS, UnifiedSample
from evaluation.metrics.counting import merge_count_evaluation
from evaluation.metrics.grounding import box_iou, grounding_deterministic_metrics
from evaluation.metrics.vqa import merge_vqa_evaluation
from evaluation.records import (
    CaptionDeterministicMetrics,
    EvaluationRecord,
    evaluation_filename_for_runtime_task,
    evaluation_task_for_runtime_task,
)
from models.base import VisionLanguageClient
from routing.router import TaskRouter
from routing.schema import RoutingDecision, TaskResolution
from workflows.artifact_writer import ArtifactWriter
from workflows.call_budget import CallBudget, CallBudgetFactory
from workflows.judge_service import JudgeService
from workflows.schema import SampleRunOutcome, SampleRunStatus
from workflows.visual_planner import VisualPlanError, VisualPlanningGate

# Hard bound on the candidate attempt plan: never run all agents.
# 候选 attempt plan 的硬上限：绝不跑所有 Agent。
MAX_ATTEMPTS = 3

# Prediction coordinate frame mandated by the Agent contract
# (agents.schema.VisualEvidence). Ground truth must declare the identical
# frame before any IoU is computed; other frames fail closed.
# Agent 契约（agents.schema.VisualEvidence）强制的预测坐标系。真值必须声明
# 相同坐标系才计算 IoU；其他坐标系一律 fail-closed。
_GROUNDING_PREDICTION_FRAME = "normalized_0_999_top_left"

# A routing fallback may deliberately execute a different task contract than
# the resolved route task. Keep this remap explicit and narrow; agent
# supported_tasks remains fail-closed. / routing fallback 可能有意执行不同于
# 已解析 route task 的任务契约。映射保持显式且窄化；Agent supported_tasks
# 继续 fail-closed。
_ROUTING_FALLBACK_TASK_REMAP = {
    ("change_qa", "general_vqa_agent"): "general_vqa",
}


def build_deterministic_evaluation(
    *,
    sample: UnifiedSample,
    execution_payload: object,
    execution_task: str | None = None,
) -> tuple[object | None, str | None]:
    """Shared deterministic evaluator dispatch consumed by both fresh runs and
    resume supplements, so the two paths can never drift. Returns
    (evaluation, filename). execution_task selects the metric family: None on
    the fresh path uses sample.task (the sample is already rebuilt for the
    executed task); the persisted/resume path passes the executed task
    explicitly, because sample.task may be the canonical resolved task after a
    candidate fallback. Grounding yields None unless prediction and ground
    truth agree on normalized_0_999_top_left with 4-value xyxy boxes; counting
    yields None for non-CountingResult payloads; caption without references
    yields None. 供 fresh run 与 resume 补判共用的确定性评估分派，两条路径
    永不漂移。返回（evaluation, filename）。execution_task 选择指标族：
    fresh 路径传 None 使用 sample.task（样本已为执行任务重建）；持久化/
    resume 路径显式传执行任务，因为候选兜底后 sample.task 可能是 canonical
    resolved task。grounding 仅在预测与真值同为 normalized_0_999_top_left
    且均为 4-value xyxy 时产出；counting 对非 CountingResult 载荷返回
    None；caption 无参考答案返回 None。"""

    task = execution_task or sample.task
    family = evaluation_task_for_runtime_task(task)
    filename = evaluation_filename_for_runtime_task(task)
    if family is None or filename is None:
        return None, None
    if family == "general_vqa":
        candidate_answer = str(getattr(execution_payload, "answer", ""))
        references = (
            list(sample.ground_truth.answers)
            if sample.ground_truth is not None
            else []
        )
        evaluation = merge_vqa_evaluation(
            sample_id=sample.sample_id,
            question=sample.question,
            reference_answers=references,
            candidate_answer=candidate_answer,
        )
        return evaluation, filename
    if family == "counting":
        if not isinstance(execution_payload, CountingResult):
            return None, None  # fail closed: never fabricate counting metrics
        evaluation = merge_count_evaluation(
            sample_id=sample.sample_id,
            counting=execution_payload,
            ground_truth=sample.ground_truth,
        )
        return evaluation, filename
    if family == "grounding":
        evaluation = _grounding_evaluation(sample, execution_payload)
        return evaluation, filename if evaluation is not None else None
    if family == "caption":
        evaluation = _caption_evaluation(sample, execution_payload)
        return evaluation, filename if evaluation is not None else None
    return None, None


@dataclass(frozen=True)
class _AgentAttempt:
    """One concrete agent execution with its materialized task and sample.
    一次具体 Agent 执行及其已物化 task/sample。"""

    agent_name: str
    task: str
    sample: UnifiedSample


@dataclass(frozen=True)
class _Attempt:
    """One executable attempt: a task, its deduplicated agent list, the sample
    rebuilt for that task, and the routing decision. 一次可执行尝试：任务、
    去重后的 Agent 列表、为该任务重建的样本与路由决策。"""

    task: str
    agent_attempts: tuple[_AgentAttempt, ...]
    sample: UnifiedSample
    decision: RoutingDecision

    @property
    def agent_names(self) -> tuple[str, ...]:
        return tuple(item.agent_name for item in self.agent_attempts)


def sample_state_from_payload(payload: object) -> str:
    """Map persisted payload status to the only allowed sample-state mapping;
    unknown statuses fail closed as failed. 使用唯一允许的映射将持久化载荷
    状态转换为样本状态；未知状态以 failed 关闭失败。"""

    status = str(getattr(payload, "status", "completed"))
    mapping = {
        "completed": "succeeded",
        "completed_with_warnings": "succeeded",
        "partial": "partial",
        "failed": "failed",
    }
    return mapping.get(status, "failed")


def failed_sample_status(sample: UnifiedSample, error: Exception) -> SampleRunStatus:
    """Create a visible failed status carrying only stable codes.
    创建只携带稳定 code 的可见失败状态。"""

    return _status(
        sample,
        "failed",
        error_code=_stable_error_code(error),
        error_message=type(error).__name__,
    )


def _stable_error_code(error: Exception) -> str:
    """Prefer an explicit stable code on the exception, else the class name;
    never the raw message. 优先取异常上的显式稳定 code，否则用类名；绝不取
    原始消息。"""

    code = getattr(error, "code", None)
    if isinstance(code, str) and code:
        return code
    return type(error).__name__


def _status(
    sample: UnifiedSample,
    state: str,
    *,
    result_path: Path | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> SampleRunStatus:
    """Build one timestamped status without inferring business success.
    构建带时间戳的状态，不推断业务成功。"""

    return SampleRunStatus(
        sample_id=sample.sample_id,
        task=sample.task,
        state=state,
        error_code=error_code,
        error_message=error_message,
        result_path=result_path,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


class SampleRunner:
    """Route and execute one sample using only injected runtime components.
    仅使用注入的运行时组件路由并执行一条样本。"""

    def __init__(
        self,
        registry: AgentRegistry,
        router: TaskRouter,
        qwen_client: VisionLanguageClient,
        artifact_writer: ArtifactWriter,
        call_budget_factory: CallBudgetFactory,
        judge_service: JudgeService | None = None,
        fallback_on_partial: bool = False,
        data_root: Path | None = None,
        visual_planning: VisualPlanningGate | None = None,
    ) -> None:
        self.agent_registry = registry
        self.router = router
        self.qwen_client = qwen_client
        self.artifact_writer = artifact_writer
        self.call_budget_factory = call_budget_factory
        self.judge_service = judge_service
        self.fallback_on_partial = fallback_on_partial
        self.data_root = data_root
        # Feature-flagged first-Qwen planning gate; None keeps every existing
        # call count, result, trace, and artifact byte-identical.
        # 特性开关第一 Qwen 规划门；None 时现有调用次数、结果、trace 与产物
        # 逐字节一致。
        self.visual_planning = visual_planning

    async def run_one(
        self,
        sample: UnifiedSample,
        sample_dir: Path,
        *,
        resolution: TaskResolution | None = None,
        judge_policy: str = "none",
        budget: CallBudget | None = None,
        evaluate: bool = True,
    ) -> SampleRunOutcome:
        """Execute routing, attempt plan, optional judge, and persistence.
        Sample-level failures are converted to a failed status with stable
        codes and never raise raw exceptions. An external budget (e.g. one
        already consumed by a TaskResolver) is shared across every attempt
        and judge call; otherwise a fresh per-sample budget is created.
        evaluate=False writes inference artifacts only and skips the
        deterministic evaluation artifact — the narrow switch used by
        count-image; dataset execution keeps the default True.
        执行路由、attempt plan、可选 judge 与持久化。样本级失败转换为携带
        稳定 code 的 failed 状态，绝不抛出原始异常。外部预算（如已被
        TaskResolver 消费的预算）贯穿所有 attempt 与 judge 调用；否则创建
        新的逐样本预算。evaluate=False 只写推理产物并跳过确定性评估产物——
        这是 count-image 使用的窄开关；数据集执行保持默认 True。"""

        self.artifact_writer.write_sample(sample_dir, sample)
        self.artifact_writer.write_running_status(sample_dir, _status(sample, "running"))
        started_at = time.perf_counter()
        base_task = resolution.task if resolution is not None else sample.task
        budget = budget if budget is not None else self.call_budget_factory.create_for_sample(base_task)
        visual_plan = None
        if self.visual_planning is not None:
            try:
                visual_plan = await self.visual_planning.plan_sample(
                    sample,
                    data_root=self.data_root,
                    artifact_dir=sample_dir,
                    budget=budget,
                )
            except VisualPlanError as error:
                # Frozen planner failure policy (14A2 gate 2): strict failure,
                # no retry, no legacy fallback, no sample.task rewrite. The
                # public message carries only the stable code.
                # 冻结规划器失败策略（14A2 门禁 2）：严格失败，无重试、无旧路径
                # 回退、不改 sample.task。公共消息只携带稳定 code。
                return self._finish_failed(
                    sample=sample,
                    sample_dir=sample_dir,
                    base_task=base_task,
                    resolution=resolution,
                    started_at=started_at,
                    attempts=[],
                    skipped=[],
                    error_code=str(error),
                    error_message="VisualPlanError",
                )
            except Exception as error:
                # Defensive bound: any unexpected gate failure is still a
                # stable-code sample failure, never a raw exception escape.
                # 防御性兜底：任何意外 gate 失败仍是稳定 code 的样本失败，
                # 绝不外泄原始异常。
                return self._finish_failed(
                    sample=sample,
                    sample_dir=sample_dir,
                    base_task=base_task,
                    resolution=resolution,
                    started_at=started_at,
                    attempts=[],
                    skipped=[],
                    error_code=type(error).__name__,
                )
            if visual_plan is not None:
                # Persist the validated plan (validated schema only, never a
                # raw model body). / 持久化已验证计划（只存已校验 schema，
                # 绝不存原始模型正文）。
                self.artifact_writer.write_visual_plan(sample_dir, visual_plan)
        attempts, skipped = self._build_attempt_plan(sample, resolution)
        if not attempts:
            return self._finish_failed(
                sample=sample,
                sample_dir=sample_dir,
                base_task=base_task,
                resolution=resolution,
                started_at=started_at,
                attempts=[],
                skipped=skipped,
                error_code="NO_EXECUTABLE_ATTEMPTS",
            )
        self.artifact_writer.write_routing(sample_dir, attempts[0].decision)
        context = AgentContext(
            artifact_dir=sample_dir,
            qwen_client=self.qwen_client,
            call_budget=budget,
            data_root=self.data_root,
            judge_client=(
                self.judge_service.judge_client
                if self.judge_service is not None
                else None
            ),
            visual_plan=visual_plan,
            visual_bindings=(
                self.visual_planning.bindings
                if self.visual_planning is not None
                else None
            ),
        )

        execution: AgentExecution | None = None
        executed_attempt: _Attempt | None = None
        executed_agent_attempt: _AgentAttempt | None = None
        failure_code: str | None = None
        primary_reason: str | None = None
        fallback_used = False
        for candidate_index, attempt in enumerate(attempts):
            (
                candidate_execution,
                attempt_failure,
                attempt_fallback,
                candidate_agent_attempt,
            ) = (
                await self._run_attempt(attempt, context)
            )
            if candidate_execution is None:
                failure_code = attempt_failure
                continue
            execution = candidate_execution
            executed_attempt = attempt
            executed_agent_attempt = candidate_agent_attempt
            fallback_used = (
                fallback_used or candidate_index > 0 or attempt_fallback
            )
            primary_reason = attempt_failure
            if sample_state_from_payload(execution.payload) != "failed":
                break
            failure_code = attempt_failure or "AGENT_STATUS_FAILED"
            execution = None
            continue

        if (
            execution is None
            or executed_attempt is None
            or executed_agent_attempt is None
        ):
            return self._finish_failed(
                sample=sample,
                sample_dir=sample_dir,
                base_task=base_task,
                resolution=resolution,
                started_at=started_at,
                attempts=attempts,
                skipped=skipped,
                error_code=failure_code or "ALL_ATTEMPTS_FAILED",
                fallback_used=fallback_used,
            )

        # The routing artifact must reflect the task that actually executed,
        # never the failed top candidate; the full candidate history stays in
        # the trace. / routing 产物必须反映实际执行的任务，绝不停留在失败的
        # top candidate；完整候选历史留在 trace。
        self.artifact_writer.write_routing(sample_dir, executed_attempt.decision)
        self.artifact_writer.write_execution(sample_dir, execution)
        evaluation = None
        if evaluate:
            evaluation = await self._persist_evaluation(
                executed_agent_attempt.sample,
                execution,
                sample_dir,
                budget=budget,
                judge_policy=judge_policy,
            )
        trace = _trace_payload(
            execution,
            executed_attempt,
            executed_task=executed_agent_attempt.task,
            resolution=resolution,
            resolved_task=base_task,
            inference_seconds=round(time.perf_counter() - started_at, 6),
            evaluation=evaluation,
            fallback_used=fallback_used,
            primary_reason=primary_reason,
            attempts=attempts,
            skipped=skipped,
            failure_code=None,
        )
        self.artifact_writer.write_trace(sample_dir, trace)
        # result_path is the sample-relative result artifact (the declared
        # result basename); machine absolute paths never enter the status.
        # result_path 是样本相对的结果产物（声明的结果 basename）；机器绝对
        # 路径绝不进入状态。
        final = _status(
            executed_agent_attempt.sample,
            sample_state_from_payload(execution.payload),
            result_path=Path(execution.result_filename),
        )
        self.artifact_writer.write_final_status(sample_dir, final)
        return SampleRunOutcome(
            execution=execution,
            status=final,
            routing=executed_attempt.decision,
            evaluation=evaluation,
            fallback_used=fallback_used,
        )

    def _build_attempt_plan(
        self,
        sample: UnifiedSample,
        resolution: TaskResolution | None,
    ) -> tuple[list[_Attempt], list[dict[str, str]]]:
        """Build the bounded, deduplicated attempt plan. High-confidence or
        absent resolutions run only the top task; low-confidence resolutions
        expand up to MAX_ATTEMPTS candidate tasks, route each, and deduplicate
        agent names stably. Incompatible candidate samples are skipped with a
        stable reason. 构建有界、去重的 attempt plan。高置信度或缺省 resolution
        只跑 top task；低置信度扩展最多 MAX_ATTEMPTS 个候选任务，逐个路由并
        稳定去重 Agent 名。不兼容候选样本以稳定 reason 跳过。"""

        base_task = resolution.task if resolution is not None else sample.task
        if resolution is not None and resolution.needs_candidate_fallback:
            candidate_tasks = list(dict.fromkeys([base_task, *resolution.candidate_tasks]))[:MAX_ATTEMPTS]
        else:
            candidate_tasks = [base_task]
        attempts: list[_Attempt] = []
        seen_agents: set[str] = set()
        skipped: list[dict[str, str]] = []
        for task in candidate_tasks:
            try:
                decision = self.router.route(task)
            except KeyError:
                skipped.append({"task": task, "reason": "UNROUTABLE_TASK"})
                continue
            candidate_sample = (
                sample
                if task == sample.task
                else _rebuild_sample_for_task(sample, task)
            )
            if candidate_sample is None:
                skipped.append({"task": task, "reason": "INCOMPATIBLE_SAMPLE"})
                continue
            agent_names = tuple(
                agent_name
                for agent_name in (decision.primary_agent, *decision.fallback_agents)
                if agent_name not in seen_agents
            )
            if not agent_names:
                skipped.append({"task": task, "reason": "AGENTS_DEDUPLICATED"})
                continue
            agent_attempts: list[_AgentAttempt] = []
            for agent_name in agent_names:
                execution_task = _ROUTING_FALLBACK_TASK_REMAP.get(
                    (task, agent_name),
                    task,
                )
                execution_sample = (
                    candidate_sample
                    if execution_task == task
                    else _rebuild_sample_for_task(candidate_sample, execution_task)
                )
                if execution_sample is None:
                    skipped.append(
                        {
                            "task": execution_task,
                            "reason": "INCOMPATIBLE_FALLBACK_SAMPLE",
                        }
                    )
                    continue
                agent_attempts.append(
                    _AgentAttempt(
                        agent_name=agent_name,
                        task=execution_task,
                        sample=execution_sample,
                    )
                )
            if not agent_attempts:
                continue
            seen_agents.update(item.agent_name for item in agent_attempts)
            attempts.append(
                _Attempt(
                    task=task,
                    agent_attempts=tuple(agent_attempts),
                    sample=candidate_sample,
                    decision=decision,
                )
            )
            if len(attempts) >= MAX_ATTEMPTS:
                break
        return attempts, skipped

    async def _run_attempt(
        self,
        attempt: _Attempt,
        context: AgentContext,
    ) -> tuple[AgentExecution | None, str | None, bool, _AgentAttempt | None]:
        """Execute one attempt's deduplicated agent list. A primary exception
        or a partial result under fallback_on_partial triggers the declared
        routing-fallback agents; all failures collapse into a stable code.
        执行一次尝试的去重 Agent 列表。primary 异常或在 fallback_on_partial
        下的 partial 结果触发声明的 routing-fallback Agent；所有失败收敛为
        稳定 code。"""

        last_failure: str | None = None
        fallback_used = False
        for index, agent_attempt in enumerate(attempt.agent_attempts):
            try:
                execution = await self.agent_registry.get(
                    agent_attempt.agent_name
                ).run(
                    agent_attempt.sample, context
                )
            except Exception as error:
                last_failure = _stable_error_code(error)
                continue
            if (
                index == 0
                and execution.status == "partial"
                and self.fallback_on_partial
                and len(attempt.agent_names) > 1
            ):
                last_failure = "PRIMARY_PARTIAL"
                fallback_used = True
                continue
            if index > 0:
                fallback_used = True
            return execution, last_failure, fallback_used, agent_attempt
        return None, last_failure or "ATTEMPT_FAILED", fallback_used, None

    async def _persist_evaluation(
        self,
        sample: UnifiedSample,
        execution: AgentExecution,
        sample_dir: Path,
        *,
        budget: CallBudget,
        judge_policy: str,
    ) -> object | None:
        """Persist the sample-level deterministic evaluation through the
        shared dispatch helper, then add the optional text-only judge for
        general_vqa only. The judge runs off the asyncio event loop and never
        fails the sample, never replaces the deterministic match, and never
        leaks raw exception text. 经共享分派 helper 持久化样本级确定性评估，
        然后仅对 general_vqa 追加可选仅文本 judge。judge 在 asyncio 事件循环
        之外运行，绝不让样本失败、绝不替换确定性匹配、绝不泄漏原始异常文本。"""

        task = sample.task
        evaluation, filename = build_deterministic_evaluation(
            sample=sample, execution_payload=execution.payload
        )
        if (
            evaluation_task_for_runtime_task(task) == "general_vqa"
            and self.judge_service is not None
        ):
            candidate_answer = str(getattr(execution.payload, "answer", ""))
            try:
                evaluation = await asyncio.to_thread(
                    self.judge_service.judge_vqa,
                    sample=sample,
                    candidate_answer=candidate_answer,
                    sample_dir=sample_dir,
                    judge_policy=judge_policy,
                    call_budget=budget,
                )
            except Exception as error:
                evaluation = {
                    "sample_id": sample.sample_id,
                    "judge_status": "failed",
                    "judge_error": type(error).__name__,
                }
        if filename is not None:
            self.artifact_writer.write_evaluation(
                sample_dir, evaluation, filename=filename
            )
        return evaluation

    def _finish_failed(
        self,
        *,
        sample: UnifiedSample,
        sample_dir: Path,
        base_task: str,
        resolution: TaskResolution | None,
        started_at: float,
        attempts: list[_Attempt],
        skipped: list[dict[str, str]],
        error_code: str,
        error_message: str | None = None,
        fallback_used: bool = False,
    ) -> SampleRunOutcome:
        """Write the failure trace and the final failed status with only
        stable codes, then return the failed outcome. 写入失败 trace 与只含
        稳定 code 的最终 failed 状态，并返回失败结果。"""

        trace = _trace_payload(
            None,
            None,
            executed_task=None,
            resolution=resolution,
            resolved_task=base_task,
            inference_seconds=round(time.perf_counter() - started_at, 6),
            evaluation=None,
            fallback_used=fallback_used,
            primary_reason=None,
            attempts=attempts,
            skipped=skipped,
            failure_code=error_code,
        )
        self.artifact_writer.write_trace(sample_dir, trace)
        final = _status(
            sample,
            "failed",
            error_code=error_code,
            error_message=error_message or error_code,
        )
        self.artifact_writer.write_final_status(sample_dir, final)
        routing = attempts[0].decision if attempts else None
        return SampleRunOutcome(
            execution=None,
            status=final,
            routing=routing,
            evaluation=None,
            fallback_used=fallback_used,
        )


def _rebuild_sample_for_task(
    sample: UnifiedSample,
    task: str,
) -> UnifiedSample | None:
    """Rebuild a sample for a candidate task: task replaced, normalization
    dropped, image roles rebuilt (t1/t2/context for change tasks, image/
    context otherwise). Returns None when the sample is incompatible with the
    candidate task (e.g. a change task on a single image); the original sample
    is never mutated. 为候选任务重建样本：替换 task、清空 normalization、重建
    图像角色（变化任务 t1/t2/context，其余 image/context）。样本与候选任务
    不兼容时（如单图上的变化任务）返回 None；原样本绝不修改。"""

    change_task = task in CHANGE_TASKS
    if change_task and len(sample.images) < 2:
        return None
    roles = ["t1", "t2"] if change_task else ["image"]
    roles.extend("context" for _ in range(len(sample.images) - len(roles)))
    try:
        data = sample.model_dump(mode="json")
        data["task"] = task
        data["normalization"] = None
        images = data["images"]
        for image_data, role in zip(images, roles):
            image_data["role"] = role
        data["images"] = images
        return UnifiedSample.model_validate(data)
    except (ValidationError, ValueError):
        return None


def _grounding_evaluation(
    sample: UnifiedSample,
    execution_payload: object,
) -> EvaluationRecord | None:
    """Axis-aligned grounding IoU, computed only when prediction and ground
    truth agree on the normalized_0_999_top_left frame with 4-value xyxy
    boxes. source_pixels_top_left, unlabelled frames, 8-value polygons, and
    non-4 prediction boxes all yield None — a fake IoU is never fabricated,
    official oriented metrics stay upstream. 轴对齐 grounding IoU，仅在预测与
    真值同为 normalized_0_999_top_left 且均为 4-value xyxy 时计算。
    source_pixels_top_left、未声明坐标系、8-value 多边形与非 4-value 预测框
    一律返回 None——绝不伪造 IoU，官方 oriented 指标留在上游评测器。"""

    ground_truth = sample.ground_truth
    gt_boxes = ground_truth.boxes if ground_truth is not None else []
    prediction_boxes = getattr(execution_payload, "boxes", None) or []
    if not prediction_boxes or not gt_boxes:
        return None
    # Frame contract: the Agent contract always emits normalized_0_999_top_left
    # boxes; the ground truth must declare the identical frame explicitly.
    # 坐标系契约：Agent 契约恒输出 normalized_0_999_top_left 框；真值必须
    # 显式声明相同坐标系。
    if ground_truth is None or ground_truth.coordinate_frame != _GROUNDING_PREDICTION_FRAME:
        return None
    prediction = prediction_boxes[0]
    truth = gt_boxes[0]
    if not isinstance(prediction, (list, tuple)) or len(prediction) != 4:
        return None
    if not isinstance(truth, (list, tuple)) or len(truth) != 4:
        return None
    try:
        iou = box_iou(prediction, truth)
    except (TypeError, IndexError, ValueError):
        return None
    return EvaluationRecord(
        sample_id=sample.sample_id,
        task="grounding",
        deterministic_metrics=grounding_deterministic_metrics(iou),
        judge_status="not_requested",
    )


def _caption_evaluation(
    sample: UnifiedSample,
    execution_payload: object,
) -> EvaluationRecord | None:
    """Per-sample caption record: candidate plus references. Corpus-level
    BLEU/METEOR/ROUGE/CIDEr stays with the report layer; no references means
    no record is fabricated. 逐样本 caption 记录：候选与参考答案。语料级
    BLEU/METEOR/ROUGE/CIDEr 留在报告层；无参考答案时不伪造记录。"""

    ground_truth = sample.ground_truth
    references = ground_truth.answers if ground_truth is not None else []
    if not references:
        return None
    return EvaluationRecord(
        sample_id=sample.sample_id,
        task="caption",
        deterministic_metrics=CaptionDeterministicMetrics(
            candidate=str(getattr(execution_payload, "answer", "")),
            references=list(references),
        ),
        judge_status="not_requested",
    )


def _trace_payload(
    execution: AgentExecution | None,
    attempt: _Attempt | None,
    *,
    executed_task: str | None,
    resolution: TaskResolution | None,
    resolved_task: str,
    inference_seconds: float,
    evaluation: object | None,
    fallback_used: bool,
    primary_reason: str | None,
    attempts: list[_Attempt],
    skipped: list[dict[str, str]],
    failure_code: str | None,
) -> dict[str, Any]:
    """Compose the auditable trace from already-decided runtime facts; every
    value is JSON-safe and carries no raw paths, secrets, or exception text.
    task_type is fixed to the resolved task; resolved_task and execution_task
    are explicit so candidate fallback (resolved != executed) stays auditable.
    根据已确定的运行时事实组装可审计 trace；所有值 JSON 安全且不含原始路径、
    密钥或异常文本。task_type 固定为解析任务；resolved_task 与 execution_task
    显式记录，使候选兜底（解析 != 执行）保持可审计。"""

    judge_status = getattr(evaluation, "judge_status", None)
    if isinstance(evaluation, dict):
        judge_status = evaluation.get("judge_status")
    trace: dict[str, Any] = dict(execution.trace) if execution is not None else {}
    trace.update(
        {
            "router_used": True,
            "task_type": resolved_task,
            "resolved_task": resolved_task,
            "qwen_backend": "transformers",
            "inference_seconds": inference_seconds,
            "execution_task": executed_task,
            "execution_agent": execution.agent_name if execution is not None else None,
            "resolution_source": resolution.source if resolution is not None else "dataset_task",
            "low_confidence": (
                resolution.needs_candidate_fallback if resolution is not None else False
            ),
            "candidate_tasks": [item.task for item in attempts],
            "attempt_agents": [list(item.agent_names) for item in attempts],
            "skipped_candidates": skipped,
            "execution_mode": (
                attempt.decision.execution_mode if attempt is not None else None
            ),
            "fallback_agents": (
                attempt.decision.fallback_agents if attempt is not None else []
            ),
            "fallback_used": fallback_used,
            "judge_status": judge_status or "not_requested",
        }
    )
    if primary_reason is not None:
        trace["primary_reason"] = primary_reason
    if attempt is not None and executed_task is not None and executed_task != attempt.task:
        trace["fallback_from_task"] = attempt.task
    if failure_code is not None:
        trace["failure_code"] = failure_code
    return trace
