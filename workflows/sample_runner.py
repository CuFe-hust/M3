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
from evaluation.metrics.vqa import merge_vqa_evaluation
from models.base import VisionLanguageClient
from routing.router import TaskRouter
from routing.schema import RoutingDecision, TaskResolution
from workflows.artifact_writer import ArtifactWriter
from workflows.call_budget import CallBudget, CallBudgetFactory
from workflows.judge_service import JudgeService
from workflows.schema import SampleRunOutcome, SampleRunStatus

# Hard bound on the candidate attempt plan: never run all agents.
# 候选 attempt plan 的硬上限：绝不跑所有 Agent。
MAX_ATTEMPTS = 3

_VQA_EVALUATION_FILENAME = "vqa_evaluation.json"
_COUNTING_EVALUATION_FILENAME = "counting_evaluation.json"


@dataclass(frozen=True)
class _Attempt:
    """One executable attempt: a task, its deduplicated agent list, the sample
    rebuilt for that task, and the routing decision. 一次可执行尝试：任务、
    去重后的 Agent 列表、为该任务重建的样本与路由决策。"""

    task: str
    agent_names: tuple[str, ...]
    sample: UnifiedSample
    decision: RoutingDecision


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
    ) -> None:
        self.agent_registry = registry
        self.router = router
        self.qwen_client = qwen_client
        self.artifact_writer = artifact_writer
        self.call_budget_factory = call_budget_factory
        self.judge_service = judge_service
        self.fallback_on_partial = fallback_on_partial
        self.data_root = data_root

    async def run_one(
        self,
        sample: UnifiedSample,
        sample_dir: Path,
        *,
        resolution: TaskResolution | None = None,
        judge_policy: str = "none",
        budget: CallBudget | None = None,
    ) -> SampleRunOutcome:
        """Execute routing, attempt plan, optional judge, and persistence.
        Sample-level failures are converted to a failed status with stable
        codes and never raise raw exceptions. An external budget (e.g. one
        already consumed by a TaskResolver) is shared across every attempt
        and judge call; otherwise a fresh per-sample budget is created.
        执行路由、attempt plan、可选 judge 与持久化。样本级失败转换为携带
        稳定 code 的 failed 状态，绝不抛出原始异常。外部预算（如已被
        TaskResolver 消费的预算）贯穿所有 attempt 与 judge 调用；否则创建
        新的逐样本预算。"""

        self.artifact_writer.write_sample(sample_dir, sample)
        self.artifact_writer.write_running_status(sample_dir, _status(sample, "running"))
        started_at = time.perf_counter()
        base_task = resolution.task if resolution is not None else sample.task
        budget = budget if budget is not None else self.call_budget_factory.create_for_sample(base_task)
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
        )

        execution: AgentExecution | None = None
        executed_attempt: _Attempt | None = None
        failure_code: str | None = None
        primary_reason: str | None = None
        fallback_used = False
        for attempt in attempts:
            candidate_execution, attempt_failure, attempt_fallback = (
                await self._run_attempt(attempt, context)
            )
            if candidate_execution is None:
                failure_code = attempt_failure
                continue
            execution = candidate_execution
            executed_attempt = attempt
            fallback_used = attempt_fallback
            primary_reason = attempt_failure
            if sample_state_from_payload(execution.payload) != "failed":
                break
            failure_code = attempt_failure or "AGENT_STATUS_FAILED"
            execution = None
            continue

        if execution is None or executed_attempt is None:
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

        result_path = self.artifact_writer.write_execution(sample_dir, execution)
        evaluation = await self._persist_evaluation(
            executed_attempt.sample,
            execution,
            sample_dir,
            budget=budget,
            judge_policy=judge_policy,
        )
        trace = _trace_payload(
            execution,
            executed_attempt,
            resolution=resolution,
            inference_seconds=round(time.perf_counter() - started_at, 6),
            evaluation=evaluation,
            fallback_used=fallback_used,
            primary_reason=primary_reason,
            attempts=attempts,
            skipped=skipped,
            failure_code=None,
        )
        self.artifact_writer.write_trace(sample_dir, trace)
        final = _status(
            executed_attempt.sample,
            sample_state_from_payload(execution.payload),
            result_path=result_path,
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
            agents = tuple(
                agent_name
                for agent_name in (decision.primary_agent, *decision.fallback_agents)
                if agent_name not in seen_agents
            )
            if not agents:
                skipped.append({"task": task, "reason": "AGENTS_DEDUPLICATED"})
                continue
            seen_agents.update(agents)
            attempts.append(
                _Attempt(
                    task=task,
                    agent_names=agents,
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
    ) -> tuple[AgentExecution | None, str | None, bool]:
        """Execute one attempt's deduplicated agent list. A primary exception
        or a partial result under fallback_on_partial triggers the declared
        routing-fallback agents; all failures collapse into a stable code.
        执行一次尝试的去重 Agent 列表。primary 异常或在 fallback_on_partial
        下的 partial 结果触发声明的 routing-fallback Agent；所有失败收敛为
        稳定 code。"""

        last_failure: str | None = None
        fallback_used = False
        for index, agent_name in enumerate(attempt.agent_names):
            try:
                execution = await self.agent_registry.get(agent_name).run(
                    attempt.sample, context
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
            return execution, last_failure, fallback_used
        return None, last_failure or "ATTEMPT_FAILED", fallback_used

    async def _persist_evaluation(
        self,
        sample: UnifiedSample,
        execution: AgentExecution,
        sample_dir: Path,
        *,
        budget: CallBudget,
        judge_policy: str,
    ) -> object | None:
        """Persist the sample-level deterministic evaluation; general_vqa adds
        the optional text-only judge. Judge failures are recorded as stable
        judge_status=failed and never replace the deterministic match, never
        fail the sample, and never leak raw exception text.
        持久化样本级确定性评估；general_vqa 追加可选仅文本 judge。Judge 失败
        记录为稳定 judge_status=failed，绝不替换确定性匹配、绝不让样本失败、
        绝不泄漏原始异常文本。"""

        if sample.task == "general_vqa":
            candidate_answer = (
                str(execution.payload.final_count)
                if isinstance(execution.payload, CountingResult)
                else str(execution.payload.answer)
            )
            references = (
                list(sample.ground_truth.answers)
                if sample.ground_truth is not None
                else []
            )
            if self.judge_service is not None:
                try:
                    evaluation = self.judge_service.judge_vqa(
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
            else:
                evaluation = merge_vqa_evaluation(
                    sample_id=sample.sample_id,
                    question=sample.question,
                    reference_answers=references,
                    candidate_answer=candidate_answer,
                )
            self.artifact_writer.write_evaluation(
                sample_dir, evaluation, filename=_VQA_EVALUATION_FILENAME
            )
            return evaluation
        if sample.task == "counting":
            evaluation = merge_count_evaluation(
                sample_id=sample.sample_id,
                counting=execution.payload,
                ground_truth=sample.ground_truth,
            )
            self.artifact_writer.write_evaluation(
                sample_dir, evaluation, filename=_COUNTING_EVALUATION_FILENAME
            )
            return evaluation
        return None

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
        fallback_used: bool = False,
    ) -> SampleRunOutcome:
        """Write the failure trace and the final failed status with only
        stable codes, then return the failed outcome. 写入失败 trace 与只含
        稳定 code 的最终 failed 状态，并返回失败结果。"""

        trace = _trace_payload(
            None,
            None,
            resolution=resolution,
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
            error_message=error_code,
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


def _trace_payload(
    execution: AgentExecution | None,
    attempt: _Attempt | None,
    *,
    resolution: TaskResolution | None,
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
    根据已确定的运行时事实组装可审计 trace；所有值 JSON 安全且不含原始路径、
    密钥或异常文本。"""

    judge_status = getattr(evaluation, "judge_status", None)
    if isinstance(evaluation, dict):
        judge_status = evaluation.get("judge_status")
    trace: dict[str, Any] = dict(execution.trace) if execution is not None else {}
    trace.update(
        {
            "router_used": True,
            "task_type": resolution.task if resolution is not None else (
                attempt.task if attempt is not None else "unknown"
            ),
            "qwen_backend": "transformers",
            "inference_seconds": inference_seconds,
            "execution_task": attempt.task if attempt is not None else None,
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
    if failure_code is not None:
        trace["failure_code"] = failure_code
    return trace
