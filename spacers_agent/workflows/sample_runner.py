"""Single-sample execution for the injected Agent runtime.
使用注入依赖的 Agent 运行时单样本执行。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from spacers_agent.agents.base import AgentExecution, AgentPayload, AgentContext
from spacers_agent.agents.registry import AgentRegistry
from models.base import VisionLanguageClient
from spacers_agent.prompt_catalog import PromptCatalog
from spacers_agent.routing import CallBudget, CallBudgetFactory, TaskRouter
from spacers_agent.routing.schemas import RoutingDecision, normalize_agent_name
from spacers_agent.schemas import CountingResult, SampleRunStatus, UnifiedSample
from spacers_agent.settings import AppSettings
from spacers_agent.workflows.artifact_writer import ArtifactWriter
from spacers_agent.workflows.judge_service import JudgeService


SampleState = Literal["succeeded", "partial", "failed"]


@dataclass(frozen=True)
class DatasetRunOptions:
    """Typed dataset run options for resume and fresh runs.
    用于 resume 和新运行的定型数据集运行选项。
    None values do not participate in numeric comparisons.
    None 值不参与数值比较。
    """

    dataset: str
    root: Path
    split: str
    tasks: tuple[str, ...]
    run_id: str | None = None
    resume: bool = False
    limit: int | None = None
    start_index: int = 0
    shard_index: int = 0
    shard_count: int = 1
    sample_concurrency: int = 1
    sample_ids: set[str] | None = None
    evaluate: bool = False
    judge_policy: str = "none"
    fail_fast: bool = False


@dataclass(frozen=True)
class SampleRunOutcome:
    """All observable outputs from one SampleRunner invocation.
    一次 SampleRunner 调用产生的全部可观察输出。
    """

    execution: AgentExecution | None
    status: SampleRunStatus
    routing: RoutingDecision | None
    evaluation: object | None
    fallback_used: bool


def sample_state_from_payload(payload: AgentPayload) -> SampleState:
    """Map persisted payload status to the only allowed sample-state mapping.
    使用唯一允许的映射将持久化载荷状态转换为样本状态。
    """

    if isinstance(payload, CountingResult):
        return {
            "completed": "succeeded",
            "completed_with_warnings": "succeeded",
            "partial": "partial",
            "failed": "failed",
        }[payload.status]
    return {
        "completed": "succeeded",
        "partial": "partial",
        "failed": "failed",
    }[payload.status]


class SampleRunner:
    """Route and execute one sample using only injected runtime components.
    仅使用注入的运行时组件路由并执行一条样本。
    """

    def __init__(
        self,
        settings: AppSettings,
        agent_registry: AgentRegistry,
        qwen_client: VisionLanguageClient,
        prompt_catalog: PromptCatalog,
        *,
        router: TaskRouter,
        judge_service: JudgeService,
        artifact_writer: ArtifactWriter,
        call_budget_factory: CallBudgetFactory,
        fallback_on_partial: bool = False,
    ) -> None:
        self.settings = settings
        self.agent_registry = agent_registry
        self.qwen_client = qwen_client
        self.prompt_catalog = prompt_catalog
        self.router = router
        self.judge_service = judge_service
        self.artifact_writer = artifact_writer
        self.call_budget_factory = call_budget_factory
        self.fallback_on_partial = fallback_on_partial

    async def run_one(
        self,
        sample: UnifiedSample,
        sample_dir: Path,
        *,
        judge_policy: str = "none",
    ) -> SampleRunOutcome:
        """Execute route, Agent, optional fallback, Judge, and persistence.
        执行路由、Agent、可选兜底、审核与持久化。
        """

        self.artifact_writer.write_sample(sample_dir, sample)
        running = _status(sample, "running")
        self.artifact_writer.write_running_status(sample_dir, running)
        started_at = time.perf_counter()
        budget = self.call_budget_factory.create_for_sample(sample.task)
        high_resolution = any(
            (image.width or 0) * (image.height or 0) > self.settings.counting.max_pixels_without_tiling
            for image in sample.images
        )
        decision = await self.router.route_sample(
            sample,
            budget=budget,
            high_resolution=high_resolution,
            artifact_dir=sample_dir,
        )
        self.artifact_writer.write_routing(sample_dir, decision)
        context = AgentContext(
            artifact_dir=sample_dir,
            settings=self.settings,
            qwen_client=self.qwen_client,
            call_budget=budget,
            prompt_catalog=self.prompt_catalog,
            judge_client=self.judge_service.judge_client,
        )

        fallback_used = False
        primary_reason: str | None = None
        try:
            execution = await self.agent_registry.get(
                normalize_agent_name(decision.primary_agent)
            ).run(sample, context)
        except Exception as error:
            if decision.execution_mode != "fallback" or not decision.fallback_agents:
                raise
            primary_reason = f"{type(error).__name__}: {error}"
            execution = await self._run_fallback(sample, decision, context, primary_reason)
            fallback_used = True

        if self.fallback_on_partial and execution.status == "partial" and decision.fallback_agents:
            primary_reason = "primary_partial"
            execution = await self._run_fallback(sample, decision, context, primary_reason)
            fallback_used = True

        result_path = self.artifact_writer.write_execution(sample_dir, execution)
        evaluation = await self._judge_vqa(
            sample,
            execution,
            sample_dir,
            budget=budget,
            judge_policy=judge_policy,
        )
        trace = _trace_payload(
            execution,
            decision,
            inference_seconds=round(time.perf_counter() - started_at, 6),
            settings=self.settings,
            evaluation=evaluation,
            fallback_used=fallback_used,
            primary_reason=primary_reason,
        )
        self.artifact_writer.write_trace(sample_dir, trace)
        final = _status(
            sample,
            sample_state_from_payload(execution.payload),
            result_path=result_path,
        )
        self.artifact_writer.write_final_status(sample_dir, final)
        return SampleRunOutcome(
            execution=execution,
            status=final,
            routing=decision,
            evaluation=evaluation,
            fallback_used=fallback_used,
        )

    async def _run_fallback(
        self,
        sample: UnifiedSample,
        decision: RoutingDecision,
        context: AgentContext,
        primary_reason: str,
    ) -> AgentExecution:
        """Execute declared fallback Agents sequentially and expose total failure.
        顺序执行声明的兜底 Agent，并显式暴露全部失败。
        """

        last_error = primary_reason
        for fallback_name in decision.fallback_agents:
            try:
                return await self.agent_registry.get(
                    normalize_agent_name(fallback_name)
                ).run(sample, context)
            except Exception as error:
                last_error = f"{type(error).__name__}: {error}"
        raise RuntimeError(
            "All agents failed "
            f"(primary={decision.primary_agent}, fallback={decision.fallback_agents}): {last_error}"
        )

    async def _judge_vqa(
        self,
        sample: UnifiedSample,
        execution: AgentExecution,
        sample_dir: Path,
        *,
        budget: CallBudget,
        judge_policy: str,
    ) -> object | None:
        """Persist a text-only VQA evaluation while retaining Agent output on errors.
        持久化纯文本 VQA 评测，并在审核错误时保留 Agent 输出。
        """

        if sample.task != "general_vqa":
            return None
        candidate_answer = (
            str(execution.payload.final_count)
            if isinstance(execution.payload, CountingResult)
            else str(execution.payload.answer)
        )
        try:
            evaluation = await self.judge_service.judge_vqa(
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
                "judge_error": f"{type(error).__name__}: {error}",
            }
        self.artifact_writer.write_evaluation(
            sample_dir,
            evaluation,
            filename="vqa_evaluation.json",
        )
        return evaluation


def _status(
    sample: UnifiedSample,
    state: str,
    *,
    result_path: Path | None = None,
    error: Exception | None = None,
) -> SampleRunStatus:
    """Build one timestamped status without inferring business success.
    构建带时间戳的状态，不推断业务成功。
    """

    return SampleRunStatus(
        sample_id=sample.sample_id,
        task=sample.task,
        state=state,
        error_code=type(error).__name__ if error is not None else None,
        error_message=str(error) if error is not None else None,
        result_path=result_path,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


def failed_sample_status(sample: UnifiedSample, error: Exception) -> SampleRunStatus:
    """Create a visible failed status for an execution exception.
    为执行异常创建可见的失败状态。
    """

    return _status(sample, "failed", error=error)


def _trace_payload(
    execution: AgentExecution,
    decision: RoutingDecision,
    *,
    inference_seconds: float,
    settings: AppSettings,
    evaluation: object | None,
    fallback_used: bool,
    primary_reason: str | None,
) -> dict[str, object]:
    """Compose the auditable trace from already-decided runtime facts.
    根据已确定的运行时事实组装可审计轨迹。
    """

    trace: dict[str, object] = dict(execution.trace)
    judge_status = getattr(evaluation, "judge_status", None)
    if isinstance(evaluation, dict):
        judge_status = evaluation.get("judge_status")
    trace.update(
        {
            "router_used": True,
            "task_type": decision.task,
            "qwen_backend": "transformers",
            "inference_seconds": inference_seconds,
            "execution_task": decision.task,
            "routing_source": decision.router_source,
            "execution_mode": decision.execution_mode,
            "fallback_agents": decision.fallback_agents,
            "fallback_used": fallback_used,
            "judge_status": judge_status or "not_requested",
        }
    )
    if primary_reason is not None:
        trace["primary_reason"] = primary_reason
    return trace
