"""Single-sample execution: route → dispatch → fallback → judge → persist.
单样本执行：路由 → 分派 → fallback → 审核 → 持久化。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from spacers_agent.agents.base import AgentContext, AgentExecution
from spacers_agent.agents.registry import AgentRegistry
from spacers_agent.clients.base import VisionLanguageClient
from spacers_agent.routing import CallBudget, TaskRouter
from spacers_agent.routing.schemas import normalize_agent_name
from spacers_agent.schemas import UnifiedSample
from spacers_agent.settings import AppSettings
from spacers_agent.workflow import atomic_write_json


class SampleRunner:
    """Execute one sample: route → agent.run → judge → persist.
    执行一条样本：路由 → agent.run → 审核 → 持久化。
    """

    def __init__(
        self,
        settings: AppSettings,
        agent_registry: AgentRegistry,
        client: VisionLanguageClient,
        prompts: dict[str, str],
        *,
        judge_service: Any | None = None,
        judge_policy: str = "none",
        fallback_on_partial: bool = False,
    ) -> None:
        self.settings = settings
        self.agent_registry = agent_registry
        self.client = client
        self.prompts = prompts
        self.judge_service = judge_service
        self.judge_policy = judge_policy
        self.fallback_on_partial = fallback_on_partial

    async def run_one(self, sample: UnifiedSample, sample_dir: Path) -> AgentExecution:
        """Full single-sample pipeline. / 完整单样本管线。"""

        atomic_write_json(sample_dir / "sample.json", sample.model_dump(mode="json"))
        started_at = time.perf_counter()

        budget = CallBudget(max_qwen_calls=50, max_deepseek_calls=10)

        high_resolution = any(
            (ref.width or 0) * (ref.height or 0) > self.settings.counting.max_pixels_without_tiling
            for ref in sample.images
        )

        # ── route / 路由 ──────────────────────────────────────────────
        router = TaskRouter()
        decision = await router.route_sample(
            sample, budget=budget, high_resolution=high_resolution,
            artifact_dir=sample_dir,
        )
        atomic_write_json(sample_dir / "routing_decision.json", decision.model_dump(mode="json"))

        # ── primary agent execution / 主 Agent 执行 ──────────────────
        primary_name = normalize_agent_name(decision.primary_agent)
        ctx = AgentContext(
            artifact_dir=sample_dir, settings=self.settings,
            qwen_client=self.client, call_budget=budget, prompts=self.prompts,
        )

        try:
            agent = self.agent_registry.get(primary_name)
            execution = await agent.run(sample, ctx)
        except Exception as error:
            # Primary failed — try fallback / 主 Agent 失败 — 尝试 fallback
            if decision.execution_mode == "fallback" and decision.fallback_agents:
                return await self._run_fallback(sample, decision, ctx, sample_dir, str(error))
            raise

        # ── persist payload / 持久化 payload ─────────────────────────
        _write_payload(sample_dir, execution)

        # ── fallback on partial (if configured) / partial 时 fallback ──
        if self.fallback_on_partial and execution.status == "partial" and decision.fallback_agents:
            return await self._run_fallback(sample, decision, ctx, sample_dir, "primary_partial")

        # ── VQA judge / VQA 审核 ──────────────────────────────────────
        if sample.task == "general_vqa" and self.judge_service and self.judge_policy != "none":
            await self._judge_vqa(sample, execution, sample_dir)

        # ── trace / trace ────────────────────────────────────────────
        inference_seconds = round(time.perf_counter() - started_at, 6)
        _persist_trace(sample_dir, execution, decision, inference_seconds, self.settings)

        return execution

    # ── fallback / fallback ─────────────────────────────────────────────

    async def _run_fallback(
        self, sample: UnifiedSample, decision, ctx: AgentContext,
        sample_dir: Path, primary_reason: str,
    ) -> AgentExecution:
        """Execute fallback agents sequentially. / 顺序执行 fallback Agent。"""
        last_error = primary_reason
        for fallback_name in decision.fallback_agents:
            try:
                agent_name = normalize_agent_name(fallback_name)
                agent = self.agent_registry.get(agent_name)
                execution = await agent.run(sample, ctx)
                _write_payload(sample_dir, execution)
                execution.trace["fallback_used"] = True
                execution.trace["primary_reason"] = primary_reason
                return execution
            except Exception as error:
                last_error = str(error)
                continue

        raise RuntimeError(f"All agents failed (primary={decision.primary_agent}, fallback={decision.fallback_agents}): {last_error}")

    # ── VQA judge / VQA 审核 ──────────────────────────────────────────

    async def _judge_vqa(self, sample: UnifiedSample, execution: AgentExecution, sample_dir: Path) -> None:
        """Run text-only VQA judge if service is available. / 如服务可用则运行仅文本 VQA 审核。"""
        try:
            evaluation = await self.judge_service.judge_vqa(
                sample=sample,
                candidate_answer=str(execution.payload.answer),  # type: ignore[union-attr]
                sample_dir=sample_dir,
            )
            atomic_write_json(sample_dir / "vqa_evaluation.json", evaluation.model_dump(mode="json"))
        except Exception:
            pass  # Judge is optional — never fail a sample for judge errors


# ── helpers / 辅助函数 ───────────────────────────────────────────────────


def _write_payload(sample_dir: Path, execution: AgentExecution) -> None:
    """Persist agent payload to the expected artifact filename. / 将 Agent payload 持久化为预期产物文件名。"""
    data = execution.payload.model_dump(mode="json")  # type: ignore[union-attr]
    atomic_write_json(sample_dir / execution.result_filename, data)


def _persist_trace(
    sample_dir: Path, execution: AgentExecution, decision, inference_seconds: float, settings: AppSettings,
) -> None:
    """Write auditable agent_trace.json. / 写入可审计 agent_trace.json。"""
    trace = dict(execution.trace)
    trace.update({
        "router_used": True,
        "task_type": decision.task,
        "qwen_backend": getattr(settings.models.qwen, "backend", "unknown"),
        "inference_seconds": inference_seconds,
        "execution_task": decision.task,
        "routing_source": decision.router_source,
        "execution_mode": decision.execution_mode,
        "fallback_agents": decision.fallback_agents,
    })
    atomic_write_json(sample_dir / "agent_trace.json", trace)
