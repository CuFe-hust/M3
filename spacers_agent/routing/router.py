"""TaskRouter — deterministic and text-based routing with unified entry point.
TaskRouter — 确定性 + 基于文本的路由，统一入口。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from spacers_agent.clients.base import RequestMeta, VisionLanguageClient, build_request_hash
from spacers_agent.counting import PointCountingOrchestrator
from spacers_agent.routing.budget import CallBudget
from spacers_agent.routing.policies import ROUTES, needs_tiling
from spacers_agent.routing.schemas import (
    AgentName,
    RoutableTask,
    RoutingDecision,
    normalize_agent_name,
)
from spacers_agent.schemas import CountTargetSpec, CountingResult
from spacers_agent.vqa_geometry import execution_task_for_vrsbench


class TaskRouter:
    """Route samples to agents via deterministic rules or text-based model.
    通过确定性规则或基于文本的模型将样本路由到 Agent。
    """

    def __init__(self, router_client: VisionLanguageClient | None = None, *, router_prompt: str = "") -> None:
        self.router_client = router_client
        self.router_prompt = router_prompt

    # ── unified entry / 统一入口 ────────────────────────────────────────

    async def route_sample(
        self,
        sample: Any,  # UnifiedSample
        *,
        budget: CallBudget,
        high_resolution: bool = False,
        artifact_dir: Path | None = None,
    ) -> RoutingDecision:
        """Route one sample: VRSBench → known → unknown → rule fallback.
        路由单条样本：VRSBench → 已知 → 未知 → 规则兜底。
        """

        question_type = str(getattr(sample, "metadata", {}).get("question_type", ""))
        dataset = str(getattr(sample, "dataset", ""))
        task = str(getattr(sample, "task", ""))

        # 1. VRSBench VQA — conservative semantic routing / VRSBench VQA 保守语义路由
        if dataset == "VRSBench" and task == "general_vqa":
            return self.route_vrsbench_vqa(question_type, question=sample.question, high_resolution=high_resolution)

        # 2. Known task — deterministic route / 已知 task — 确定性路由
        if task in ROUTES:
            return self.route_known(task, high_resolution=high_resolution)  # type: ignore[arg-type]

        # 3. Unknown task — model-based routing / 未知 task — 基于模型的路由
        if self.router_client is not None:
            try:
                return await self.route_unknown(
                    sample.question,
                    budget=budget,
                    sample_id=getattr(sample, "sample_id", "unknown"),
                    artifact_dir=artifact_dir,
                )
            except Exception:
                pass  # fall through to rule fallback

        # 4. Rule fallback — conservative general_vqa / 规则兜底 — 保守 general_vqa
        return self._rule_fallback(high_resolution=high_resolution)

    # ── known task / 已知 task ──────────────────────────────────────────

    def route_known(self, task: RoutableTask | str, *, high_resolution: bool = False) -> RoutingDecision:
        """Route by task name only; no model call. / 仅按 task 名路由；不调用模型。"""
        task_str = str(task)
        agents = ROUTES.get(task_str, ("general_vqa_agent",))  # type: ignore[arg-type]
        primary = agents[0]
        fallback = list(agents[1:]) if agents[1:] else []

        return RoutingDecision(
            task=task_str,  # type: ignore[arg-type]
            primary_agent=primary,
            fallback_agents=fallback,
            execution_mode="fallback" if fallback else "single",
            requires_tiling=needs_tiling(task_str),
            reason_codes=[f"task_{task_str}"] + (["high_resolution"] if high_resolution else []),
            router_source="dataset_task",
        )

    # ── VRSBench / VRSBench ─────────────────────────────────────────────

    def route_vrsbench_vqa(
        self, question_type: str, *, question: str | None = None, high_resolution: bool = False
    ) -> RoutingDecision:
        """Conservative VRSBench routing from question semantics. / 根据问题语义保守路由 VRSBench。"""
        task = execution_task_for_vrsbench(question_type, question)
        decision = self.route_known(task, high_resolution=high_resolution)
        normalized_type = "_".join(question_type.casefold().split()) or "unspecified"
        reason_codes = list(decision.reason_codes) + [
            f"vrsbench_type_{normalized_type}",
            f"vrsbench_semantic_{task}" if question else "vrsbench_conservative_fallback",
        ]
        return decision.model_copy(update={
            "reason_codes": reason_codes,
            "router_source": "vrsbench_semantic_rule",
        })

    # ── unknown task / 未知 task ────────────────────────────────────────

    async def route_unknown(
        self, question: str, *, budget: CallBudget, sample_id: str, artifact_dir: Path | None = None,
    ) -> RoutingDecision:
        """Text-only model-based routing. / 仅文本基于模型的路由。"""
        if self.router_client is None:
            raise ValueError("unknown tasks require an injected router client")
        budget.reserve_qwen()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.router_prompt},
            {"role": "user", "content": question},
        ]
        request_hash = build_request_hash(
            model="router", generation={"temperature": 0.0},
            prompt_version="router-v1", messages=messages, image_sha256=None,
        )
        return await self.router_client.complete_json(
            messages=messages, response_model=RoutingDecision,
            request_meta=RequestMeta(
                request_id=f"{sample_id}:router", request_hash=request_hash,
                prompt_version="router-v1", sample_id=sample_id, artifact_dir=artifact_dir,
            ),
        )

    # ── rule fallback / 规则兜底 ────────────────────────────────────────

    def _rule_fallback(self, *, high_resolution: bool = False) -> RoutingDecision:
        """Conservative fallback when no route is available. / 无可用路由时的保守兜底。"""
        return RoutingDecision(
            task="general_vqa", primary_agent="general_vqa_agent",
            execution_mode="single", requires_tiling=False,
            reason_codes=["rule_fallback"] + (["high_resolution"] if high_resolution else []),
            router_source="rule_fallback",
        )


# ── legacy CountingExpert (preserved for backward compat) / 保留 CountingExpert 向后兼容 ──

class CountingExpertAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str
    complete: bool
    counting_result: CountingResult


class CountingExpert:
    def __init__(self, pipeline: PointCountingOrchestrator) -> None:
        self.pipeline = pipeline

    async def answer(self, image: Image.Image, *, sample_id: str, question: str,
                     target: CountTargetSpec, minimum_scan_depth: int = 0,
                     review_empty: bool = False, upscale_max_side: int | None = None,
                     ) -> CountingExpertAnswer:
        result = await self.pipeline.count_image(
            image, sample_id=sample_id, question=question, target=target,
            minimum_scan_depth=minimum_scan_depth, review_empty=review_empty,
            upscale_max_side=upscale_max_side,
        )
        total = len(result.succeeded_tiles) + len(result.failed_tiles)
        if result.status in {"partial", "failed"}:
            return CountingExpertAnswer(
                answer=f"Completed {len(result.succeeded_tiles)}/{total} tiles and confirmed {result.final_count} instances; the result is incomplete.",
                complete=False, counting_result=result,
            )
        return CountingExpertAnswer(
            answer=f"Based on {result.final_count} accepted global instance points, the image contains {result.final_count} {target.canonical_label}(s).",
            complete=True, counting_result=result,
        )


def attach_qwen_budget(pipeline: PointCountingOrchestrator, budget: CallBudget) -> PointCountingOrchestrator:
    pipeline.before_qwen_call = budget.reserve_qwen
    return pipeline
