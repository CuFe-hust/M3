"""Rule-first sparse routing, bounded call budgets, and expert-facing results.
规则优先的稀疏路由、受限调用预算和面向专家的结果封装。
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from spacers_agent.clients.base import RequestMeta, VisionLanguageClient, build_request_hash
from spacers_agent.counting import PointCountingOrchestrator
from spacers_agent.schemas import CountTargetSpec, CountingResult


ExpertName = Literal[
    "counting_expert",
    "change_expert",
    "grounding_expert",
    "spatial_expert",
    "general_vqa_expert",
]
RoutableTask = Literal[
    "counting",
    "fine_grained_counting",
    "change_caption",
    "change_qa",
    "grounding",
    "spatial_relation",
    "scene_classification",
    "general_vqa",
    "multiple_choice_vqa",
]

ROUTES: dict[RoutableTask, tuple[ExpertName, ...]] = {
    "counting": ("counting_expert",),
    "fine_grained_counting": ("counting_expert", "spatial_expert"),
    "change_caption": ("change_expert",),
    "change_qa": ("change_expert", "general_vqa_expert"),
    "grounding": ("grounding_expert",),
    "spatial_relation": ("spatial_expert",),
    "scene_classification": ("general_vqa_expert",),
    "general_vqa": ("general_vqa_expert",),
    "multiple_choice_vqa": ("general_vqa_expert",),
}


class ExpertAssignment(BaseModel):
    """One selected expert with a normalized routing weight.
    一个带有归一化路由权重的选中专家。
    """

    model_config = ConfigDict(extra="forbid")

    name: ExpertName
    weight: float = Field(gt=0.0, le=1.0)


class RoutingDecision(BaseModel):
    """Short auditable route with discrete reason codes and no hidden reasoning.
    具有离散原因代码且不保存隐藏推理的简短可审计路由。
    """

    model_config = ConfigDict(extra="forbid")

    task: RoutableTask
    experts: list[ExpertAssignment] = Field(min_length=1)
    requires_tiling: bool
    reason_codes: list[str] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_expert_weights(self) -> "RoutingDecision":
        """Require a stable normalized distribution over selected experts.
        要求选中专家形成稳定的归一化分布。
        """

        if abs(sum(expert.weight for expert in self.experts) - 1.0) > 1e-6:
            raise ValueError("expert weights must sum to 1.0")
        expected = set(ROUTES[self.task])
        actual = {expert.name for expert in self.experts}
        if actual != expected:
            raise ValueError(f"experts do not match fixed route for {self.task}")
        return self


class CallBudgetExceeded(RuntimeError):
    """Raised before an operation would exceed its explicit model-call budget.
    在操作将超出明确模型调用预算前抛出。
    """


class CallBudget(BaseModel):
    """Mutable per-sample model budget shared by routing and expert workflows.
    在路由和专家工作流之间共享的可变单样本模型调用预算。
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    max_qwen_calls: int = Field(ge=0)
    max_deepseek_calls: int = Field(ge=0)
    qwen_calls_used: int = Field(default=0, ge=0)
    deepseek_calls_used: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_usage(self) -> "CallBudget":
        """Reject persisted budgets whose used counters exceed their limits.
        拒绝已用计数超过限制的持久化预算。
        """

        if self.qwen_calls_used > self.max_qwen_calls or self.deepseek_calls_used > self.max_deepseek_calls:
            raise ValueError("used calls must not exceed call budget")
        return self

    def reserve_qwen(self) -> None:
        """Consume one Qwen call before issuing it.
        在发起调用前消耗一次 Qwen 预算。
        """

        if self.qwen_calls_used >= self.max_qwen_calls:
            raise CallBudgetExceeded("Qwen call budget exhausted")
        self.qwen_calls_used += 1

    def reserve_deepseek(self) -> None:
        """Consume one DeepSeek call before issuing it.
        在发起调用前消耗一次 DeepSeek 预算。
        """

        if self.deepseek_calls_used >= self.max_deepseek_calls:
            raise CallBudgetExceeded("DeepSeek call budget exhausted")
        self.deepseek_calls_used += 1


class TaskRouter:
    """Use deterministic routes for known tasks and an optional text router otherwise.
    对已知任务使用确定性路由，对未知任务可选地使用文本路由器。
    """

    def __init__(self, router_client: VisionLanguageClient | None = None, *, router_prompt: str = "") -> None:
        self.router_client = router_client
        self.router_prompt = router_prompt

    def route_known(self, task: RoutableTask, *, high_resolution: bool = False) -> RoutingDecision:
        """Route a declared task without an additional model call.
        在不增加模型调用的情况下路由已声明任务。
        """

        experts = ROUTES[task]
        weight = 1.0 / len(experts)
        reason_codes = [f"task_{task}"]
        if high_resolution:
            reason_codes.append("high_resolution")
        return RoutingDecision(
            task=task,
            experts=[ExpertAssignment(name=expert, weight=weight) for expert in experts],
            requires_tiling=task in {"counting", "fine_grained_counting", "grounding", "change_caption", "change_qa"},
            reason_codes=reason_codes,
        )

    async def route_unknown(
        self,
        question: str,
        *,
        budget: CallBudget,
        sample_id: str,
        artifact_dir: Path | None = None,
    ) -> RoutingDecision:
        """Ask an injected text-only router only when deterministic routing is unavailable.
        仅当确定性路由不可用时才请求注入的纯文本路由器。
        """

        if self.router_client is None:
            raise ValueError("unknown tasks require an injected router client")
        budget.reserve_qwen()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.router_prompt},
            {"role": "user", "content": question},
        ]
        request_hash = build_request_hash(
            model="router",
            generation={"temperature": 0.0},
            prompt_version="router-v1",
            messages=messages,
            image_sha256=None,
        )
        return await self.router_client.complete_json(
            messages=messages,
            response_model=RoutingDecision,
            request_meta=RequestMeta(
                request_id=f"{sample_id}:router",
                request_hash=request_hash,
                prompt_version="router-v1",
                sample_id=sample_id,
                artifact_dir=artifact_dir,
            ),
        )


class CountingExpertAnswer(BaseModel):
    """Presentation-safe counting answer that exposes partial completion honestly.
    如实暴露部分完成状态的安全计数展示答案。
    """

    model_config = ConfigDict(extra="forbid")

    answer: str
    complete: bool
    counting_result: CountingResult


class CountingExpert:
    """Thin counting expert that delegates all geometry to the point pipeline.
    将全部几何逻辑委托给点式管线的轻量计数专家。
    """

    def __init__(self, pipeline: PointCountingOrchestrator) -> None:
        self.pipeline = pipeline

    async def answer(
        self,
        image: Image.Image,
        *,
        sample_id: str,
        question: str,
        target: CountTargetSpec,
    ) -> CountingExpertAnswer:
        """Run the point pipeline and derive user-facing text from its final status.
        运行点式管线，并从最终状态生成面向用户的文本。
        """

        result = await self.pipeline.count_image(image, sample_id=sample_id, question=question, target=target)
        total_tiles = len(result.succeeded_tiles) + len(result.failed_tiles)
        if result.status in {"partial", "failed"}:
            answer = (
                f"Completed {len(result.succeeded_tiles)}/{total_tiles} tiles and confirmed "
                f"{result.final_count} instances; the result is incomplete."
            )
            return CountingExpertAnswer(answer=answer, complete=False, counting_result=result)
        answer = f"Based on {result.final_count} accepted global instance points, the image contains {result.final_count} {target.canonical_label}(s)."
        return CountingExpertAnswer(answer=answer, complete=True, counting_result=result)


def attach_qwen_budget(
    pipeline: PointCountingOrchestrator,
    budget: CallBudget,
) -> PointCountingOrchestrator:
    """Attach one shared Qwen budget to an existing point-counting pipeline.
    将共享 Qwen 预算附加到现有点式计数管线。
    """

    pipeline.before_qwen_call = budget.reserve_qwen
    return pipeline


def make_budget_guard(budget: CallBudget, service: Literal["qwen", "deepseek"]) -> Callable[[], None]:
    """Return a callback suitable for optional critic or judge invocations.
    返回可供可选 critic 或 judge 调用使用的预算回调。
    """

    return budget.reserve_qwen if service == "qwen" else budget.reserve_deepseek
