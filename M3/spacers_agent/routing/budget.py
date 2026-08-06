"""Mutable per-sample model-call budget shared by routing and expert workflows.
在路由和专家工作流之间共享的可变单样本模型调用预算。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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
    max_deepseek_calls: int = Field(default=0, ge=0)
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
        """Consume one Qwen call before issuing it. / 在发起调用前消耗一次 Qwen 预算。"""
        if self.qwen_calls_used >= self.max_qwen_calls:
            raise CallBudgetExceeded("Qwen call budget exhausted")
        self.qwen_calls_used += 1

    def reserve_deepseek(self) -> None:
        """Consume one DeepSeek call before issuing it. / 在发起调用前消耗一次 DeepSeek 预算。"""
        if self.deepseek_calls_used >= self.max_deepseek_calls:
            raise CallBudgetExceeded("DeepSeek call budget exhausted")
        self.deepseek_calls_used += 1


@dataclass(frozen=True)
class CallBudgetFactory:
    """Create per-sample budgets from centralized legacy-compatible limits.
    使用集中且兼容旧行为的限制创建单样本预算。
    """

    default_qwen_calls: int = 50
    default_deepseek_calls: int = 10
    task_limits: Mapping[str, tuple[int, int]] = field(default_factory=dict)

    def create_for_sample(self, task: str) -> CallBudget:
        """Return a fresh mutable budget for one task.
        为一个任务返回全新的可变预算。
        """

        qwen_calls, deepseek_calls = self.task_limits.get(
            task,
            (self.default_qwen_calls, self.default_deepseek_calls),
        )
        return CallBudget(max_qwen_calls=qwen_calls, max_deepseek_calls=deepseek_calls)


def make_budget_guard(budget: CallBudget, service: Literal["qwen", "deepseek"]) -> Callable[[], None]:
    """Return a callback suitable for optional critic or judge invocations.
    返回可供可选 critic 或 judge 调用使用的预算回调。
    """

    return budget.reserve_qwen if service == "qwen" else budget.reserve_deepseek
