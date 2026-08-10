"""Contract tests for the per-sample model call budget.

单样本模型调用预算契约测试：Qwen/DeepSeek 计数与耗尽、持久化校验、工厂
覆盖、budget guard 回调、与 agents.base.CallBudget 协议的结构兼容、无
application 依赖。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from workflows.call_budget import (
    CallBudget,
    CallBudgetExceeded,
    CallBudgetFactory,
    make_budget_guard,
)


def test_reserve_counts_and_exhaustion() -> None:
    budget = CallBudget(max_qwen_calls=2, max_deepseek_calls=1)
    budget.reserve_qwen()
    budget.reserve_qwen()
    assert budget.qwen_calls_used == 2
    with pytest.raises(CallBudgetExceeded):
        budget.reserve_qwen()
    budget.reserve_deepseek()
    assert budget.deepseek_calls_used == 1
    with pytest.raises(CallBudgetExceeded):
        budget.reserve_deepseek()


def test_persisted_usage_above_limit_rejected() -> None:
    with pytest.raises(ValueError, match="exceed"):
        CallBudget(max_qwen_calls=1, qwen_calls_used=2)
    with pytest.raises(ValueError, match="exceed"):
        CallBudget(max_qwen_calls=0, max_deepseek_calls=0, deepseek_calls_used=1)


def test_assignment_validation() -> None:
    budget = CallBudget(max_qwen_calls=1)
    with pytest.raises(ValueError):
        budget.qwen_calls_used = 5


def test_factory_defaults() -> None:
    factory = CallBudgetFactory()
    budget = factory.create_for_sample("counting")
    assert budget.max_qwen_calls == 50
    assert budget.max_deepseek_calls == 10


def test_factory_task_limits_override() -> None:
    factory = CallBudgetFactory(task_limits={"change_caption": (3, 1)})
    budget = factory.create_for_sample("change_caption")
    assert budget.max_qwen_calls == 3
    assert budget.max_deepseek_calls == 1
    assert factory.create_for_sample("other").max_qwen_calls == 50


def test_factory_returns_fresh_instances() -> None:
    factory = CallBudgetFactory(task_limits={"x": (1, 0)})
    first = factory.create_for_sample("x")
    second = factory.create_for_sample("x")
    assert first is not second
    first.reserve_qwen()
    assert second.qwen_calls_used == 0


def test_make_budget_guard_qwen_and_deepseek() -> None:
    budget = CallBudget(max_qwen_calls=1, max_deepseek_calls=1)
    qwen_guard = make_budget_guard(budget, "qwen")
    deepseek_guard = make_budget_guard(budget, "deepseek")
    qwen_guard()
    assert budget.qwen_calls_used == 1
    deepseek_guard()
    assert budget.deepseek_calls_used == 1
    with pytest.raises(CallBudgetExceeded):
        qwen_guard()


def test_budget_satisfies_agent_protocol() -> None:
    """Structural compatibility: both protocol methods exist and reserve.
    结构兼容：协议要求的两个方法存在且可调用。"""
    budget = CallBudget(max_qwen_calls=1)
    assert callable(budget.reserve_qwen)
    assert callable(budget.reserve_deepseek)
    budget.reserve_qwen()
    assert budget.qwen_calls_used == 1


def test_budget_module_has_no_application_dependency() -> None:
    source = (Path(__file__).resolve().parents[2] / "workflows" / "call_budget.py").read_text(
        encoding="utf-8"
    )
    assert "import application" not in source
    assert "spacers_agent" not in source
    assert "AppSettings" not in source
