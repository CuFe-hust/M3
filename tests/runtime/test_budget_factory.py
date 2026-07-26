"""CallBudgetFactory tests. / CallBudgetFactory 测试。"""

from spacers_agent.routing import CallBudgetFactory


def test_budget_factory_preserves_central_legacy_limits() -> None:
    factory = CallBudgetFactory()

    first = factory.create_for_sample("general_vqa")
    second = factory.create_for_sample("general_vqa")

    assert first is not second
    assert (first.max_qwen_calls, first.max_deepseek_calls) == (50, 10)
    first.reserve_qwen()
    assert second.qwen_calls_used == 0


def test_budget_factory_supports_central_task_overrides() -> None:
    factory = CallBudgetFactory(task_limits={"counting": (80, 2)})

    counting = factory.create_for_sample("counting")
    caption = factory.create_for_sample("caption")

    assert (counting.max_qwen_calls, counting.max_deepseek_calls) == (80, 2)
    assert (caption.max_qwen_calls, caption.max_deepseek_calls) == (50, 10)
