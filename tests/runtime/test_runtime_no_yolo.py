"""YOLO isolation tests for the new runtime. / 新运行时的 YOLO 隔离测试。"""

import sys

import pytest

from spacers_agent.bootstrap import assemble_runtime
from spacers_agent.settings import AppSettings


class _FakeQwen:
    async def complete_json(self, **kwargs):
        raise AssertionError("runtime construction must not call Qwen")


def test_default_runtime_registers_only_allowed_counting_backends() -> None:
    modules_before = {name for name in sys.modules if name.startswith("ultralytics")}
    runtime = assemble_runtime(AppSettings(), qwen_client=_FakeQwen())
    counting_agent = runtime.agent_registry.get("counting_agent")

    assert counting_agent._selector._registry.all_names() == [
        "qwen_point",
        "vrsbench_qwen_count",
    ]
    assert {name for name in sys.modules if name.startswith("ultralytics")} == modules_before


def test_runtime_rejects_yolo_enablement_before_registration() -> None:
    settings = AppSettings.model_validate({"backend": {"yolo": {"enabled": True}}})

    with pytest.raises(RuntimeError, match="intentionally disabled"):
        assemble_runtime(settings, qwen_client=_FakeQwen())
