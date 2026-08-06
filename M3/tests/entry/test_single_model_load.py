"""Architecture acceptance: the model is created exactly once and reused.
架构验收：模型只创建一次并被复用。

This is the most important acceptance test for the single main.py entry:
``RuntimeApplication.create`` must load the Qwen model exactly once, and
consecutive ``ask`` calls must reuse that single client instead of reloading.
这是单一 main.py 入口最重要的验收测试：``RuntimeApplication.create`` 必须
只加载一次 Qwen 模型，连续 ``ask`` 调用必须复用同一个客户端而不是重载。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from spacers_agent import application as app_module
from spacers_agent.agents.registry import AgentRegistry
from spacers_agent.application import RuntimeApplication
from spacers_agent.routing import CallBudgetFactory, TaskRouter
from spacers_agent.routing.schemas import RoutingDecision
from spacers_agent.schemas import AgentResult
from spacers_agent.settings import AppSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _FakeClient:
    def __init__(self) -> None:
        self.load_seconds = 0.0


class _EchoAgent:
    name = "general_vqa_agent"
    supported_tasks = frozenset({"general_vqa"})

    def __init__(self) -> None:
        self.run_calls = 0

    async def run(self, sample, context):
        self.run_calls += 1
        from spacers_agent.agents.base import AgentExecution

        return AgentExecution(
            agent_name=self.name,
            payload=AgentResult(
                agent_name=self.name,
                answer=f"answer-{self.run_calls}",
                status="completed",
            ),
            result_filename="agent_result.json",
            trace={},
        )


def _fake_runtime() -> SimpleNamespace:
    registry = AgentRegistry()
    registry.register(_EchoAgent())
    return SimpleNamespace(
        router=TaskRouter(),
        agent_registry=registry,
        call_budget_factory=CallBudgetFactory(),
        prompt_catalog=None,
    )


def _make_image_dir(tmp_path: Path) -> Path:
    image_dir = tmp_path / "img"
    image_dir.mkdir()
    Image.new("RGB", (16, 16), color=(1, 2, 3)).save(image_dir / "a.png")
    return image_dir


def _stub_deps(monkeypatch, loads: list[str], assembles: list[int]):
    """Install counting stubs for create_model and assemble_runtime.
    为 create_model 与 assemble_runtime 安装计数桩。
    """

    def fake_create_model(name, **kwargs):
        loads.append(name)
        return _FakeClient()

    def fake_assemble_runtime(settings, *, qwen_client, judge_client=None,
                              prompt_root=None, router_prompt=""):
        assembles.append(1)
        return _fake_runtime()

    monkeypatch.setattr(app_module, "create_model", fake_create_model)
    monkeypatch.setattr(app_module, "assemble_runtime", fake_assemble_runtime)


def test_create_loads_model_exactly_once(monkeypatch, tmp_path):
    loads: list[str] = []
    assembles: list[int] = []
    _stub_deps(monkeypatch, loads, assembles)

    settings = AppSettings()
    settings.runs.root = tmp_path / "runs"
    app = RuntimeApplication.create(settings=settings, project_root=PROJECT_ROOT)

    assert loads == ["qwen_transformers"]
    assert len(assembles) == 1


def test_three_consecutive_asks_reuse_one_model(monkeypatch, tmp_path):
    """Three asks after one create must not reload the model.
    一次 create 后连续三次 ask 不得重载模型。
    """

    loads: list[str] = []
    assembles: list[int] = []
    _stub_deps(monkeypatch, loads, assembles)

    settings = AppSettings()
    settings.runs.root = tmp_path / "runs"
    app = RuntimeApplication.create(settings=settings, project_root=PROJECT_ROOT)
    image_dir = _make_image_dir(tmp_path)

    answers = [
        asyncio.run(app.ask(image_dir=image_dir, question=f"q{index}", task="general_vqa"))
        for index in range(3)
    ]

    assert [answer.answer for answer in answers] == ["answer-1", "answer-2", "answer-3"]
    assert len({answer.request_id for answer in answers}) == 3
    assert loads == ["qwen_transformers"]
    assert len(assembles) == 1


def test_create_does_not_create_deepseek_or_vllm(monkeypatch, tmp_path):
    """create() must pass judge_client=None and never touch vLLM endpoints.
    create() 必须传入 judge_client=None 且绝不触碰 vLLM 端点。
    """

    loads: list[str] = []
    assembles: list[int] = []
    seen: dict[str, object] = {}
    original_assemble = app_module.assemble_runtime

    def fake_create_model(name, **kwargs):
        loads.append(name)
        return _FakeClient()

    def recording_assemble_runtime(settings, *, qwen_client, judge_client=None,
                                   prompt_root=None, router_prompt=""):
        assembles.append(1)
        seen["judge_client"] = judge_client
        return original_assemble(settings, qwen_client=qwen_client, judge_client=None,
                                 prompt_root=prompt_root, router_prompt=router_prompt)

    monkeypatch.setattr(app_module, "create_model", fake_create_model)
    monkeypatch.setattr(app_module, "assemble_runtime", recording_assemble_runtime)

    settings = AppSettings()
    settings.runs.root = tmp_path / "runs"
    RuntimeApplication.create(settings=settings, project_root=PROJECT_ROOT)

    assert seen["judge_client"] is None
    assert loads == ["qwen_transformers"]
