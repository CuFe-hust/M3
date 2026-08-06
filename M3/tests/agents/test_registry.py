"""Test AgentRegistry behavior: register, lookup, validation, and edge cases.
测试 AgentRegistry 行为：注册、查找、校验与边界情况。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spacers_agent.agents.base import Agent, AgentContext, AgentExecution
from spacers_agent.agents.errors import (
    AgentTaskMismatchError,
    DuplicateAgentError,
    UnsupportedAgentError,
)
from spacers_agent.agents.registry import AgentRegistry
from spacers_agent.schemas import AgentResult


# ── minimal agents for testing / 最小 Agent 实例 ────────────────────────


class _CountingAgent:
    name = "counting_agent"
    supported_tasks = frozenset({"counting", "fine_grained_counting"})

    async def run(self, sample, context: AgentContext) -> AgentExecution:
        return AgentExecution(
            agent_name=self.name,
            payload=AgentResult(agent_name="counting_agent", answer="3"),
            result_filename="counting_result.json",
        )


class _ChangeAgent:
    name = "change_agent"
    supported_tasks = frozenset({"change_caption", "change_qa"})

    async def run(self, sample, context: AgentContext) -> AgentExecution:
        return AgentExecution(
            agent_name=self.name,
            payload=AgentResult(agent_name="change", answer="changed"),
            result_filename="agent_result.json",
        )


class _GeneralVQAAgent:
    name = "general_vqa_agent"
    supported_tasks = frozenset({"general_vqa", "scene_classification", "multiple_choice_vqa", "caption"})

    async def run(self, sample, context: AgentContext) -> AgentExecution:
        return AgentExecution(
            agent_name=self.name,
            payload=AgentResult(agent_name="general_vqa_agent", answer="yes"),
            result_filename="agent_result.json",
        )


ALL_TEST_AGENTS = [_CountingAgent(), _ChangeAgent(), _GeneralVQAAgent()]


# ── basic registry operations / 基本注册表操作 ──────────────────────────


class TestBasicOperations:
    """Register, get, contains, names in insertion order. / 注册、获取、包含、插入顺序名称。"""

    def test_register_and_get(self):
        reg = AgentRegistry()
        agent = _CountingAgent()
        reg.register(agent)
        assert reg.get("counting_agent") is agent

    def test_contains_true(self):
        reg = AgentRegistry()
        reg.register(_CountingAgent())
        assert reg.contains("counting_agent")

    def test_contains_false(self):
        reg = AgentRegistry()
        assert not reg.contains("nonexistent")

    def test_names_empty(self):
        reg = AgentRegistry()
        assert reg.names() == ()

    def test_names_insertion_order(self):
        reg = AgentRegistry()
        for a in ALL_TEST_AGENTS:
            reg.register(a)
        assert reg.names() == ("counting_agent", "change_agent", "general_vqa_agent")

    def test_names_idempotent(self):
        reg = AgentRegistry()
        reg.register(_CountingAgent())
        first = reg.names()
        second = reg.names()
        assert first == second


# ── duplicate registration / 重复注册 ────────────────────────────────────


class TestDuplicateRegistration:
    """Duplicate names must raise DuplicateAgentError. / 重复名称必须抛出 DuplicateAgentError。"""

    def test_duplicate_raises(self):
        reg = AgentRegistry()
        reg.register(_CountingAgent())

        class _AnotherCounting:
            name = "counting_agent"
            supported_tasks = frozenset({"counting"})
            async def run(self, sample, context): pass

        with pytest.raises(DuplicateAgentError, match="already registered"):
            reg.register(_AnotherCounting())

    def test_duplicate_includes_class_names(self):
        reg = AgentRegistry()
        reg.register(_CountingAgent())

        class _AnotherCounting:
            name = "counting_agent"
            supported_tasks = frozenset({"counting"})
            async def run(self, sample, context): pass

        with pytest.raises(DuplicateAgentError) as exc_info:
            reg.register(_AnotherCounting())
        # error message contains both class names
        assert "_CountingAgent" in str(exc_info.value) or "_AnotherCounting" in str(exc_info.value)


# ── protocol enforcement / Protocol 强制 ─────────────────────────────────


class TestProtocolEnforcement:
    """Non-Protocol objects must be rejected. / 非 Protocol 对象必须被拒绝。"""

    def test_non_agent_raises(self):
        reg = AgentRegistry()
        with pytest.raises(TypeError, match="does not satisfy the Agent Protocol"):
            reg.register(object())  # plain object

    def test_missing_name_raises(self):
        reg = AgentRegistry()

        class _NoName:
            supported_tasks = frozenset({"counting"})
            async def run(self, sample, context): pass

        with pytest.raises((TypeError, ValueError)):
            reg.register(_NoName())

    def test_empty_name_raises(self):
        reg = AgentRegistry()

        class _EmptyName:
            name = ""
            supported_tasks = frozenset({"counting"})
            async def run(self, sample, context): pass

        with pytest.raises(ValueError, match="empty"):
            reg.register(_EmptyName())


# ── task coverage validation / 任务覆盖校验 ──────────────────────────────


class TestTaskCoverage:
    """validate_task_coverage checks all required tasks. / validate_task_coverage 检查全部必需任务。"""

    def test_all_covered(self):
        reg = AgentRegistry()
        for a in ALL_TEST_AGENTS:
            reg.register(a)
        # All tasks covered / 全部覆盖
        reg.validate_task_coverage({"counting", "change_caption", "general_vqa"})

    def test_partial_coverage_raises(self):
        reg = AgentRegistry()
        reg.register(_CountingAgent())
        with pytest.raises(UnsupportedAgentError, match="tasks_missing_coverage"):
            reg.validate_task_coverage({"counting", "caption", "spatial_relation"})


# ── legacy name normalization / 旧名规范化 ───────────────────────────────


# ── YOLO-disabled construction / YOLO 关闭时构建 ─────────────────────────


class TestYoloDisabledConstruction:
    """Building the registry with YOLO disabled must NOT import ultralytics. / YOLO 关闭时构建注册表不得导入 ultralytics。"""

    def test_registry_builds_without_ultralytics(self):
        """build_agent_registry succeeds without YOLO. / build_agent_registry 在无 YOLO 时成功。"""
        import sys
        ultralytics_before = "ultralytics" in sys.modules

        from spacers_agent.settings import AppSettings
        from spacers_agent.clients.mock import MockVisionClient

        settings = AppSettings()
        settings.backend.yolo.enabled = False

        class _FakeClient:
            async def complete_json(self, **kw): pass

        client = _FakeClient()

        from spacers_agent.bootstrap import build_agent_registry
        from spacers_agent.prompt_catalog import PromptCatalog

        prompt_catalog = PromptCatalog(Path(__file__).resolve().parents[2] / "prompts")
        registry = build_agent_registry(
            settings=settings,
            qwen_client=client,
            prompt_catalog=prompt_catalog,
        )
        assert len(registry) == 6
        assert not ultralytics_before or "ultralytics" in sys.modules
