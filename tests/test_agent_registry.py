"""Test Agent Protocol, Context, and Registry. / 测试 Agent 协议、上下文与注册表。"""

from __future__ import annotations

from pathlib import Path

import pytest

from spacers_agent.agents.base import (
    Agent,
    AgentContext,
    AgentExecution,
)
from spacers_agent.agents.registry import AgentRegistry
from spacers_agent.agents.errors import DuplicateAgentError, UnsupportedAgentError
from spacers_agent.schemas import ExpertResult


class _FakeAgent:
    """Minimal agent for registry tests. / 注册表测试用的最小 Agent。"""

    name = "counting_agent"
    supported_tasks = frozenset({"counting"})

    async def run(self, sample, context: AgentContext) -> AgentExecution:
        return AgentExecution(
            agent_name=self.name,
            payload=ExpertResult(expert="test", answer="ok"),
            result_filename="expert_result.json",
            trace={"route": "test"},
        )


class _FakeCountingAgent:
    """Another agent for duplicate tests. / 另一 Agent 用于重复测试。"""

    name = "counting_agent"
    supported_tasks = frozenset({"counting"})

    async def run(self, sample, context: AgentContext) -> AgentExecution:
        return AgentExecution(
            agent_name=self.name,
            payload=ExpertResult(expert="fake", answer="0"),
            result_filename="expert_result.json",
        )


def test_registry_register_and_get():
    """Register + retrieve work correctly. / 注册 + 获取正常工作。"""
    reg = AgentRegistry()
    agent = _FakeAgent()
    reg.register(agent)
    assert reg.get("counting_agent") is agent
    assert "counting_agent" in reg.names()


def test_registry_missing_raises_unsupported():
    """Missing agent raises UnsupportedAgentError. / 缺失 Agent 抛出 UnsupportedAgentError。"""
    reg = AgentRegistry()
    with pytest.raises(UnsupportedAgentError, match="UNREGISTERED_AGENT"):
        reg.get("nonexistent")


def test_registry_empty_names():
    """Empty registry returns empty tuple. / 空注册表返回空元组。"""
    reg = AgentRegistry()
    assert reg.names() == ()


def test_registry_duplicate_raises():
    """Re-registering raises DuplicateAgentError. / 重复注册抛出 DuplicateAgentError。"""
    reg = AgentRegistry()
    reg.register(_FakeAgent())
    with pytest.raises(DuplicateAgentError, match="already registered"):
        reg.register(_FakeCountingAgent())


def test_registry_contains():
    """contains() correctly reports registration status. / contains() 正确报告注册状态。"""
    reg = AgentRegistry()
    assert not reg.contains("counting_agent")
    reg.register(_FakeAgent())
    assert reg.contains("counting_agent")


def test_registry_supports_task():
    """supports() filter by task name. / supports() 按任务名过滤。"""
    reg = AgentRegistry()
    reg.register(_FakeAgent())
    assert "counting_agent" in reg.supports("counting")
    assert "counting_agent" not in reg.supports("general_vqa")


def test_registry_names_is_stable():
    """names() returns tuple in insertion order. / names() 按插入顺序返回元组。"""
    reg = AgentRegistry()
    reg.register(_FakeAgent())

    class _SecondAgent:
        name = "change_agent"
        supported_tasks = frozenset({"change_caption"})
        async def run(self, sample, context): pass

    reg.register(_SecondAgent())
    assert reg.names() == ("counting_agent", "change_agent")
    # Second call returns same result / 第二次调用返回相同结果
    assert reg.names() == ("counting_agent", "change_agent")


def test_registry_len():
    """len(registry) reports count. / len(registry) 报告数量。"""
    reg = AgentRegistry()
    assert len(reg) == 0
    reg.register(_FakeAgent())
    assert len(reg) == 1


def test_registry_in_operator():
    """'in' operator works. / 'in' 运算符可用。"""
    reg = AgentRegistry()
    reg.register(_FakeAgent())
    assert "counting_agent" in reg
    assert "nonexistent" not in reg


def test_registry_validate_task_coverage_passes():
    """validate_task_coverage passes when all tasks covered. / 全部任务覆盖时通过。"""
    reg = AgentRegistry()
    reg.register(_FakeAgent())  # supports "counting"
    reg.validate_task_coverage({"counting"})  # should not raise


def test_registry_validate_task_coverage_fails():
    """validate_task_coverage raises when tasks missing. / 任务缺失时抛出异常。"""
    reg = AgentRegistry()
    reg.register(_FakeAgent())  # supports "counting"
    with pytest.raises(UnsupportedAgentError, match="tasks_missing_coverage"):
        reg.validate_task_coverage({"counting", "general_vqa", "caption"})


def test_agent_execution_fields():
    """AgentExecution can be created with minimal fields. / AgentExecution 可用最少字段创建。"""
    execution = AgentExecution(
        agent_name="change_agent",
        payload=ExpertResult(expert="test", answer="ok"),
        result_filename="expert_result.json",
        trace={"route": "test -> done"},
    )
    assert execution.agent_name == "change_agent"
    assert execution.result_filename == "expert_result.json"
    assert execution.trace["route"] == "test -> done"
    assert execution.status == "completed"


def test_agent_context_create():
    """AgentContext can be created with minimal fields. / AgentContext 可用最少字段创建。"""
    ctx = AgentContext(
        artifact_dir=Path("/tmp/test_run"),
        settings=None,
        qwen_client=None,
        call_budget=None,
        prompt_catalog=None,
    )
    assert ctx.artifact_dir == Path("/tmp/test_run")
    assert ctx.prompt_catalog is None
    assert ctx.judge_client is None


def test_agent_execution_rejects_bad_filename():
    """Absolute paths and parent refs rejected. / 绝对路径和父引用被拒绝。"""
    with pytest.raises(ValueError):
        AgentExecution(agent_name="change_agent", payload=None, result_filename="/etc/passwd")
    with pytest.raises(ValueError):
        AgentExecution(agent_name="change_agent", payload=None, result_filename="../expert_result.json")


def test_agent_execution_rejects_sensitive_trace_keys():
    """Trace with api_key rejects. / 含 api_key 的 trace 被拒绝。"""
    with pytest.raises(ValueError, match="sensitive key"):
        AgentExecution(
            agent_name="change_agent",
            payload=None,
            result_filename="expert_result.json",
            trace={"api_key": "sk-secret"},
        )
