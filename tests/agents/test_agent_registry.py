"""Contract tests for the Agent registry, context, and execution validation.

Agent 注册表、上下文与执行校验测试：纯注册/查询/supports/coverage、
Protocol 检查、AgentContext 轻量约束、AgentExecution 的 result_filename /
trace 敏感键 / JSON 序列化校验。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.base import Agent, AgentContext, AgentExecution, validate_agent_execution
from agents.errors import DuplicateAgentError, UnsupportedAgentError
from agents.registry import AgentRegistry
from agents.schema import AgentResult


class _FakeBudget:
    def reserve_qwen(self) -> None:
        pass

    def reserve_deepseek(self) -> None:
        pass


class _FakeClient:
    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        return response_model.model_validate({})


class _FakeAgent:
    name = "general_vqa_agent"
    supported_tasks = frozenset({"general_vqa", "caption"})

    async def run(self, sample, context):
        return AgentExecution(
            agent_name=self.name,
            payload=AgentResult(agent_name=self.name, answer="ok"),
            result_filename="agent_result.json",
        )


# ── Registry / 注册表 ──────────────────────────────────────────────────────


def test_register_and_get() -> None:
    registry = AgentRegistry()
    agent = _FakeAgent()
    registry.register(agent)
    assert registry.get("general_vqa_agent") is agent
    assert registry.contains("general_vqa_agent")
    assert registry.names() == ("general_vqa_agent",)
    assert len(registry) == 1


def test_register_rejects_non_protocol_object() -> None:
    registry = AgentRegistry()
    with pytest.raises(TypeError, match="Protocol"):
        registry.register(object())


def test_duplicate_registration_raises() -> None:
    registry = AgentRegistry()
    registry.register(_FakeAgent())
    with pytest.raises(DuplicateAgentError, match="already registered"):
        registry.register(_FakeAgent())


def test_unknown_name_raises_with_available() -> None:
    registry = AgentRegistry()
    registry.register(_FakeAgent())
    with pytest.raises(UnsupportedAgentError, match="UNREGISTERED_AGENT"):
        registry.get("no-such-agent")


def test_supports_lists_agents_by_task() -> None:
    registry = AgentRegistry()
    registry.register(_FakeAgent())
    assert registry.supports("general_vqa") == ["general_vqa_agent"]
    assert registry.supports("counting") == []


def test_task_coverage_validation() -> None:
    registry = AgentRegistry()
    registry.register(_FakeAgent())
    registry.validate_task_coverage({"general_vqa", "caption"})
    with pytest.raises(UnsupportedAgentError, match="tasks_missing_coverage"):
        registry.validate_task_coverage({"counting"})


# ── AgentContext / 执行上下文 ───────────────────────────────────────────────


def test_context_requires_path_artifact_dir() -> None:
    with pytest.raises(ValueError, match="artifact_dir"):
        AgentContext(artifact_dir=Path("."), qwen_client=_FakeClient(), call_budget=_FakeBudget())
    with pytest.raises(TypeError, match="pathlib"):
        AgentContext(artifact_dir="x", qwen_client=_FakeClient(), call_budget=_FakeBudget())


def test_context_is_lightweight_and_immutable() -> None:
    import dataclasses

    context = AgentContext(
        artifact_dir=Path("/tmp/run"),
        qwen_client=_FakeClient(),
        call_budget=_FakeBudget(),
        request_context={"task": "general_vqa"},
    )
    assert context.artifact_dir == Path("/tmp/run")
    assert dataclasses.is_dataclass(context) and context.__dataclass_params__.frozen
    # No full AppSettings or PromptCatalog fields. / 无完整 AppSettings/PromptCatalog 字段。
    fields = {field.name for field in dataclasses.fields(context)}
    assert "settings" not in fields and "prompt_catalog" not in fields


# ── AgentExecution / 执行校验 ───────────────────────────────────────────────


def _execution(**overrides) -> AgentExecution:
    payload = AgentResult(agent_name="general_vqa_agent", answer="ok")
    values = dict(
        agent_name="general_vqa_agent",
        payload=payload,
        result_filename="agent_result.json",
        trace={"prompt_version": "v1", "geometry": {}},
    )
    values.update(overrides)
    return AgentExecution(**values)


def test_execution_accepts_valid_filename_and_trace() -> None:
    execution = _execution()
    assert execution.result_filename == "agent_result.json"
    assert execution.status == "completed"


def test_execution_rejects_absolute_and_parent_filenames() -> None:
    with pytest.raises(ValueError, match="plain basename"):
        _execution(result_filename="/etc/passwd")
    with pytest.raises(ValueError, match="not contain '..'"):
        _execution(result_filename="../agent_result.json")


def test_execution_rejects_sensitive_trace_keys() -> None:
    for key in ("api_key", "authorization", "token", "base64", "secret"):
        with pytest.raises(ValueError, match="sensitive key"):
            _execution(trace={key: "sk-fake"})


def test_execution_rejects_nested_sensitive_trace_keys() -> None:
    with pytest.raises(ValueError, match="sensitive key"):
        _execution(trace={"geometry": {"authorization": "x"}})
    with pytest.raises(ValueError, match="sensitive key"):
        _execution(trace={"items": [{"access_token": "x"}]})


def test_execution_rejects_non_json_serializable_trace() -> None:
    with pytest.raises(ValueError, match="JSON-serializable"):
        _execution(trace={b"binary": 1})


def test_execution_validates_additional_result_filenames() -> None:
    with pytest.raises(ValueError, match="safe plain basename"):
        _execution(additional_results={"../evil.json": None})


def test_validate_agent_execution_standalone() -> None:
    execution = _execution()
    validate_agent_execution(execution)  # no exception / 无异常
