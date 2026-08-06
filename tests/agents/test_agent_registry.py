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
    for bad in ("/etc/passwd", r"C:\temp\result.json", "C:/temp/result.json",
                "sub/result.json", r"..\result.json", "../agent_result.json", ".", ".."):
        with pytest.raises(ValueError, match="plain basename"):
            _execution(result_filename=bad)


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
    with pytest.raises(ValueError, match="non-string key"):
        _execution(trace={b"binary": 1})
    with pytest.raises(ValueError, match="bytes"):
        _execution(trace={"x": b"bytes"})
    with pytest.raises(ValueError, match="Path"):
        _execution(trace={"path": Path("/tmp/x")})
    with pytest.raises(ValueError, match="set"):
        _execution(trace={"s": {1, 2}})
    with pytest.raises(ValueError, match="non-finite"):
        _execution(trace={"n": float("nan")})


def test_execution_rejects_sensitive_value_prefixes() -> None:
    for value in ("sk-secret-key", "Bearer abc123", "-----BEGIN PRIVATE KEY-----"):
        with pytest.raises(ValueError, match="sensitive value"):
            _execution(trace={"answer": value})
    with pytest.raises(ValueError, match="sensitive value"):
        _execution(trace={"items": [{"raw": "data:image/png;base64,AAAA"}]})


def test_execution_payload_agent_name_must_match() -> None:
    with pytest.raises(ValueError, match="does not match"):
        _execution(agent_name="spatial_agent")


def test_execution_additional_results_must_not_collide() -> None:
    with pytest.raises(ValueError, match="collides"):
        _execution(additional_results={"agent_result.json": {"a": 1}})
    with pytest.raises(ValueError, match="plain basename"):
        _execution(additional_results={"../evil.json": None})
    execution = _execution(additional_results={"extra.json": {"x": 1}})
    assert execution.additional_results == {"extra.json": {"x": 1}}


def test_validate_agent_execution_standalone() -> None:
    execution = _execution()
    validate_agent_execution(execution)  # no exception / 无异常


# ── 安全校验补齐 / safety hardening (E) ────────────────────────────────────


def test_context_rejects_sensitive_request_context_keys() -> None:
    for bad in ({"api_key": "sk-1"}, {"authorization": "Bearer x"},
                {"nested": {"access_token": "x"}}):
        with pytest.raises(ValueError, match="sensitive key"):
            AgentContext(
                artifact_dir=Path("/tmp/run"),
                qwen_client=_FakeClient(),
                call_budget=_FakeBudget(),
                request_context=bad,
            )


def test_context_rejects_sensitive_request_context_values() -> None:
    for bad in ({"note": "sk-secret"}, {"note": "  Bearer abc"},
                {"note": "Data:Image/png;base64,AAAA"}):
        with pytest.raises(ValueError, match="sensitive value"):
            AgentContext(
                artifact_dir=Path("/tmp/run"),
                qwen_client=_FakeClient(),
                call_budget=_FakeBudget(),
                request_context=bad,
            )


def test_context_accepts_plain_request_context() -> None:
    context = AgentContext(
        artifact_dir=Path("/tmp/run"),
        qwen_client=_FakeClient(),
        call_budget=_FakeBudget(),
        request_context={"split": "test", "items": [1, 2, {"k": "v"}]},
    )
    assert context.request_context["split"] == "test"


def test_execution_rejects_sensitive_additional_result_keys() -> None:
    with pytest.raises(ValueError, match="sensitive key"):
        _execution(additional_results={"debug.json": {"api_key": "sk-1"}})
    with pytest.raises(ValueError, match="sensitive key"):
        _execution(additional_results={"debug.json": {"nested": {"authorization": "x"}}})


def test_execution_rejects_case_space_sensitive_values() -> None:
    for value in ("SK-ABC", "  Bearer abc", "  sk-secret", "Data:Image/png;base64,AAAA"):
        with pytest.raises(ValueError, match="sensitive value"):
            _execution(trace={"answer": value})
        with pytest.raises(ValueError, match="sensitive value"):
            _execution(additional_results={"debug.json": {"answer": value}})


def test_execution_accepts_plain_additional_results() -> None:
    execution = _execution(additional_results={"debug.json": {"notes": ["a", "b"], "n": 2}})
    assert execution.additional_results["debug.json"]["n"] == 2


class _DomainResult:
    """Future domain result with its own agent_name attribute.
    带自身 agent_name 属性的未来域专用结果。"""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name


def test_execution_validates_any_payload_with_agent_name() -> None:
    with pytest.raises(ValueError, match="does not match"):
        _execution(payload=_DomainResult("counting_agent"))
    execution = _execution(payload=_DomainResult("general_vqa_agent"))
    assert execution.payload.agent_name == "general_vqa_agent"


def test_execution_accepts_payload_without_agent_name() -> None:
    execution = _execution(payload={"answer": "ok"})
    assert execution.payload == {"answer": "ok"}
