"""Test AgentExecution contract enforcement and AgentContext validation.
测试 AgentExecution 契约强制与 AgentContext 校验。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spacers_agent.agents.base import (
    AgentContext,
    AgentExecution,
    validate_agent_execution,
)
from spacers_agent.schemas import CountingResult, ExpertResult


# ── AgentExecution validation / AgentExecution 校验 ───────────────────────


class TestResultFilename:
    """result_filename safety checks. / result_filename 安全性检查。"""

    def test_accepts_counting_result(self):
        exec_result = AgentExecution(
            agent_name="counting_agent",
            payload=CountingResult(
                sample_id="s1", target="car", question="?",
                source_width=100, source_height=100, tile_count=1,
                succeeded_tiles=["t1"], failed_tiles=[],
                global_points=[], merged_groups=[], unresolved_conflicts=[],
                warnings=[], final_count=0, status="completed",
            ),
            result_filename="counting_result.json",
        )
        assert exec_result.result_filename == "counting_result.json"

    def test_accepts_expert_result(self):
        exec_result = AgentExecution(
            agent_name="change_agent",
            payload=ExpertResult(expert="test", answer="ok"),
            result_filename="expert_result.json",
        )
        assert exec_result.result_filename == "expert_result.json"

    def test_rejects_absolute_path(self):
        with pytest.raises(ValueError, match="plain basename"):
            AgentExecution(
                agent_name="change_agent",
                payload=None,
                result_filename="/tmp/result.json",
            )

    def test_rejects_parent_ref(self):
        with pytest.raises(ValueError, match="not contain"):
            AgentExecution(
                agent_name="change_agent",
                payload=None,
                result_filename="../expert_result.json",
            )

    def test_rejects_backslash(self):
        with pytest.raises(ValueError, match="plain basename"):
            AgentExecution(
                agent_name="change_agent",
                payload=None,
                result_filename="sub\\result.json",
            )

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            AgentExecution(
                agent_name="change_agent",
                payload=None,
                result_filename="",
            )


class TestTraceSanitization:
    """Trace must not contain sensitive keys. / Trace 不得包含敏感键。"""

    def test_rejects_api_key(self):
        with pytest.raises(ValueError, match="sensitive key"):
            AgentExecution(
                agent_name="change_agent",
                payload=None,
                result_filename="expert_result.json",
                trace={"api_key": "sk-secret"},
            )

    def test_rejects_authorization_header(self):
        with pytest.raises(ValueError, match="sensitive key"):
            AgentExecution(
                agent_name="change_agent",
                payload=None,
                result_filename="expert_result.json",
                trace={"authorization": "Bearer token"},
            )

    def test_rejects_nested_sensitive_key(self):
        with pytest.raises(ValueError, match="sensitive key"):
            AgentExecution(
                agent_name="change_agent",
                payload=None,
                result_filename="expert_result.json",
                trace={"request": {"headers": {"api_key": "secret"}}},
            )

    def test_accepts_safe_trace(self):
        exec_result = AgentExecution(
            agent_name="change_agent",
            payload=ExpertResult(expert="test", answer="ok"),
            result_filename="expert_result.json",
            trace={
                "agent_class": "ChangeAgent",
                "route": "ChangeAgent.run -> Expert.run",
                "backend": "qwen_tile",
                "prompt_version": "change-v1",
                "model": "qwen3-vl-4b-instruct",
                "timing_seconds": 2.5,
                "selection_reason": "rule-based",
                "geometry_summary": {"boxes": 2},
            },
        )
        assert exec_result.trace["backend"] == "qwen_tile"

    def test_trace_is_json_serializable(self):
        """Trace dict must be JSON-serializable. / Trace 字典必须 JSON 可序列化。"""
        exec_result = AgentExecution(
            agent_name="change_agent",
            payload=ExpertResult(expert="test", answer="ok"),
            result_filename="expert_result.json",
            trace={"route": "test", "count": 1, "nested": {"key": "value"}},
        )
        dumped = json.dumps(exec_result.trace)
        assert "route" in dumped


class TestAgentExecutionStatus:
    """status property delegates to payload. / status 属性委托给 payload。"""

    def test_completed_status(self):
        exec_result = AgentExecution(
            agent_name="change_agent",
            payload=ExpertResult(expert="test", answer="ok", status="completed"),
            result_filename="expert_result.json",
        )
        assert exec_result.status == "completed"

    def test_partial_status(self):
        exec_result = AgentExecution(
            agent_name="change_agent",
            payload=ExpertResult(expert="test", answer="ok", status="partial"),
            result_filename="expert_result.json",
        )
        assert exec_result.status == "partial"

    def test_failed_status(self):
        exec_result = AgentExecution(
            agent_name="change_agent",
            payload=ExpertResult(expert="test", answer="ok", status="failed"),
            result_filename="expert_result.json",
        )
        assert exec_result.status == "failed"


# ── AgentContext validation / AgentContext 校验 ──────────────────────────


class TestAgentContextValidation:
    """AgentContext construction rules. / AgentContext 构造规则。"""

    def test_empty_artifact_dir_raises(self):
        # Path("") is falsy but still creates a Path; we explicitly check for falsiness.
        # Path("") 是 falsy 但仍创建 Path；我们显式检查 falsiness。
        with pytest.raises(ValueError, match="must not be empty"):
            AgentContext(
                artifact_dir=Path(""),
                settings=None,
                qwen_client=None,
                call_budget=None,
            )

    def test_no_api_keys_in_repr(self):
        ctx = AgentContext(
            artifact_dir=Path("/tmp/test"),
            settings=None,
            qwen_client=None,
            call_budget=None,
        )
        repr_str = repr(ctx)
        assert "sk-" not in repr_str.lower()
        assert "api_key" not in repr_str.lower()
