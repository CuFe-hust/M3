"""Test new RoutingDecision with primary/fallback and legacy format. / 测试新的 RoutingDecision。"""

from __future__ import annotations

import pytest

from spacers_agent.routing.schemas import RoutingDecision, normalize_agent_name


class TestNewRoutingDecision:
    """Primary/fallback validation. / primary/fallback 校验。"""

    def test_single_mode_no_fallback(self):
        d = RoutingDecision(
            task="counting", primary_agent="counting_agent", fallback_agents=[],
            execution_mode="single", requires_tiling=True,
            reason_codes=["task_counting"], router_source="dataset_task",
        )
        assert d.execution_mode == "single"

    def test_fallback_mode_with_fallback(self):
        d = RoutingDecision(
            task="change_qa", primary_agent="change_agent", fallback_agents=["general_vqa_agent"],
            execution_mode="fallback", requires_tiling=True,
            reason_codes=["task_change_qa"], router_source="dataset_task",
        )
        assert d.fallback_agents == ["general_vqa_agent"]

    def test_primary_not_in_fallback(self):
        with pytest.raises(ValueError, match="primary_agent must not appear"):
            RoutingDecision(
                task="change_qa", primary_agent="change_agent", fallback_agents=["change_agent"],
                execution_mode="fallback", requires_tiling=True,
                reason_codes=["task_change_qa"], router_source="dataset_task",
            )

    def test_single_mode_rejects_fallback(self):
        with pytest.raises(ValueError, match="empty fallback"):
            RoutingDecision(
                task="counting", primary_agent="counting_agent", fallback_agents=["change_agent"],
                execution_mode="single", requires_tiling=True,
                reason_codes=["task_counting"], router_source="dataset_task",
            )

    def test_fallback_mode_requires_fallback(self):
        with pytest.raises(ValueError, match="at least one fallback"):
            RoutingDecision(
                task="change_qa", primary_agent="change_agent", fallback_agents=[],
                execution_mode="fallback", requires_tiling=True,
                reason_codes=["task_change_qa"], router_source="dataset_task",
            )


class TestLegacyFormat:
    """Legacy `experts` list auto-converts to primary/fallback. / 旧 `experts` 列表自动转换。"""

    def test_single_expert_legacy(self):
        data = {
            "task": "counting", "experts": [{"name": "counting_expert", "weight": 1.0}],
            "requires_tiling": True, "reason_codes": ["task_counting"],
        }
        d = RoutingDecision.model_validate(data)
        assert d.primary_agent == "counting_agent"
        assert d.fallback_agents == []
        assert d.execution_mode == "single"

    def test_two_experts_legacy(self):
        data = {
            "task": "change_qa", "experts": [
                {"name": "change_expert", "weight": 0.5},
                {"name": "general_vqa_expert", "weight": 0.5},
            ],
            "requires_tiling": True, "reason_codes": ["task_change_qa"],
        }
        d = RoutingDecision.model_validate(data)
        assert d.primary_agent == "change_agent"
        assert d.fallback_agents == ["general_vqa_agent"]
        assert d.execution_mode == "fallback"

    def test_legacy_fallback_router_source(self):
        data = {
            "task": "general_vqa", "experts": [{"name": "general_vqa_expert", "weight": 1.0}],
            "requires_tiling": False, "reason_codes": ["task_general_vqa"],
        }
        d = RoutingDecision.model_validate(data)
        assert d.router_source == "rule_fallback"


class TestNormalization:
    """Agent name normalization. / Agent 名规范化。"""

    def test_expert_to_agent(self):
        assert normalize_agent_name("counting_expert") == "counting_agent"
        assert normalize_agent_name("change_expert") == "change_agent"
        assert normalize_agent_name("caption_expert") == "caption_agent"

    def test_agent_passthrough(self):
        assert normalize_agent_name("counting_agent") == "counting_agent"

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            normalize_agent_name("nonexistent")
