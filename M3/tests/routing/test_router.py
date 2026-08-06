"""Validate the strict Agent routing contract. / 验证严格的 Agent 路由契约。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spacers_agent.routing.schemas import RoutingDecision


class TestRoutingDecision:
    """Primary/fallback validation. / 主 Agent 与回退 Agent 校验。"""

    def test_single_mode_no_fallback(self) -> None:
        decision = RoutingDecision(
            task="counting",
            primary_agent="counting_agent",
            fallback_agents=[],
            execution_mode="single",
            requires_tiling=True,
            reason_codes=["task_counting"],
            router_source="dataset_task",
        )
        assert decision.execution_mode == "single"

    def test_fallback_mode_with_fallback(self) -> None:
        decision = RoutingDecision(
            task="change_qa",
            primary_agent="change_agent",
            fallback_agents=["general_vqa_agent"],
            execution_mode="fallback",
            requires_tiling=True,
            reason_codes=["task_change_qa"],
            router_source="dataset_task",
        )
        assert decision.fallback_agents == ["general_vqa_agent"]

    def test_primary_not_in_fallback(self) -> None:
        with pytest.raises(ValueError, match="primary_agent must not appear"):
            RoutingDecision(
                task="change_qa",
                primary_agent="change_agent",
                fallback_agents=["change_agent"],
                execution_mode="fallback",
                requires_tiling=True,
                reason_codes=["task_change_qa"],
                router_source="dataset_task",
            )

    def test_single_mode_rejects_fallback(self) -> None:
        with pytest.raises(ValueError, match="empty fallback"):
            RoutingDecision(
                task="counting",
                primary_agent="counting_agent",
                fallback_agents=["change_agent"],
                execution_mode="single",
                requires_tiling=True,
                reason_codes=["task_counting"],
                router_source="dataset_task",
            )

    def test_fallback_mode_requires_fallback(self) -> None:
        with pytest.raises(ValueError, match="at least one fallback"):
            RoutingDecision(
                task="change_qa",
                primary_agent="change_agent",
                fallback_agents=[],
                execution_mode="fallback",
                requires_tiling=True,
                reason_codes=["task_change_qa"],
                router_source="dataset_task",
            )

    def test_rejects_removed_experts_field(self) -> None:
        """No list-based routing fallback exists. / 不再支持基于列表的路由回退。"""
        with pytest.raises(ValidationError, match="experts"):
            RoutingDecision.model_validate(
                {
                    "task": "counting",
                    "primary_agent": "counting_agent",
                    "fallback_agents": [],
                    "execution_mode": "single",
                    "requires_tiling": True,
                    "reason_codes": ["task_counting"],
                    "router_source": "dataset_task",
                    "experts": [{"name": "counting_agent"}],
                }
            )

    @pytest.mark.parametrize("agent_name", ["counting_expert", "unknown_agent"])
    def test_rejects_noncanonical_agent_names(self, agent_name: str) -> None:
        with pytest.raises(ValidationError):
            RoutingDecision(
                task="counting",
                primary_agent=agent_name,
                fallback_agents=[],
                execution_mode="single",
                requires_tiling=True,
                reason_codes=["task_counting"],
                router_source="dataset_task",
            )
