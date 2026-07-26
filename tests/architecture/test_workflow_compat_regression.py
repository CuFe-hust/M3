"""Regression test — old workflow.py must work with new RoutingDecision (primary_agent).
回归测试 — 旧 workflow.py 必须与新 RoutingDecision (primary_agent) 兼容。
"""

from __future__ import annotations

import pytest

from spacers_agent.routing import TaskRouter, RoutingDecision
from spacers_agent.routing.schemas import AGENT_TO_EXPERT


def test_new_routing_decision_has_primary_agent_not_experts():
    """New RoutingDecision uses primary_agent, not experts."""
    router = TaskRouter()
    decision = router.route_known("change_caption")
    assert decision.primary_agent == "change_agent"
    # Old code used decision.experts[0].name → must handle gracefully
    assert not hasattr(decision, "experts"), "New RoutingDecision must NOT have 'experts' attr"


def test_ag_to_expert_map_is_complete():
    """Every agent name in ROUTES has a corresponding expert name."""
    from spacers_agent.routing import ROUTES
    for agents in ROUTES.values():
        for agent_name in agents:
            expert = AGENT_TO_EXPERT.get(agent_name)
            assert expert is not None, f"No expert mapping for agent {agent_name}"


def test_legacy_workflow_compat_pattern():
    """Simulate old workflow.py pattern with new RoutingDecision."""
    router = TaskRouter()
    decision = router.route_known("general_vqa")
    # Old pattern: expert_name = decision.experts[0].name
    # New pattern:
    expert_name = AGENT_TO_EXPERT.get(decision.primary_agent, decision.primary_agent)
    assert expert_name == "general_vqa_expert"

    # The old if-branch against counting_expert still works
    assert expert_name != "counting_expert"  # general_vqa → general_vqa_expert, not counting


def test_vrsbench_routing_also_works():
    """VRSBench routing also returns new format."""
    router = TaskRouter()
    decision = router.route_vrsbench_vqa("object quantity", question="How many cars?")
    assert decision.primary_agent == "counting_agent"
    expert_name = AGENT_TO_EXPERT.get(decision.primary_agent, decision.primary_agent)
    assert expert_name == "counting_expert"


def test_workflow_service_uses_registry_without_legacy_expert_dictionary():
    """The compatibility facade must not retain the former expert dictionary.
    兼容门面不得保留原专家字典。
    """

    from spacers_agent.workflow import WorkflowService

    prompts = {
        "change": "change",
        "general": "general",
        "spatial": "spatial",
        "spatial_grid": "grid",
        "spatial_review": "review",
        "spatial_grid_review": "grid review",
        "caption": "caption",
    }
    service = WorkflowService(None, prompts, "model")

    assert not hasattr(service, "experts")
    assert type(service.get_agent("change_expert")).__name__ == "ChangeAgent"
    assert type(service.get_agent("caption_expert")).__name__ == "CaptionAgent"
