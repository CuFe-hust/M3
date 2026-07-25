"""Counting agent and its components.
计数 Agent 及其组件。
"""

from spacers_agent.agents.counting.agent import CountingAgent
from spacers_agent.agents.counting.backends import BackendRegistry, BackendSelector
from spacers_agent.agents.counting.evidence import (
    accepted_count_evidence,
    box_evidence,
    global_count_point,
    parse_count_answer,
)
from spacers_agent.agents.counting.target_parser import CountTargetParser, TargetParser

__all__ = [
    "BackendRegistry",
    "BackendSelector",
    "CountTargetParser",
    "CountingAgent",
    "TargetParser",
    "accepted_count_evidence",
    "box_evidence",
    "global_count_point",
    "parse_count_answer",
]
