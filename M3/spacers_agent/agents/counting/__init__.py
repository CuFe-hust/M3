"""Lazy public exports for the counting Agent package.
计数 Agent 包的延迟公共导出。

Keeping package initialization side-effect free avoids runtime import cycles.
保持包初始化无副作用，避免运行时循环导入。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spacers_agent.agents.counting.agent import CountingAgent
    from spacers_agent.agents.counting.backends.registry import BackendRegistry
    from spacers_agent.agents.counting.backends.selector import BackendSelector
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


def __getattr__(name: str) -> Any:
    """Resolve lazy public exports only when a caller requests them.
    仅在调用方请求时解析延迟公共导出。
    """

    if name == "CountingAgent":
        from spacers_agent.agents.counting.agent import CountingAgent

        return CountingAgent
    if name in {"BackendRegistry", "BackendSelector"}:
        from spacers_agent.agents.counting.backends.registry import BackendRegistry
        from spacers_agent.agents.counting.backends.selector import BackendSelector

        return {"BackendRegistry": BackendRegistry, "BackendSelector": BackendSelector}[name]
    if name in {"CountTargetParser", "TargetParser"}:
        from spacers_agent.agents.counting.target_parser import CountTargetParser, TargetParser

        return {"CountTargetParser": CountTargetParser, "TargetParser": TargetParser}[name]
    if name in {
        "accepted_count_evidence",
        "box_evidence",
        "global_count_point",
        "parse_count_answer",
    }:
        from spacers_agent.agents.counting import evidence

        return getattr(evidence, name)
    raise AttributeError(name)
