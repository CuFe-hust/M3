"""Agent Registry — maps names to Agent implementations.
Agent 注册表 — 将名称映射到 Agent 实现。

The registry is a pure data structure. It never routes, reads config,
creates agents, calls models, or writes to disk. Bootstrap constructs and
registers agent instances.
注册表是纯数据结构。它不路由、不读配置、不创建 Agent、不调用模型、不写盘。
Bootstrap 负责构造 Agent 并注册。
"""

from __future__ import annotations

from agents.base import Agent
from agents.errors import DuplicateAgentError, UnsupportedAgentError


class AgentRegistry:
    """Map agent names to implementations; never loads models or reads weights.
    映射 Agent 名到实现；绝不加载模型或读取权重。"""

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}
        # Stable insertion order for deterministic output / 确定性输出的稳定插入顺序
        self._order: list[str] = []

    def register(self, agent: Agent) -> None:
        """Register an agent. Raises DuplicateAgentError on name conflict.
        注册 Agent。名称冲突时抛出 DuplicateAgentError。"""
        if not isinstance(agent, Agent):
            raise TypeError(
                f"Object {agent!r} does not satisfy the Agent Protocol. "
                "Expected attributes: name (AgentName), supported_tasks (frozenset[str]), "
                "async run(sample, context) -> AgentExecution."
            )
        name = agent.name
        if not name:
            raise ValueError("Agent name must not be empty")
        if name in self._agents:
            existing = type(self._agents[name]).__name__
            new = type(agent).__name__
            raise DuplicateAgentError(name, existing_class=existing, new_class=new)
        self._agents[name] = agent
        self._order.append(name)

    def get(self, name: str) -> Agent:
        """Resolve a registered agent by name. / 按名称解析已注册 Agent。"""
        try:
            return self._agents[name]
        except KeyError as error:
            raise UnsupportedAgentError(name, available=list(self._agents)) from error

    def contains(self, name: str) -> bool:
        """Return whether an agent is registered under the given name.
        返回给定名称下是否注册了 Agent。"""
        return name in self._agents

    def names(self) -> tuple[str, ...]:
        """Return registered agent names in stable insertion order.
        按稳定插入顺序返回注册 Agent 名。"""
        return tuple(self._order)

    def supports(self, task: str) -> list[str]:
        """Return agent names that declare support for a given task.
        返回声明支持某任务的 Agent 名列表。"""
        return [
            name
            for name in self._order
            if name in self._agents and task in self._agents[name].supported_tasks
        ]

    def validate_task_coverage(self, tasks: set[str]) -> None:
        """Validate that every required task has at least one registered agent.
        校验每个必需任务至少有注册 Agent。"""
        covered: set[str] = set()
        for agent in self._agents.values():
            covered.update(agent.supported_tasks)
        missing = tasks - covered
        if missing:
            raise UnsupportedAgentError(
                f"tasks_missing_coverage:{sorted(missing)}",
                available=sorted(covered),
            )

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        return self.contains(name)

    def __repr__(self) -> str:
        return f"AgentRegistry({list(self._order)})"
