"""Agent Registry — maps names to Agent implementations.
Agent 注册表 — 将名称映射到 Agent 实现。

The registry is a pure data structure. It never routes, reads config,
creates agents, calls models, or writes to disk.
注册表是纯数据结构。它不路由、不读配置、不创建 Agent、不调用模型、不写盘。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spacers_agent.agents.base import Agent, AgentName
from spacers_agent.agents.errors import DuplicateAgentError, UnsupportedAgentError

if TYPE_CHECKING:
    pass


class AgentRegistry:
    """Map agent names to implementations; never loads models or reads weights.
    映射 Agent 名到实现；绝不加载模型或读取权重。

    Registry 不负责实例化；Bootstrap 负责构造 Agent 并注册。
    The registry does not instantiate agents; bootstrap creates and registers them.
    """

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}
        # Stable insertion order for deterministic output / 确定性输出的稳定插入顺序
        self._order: list[str] = []

    # ── registration / 注册 ──────────────────────────────────────────────

    def register(self, agent: Agent) -> None:
        """Register an agent. Raises DuplicateAgentError on name conflict.
        注册 Agent。名称冲突时抛出 DuplicateAgentError。
        """

        # Protocol check — must come BEFORE any attribute access
        # Protocol 检查 — 必须在任何属性访问之前
        if not isinstance(agent, Agent):
            raise TypeError(
                f"Object {agent!r} does not satisfy the Agent Protocol. "
                f"Expected attributes: name (AgentName), supported_tasks (frozenset[str]), "
                f"async run(sample, context) -> AgentExecution."
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

    # ── lookup / 查找 ────────────────────────────────────────────────────

    def get(self, name: str) -> Agent:
        """Resolve a registered agent by name. / 按名称解析已注册 Agent。

        Raises UnsupportedAgentError when the name is not registered.
        名称未注册时抛出 UnsupportedAgentError。
        """
        try:
            return self._agents[name]
        except KeyError as error:
            raise UnsupportedAgentError(name, available=list(self._agents)) from error

    def contains(self, name: str) -> bool:
        """Return whether an agent is registered under the given name.
        返回给定名称下是否注册了 Agent。
        """
        return name in self._agents

    # ── introspection / 内省 ─────────────────────────────────────────────

    def names(self) -> tuple[str, ...]:
        """Return registered agent names in stable insertion order.
        按稳定插入顺序返回注册 Agent 名。

        The order is deterministic — it reflects the order in which agents
        were registered, which is controlled by bootstrap.
        顺序是确定性的 — 反映 Agent 注册顺序，由 bootstrap 控制。
        """
        return tuple(self._order)

    def supports(self, task: str) -> list[str]:
        """Return agent names that declare support for a given task.
        返回声明支持某任务的 Agent 名列表。
        """
        return [
            name
            for name in self._order
            if name in self._agents and task in self._agents[name].supported_tasks
        ]

    # ── validation / 校验 ────────────────────────────────────────────────

    def validate_task_coverage(self, tasks: set[str]) -> None:
        """Validate that every required task has at least one registered agent.
        校验每个必需任务至少有注册 Agent。

        Raises UnsupportedAgentError if any task lacks coverage.
        若任何任务缺乏覆盖则抛出 UnsupportedAgentError。
        """
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
