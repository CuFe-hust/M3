"""TaskRouter — fully deterministic, synchronous task-to-policy routing.

TaskRouter — 完全确定性、同步的 task 到策略路由。Router 只将规范化
TaskName 映射为固定策略；不读取问题、不调用模型、不消费 budget。
"""

from __future__ import annotations

from routing.policies import policy_for
from routing.schema import RoutingDecision, SampleCapabilities


class TaskRouter:
    """Map normalized task names to fixed policies; never reads the question,
    never calls a model. 将规范化 task 名映射到固定策略；绝不读问题、绝不
    调用模型。"""

    def route(
        self,
        task: str,
        *,
        capabilities: SampleCapabilities | None = None,
    ) -> RoutingDecision:
        """Synchronous deterministic routing for one normalized task. Unknown
        tasks raise KeyError — there is no general_vqa guessing.
        单条规范化 task 的同步确定性路由。未知 task 抛 KeyError——不做
        general_vqa 猜测。"""
        policy = policy_for(task)
        reason_codes = [f"task_{policy.task}"]
        if capabilities is not None and capabilities.high_resolution:
            reason_codes.append("high_resolution")
        return RoutingDecision(
            task=policy.task,
            primary_agent=policy.primary_agent,
            fallback_agents=list(policy.fallback_agents),
            execution_mode=policy.execution_mode,
            requires_tiling=policy.requires_tiling,
            reason_codes=reason_codes,
        )
