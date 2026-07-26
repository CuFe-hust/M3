"""CountingAgent selects the explicit backend and publishes accepted-point truth.
CountingAgent 选择明确后端并发布以接受点为准的事实。
"""

from __future__ import annotations

from spacers_agent.agents.base import AgentContext, AgentExecution, AgentName
from spacers_agent.agents.counting.backends.base import CountingRequest
from spacers_agent.agents.counting.backends.registry import BackendRegistry
from spacers_agent.agents.counting.backends.selector import BackendSelector, is_vrsbench_quantity
from spacers_agent.agents.counting.target_parser import CountTargetParser
from spacers_agent.imaging import read_normalized_image
from spacers_agent.schemas import CountTargetSpec
from spacers_agent.vqa_geometry import vrsbench_count_target


class CountingAgent:
    """Run native or VRSBench counting through one selected backend.
    通过一个选定后端运行原生或 VRSBench 计数。
    """

    name: AgentName = "counting_agent"
    supported_tasks: frozenset[str] = frozenset({"counting", "fine_grained_counting"})

    def __init__(
        self,
        client,
        prompts: dict[str, str],
        model: str,
        backend_registry: BackendRegistry,
    ) -> None:
        self._client = client
        self._target_prompt = prompts["target"]
        self._model = model
        self._selector = BackendSelector(backend_registry)

    async def run(self, sample, context: AgentContext) -> AgentExecution:
        """Select the target and backend without hidden artifact writes.
        在不进行隐藏产物写入的情况下选择目标与后端。
        """

        override = sample.metadata.get("count_target_spec")
        if override is not None:
            target = CountTargetSpec.model_validate(override)
        elif is_vrsbench_quantity(sample):
            target = vrsbench_count_target(sample.question)
        else:
            context.call_budget.reserve_qwen()
            target = await CountTargetParser(
                self._client,
                self._target_prompt,
                self._model,
            ).parse(
                sample.question,
                sample_id=sample.sample_id,
                artifact_dir=context.artifact_dir,
            )

        selection = self._selector.select(target, sample)
        if selection is None:
            raise RuntimeError(
                f"No counting backend available for task={sample.task!r}, target={target.canonical_label!r}"
            )
        backend = self._selector.backend(selection)
        request = CountingRequest(
            sample=sample,
            image=read_normalized_image(sample.images[0].path),
            target=target,
            artifact_dir=context.artifact_dir,
        )
        outcome = await backend.count(request, context)

        trace: dict[str, object] = {
            "agent_class": "spacers_agent.agents.counting.agent.CountingAgent",
            "entrypoint": "run",
            "route": (
                f"CountingAgent.run -> BackendSelector.select -> {backend.name} -> "
                f"{type(self._client).__name__}.complete_json"
            ),
            "backend": backend.name,
            "selection_reason": list(selection.reason_codes),
            "target": target.canonical_label,
            "status": outcome.counting.status,
        }
        if outcome.trace:
            trace.update(outcome.trace)

        if outcome.expert_result is not None:
            return AgentExecution(
                agent_name=self.name,
                payload=outcome.expert_result,
                result_filename="expert_result.json",
                additional_results={"counting_result.json": outcome.counting},
                trace=trace,
            )
        return AgentExecution(
            agent_name=self.name,
            payload=outcome.counting,
            result_filename="counting_result.json",
            trace=trace,
        )
