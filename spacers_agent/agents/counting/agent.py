"""CountingAgent — selects best backend, derives count from accepted points.
CountingAgent — 选择最优后端，从接受点导出计数。
"""

from __future__ import annotations

from spacers_agent.agents.base import Agent, AgentContext, AgentExecution, AgentName
from spacers_agent.agents.counting.backends.registry import BackendRegistry
from spacers_agent.agents.counting.backends.selector import BackendSelector
from spacers_agent.agents.counting.target_parser import CountTargetParser
from spacers_agent.imaging import read_normalized_image
from spacers_agent.schemas import ExpertResult
from spacers_agent.workflow import atomic_write_json


class CountingAgent:
    """Agent that selects a backend and enforces final_count == accepted points.
    选择后端并强制 final_count == accepted points 的 Agent。
    """

    name: AgentName = "counting_agent"
    supported_tasks: frozenset[str] = frozenset({"counting", "fine_grained_counting"})

    def __init__(self, client, prompts: dict[str, str], model: str, backend_registry: BackendRegistry) -> None:
        self._client = client
        self._prompts = prompts
        self._model = model
        self._selector = BackendSelector(backend_registry)

    async def run(self, sample, context: AgentContext) -> AgentExecution:
        sample_dir = context.artifact_dir
        settings = context.settings

        # 1. Parse target / 解析目标
        target = await CountTargetParser(self._client, self._prompts["target"], self._model).parse(
            sample.question, sample_id=sample.sample_id, artifact_dir=sample_dir,
        )

        # 2. Select backend / 选择后端
        selection = self._selector.select(target, sample)
        if selection is None:
            raise RuntimeError("No counting backend available")

        # 3. Read image + run backend / 读取图像 + 运行后端
        image = read_normalized_image(sample.images[0].path)
        from spacers_agent.agents.counting.backends.base import CountingRequest
        request = CountingRequest(sample=sample, image=image, target=target, artifact_dir=sample_dir)

        backend = None
        for b in self._selector._registry._backends:
            if b.name == selection.backend_name:
                backend = b
                break
        if backend is None:
            raise RuntimeError(f"Backend {selection.backend_name} not found in registry")

        result = await backend.count(request, context)

        # 4. Persist / 持久化
        atomic_write_json(sample_dir / "counting_result.json", result.model_dump(mode="json"))

        route = (
            f"CountingAgent.run -> BackendSelector.select -> {backend.name}"
            f" -> {type(self._client).__name__}.complete_json"
        )

        return AgentExecution(
            agent_name=self.name, payload=result, result_filename="counting_result.json",
            trace={
                "agent_class": "spacers_agent.agents.counting.agent.CountingAgent",
                "route": route, "backend": backend.name,
                "selection_reason": list(selection.reason_codes),
                "target": target.canonical_label, "status": result.status,
            },
        )
