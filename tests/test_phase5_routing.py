from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from spacers_agent.clients.base import RequestMeta
from spacers_agent.clients.mock import MockVisionClient
from spacers_agent.counting import PointCountingOrchestrator
from spacers_agent.routing import (
    CallBudget,
    CallBudgetExceeded,
    CountingExpert,
    TaskRouter,
    attach_qwen_budget,
)
from spacers_agent.schemas import CountTargetSpec, TileCountResponse
from spacers_agent.settings import CountingSettings, QwenSettings


class TileClient:
    """Return one offline point response and preserve request metadata for assertions.
    返回一个离线点响应，并保留请求元数据供断言使用。
    """

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[RequestMeta] = []

    async def complete_json(
        self,
        *,
        messages: list[dict[str, Any]],
        response_model: type[TileCountResponse],
        request_meta: RequestMeta,
    ) -> TileCountResponse:
        self.calls.append(request_meta)
        return response_model.model_validate(self.response)


def _target() -> CountTargetSpec:
    return CountTargetSpec(
        canonical_label="building",
        inclusion_rule="Count each independent building.",
        exclusion_rule="Exclude shadows.",
    )


def _pipeline(client: TileClient, run_dir: Path) -> PointCountingOrchestrator:
    return PointCountingOrchestrator(
        client,
        counting=CountingSettings(tile_core_size=64, halo_size=0, model_max_side=64),
        qwen=QwenSettings(model="mock-qwen"),
        system_prompt="count points",
        run_dir=run_dir,
    )


def test_known_tasks_route_without_model_calls() -> None:
    router = TaskRouter()
    decision = router.route_known("fine_grained_counting", high_resolution=True)

    assert [expert.name for expert in decision.experts] == ["counting_expert", "spatial_expert"]
    assert [expert.weight for expert in decision.experts] == [0.5, 0.5]
    assert decision.requires_tiling
    assert decision.reason_codes == ["task_fine_grained_counting", "high_resolution"]


@pytest.mark.asyncio
async def test_unknown_task_uses_one_text_router_call_and_budget(tmp_path: Path) -> None:
    mock = MockVisionClient(
        {
            "sample:router": {
                "task": "general_vqa",
                "experts": [{"name": "general_vqa_expert", "weight": 1.0}],
                "requires_tiling": False,
                "reason_codes": ["free_form_question"],
            }
        }
    )
    budget = CallBudget(max_qwen_calls=1, max_deepseek_calls=0)
    decision = await TaskRouter(mock, router_prompt="route").route_unknown(
        "What is shown?", budget=budget, sample_id="sample", artifact_dir=tmp_path
    )

    assert decision.task == "general_vqa"
    assert budget.qwen_calls_used == 1
    assert len(mock.calls) == 1


@pytest.mark.asyncio
async def test_unknown_task_refuses_to_call_when_budget_is_exhausted() -> None:
    mock = MockVisionClient({})
    with pytest.raises(CallBudgetExceeded):
        await TaskRouter(mock, router_prompt="route").route_unknown(
            "unknown", budget=CallBudget(max_qwen_calls=0, max_deepseek_calls=0), sample_id="sample"
        )
    assert mock.calls == []


@pytest.mark.asyncio
async def test_counting_expert_uses_global_points_and_shared_budget(tmp_path: Path) -> None:
    client = TileClient(
        {
            "target": "building",
            "tile_id": "r000_c000",
            "points": [
                {
                    "local_id": "p1",
                    "x": 500,
                    "y": 500,
                    "confidence": 0.9,
                    "radius": 0,
                    "short_evidence": "roof",
                }
            ],
            "reported_count": 1,
        }
    )
    budget = CallBudget(max_qwen_calls=1, max_deepseek_calls=0)
    expert = CountingExpert(attach_qwen_budget(_pipeline(client, tmp_path), budget))

    result = await expert.answer(
        Image.new("RGB", (16, 16)), sample_id="sample", question="count buildings", target=_target()
    )

    assert result.complete
    assert result.counting_result.final_count == 1
    assert "1 accepted global instance points" in result.answer
    assert budget.qwen_calls_used == 1

    resumed = await expert.answer(
        Image.new("RGB", (16, 16)), sample_id="sample", question="count buildings", target=_target()
    )
    assert resumed.complete
    assert len(client.calls) == 1
