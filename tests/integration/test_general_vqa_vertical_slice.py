"""Vertical slice: UnifiedSample → TaskRouter → GeneralVQAAgent → AgentExecution.

垂直切片：UnifiedSample → TaskRouter → GeneralVQAAgent → AgentExecution。
使用 fake VisionLanguageClient 离线打通最小新链；不接入 DatasetRunner，
不导入任何旧包。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from PIL import Image

from agents.base import AgentContext, AgentExecution
from agents.general_vqa import GeneralVQAAgent
from agents.schema import AgentResult
from data.schema import GroundTruth, ImageRef, TaskNormalization, UnifiedSample
from models.base import ModelCacheIdentity
from routing.router import TaskRouter
from routing.schema import RoutingDecision


class _FakeBudget:
    def reserve_qwen(self) -> None:
        pass

    def reserve_deepseek(self) -> None:
        pass


class _FakeClient:
    """Minimal VisionLanguageClient with a stable cache identity.
    带稳定缓存身份的最小 VisionLanguageClient。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="fake-model",
            generation={"temperature": 0.0, "do_sample": False, "max_tokens": 128},
            client_version="1",
        )

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls.append({"messages": messages, "request_hash": request_meta.request_hash})
        return response_model.model_validate(
            {"agent_name": "general_vqa_agent", "answer": "ok", "status": "completed"}
        )


def _make_image(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (10, 20, 30)).save(root / "img.png", format="PNG")


def _sample(
    root: Path,
    *,
    task: str,
    constraints: dict[str, Any] | None = None,
) -> UnifiedSample:
    _make_image(root)
    normalization = None
    if constraints is not None:
        normalization = TaskNormalization(
            source_task=task,
            normalized_task=task,  # type: ignore[arg-type]
            normalizer="test", version="1",
            answer_constraints=constraints,
        )
    return UnifiedSample(
        sample_id="slice-1",
        dataset="parity",
        split="test",
        task=task,  # type: ignore[arg-type]
        images=[ImageRef(image_id="i1", path="img.png", role="image")],
        question="Describe the scene.",
        ground_truth=GroundTruth(answers=["ok"]),
        normalization=normalization,
    )


def _context(root: Path) -> AgentContext:
    return AgentContext(
        artifact_dir=root / "artifacts",
        qwen_client=None,
        call_budget=_FakeBudget(),
        data_root=root,
    )


def _run_vertical_slice(task: str, root: Path, constraints: dict[str, Any] | None = None):
    """Run the full minimal chain and return (decision, execution, client).
    运行完整最小链并返回（决策、执行、client）。"""
    sample = _sample(root, task=task, constraints=constraints)
    router = TaskRouter()
    decision = router.route(task)
    client = _FakeClient()
    agent = GeneralVQAAgent(client)
    execution = asyncio.run(agent.run(sample, _context(root)))
    return sample, decision, execution, client


def test_vertical_slice_general_vqa(tmp_path: Path) -> None:
    sample, decision, execution, client = _run_vertical_slice("general_vqa", tmp_path)
    assert isinstance(decision, RoutingDecision)
    assert decision.primary_agent == "general_vqa_agent"
    assert execution.agent_name == decision.primary_agent
    assert isinstance(execution.payload, AgentResult)
    assert execution.payload.answer == "ok"
    assert execution.result_filename == "agent_result.json"
    assert len(client.calls) == 1


def test_vertical_slice_scene_classification(tmp_path: Path) -> None:
    sample, decision, execution, client = _run_vertical_slice("scene_classification", tmp_path)
    assert decision.primary_agent == "general_vqa_agent"
    assert execution.payload.agent_name == "general_vqa_agent"
    assert len(client.calls) == 1


def test_vertical_slice_multiple_choice(tmp_path: Path) -> None:
    constraints = {"type": "closed_vocabulary", "values": ["A", "B", "C", "D"]}
    sample, decision, execution, client = _run_vertical_slice(
        "multiple_choice_vqa", tmp_path, constraints
    )
    assert decision.primary_agent == "general_vqa_agent"
    # The choice constraints travel through the agent payload.
    # 选项约束经 Agent 载荷传递。
    messages = client.calls[0]["messages"]
    payload = json.loads(messages[1]["content"][-1]["text"])
    assert payload["choices"] == ["A", "B", "C", "D"]
    assert payload["allow_multiple"] is False
    assert execution.result_filename == "agent_result.json"


def test_vertical_slice_router_decision_drives_agent(tmp_path: Path) -> None:
    """The router decision's primary agent must be the agent that runs.
    路由决策的 primary agent 必须是实际运行的 Agent。"""
    for task in ("general_vqa", "scene_classification", "multiple_choice_vqa"):
        constraints = None
        if task == "multiple_choice_vqa":
            constraints = {"type": "closed_vocabulary", "values": ["A", "B"]}
        sample, decision, execution, client = _run_vertical_slice(
            task, tmp_path / task, constraints
        )
        assert decision.primary_agent == execution.agent_name
        assert decision.primary_agent == "general_vqa_agent"


def test_vertical_slice_has_no_legacy_imports(tmp_path: Path) -> None:
    source = (Path(__file__).resolve().parents[2] / "agents" / "general_vqa" / "agent.py").read_text(
        encoding="utf-8"
    )
    assert "spacers_agent" not in source
    assert "VRSBench" not in source
