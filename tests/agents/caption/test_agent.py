"""Offline unit tests for the Caption agent.

图像描述 Agent 离线单测：只支持 caption、完整 run 输出 AgentExecution/
AgentResult/agent_result.json、trace 含稳定 agent class/route/prompt version、
不支持 task 前置失败、无指标计算、无旧包依赖。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from agents.base import AgentContext, AgentExecution
from agents.caption import CaptionAgent
from agents.errors import AgentTaskMismatchError
from agents.schema import AgentResult
from data.schema import GroundTruth, ImageRef, UnifiedSample
from models.base import ModelCacheIdentity


class _FakeBudget:
    def __init__(self) -> None:
        self.qwen_calls = 0

    def reserve_qwen(self) -> None:
        self.qwen_calls += 1

    def reserve_deepseek(self) -> None:
        pass


class _RecordingClient:
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
        self.calls.append({"request_hash": request_meta.request_hash})
        return response_model.model_validate(
            {"agent_name": "caption_agent", "answer": "A coastal city.", "status": "completed"}
        )


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (1, 2, 3)).save(path, format="PNG")


def _sample(root: Path, *, task: str = "caption") -> UnifiedSample:
    _make_image(root / "img.png")
    return UnifiedSample(
        sample_id="s1",
        dataset="parity",
        split="test",
        task=task,  # type: ignore[arg-type]
        images=[ImageRef(image_id="i1", path="img.png", role="image")],
        question="Describe the scene.",
        ground_truth=GroundTruth(answers=["A coastal city."]),
    )


def _context(root: Path, budget: _FakeBudget | None = None) -> AgentContext:
    return AgentContext(
        artifact_dir=root / "artifacts",
        qwen_client=None,
        call_budget=budget or _FakeBudget(),
        data_root=root,
    )


def test_agent_identity_and_single_task() -> None:
    agent = CaptionAgent(_RecordingClient())
    assert agent.name == "caption_agent"
    assert agent.supported_tasks == frozenset({"caption"})


def test_run_returns_agent_execution(tmp_path: Path) -> None:
    client = _RecordingClient()
    agent = CaptionAgent(client)
    execution = asyncio.run(agent.run(_sample(tmp_path), _context(tmp_path)))
    assert isinstance(execution, AgentExecution)
    assert isinstance(execution.payload, AgentResult)
    assert execution.payload.answer == "A coastal city."
    assert execution.result_filename == "agent_result.json"
    assert len(client.calls) == 1


def test_trace_contains_stable_class_route_and_prompt_version(tmp_path: Path) -> None:
    agent = CaptionAgent(_RecordingClient())
    execution = asyncio.run(agent.run(_sample(tmp_path), _context(tmp_path)))
    assert execution.trace["agent_class"] == "agents.caption.agent.CaptionAgent"
    assert execution.trace["route"].startswith("CaptionAgent.run -> VisualAgentBase.run")
    assert execution.trace["prompt_version"] == "caption_v1"
    assert execution.trace["model"] == "fake-model"


def test_unsupported_task_fails_before_model_call(tmp_path: Path) -> None:
    client = _RecordingClient()
    budget = _FakeBudget()
    with pytest.raises(AgentTaskMismatchError):
        asyncio.run(CaptionAgent(client).run(_sample(tmp_path, task="general_vqa"), _context(tmp_path, budget)))
    assert budget.qwen_calls == 0
    assert client.calls == []


def test_no_metric_computation_in_agent() -> None:
    source = (Path(__file__).resolve().parents[3] / "agents" / "caption" / "agent.py").read_text(
        encoding="utf-8"
    )
    for token in ("metric", "iou", "IoU", "CIDEr", "cider", "bleu"):
        assert token not in source, token


def test_no_dataset_branch_and_no_legacy_import() -> None:
    source = (Path(__file__).resolve().parents[3] / "agents" / "caption" / "agent.py").read_text(
        encoding="utf-8"
    )
    assert "VRSBench" not in source
    assert "dataset" not in source
    assert "spacers_agent" not in source


def test_import_does_not_load_legacy_packages() -> None:
    import agents.caption  # noqa: F401

    for legacy in ("spacers_agent", "eval"):
        assert legacy not in sys.modules
