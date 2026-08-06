"""Offline unit tests for the Grounding agent.

定位 Agent 离线单测：只支持 grounding、完整 run 输出 AgentExecution/
AgentResult、证据使用统一 0..999 坐标、trace 含稳定 agent class/route/
prompt version、不支持 task 前置失败、无指标计算、无旧包依赖。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from agents.base import AgentContext, AgentExecution
from agents.errors import AgentTaskMismatchError
from agents.grounding import GroundingAgent
from agents.schema import AgentResult, VisualEvidence
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
            {
                "agent_name": "grounding_agent",
                "answer": "The building.",
                "evidence_items": [
                    {"label": "building", "box": [120, 80, 340, 260], "confidence": 0.9}
                ],
                "status": "completed",
            }
        )


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (1, 2, 3)).save(path, format="PNG")


def _sample(root: Path, *, task: str = "grounding") -> UnifiedSample:
    _make_image(root / "img.png")
    return UnifiedSample(
        sample_id="s1",
        dataset="parity",
        split="test",
        task=task,  # type: ignore[arg-type]
        images=[ImageRef(image_id="i1", path="img.png", role="image")],
        question="Locate the building.",
        ground_truth=GroundTruth(answers=["building"], boxes=[[120, 80, 340, 260]]),
    )


def _context(root: Path, budget: _FakeBudget | None = None) -> AgentContext:
    return AgentContext(
        artifact_dir=root / "artifacts",
        qwen_client=None,
        call_budget=budget or _FakeBudget(),
        data_root=root,
    )


def test_agent_identity_and_single_task() -> None:
    agent = GroundingAgent(_RecordingClient())
    assert agent.name == "grounding_agent"
    assert agent.supported_tasks == frozenset({"grounding"})


def test_run_returns_agent_execution(tmp_path: Path) -> None:
    client = _RecordingClient()
    agent = GroundingAgent(client)
    execution = asyncio.run(agent.run(_sample(tmp_path), _context(tmp_path)))
    assert isinstance(execution, AgentExecution)
    assert isinstance(execution.payload, AgentResult)
    assert execution.result_filename == "agent_result.json"
    assert len(client.calls) == 1


def test_evidence_uses_unified_0_999_coordinates(tmp_path: Path) -> None:
    """Grounding evidence must live in the unified 0..999 normalized frame.
    定位证据必须处于统一 0..999 归一化坐标系。"""
    agent = GroundingAgent(_RecordingClient())
    execution = asyncio.run(agent.run(_sample(tmp_path), _context(tmp_path)))
    payload = execution.payload
    assert isinstance(payload, AgentResult)
    assert len(payload.evidence_items) == 1
    evidence = payload.evidence_items[0]
    assert isinstance(evidence, VisualEvidence)
    assert evidence.box == [120, 80, 340, 260]
    assert all(0 <= value <= 999 for value in evidence.box)
    assert evidence.coordinate_frame == "normalized_0_999_top_left"
    # Labeled evidence boxes are retained in the canonical box list.
    # 带标签证据框保留在统一框列表中。
    assert payload.boxes == [[120.0, 80.0, 340.0, 260.0]]


def test_trace_contains_stable_class_route_and_prompt_version(tmp_path: Path) -> None:
    agent = GroundingAgent(_RecordingClient())
    execution = asyncio.run(agent.run(_sample(tmp_path), _context(tmp_path)))
    assert execution.trace["agent_class"] == "agents.grounding.agent.GroundingAgent"
    assert execution.trace["route"].startswith("GroundingAgent.run -> VisualAgentBase.run")
    assert execution.trace["prompt_version"] == "general_vqa_v2"
    assert execution.trace["model"] == "fake-model"


def test_unsupported_task_fails_before_model_call(tmp_path: Path) -> None:
    client = _RecordingClient()
    budget = _FakeBudget()
    with pytest.raises(AgentTaskMismatchError):
        asyncio.run(
            GroundingAgent(client).run(_sample(tmp_path, task="spatial_relation"), _context(tmp_path, budget))
        )
    assert budget.qwen_calls == 0
    assert client.calls == []


def test_no_metric_computation_in_agent() -> None:
    source = (Path(__file__).resolve().parents[3] / "agents" / "grounding" / "agent.py").read_text(
        encoding="utf-8"
    )
    for token in ("metric", "iou", "IoU", "cider", "CIDEr"):
        assert token not in source, token


def test_no_dataset_branch_and_no_legacy_import() -> None:
    source = (Path(__file__).resolve().parents[3] / "agents" / "grounding" / "agent.py").read_text(
        encoding="utf-8"
    )
    assert "VRSBench" not in source
    assert "dataset" not in source
    assert "spacers_agent" not in source


def test_import_does_not_load_legacy_packages() -> None:
    import agents.grounding  # noqa: F401

    for legacy in ("spacers_agent", "eval"):
        assert legacy not in sys.modules
