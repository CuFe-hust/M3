"""Contract tests for the spatial agent.

空间 Agent 契约测试：普通/grid prompt 选择、候选复核接入、通用几何后处理、
trace 记录 review 使用与 prompt version、无数据集分支、不重复 repair。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from agents.base import AgentContext, AgentExecution
from agents.schema import AgentResult, VisualEvidence
from agents.spatial import SpatialAgent
from agents.spatial.candidate_review import SpatialCandidateReviewResult
from agents.visual_base import PromptBinding
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
    def __init__(self, first_evidence_count: int = 2) -> None:
        self.calls: list[Any] = []
        self._first_evidence_count = first_evidence_count

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="fake-model",
            generation={"temperature": 0.0, "do_sample": False, "max_tokens": 128},
            client_version="1",
        )

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls.append(request_meta.request_id)
        if response_model is SpatialCandidateReviewResult:
            return response_model.model_validate(
                {"boxes": [], "complete": True}
            )
        items = [
            {"label": "small-vehicle", "box": [100, 700, 200, 800], "confidence": 0.9},
            {"label": "small-vehicle", "box": [100, 60, 200, 120], "confidence": 0.9},
        ]
        return response_model.model_validate(
            {
                "agent_name": "spatial_agent",
                "answer": "small-vehicle",
                "evidence_items": items[: self._first_evidence_count],
                "status": "completed",
            }
        )


def _sample(root: Path, *, spatial_query: dict[str, Any] | None = None) -> UnifiedSample:
    return UnifiedSample(
        sample_id="s1",
        dataset="parity",
        split="test",
        task="spatial_relation",
        images=[ImageRef(image_id="i1", path="img.png", role="image")],
        question="Which vehicle is at the top?",
        ground_truth=GroundTruth(answers=["small-vehicle"]),
        metadata={"spatial_query": spatial_query} if spatial_query else {},
    )


def _image(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100), (1, 2, 3)).save(root / "img.png", format="PNG")


def _agent(client: _RecordingClient, **overrides) -> SpatialAgent:
    values = dict(
        prompt=PromptBinding(text="Answer the spatial question.", version="spatial-v1"),
        grid_prompt=PromptBinding(text="Locate in the grid.", version="spatial-grid-v1"),
        review_prompt="Enumerate candidates.",
        review_prompt_version="candidate-review-v1",
    )
    values.update(overrides)
    return SpatialAgent(client, **values)


def _context(root: Path, budget: _FakeBudget | None = None) -> AgentContext:
    return AgentContext(
        artifact_dir=root / "artifacts",
        qwen_client=None,
        call_budget=budget or _FakeBudget(),
        data_root=root,
    )


def _spatial_query(operation: str = "extreme_category", **extra) -> dict[str, Any]:
    values = {
        "operation": operation,
        "target_label": "small-vehicle",
        "target_hint": "top",
    }
    values.update(extra)
    return values


def test_agent_identity_and_tasks() -> None:
    agent = _agent(_RecordingClient())
    assert agent.name == "spatial_agent"
    assert agent.supported_tasks == frozenset({"spatial_relation"})


def test_run_reviews_and_applies_geometry(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _image(root)
    client = _RecordingClient()
    sample = _sample(root, spatial_query=_spatial_query())
    execution = asyncio.run(_agent(client).run(sample, _context(root)))
    assert isinstance(execution, AgentExecution)
    assert isinstance(execution.payload, AgentResult)
    assert execution.result_filename == "agent_result.json"
    # Geometry override: top-most selected. / 几何覆盖：选择最上方。
    assert execution.payload.answer == "small-vehicle"
    assert execution.payload.geometry["answer_source"] == "deterministic_geometry"
    assert execution.trace["candidate_review_used"] is True
    assert execution.trace["prompt_version"] == "spatial-v1"


def test_grid_prompt_selected_for_grid_position(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _image(root)
    client = _RecordingClient(first_evidence_count=1)
    sample = _sample(root, spatial_query=_spatial_query(operation="grid_position"))
    execution = asyncio.run(_agent(client).run(sample, _context(root)))
    assert execution.trace["prompt_version"] == "spatial-grid-v1"
    # Single grid candidate → deterministic position. / 单一网格候选 → 确定性位置。
    assert execution.payload.geometry["rule"] == "three_by_three_box_center"


def test_no_spatial_query_skips_geometry(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _image(root)
    client = _RecordingClient()
    execution = asyncio.run(_agent(client).run(_sample(root), _context(root)))
    # No spec → no review, no geometry override. / 无 spec → 无复核、无几何覆盖。
    assert execution.trace["candidate_review_used"] is False
    assert execution.payload.answer == "small-vehicle"
    assert execution.payload.geometry.get("answer_source") is None


def test_review_never_duplicates_geometry_repair(tmp_path: Path) -> None:
    """The final geometry post-processing runs exactly once on the reviewed
    result — no duplicated repair. 最终几何后处理在复核结果上恰好运行一次——
    无重复修复。"""
    root = tmp_path / "data"
    _image(root)
    client = _RecordingClient()
    sample = _sample(root, spatial_query=_spatial_query())
    execution = asyncio.run(_agent(client).run(sample, _context(root)))
    geometry = execution.payload.geometry
    assert geometry["version"] == "evidence-geometry-v1"
    assert geometry["answer_source"] == "deterministic_geometry"
    # The review audit survives next to the geometry audit. / 复核审计与几何审计并存。
    assert geometry["candidate_review_used"] is True


def test_trace_route_records_review(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _image(root)
    client = _RecordingClient()
    sample = _sample(root, spatial_query=_spatial_query())
    execution = asyncio.run(_agent(client).run(sample, _context(root)))
    assert "SpatialCandidateReviewer.review" in execution.trace["route"]
    assert execution.trace["agent_class"] == "agents.spatial.agent.SpatialAgent"


def test_agent_has_no_dataset_branch_or_evaluation(tmp_path: Path) -> None:
    source = (Path(__file__).resolve().parents[3] / "agents" / "spatial" / "agent.py").read_text(
        encoding="utf-8"
    )
    assert "VRSBench" not in source
    assert "vrsbench" not in source
    assert "spacers_agent" not in source
    assert "evaluate" not in source
    assert "metric" not in source
