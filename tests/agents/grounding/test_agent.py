"""Offline tests for direct and v2 evidence-backed GroundingAgent paths.

定位 Agent 离线测试：直接路径保持一次模型调用，证据路径只消费 v2 计划和
物化视图，并把确定性整图框写回统一 AgentResult。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from agents.base import AgentContext, AgentExecution, VisualPlanBindings
from agents.errors import AgentTaskMismatchError
from agents.grounding import GroundingAgent
from agents.grounding.evidence import (
    GroundingEvidenceBundle,
    GroundingEvidenceResult,
    GroundingRoiRecord,
    WholeImageBox,
)
from agents.schema import AgentResult, MaterializedVisualView, VisualEvidence, VisualTaskPlan
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
        return ModelCacheIdentity(model="fake-model", generation={"temperature": 0.0}, client_version="test")

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls.append({"messages": messages, "request_meta": request_meta})
        return response_model.model_validate(
            {
                "agent_name": "grounding_agent",
                "answer": "building",
                "evidence_items": [
                    {"label": "building", "box": [120, 80, 340, 260], "confidence": 0.9}
                ],
                "status": "completed",
            }
        )


def _sample(root: Path, *, task: str = "grounding") -> UnifiedSample:
    Image.new("RGB", (8, 6), (1, 2, 3)).save(root / "img.png", format="PNG")
    return UnifiedSample(
        sample_id="s1",
        dataset="demo",
        split="test",
        task=task,  # type: ignore[arg-type]
        images=[ImageRef(image_id="img1", path="img.png", role="image")],
        question="Locate the building.",
        ground_truth=GroundTruth(answers=["building"], boxes=[[120, 80, 340, 260]]),
    )


def _context(
    root: Path,
    client: _RecordingClient,
    *,
    plan: VisualTaskPlan | None = None,
    views: tuple[MaterializedVisualView, ...] = (),
    bindings: VisualPlanBindings | None = None,
) -> AgentContext:
    return AgentContext(
        artifact_dir=root / "artifacts",
        qwen_client=client,
        call_budget=_FakeBudget(),
        data_root=root,
        visual_task_plan=plan,
        visual_views=views,
        visual_bindings=bindings,
    )


def _plan() -> VisualTaskPlan:
    return VisualTaskPlan(
        version="visual-task-plan-v2",
        task="grounding",
        needs_visual_assistance=True,
        object_categories=["vehicle"],
        confidence=0.9,
        reason_codes=["test"],
    )


def _view() -> MaterializedVisualView:
    return MaterializedVisualView(
        image_id="img1",
        view_mode="full_image",
        source_size=(8, 6),
        crop_xyxy=(0, 0, 8, 6),
        crop_size=(8, 6),
    )


class _FakeGroundingService:
    def __init__(self, result: GroundingEvidenceResult) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def run(self, plan, sample, images, *, fallback_image_id, artifact_dir, budget, materialized_views):
        self.calls.append(
            {
                "plan": plan,
                "sample": sample,
                "images": images,
                "materialized_views": materialized_views,
            }
        )
        return self.result


def _result() -> GroundingEvidenceResult:
    roi = GroundingRoiRecord(
        roi_id="full",
        image_id="img1",
        source_size=(8, 6),
        core_xyxy=(0, 0, 8, 6),
        expanded_xyxy=(0, 0, 8, 6),
        crop_size=(8, 6),
    )
    bundle = GroundingEvidenceBundle(
        catalog_version="test-catalog-v1",
        rois=[roi],
        leaf_states={"building_outline": "hit"},
        selected_box_ids=["box-1"],
    )
    return GroundingEvidenceResult(
        bundle=bundle,
        whole_image_boxes=[WholeImageBox(label="building", box=(120, 80, 340, 260))],
    )


def test_direct_grounding_path_returns_unified_result(tmp_path: Path) -> None:
    client = _RecordingClient()
    execution = asyncio.run(
        GroundingAgent(client).run(_sample(tmp_path), _context(tmp_path, client))
    )
    assert isinstance(execution, AgentExecution)
    assert isinstance(execution.payload, AgentResult)
    assert execution.payload.evidence_items[0].box == [120, 80, 340, 260]
    assert execution.payload.boxes == [[120, 80, 340, 260]]
    assert len(client.calls) == 1


def test_v2_grounding_path_consumes_plan_and_views(tmp_path: Path) -> None:
    client = _RecordingClient()
    service = _FakeGroundingService(_result())
    plan = _plan()
    views = (_view(),)
    execution = asyncio.run(
        GroundingAgent(client).run(
            _sample(tmp_path),
            _context(
                tmp_path,
                client,
                plan=plan,
                views=views,
                bindings=VisualPlanBindings(grounding_evidence=service),
            ),
        )
    )
    assert len(service.calls) == 1
    assert service.calls[0]["plan"] is plan
    assert service.calls[0]["materialized_views"] == views
    assert execution.payload.boxes == [[120, 80, 340, 260]]
    assert execution.additional_results["grounding_evidence.json"]["rois"][0]["expanded_xyxy"] == [0, 0, 8, 6]
    assert client.calls == []


def test_unsupported_task_fails_before_model_call(tmp_path: Path) -> None:
    client = _RecordingClient()
    with pytest.raises(AgentTaskMismatchError):
        asyncio.run(
            GroundingAgent(client).run(
                _sample(tmp_path, task="general_vqa"),
                _context(tmp_path, client),
            )
        )
    assert client.calls == []


def test_agent_source_has_no_v1_plan_contract_or_legacy_import() -> None:
    source = (Path(__file__).resolve().parents[3] / "agents" / "grounding" / "agent.py").read_text(
        encoding="utf-8"
    )
    assert "spacers_agent" not in source
    assert "visual_plan" not in source
    assert "joint_visual_plan" not in source


def test_import_does_not_load_legacy_packages() -> None:
    import agents.grounding  # noqa: F401

    assert "spacers_agent" not in sys.modules
    assert "eval" not in sys.modules
