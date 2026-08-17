"""Offline tests for direct and v2 evidence-backed GeneralVQAAgent paths.

通用 VQA Agent 离线测试：直接路径继续只做一次模型调用，证据路径只接受
VisualTaskPlan 与 MaterializedVisualView，不再构造旧视觉计划或候选回退。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from agents.base import AgentContext, AgentExecution, VisualPlanBindings
from agents.errors import AgentTaskMismatchError
from agents.general_vqa import GeneralVQAAgent
from agents.general_vqa.evidence.executor import EvidenceExecution
from agents.general_vqa.evidence.schema import (
    LayerStateRecord,
    RoiEvidenceRecord,
    VqaEvidenceBundle,
)
from agents.schema import AgentResult, MaterializedVisualView, VisualTaskPlan
from data.schema import GroundTruth, ImageRef, TaskNormalization, UnifiedSample
from models.base import ModelCacheIdentity


class _FakeBudget:
    def __init__(self) -> None:
        self.qwen_calls = 0

    def reserve_qwen(self) -> None:
        self.qwen_calls += 1

    def reserve_deepseek(self) -> None:
        pass


class _RecordingClient:
    """Return a schema-valid answer and record every final-Qwen call.
    返回通过 schema 校验的答案，并记录每次最终 Qwen 调用。"""

    def __init__(self, answer: str = "yes") -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="fake-model",
            generation={"temperature": 0.0},
            client_version="test",
        )

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls.append({"messages": messages, "request_meta": request_meta})
        return response_model.model_validate(
            {"agent_name": "general_vqa_agent", "answer": self.answer, "status": "completed"}
        )


def _sample(root: Path, *, task: str = "general_vqa") -> UnifiedSample:
    Image.new("RGB", (8, 6), (1, 2, 3)).save(root / "img.png", format="PNG")
    normalization = None
    if task == "multiple_choice_vqa":
        normalization = TaskNormalization(
            source_task=task,
            normalized_task=task,  # type: ignore[arg-type]
            normalizer="test",
            version="1",
            answer_constraints={"choices": ["yes", "no"]},
        )
    return UnifiedSample(
        sample_id="s1",
        dataset="demo",
        split="test",
        task=task,  # type: ignore[arg-type]
        images=[ImageRef(image_id="img1", path="img.png", role="image")],
        question="What is in the image?",
        ground_truth=GroundTruth(answers=["yes"]),
        normalization=normalization,
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
        version="visual-task-plan-v4",
        task="general_vqa",
        needs_visual_assistance=True,
        object_categories=["small-vehicle"],
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


class _FakeVqaEvidenceService:
    def __init__(self, bundle: VqaEvidenceBundle) -> None:
        self.bundle = bundle
        self.calls: list[dict[str, Any]] = []

    def execute(self, plan, images, *, fallback_image_id, materialized_views):
        self.calls.append(
            {
                "plan": plan,
                "images": images,
                "fallback_image_id": fallback_image_id,
                "materialized_views": materialized_views,
            }
        )
        return EvidenceExecution(
            bundle=self.bundle,
            layer_states=(),
            outcomes=(),
            masks={},
        )


def _bundle() -> VqaEvidenceBundle:
    roi = RoiEvidenceRecord(
        roi_id="full",
        image_id="img1",
        source_size=(8, 6),
        core_xyxy=(0, 0, 8, 6),
        expanded_xyxy=(0, 0, 8, 6),
        crop_size=(8, 6),
    )
    return VqaEvidenceBundle(
        catalog_version="test-catalog-v1",
        rois=[roi],
        missing_leaves=["small_vehicle"],
        leaf_states={"small_vehicle": "missing"},
    )


def test_agent_identity_and_direct_path(tmp_path: Path) -> None:
    client = _RecordingClient()
    agent = GeneralVQAAgent(client)
    execution = asyncio.run(agent.run(_sample(tmp_path), _context(tmp_path, client)))
    assert agent.name == "general_vqa_agent"
    assert "general_vqa" in agent.supported_tasks
    assert isinstance(execution, AgentExecution)
    assert isinstance(execution.payload, AgentResult)
    assert execution.result_filename == "agent_result.json"
    assert len(client.calls) == 1


def test_multiple_choice_constraints_remain_on_direct_path(tmp_path: Path) -> None:
    client = _RecordingClient(answer="yes")
    execution = asyncio.run(
        GeneralVQAAgent(client).run(
            _sample(tmp_path, task="multiple_choice_vqa"),
            _context(tmp_path, client),
        )
    )
    payload = json.loads(client.calls[0]["messages"][1]["content"][-1]["text"])
    assert payload["choices"] == ["yes", "no"]
    assert execution.payload.answer == "yes"


def test_unsupported_task_fails_before_model_call(tmp_path: Path) -> None:
    client = _RecordingClient()
    with pytest.raises(AgentTaskMismatchError):
        asyncio.run(
            GeneralVQAAgent(client).run(
                _sample(tmp_path, task="grounding"),
                _context(tmp_path, client),
            )
        )
    assert client.calls == []


def test_v2_evidence_path_consumes_plan_and_materialized_views(tmp_path: Path) -> None:
    client = _RecordingClient()
    service = _FakeVqaEvidenceService(_bundle())
    plan = _plan()
    views = (_view(),)
    execution = asyncio.run(
        GeneralVQAAgent(client).run(
            _sample(tmp_path),
            _context(
                tmp_path,
                client,
                plan=plan,
                views=views,
                bindings=VisualPlanBindings(vqa_evidence=service),
            ),
        )
    )
    assert len(service.calls) == 1
    assert service.calls[0]["plan"] is plan
    assert service.calls[0]["materialized_views"] == views
    assert execution.additional_results["vqa_evidence.json"]["rois"][0]["expanded_xyxy"] == [0, 0, 8, 6]
    assert len(client.calls) == 1


def test_v2_assistance_is_rejected_for_non_general_vqa(tmp_path: Path) -> None:
    client = _RecordingClient()
    plan = _plan()
    with pytest.raises(Exception, match="visual_assistance_forbidden"):
        asyncio.run(
            GeneralVQAAgent(client).run(
                _sample(tmp_path, task="scene_classification"),
                _context(tmp_path, client, plan=plan, views=(_view(),)),
            )
        )
    assert client.calls == []


def test_agent_source_has_no_legacy_package_or_v1_plan_contract() -> None:
    source = (Path(__file__).resolve().parents[3] / "agents" / "general_vqa" / "agent.py").read_text(
        encoding="utf-8"
    )
    assert "spacers_agent" not in source
    assert "visual_plan" not in source
    assert "joint_visual_plan" not in source


def test_import_does_not_load_legacy_packages() -> None:
    import agents.general_vqa  # noqa: F401

    assert "spacers_agent" not in sys.modules
    assert "eval" not in sys.modules
