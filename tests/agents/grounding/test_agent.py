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

from agents.base import AgentContext, AgentExecution, VisualPlanBindings
from agents.errors import AgentExecutionError, AgentTaskMismatchError
from agents.grounding import GroundingAgent
from agents.grounding.evidence import (
    GroundingEvidenceBundle,
    GroundingEvidenceError,
    GroundingEvidenceResult,
    WholeImageBox,
)
from agents.schema import (
    AgentResult,
    FirstQwenVisualPlan,
    ObjectEvidenceRequest,
    RoiPlan,
    VisualEvidence,
)
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
    assert execution.trace["prompt_version"] == "general_vqa_v3"
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


# ── grounding evidence path (C7, 14A2 §4.3) / 定位证据路径 ────────────────


class _FakeGroundingService:
    """GroundingEvidenceService protocol fake with a call record.
    GroundingEvidenceService 协议 fake，记录调用。"""

    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.calls: list[tuple[object, object, object, object, object, object]] = []

    async def run(self, plan, sample, images, *, fallback_image_id, artifact_dir, budget):
        self.calls.append((plan, sample, dict(images), fallback_image_id, artifact_dir, budget))
        if self.error is not None:
            raise self.error
        return self.result


def _grounding_plan() -> FirstQwenVisualPlan:
    """A validated grounding plan: object_evidence_vqa family carries the
    evidence request (14C evidence seam contract). 合法 grounding 计划：
    object_evidence_vqa 家族携带 evidence request（14C 证据 seam 契约）。"""
    return FirstQwenVisualPlan(
        version="first-qwen-plan-v1",
        execution_family="object_evidence_vqa",
        confidence=0.9,
        roi_plan=RoiPlan(rois=[]),
        evidence_request=ObjectEvidenceRequest(composite_categories=["building"]),
    )


def _direct_plan() -> FirstQwenVisualPlan:
    """A direct_vqa plan (no evidence request) keeps the legacy path.
    direct_vqa 计划（无 evidence request）保持旧路径。"""
    return FirstQwenVisualPlan(
        version="first-qwen-plan-v1",
        execution_family="direct_vqa",
        confidence=0.9,
        roi_plan=RoiPlan(rois=[]),
    )


def _evidence_result() -> GroundingEvidenceResult:
    return GroundingEvidenceResult(
        bundle=GroundingEvidenceBundle(catalog_version="first-qwen-grounding-v1"),
        whole_image_boxes=[WholeImageBox(label="building", box=(120, 80, 340, 260))],
    )


def _evidence_context(
    root: Path,
    budget: _FakeBudget | None = None,
    *,
    visual_plan: FirstQwenVisualPlan | None = None,
    visual_bindings: VisualPlanBindings | None = None,
) -> AgentContext:
    return AgentContext(
        artifact_dir=root / "artifacts",
        qwen_client=None,
        call_budget=budget or _FakeBudget(),
        data_root=root,
        visual_plan=visual_plan,
        visual_bindings=visual_bindings,
    )


def test_evidence_path_serializes_deterministic_boxes_into_agent_result(
    tmp_path: Path,
) -> None:
    """The C6 executor's whole-image boxes become the existing AgentResult
    grounding contract: boxes, evidence_items, and a label-summary answer —
    never a free-text coordinate answer (14C §8).
    C6 执行器的整图框进入现有 AgentResult 定位契约：boxes、evidence_items 与
    标签汇总 answer——绝不使用自由文本坐标答案（14C §8）。"""
    client = _RecordingClient()
    budget = _FakeBudget()
    service = _FakeGroundingService(result=_evidence_result())
    bindings = VisualPlanBindings(grounding_evidence=service)
    execution = asyncio.run(
        GroundingAgent(client).run(
            _sample(tmp_path),
            _evidence_context(
                tmp_path,
                budget,
                visual_plan=_grounding_plan(),
                visual_bindings=bindings,
            ),
        )
    )
    assert isinstance(execution, AgentExecution)
    payload = execution.payload
    assert isinstance(payload, AgentResult)
    assert payload.answer == "building"
    assert payload.boxes == [[120, 80, 340, 260]]
    assert payload.status == "completed"
    assert len(payload.evidence_items) == 1
    evidence = payload.evidence_items[0]
    assert evidence.label == "building"
    assert evidence.box == [120, 80, 340, 260]
    assert evidence.coordinate_frame == "normalized_0_999_top_left"
    assert execution.result_filename == "agent_result.json"


def test_evidence_path_calls_service_with_full_context(tmp_path: Path) -> None:
    """The seam receives the plan, sample, decoded images, fallback image id,
    artifact dir, and the shared budget. seam 收到 plan、sample、解码图像、
    fallback image id、artifact 目录与共享预算。"""
    client = _RecordingClient()
    budget = _FakeBudget()
    service = _FakeGroundingService(result=_evidence_result())
    bindings = VisualPlanBindings(grounding_evidence=service)
    sample = _sample(tmp_path)
    asyncio.run(
        GroundingAgent(client).run(
            sample,
            _evidence_context(
                tmp_path,
                budget,
                visual_plan=_grounding_plan(),
                visual_bindings=bindings,
            ),
        )
    )
    assert len(service.calls) == 1
    plan, called_sample, images, fallback_id, artifact_dir, called_budget = service.calls[0]
    assert plan == _grounding_plan()
    assert called_sample.sample_id == sample.sample_id
    assert set(images) == {"i1"}
    assert fallback_id == "i1"
    assert artifact_dir == tmp_path / "artifacts"
    assert called_budget is budget
    # The final Grounding Qwen call happens inside the seam; the agent itself
    # never talks to the model. 最终 Grounding Qwen 调用发生在 seam 内部；Agent
    # 本身绝不触达模型。
    assert client.calls == []


def test_evidence_path_trace_and_bundle(tmp_path: Path) -> None:
    """The trace names the grounding_evidence workflow and catalog version;
    the JSON-safe bundle persists under the protocol owner's additional
    results. trace 标注 grounding_evidence 工作流与 catalog 版本；JSON 安全
    bundle 在协议 owner 的附加结果名下持久化。"""
    client = _RecordingClient()
    service = _FakeGroundingService(result=_evidence_result())
    bindings = VisualPlanBindings(grounding_evidence=service)
    execution = asyncio.run(
        GroundingAgent(client).run(
            _sample(tmp_path),
            _evidence_context(
                tmp_path,
                visual_plan=_grounding_plan(),
                visual_bindings=bindings,
            ),
        )
    )
    assert execution.trace["workflow"] == "grounding_evidence"
    assert execution.trace["catalog_version"] == "first-qwen-grounding-v1"
    assert execution.trace["agent_class"] == "agents.grounding.agent.GroundingAgent"
    assert "GroundingEvidenceExecutor.run" in execution.trace["route"]
    bundle_json = execution.additional_results["grounding_evidence.json"]
    assert bundle_json["workflow"] == "grounding_evidence"
    assert bundle_json["catalog_version"] == "first-qwen-grounding-v1"


def test_evidence_path_service_error_maps_to_stable_code(tmp_path: Path) -> None:
    """GroundingEvidenceError becomes AgentExecutionError with the stable
    code; an unexpected exception maps to its type name. GroundingEvidenceError
    转为携带稳定 code 的 AgentExecutionError；意外异常映射为类型名。"""
    service = _FakeGroundingService(error=GroundingEvidenceError("YOLO_UNAVAILABLE"))
    bindings = VisualPlanBindings(grounding_evidence=service)
    with pytest.raises(AgentExecutionError) as exc_info:
        asyncio.run(
            GroundingAgent(_RecordingClient()).run(
                _sample(tmp_path),
                _evidence_context(
                    tmp_path,
                    visual_plan=_grounding_plan(),
                    visual_bindings=bindings,
                ),
            )
        )
    assert "grounding_evidence_failed:YOLO_UNAVAILABLE" in str(exc_info.value)

    service = _FakeGroundingService(error=RuntimeError("raw internal detail"))
    bindings = VisualPlanBindings(grounding_evidence=service)
    with pytest.raises(AgentExecutionError) as exc_info:
        asyncio.run(
            GroundingAgent(_RecordingClient()).run(
                _sample(tmp_path),
                _evidence_context(
                    tmp_path,
                    visual_plan=_grounding_plan(),
                    visual_bindings=bindings,
                ),
            )
        )
    assert "grounding_evidence_failed:RuntimeError" in str(exc_info.value)


def test_plan_without_evidence_request_keeps_legacy_path(tmp_path: Path) -> None:
    """A direct_vqa plan (feature on, no evidence request) runs the legacy
    direct path: no seam call, no additional results.
    direct_vqa 计划（特性开启但无 evidence request）运行旧直接路径：无 seam
    调用、无附加结果。"""
    client = _RecordingClient()
    budget = _FakeBudget()
    service = _FakeGroundingService(result=_evidence_result())
    bindings = VisualPlanBindings(grounding_evidence=service)
    execution = asyncio.run(
        GroundingAgent(client).run(
            _sample(tmp_path),
            _evidence_context(
                tmp_path,
                budget,
                visual_plan=_direct_plan(),
                visual_bindings=bindings,
            ),
        )
    )
    assert service.calls == []
    assert len(client.calls) == 1  # one legacy direct Qwen call / 一次旧直接 Qwen 调用
    assert budget.qwen_calls == 1
    assert execution.additional_results == {}
    assert execution.trace["route"].startswith(
        "GroundingAgent.run -> VisualAgentBase.run"
    )


def test_feature_off_keeps_legacy_path(tmp_path: Path) -> None:
    """No plan in context: legacy run, stable route trace, no seam.
    context 无 plan：旧运行、稳定 route trace、无 seam。"""
    client = _RecordingClient()
    budget = _FakeBudget()
    execution = asyncio.run(
        GroundingAgent(client).run(_sample(tmp_path), _context(tmp_path, budget))
    )
    assert len(client.calls) == 1
    assert budget.qwen_calls == 1
    assert execution.additional_results == {}
    assert execution.trace["route"].startswith(
        "GroundingAgent.run -> VisualAgentBase.run"
    )
