"""Offline unit tests for the General VQA agent.

通用 VQA Agent 离线单测：注入带 cache_identity 的 fake client，覆盖四个
受支持 task 的完整 run、选择题载荷（choices/allow_multiple）、task mismatch
前置失败、无 VRSBench geometry、无 spacers_agent 依赖。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from agents.base import AgentContext, AgentExecution, VisualPlanBindings
from agents.errors import AgentExecutionError, AgentTaskMismatchError
from agents.general_vqa import GeneralVQAAgent
from agents.general_vqa.evidence.executor import EvidenceExecution
from agents.general_vqa.evidence.schema import (
    LayerStateRecord,
    ModelCallAudit,
    RoiEvidenceRecord,
    SegFormerEvidenceRecord,
    VqaEvidenceBundle,
    YoloDetectionRecord,
)
from agents.schema import (
    AgentResult,
    FirstQwenVisualPlan,
    ObjectEvidenceRequest,
    RoiPlan,
)
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
    """Records messages and request meta; returns a stable AgentResult.
    记录消息与请求元数据；返回稳定的 AgentResult。"""

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
            {"agent_name": "general_vqa_agent", "answer": "yes", "status": "completed"}
        )


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (1, 2, 3)).save(path, format="PNG")


def _sample(
    root: Path,
    *,
    task: str = "general_vqa",
    constraints: dict[str, Any] | None = None,
) -> UnifiedSample:
    _make_image(root / "img.png")
    normalization = None
    if constraints is not None:
        normalization = TaskNormalization(
            source_task=task,
            normalized_task=task,  # type: ignore[arg-type]
            normalizer="test", version="1",
            answer_constraints=constraints,
        )
    return UnifiedSample(
        sample_id="s1",
        dataset="parity",
        split="test",
        task=task,  # type: ignore[arg-type]
        images=[ImageRef(image_id="i1", path="img.png", role="image")],
        question="What is in the image?",
        ground_truth=GroundTruth(answers=["yes"]),
        normalization=normalization,
    )


def _agent(client: _RecordingClient | None = None) -> GeneralVQAAgent:
    return GeneralVQAAgent(client or _RecordingClient())


def _context(
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


def _last_user_payload(client: _RecordingClient) -> dict[str, Any]:
    """Parse the JSON payload embedded in the last recorded user message.
    解析最后一条已记录 user 消息中的 JSON 载荷。"""
    messages = client.calls[-1]["messages"]
    user_content = messages[1]["content"]
    text = user_content[-1]["text"]
    return json.loads(text)


# ── 协议 / protocol ────────────────────────────────────────────────────────


def test_agent_identity_and_tasks() -> None:
    agent = _agent()
    assert agent.name == "general_vqa_agent"
    assert agent.supported_tasks == frozenset(
        {"general_vqa", "scene_classification", "multiple_choice_vqa", "spatial_relation"}
    )


def test_run_returns_agent_execution_with_default_filename(tmp_path: Path) -> None:
    client = _RecordingClient()
    agent = _agent(client)
    execution = asyncio.run(agent.run(_sample(tmp_path), _context(tmp_path)))
    assert isinstance(execution, AgentExecution)
    assert isinstance(execution.payload, AgentResult)
    assert execution.agent_name == "general_vqa_agent"
    assert execution.result_filename == "agent_result.json"
    assert execution.trace["model"] == "fake-model"
    assert len(client.calls) == 1


@pytest.mark.parametrize("task", ["general_vqa", "scene_classification", "multiple_choice_vqa", "spatial_relation"])
def test_all_supported_tasks_run(task: str, tmp_path: Path) -> None:
    client = _RecordingClient()
    constraints = None
    if task == "multiple_choice_vqa":
        constraints = {"type": "closed_vocabulary", "values": ["A", "B", "C", "D"]}
    execution = asyncio.run(
        _agent(client).run(_sample(tmp_path, task=task, constraints=constraints), _context(tmp_path))
    )
    assert execution.payload.answer == "yes"
    assert len(client.calls) == 1


def test_spatial_relation_uses_general_prompt_single_call_and_general_agent(tmp_path: Path) -> None:
    """A spatial_relation sample runs through the generic single-call VQA path:
    one Qwen call, the default general_vqa_v2 prompt, agent_result.json, and
    the general_vqa_agent identity — no spatial candidate review or geometry
    rewrite. spatial_relation 样本走通用单次调用 VQA 路径：恰好一次 Qwen 调用、
    默认 general_vqa_v2 Prompt、agent_result.json 与 general_vqa_agent 身份——
    无 spatial 候选复核或几何改写。"""
    client = _RecordingClient()
    budget = _FakeBudget()
    agent = _agent(client)
    sample = _sample(tmp_path, task="spatial_relation")
    execution = asyncio.run(agent.run(sample, _context(tmp_path, budget)))
    assert execution.agent_name == "general_vqa_agent"
    assert execution.payload.agent_name == "general_vqa_agent"
    assert execution.result_filename == "agent_result.json"
    assert execution.trace["prompt_version"] == "general_vqa_v2"
    assert len(client.calls) == 1
    assert budget.qwen_calls == 1
    payload = _last_user_payload(client)
    assert payload["task"] == "spatial_relation"
    # No spatial candidate review or geometry branch in the generic path.
    # 通用路径不含 spatial 候选复核或几何分支。
    assert "candidate_review" not in execution.trace
    assert "agent_class" not in execution.trace


def test_unsupported_task_fails_before_model_call(tmp_path: Path) -> None:
    client = _RecordingClient()
    budget = _FakeBudget()
    with pytest.raises(AgentTaskMismatchError):
        asyncio.run(
            _agent(client).run(_sample(tmp_path, task="counting"), _context(tmp_path, budget))
        )
    assert budget.qwen_calls == 0
    assert client.calls == []


# ── 选择题载荷 / multiple-choice payload ───────────────────────────────────


def test_multiple_choice_payload_contains_choices_and_constraint(tmp_path: Path) -> None:
    client = _RecordingClient()
    agent = _agent(client)
    sample = _sample(
        tmp_path,
        task="multiple_choice_vqa",
        constraints={"type": "closed_vocabulary", "values": ["A", "B", "C", "D"]},
    )
    asyncio.run(agent.run(sample, _context(tmp_path)))
    payload = _last_user_payload(client)
    assert payload["choices"] == ["A", "B", "C", "D"]
    assert payload["allow_multiple"] is False
    # Ground truth is never leaked. / ground truth 绝不泄漏。
    assert "ground_truth" not in payload


def test_multiple_choice_payload_allow_multiple(tmp_path: Path) -> None:
    client = _RecordingClient()
    agent = _agent(client)
    sample = _sample(
        tmp_path,
        task="multiple_choice_vqa",
        constraints={"type": "closed_vocabulary", "values": ["A", "B"], "allow_multiple": True},
    )
    asyncio.run(agent.run(sample, _context(tmp_path)))
    payload = _last_user_payload(client)
    assert payload["choices"] == ["A", "B"]
    assert payload["allow_multiple"] is True


def test_multiple_choice_payload_without_constraints(tmp_path: Path) -> None:
    """A multiple-choice sample without extractable choices now fails with a
    stable input-contract error instead of being treated as open-ended VQA.
    无可用 choices 的多选题样本现在以稳定输入契约错误失败，而非按开放问答
    处理。"""
    from agents.errors import AgentExecutionError

    client = _RecordingClient()
    agent = _agent(client)
    sample = _sample(tmp_path, task="multiple_choice_vqa")
    with pytest.raises(AgentExecutionError, match="without_choices"):
        asyncio.run(agent.run(sample, _context(tmp_path)))


def test_non_choice_payload_has_no_choices_key(tmp_path: Path) -> None:
    client = _RecordingClient()
    agent = _agent(client)
    asyncio.run(agent.run(_sample(tmp_path, task="scene_classification"), _context(tmp_path)))
    payload = _last_user_payload(client)
    assert "choices" not in payload
    assert "allow_multiple" not in payload


# ── MCQ 输出约束 / multiple-choice output constraint (25.5) ───────────────


class _AnswerClient(_RecordingClient):
    """Returns a configurable answer. / 返回可配置答案的客户端。"""

    def __init__(self, answer: str) -> None:
        super().__init__()
        self._answer = answer

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls.append({"request_hash": request_meta.request_hash})
        return response_model.model_validate(
            {"agent_name": "general_vqa_agent", "answer": self._answer, "status": "completed"}
        )


def _mcq_sample(tmp_path: Path, *, values, allow_multiple: bool = False) -> UnifiedSample:
    return _sample(
        tmp_path,
        task="multiple_choice_vqa",
        constraints={
            "type": "closed_vocabulary",
            "values": values,
            "allow_multiple": allow_multiple,
        },
    )


def test_mcq_single_choice_letter_maps(tmp_path: Path) -> None:
    execution = asyncio.run(
        _agent(_AnswerClient("B")).run(_mcq_sample(tmp_path, values=["A", "B", "C", "D"]), _context(tmp_path))
    )
    assert execution.payload.status == "completed"
    assert execution.payload.answer == "B"


def test_mcq_single_choice_full_text_maps(tmp_path: Path) -> None:
    values = ["A. Building", "B. Road"]
    execution = asyncio.run(
        _agent(_AnswerClient("Road")).run(_mcq_sample(tmp_path, values=values), _context(tmp_path))
    )
    assert execution.payload.status == "completed"


def test_mcq_single_choice_case_and_whitespace(tmp_path: Path) -> None:
    execution = asyncio.run(
        _agent(_AnswerClient("  b ")).run(_mcq_sample(tmp_path, values=["A", "B"]), _context(tmp_path))
    )
    assert execution.payload.status == "completed"


def test_mcq_invalid_single_choice_is_partial(tmp_path: Path) -> None:
    execution = asyncio.run(
        _agent(_AnswerClient("yes")).run(_mcq_sample(tmp_path, values=["A", "B"]), _context(tmp_path))
    )
    assert execution.payload.status == "partial"
    assert execution.payload.geometry["answer_constraint_violation"]
    # Ground truth never leaks into the violation. / ground truth 绝不泄漏。
    assert "answers" not in str(execution.payload.geometry)


def test_mcq_multiple_choice_dedup_and_stable_order(tmp_path: Path) -> None:
    values = ["A. Car", "B. Truck", "C. Ship"]
    execution = asyncio.run(
        _agent(_AnswerClient("B, A, B")).run(
            _mcq_sample(tmp_path, values=values, allow_multiple=True), _context(tmp_path)
        )
    )
    assert execution.payload.status == "completed"
    assert execution.payload.answer == "A. Car, B. Truck"  # dedup + choice order / 去重 + 选项顺序


def test_mcq_multiple_choice_invalid_item_is_partial(tmp_path: Path) -> None:
    execution = asyncio.run(
        _agent(_AnswerClient("A, plane")).run(
            _mcq_sample(tmp_path, values=["A", "B"], allow_multiple=True), _context(tmp_path)
        )
    )
    assert execution.payload.status == "partial"
    assert "not among the choices" in execution.payload.geometry["answer_constraint_violation"]


def test_mcq_multiple_choice_empty_answer_is_partial(tmp_path: Path) -> None:
    execution = asyncio.run(
        _agent(_AnswerClient("")).run(
            _mcq_sample(tmp_path, values=["A", "B"], allow_multiple=True), _context(tmp_path)
        )
    )
    assert execution.payload.status == "partial"
    assert "empty" in execution.payload.geometry["answer_constraint_violation"]


# ── 边界 / boundaries ──────────────────────────────────────────────────────


def test_agent_has_no_vrsbench_geometry() -> None:
    source = (Path(__file__).resolve().parents[3] / "agents" / "general_vqa" / "agent.py").read_text(
        encoding="utf-8"
    )
    assert "VRSBench" not in source
    assert "vrsbench" not in source
    assert "apply_vrsbench" not in source


def test_agent_has_no_spacers_agent_import() -> None:
    source = (Path(__file__).resolve().parents[3] / "agents" / "general_vqa" / "agent.py").read_text(
        encoding="utf-8"
    )
    assert "spacers_agent" not in source
    assert "eval" not in source


def test_import_agent_does_not_load_legacy_packages() -> None:
    import agents.general_vqa  # noqa: F401

    for legacy in ("spacers_agent", "eval"):
        assert legacy not in sys.modules


class _FakeVqaEvidenceService:
    """VqaEvidenceService protocol fake: returns a configured EvidenceExecution.
    VqaEvidenceService 协议 fake：返回配置好的 EvidenceExecution。"""

    def __init__(self, execution: EvidenceExecution | None = None) -> None:
        self.execution = execution
        self.calls: list[tuple[object, dict[str, Image.Image], str]] = []

    def execute(self, plan, images, *, fallback_image_id):
        self.calls.append((plan, dict(images), fallback_image_id))
        return self.execution


# 200x160 image: below the 1080 preview shrink, so the full-image ROI crop and
# the presence mask keep their native size (mask shape is (H, W) = (160, 200)).
# 200x160 图像：低于 1080 预览缩放下限，整图 ROI 裁切与 presence mask 保持
# 原始尺寸（mask 形状为 (H, W) = (160, 200)）。
_EVIDENCE_IMAGE_SIZE = (200, 160)


def _evidence_sample(root: Path, *, task: str = "general_vqa") -> UnifiedSample:
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", _EVIDENCE_IMAGE_SIZE, (30, 60, 90)).save(root / "img.png", format="PNG")
    return UnifiedSample(
        sample_id="s1",
        dataset="parity",
        split="test",
        task=task,  # type: ignore[arg-type]
        images=[ImageRef(image_id="i1", path="img.png", role="image")],
        question="What is in the image?",
        ground_truth=GroundTruth(answers=["yes"]),
    )


def _object_plan() -> FirstQwenVisualPlan:
    """object_evidence_vqa plan with no ROIs (maps to the unique full-image
    ROI) and the closed composite categories. 无 ROI 的 object_evidence_vqa
    计划（映射为唯一整图 ROI）与封闭组合类别。"""
    return FirstQwenVisualPlan(
        version="first-qwen-plan-v1",
        execution_family="object_evidence_vqa",
        confidence=0.9,
        roi_plan=RoiPlan(rois=[]),
        evidence_request=ObjectEvidenceRequest(
            composite_categories=["small_vehicle", "large_vehicle", "building_outline"]
        ),
    )


def _direct_plan() -> FirstQwenVisualPlan:
    """Valid direct_vqa plan (feature on, legacy-equivalent family).
    合法 direct_vqa 计划（特性开启但等价于旧路径的家族）。"""
    return FirstQwenVisualPlan(
        version="first-qwen-plan-v1",
        execution_family="direct_vqa",
        confidence=0.9,
        roi_plan=RoiPlan(rois=[]),
    )


def _bundle() -> VqaEvidenceBundle:
    """One full-image ROI with one YOLO detection, one SegFormer mask hit,
    one missing leaf, and a full call audit. 一个整图 ROI，含一条 YOLO 检测、
    一条 SegFormer 掩膜命中、一个缺失叶子与完整调用审计。"""
    return VqaEvidenceBundle(
        rois=[
            RoiEvidenceRecord(
                roi_id="full",
                image_id="i1",
                source_size=_EVIDENCE_IMAGE_SIZE,
                core_xyxy=(0, 0, 200, 160),
                expanded_xyxy=(0, 0, 200, 160),
                crop_size=_EVIDENCE_IMAGE_SIZE,
            )
        ],
        detections=[
            YoloDetectionRecord(
                leaf_category="small_vehicle",
                roi_id="full",
                local_xyxy=(10.0, 10.0, 50.0, 40.0),
                local_roi_size=_EVIDENCE_IMAGE_SIZE,
                global_xyxy=(10.0, 10.0, 50.0, 40.0),
                global_image_size=_EVIDENCE_IMAGE_SIZE,
            )
        ],
        segments=[
            SegFormerEvidenceRecord(leaf_category="building_outline", roi_id="full")
        ],
        missing_leaves=["large_vehicle"],
        leaf_states={
            "small_vehicle": "hit",
            "building_outline": "hit",
            "large_vehicle": "missing",
        },
        call_audit=[
            ModelCallAudit(
                layer="yolo",
                roi_id="full",
                input_size=_EVIDENCE_IMAGE_SIZE,
                logical_model_id="fake-yolo",
            )
        ],
    )


def _execution() -> EvidenceExecution:
    """In-memory mask for the building_outline leaf only: shape (H, W) must
    match the rendered crop, exactly as the executor's presence masks do.
    仅为 building_outline 叶子提供内存掩膜：形状 (H, W) 必须与渲染裁切一致，
    与执行器 presence mask 相同。"""
    return EvidenceExecution(
        bundle=_bundle(),
        layer_states=(
            LayerStateRecord(leaf_category="small_vehicle", layer="yolo", state="hit"),
            LayerStateRecord(leaf_category="building_outline", layer="segformer", state="hit"),
            LayerStateRecord(leaf_category="large_vehicle", layer="yolo", state="missing"),
        ),
        outcomes=(),
        masks={("full", "building_outline"): np.ones((160, 200), dtype=bool)},
    )


def _content(client: _RecordingClient) -> list[dict[str, Any]]:
    """User content of the last recorded call. 最后一条已记录调用的 user content。"""
    return client.calls[-1]["messages"][1]["content"]


def _text_payload(client: _RecordingClient) -> dict[str, Any]:
    """JSON payload of the trailing text block. 末尾文本块的 JSON 载荷。"""
    return json.loads(_content(client)[-1]["text"])


# ── object-evidence path / 对象证据路径 ────────────────────────────────────


def test_object_evidence_exactly_one_final_qwen_call(tmp_path: Path) -> None:
    client = _RecordingClient()
    budget = _FakeBudget()
    service = _FakeVqaEvidenceService(execution=_execution())
    bindings = VisualPlanBindings(vqa_evidence=service)
    execution = asyncio.run(
        _agent(client).run(
            _evidence_sample(tmp_path),
            _context(
                tmp_path,
                budget,
                visual_plan=_object_plan(),
                visual_bindings=bindings,
            ),
        )
    )
    assert isinstance(execution, AgentExecution)
    # Exactly one final-Qwen budget entry and one client call; the evidence
    # service never talks to the model.
    # 恰好一次最终 Qwen budget 条目与一次 client 调用；证据服务绝不触达模型。
    assert budget.qwen_calls == 1
    assert len(client.calls) == 1
    assert len(service.calls) == 1
    plan, images, fallback_image_id = service.calls[0]
    assert plan == _object_plan()
    assert set(images) == {"i1"}
    assert fallback_image_id == "i1"
    assert execution.payload.answer == "yes"
    assert execution.result_filename == "agent_result.json"


def test_object_evidence_content_follows_14b_section_10_order(tmp_path: Path) -> None:
    """Final-Qwen user content: clean ROI image, then the per-ROI overlay,
    then the text evidence block (14B §10). 最终 Qwen 用户内容：干净 ROI 图、
    逐 ROI overlay、最后文本证据块（14B §10）。"""
    client = _RecordingClient()
    service = _FakeVqaEvidenceService(execution=_execution())
    bindings = VisualPlanBindings(vqa_evidence=service)
    asyncio.run(
        _agent(client).run(
            _evidence_sample(tmp_path),
            _context(
                tmp_path,
                visual_plan=_object_plan(),
                visual_bindings=bindings,
            ),
        )
    )
    content = _content(client)
    assert [block["type"] for block in content] == ["image_url", "image_url", "text"]
    # Overlay must differ from the clean crop. / overlay 必须与干净裁切不同。
    urls = [block["image_url"]["url"] for block in content[:2]]
    assert urls[0].startswith("data:image/png;base64,")
    assert urls[1].startswith("data:image/png;base64,")
    assert urls[0] != urls[1]


def test_object_evidence_text_payload_is_safe_and_complete(tmp_path: Path) -> None:
    """Text evidence carries geometry and text records only: no confidence,
    no box-drawn images, no local_roi_size, and the full SegFormer legend
    with stable per-leaf colors. 文本证据只携带几何与文本记录：无 confidence、
    无绘制框图、无 local_roi_size，含带稳定每叶子颜色的完整 SegFormer 图例。"""
    client = _RecordingClient()
    service = _FakeVqaEvidenceService(execution=_execution())
    bindings = VisualPlanBindings(vqa_evidence=service)
    asyncio.run(
        _agent(client).run(
            _evidence_sample(tmp_path),
            _context(
                tmp_path,
                visual_plan=_object_plan(),
                visual_bindings=bindings,
            ),
        )
    )
    payload = _text_payload(client)
    assert payload["question"] == "What is in the image?"
    assert payload["task"] == "general_vqa"
    assert payload["coordinate_frame"] == "normalized_0_999_top_left"
    assert payload["images"] == [{"image_id": "i1", "width": 200, "height": 160}]
    assert payload["rois"] == [
        {
            "roi_id": "full",
            "image_id": "i1",
            "source_size": [200, 160],
            "crop_xyxy": [0, 0, 200, 160],
            "crop_size": [200, 160],
        }
    ]
    assert payload["yolo_detections"] == [
        {
            "leaf_category": "small_vehicle",
            "roi_id": "full",
            "local_xyxy": [10.0, 10.0, 50.0, 40.0],
            "global_xyxy": [10.0, 10.0, 50.0, 40.0],
        }
    ]
    assert payload["segformer_hits"] == [
        {"roi_id": "full", "leaf_category": "building_outline"}
    ]
    legend = payload["segformer_legend"]
    assert legend == [{"leaf_category": "building_outline", "color_rgb": list(legend[0]["color_rgb"])}]
    assert all(len(entry["color_rgb"]) == 3 for entry in legend)
    assert all(0 <= channel <= 255 for entry in legend for channel in entry["color_rgb"])
    assert payload["missing_leaves"] == ["large_vehicle"]
    # Confidence and detection-box imagery never appear. / 绝不出现 confidence
    # 与检测框图像。
    assert "confidence" not in json.dumps(payload).lower()


def test_object_evidence_bundle_persists_as_additional_result(tmp_path: Path) -> None:
    client = _RecordingClient()
    service = _FakeVqaEvidenceService(execution=_execution())
    bindings = VisualPlanBindings(vqa_evidence=service)
    execution = asyncio.run(
        _agent(client).run(
            _evidence_sample(tmp_path),
            _context(
                tmp_path,
                visual_plan=_object_plan(),
                visual_bindings=bindings,
            ),
        )
    )
    bundle_json = execution.additional_results["vqa_evidence.json"]
    assert bundle_json["workflow"] == "object_evidence_vqa"
    assert bundle_json["detections"][0]["leaf_category"] == "small_vehicle"
    assert bundle_json["missing_leaves"] == ["large_vehicle"]
    assert execution.trace["workflow"] == "object_evidence_vqa"
    assert execution.trace["model"] == "fake-model"
    # The honest digest covers both rendered images. / 真实摘要覆盖两张渲染图。
    assert len(execution.trace["image_sha256"]) == 2
    assert all(len(digest) == 64 for digest in execution.trace["image_sha256"])


def test_object_evidence_request_hash_covers_the_actual_call(tmp_path: Path) -> None:
    """The request hash covers messages (including the rendered image digests)
    and differs from the legacy direct run over the same sample.
    request hash 覆盖消息（含渲染图摘要），并与同一样本的旧直接路径不同。"""
    client = _RecordingClient()
    service = _FakeVqaEvidenceService(execution=_execution())
    bindings = VisualPlanBindings(vqa_evidence=service)
    asyncio.run(
        _agent(client).run(
            _evidence_sample(tmp_path),
            _context(
                tmp_path,
                visual_plan=_object_plan(),
                visual_bindings=bindings,
            ),
        )
    )
    evidence_hash = client.calls[0]["request_hash"]
    assert len(evidence_hash) == 64
    direct_client = _RecordingClient()
    asyncio.run(
        _agent(direct_client).run(_evidence_sample(tmp_path), _context(tmp_path))
    )
    direct_hash = direct_client.calls[0]["request_hash"]
    assert len(direct_hash) == 64
    assert evidence_hash != direct_hash


def test_object_evidence_system_prompt_matches_legacy_suffix(tmp_path: Path) -> None:
    """The evidence final call keeps the format-identical structured prompt:
    same JSON-only instruction and same agent_name binding.
    证据最终调用保持格式一致的结构化 prompt：同样的 JSON-only 指令与
    agent_name 绑定。"""
    client = _RecordingClient()
    service = _FakeVqaEvidenceService(execution=_execution())
    bindings = VisualPlanBindings(vqa_evidence=service)
    asyncio.run(
        _agent(client).run(
            _evidence_sample(tmp_path),
            _context(
                tmp_path,
                visual_plan=_object_plan(),
                visual_bindings=bindings,
            ),
        )
    )
    system = client.calls[0]["messages"][0]["content"]
    assert "Return valid JSON only." in system
    assert "general_vqa_agent" in system


# ── compatibility matrix / 兼容矩阵 ────────────────────────────────────────


def test_object_evidence_plan_forbidden_for_non_general_vqa(tmp_path: Path) -> None:
    """A plan selecting object_evidence_vqa for scene_classification fails
    stably instead of silently degrading to the direct path.
    为 scene_classification 选择 object_evidence_vqa 的计划稳定失败，而不
    静默降级到直接路径。"""
    service = _FakeVqaEvidenceService(execution=_execution())
    bindings = VisualPlanBindings(vqa_evidence=service)
    sample = _evidence_sample(tmp_path, task="scene_classification")
    with pytest.raises(AgentExecutionError) as exc_info:
        asyncio.run(
            _agent().run(
                sample,
                _context(
                    tmp_path,
                    visual_plan=_object_plan(),
                    visual_bindings=bindings,
                ),
            )
        )
    assert "object_evidence_plan_forbidden_for_task" in str(exc_info.value)
    assert service.calls == []


def test_object_evidence_requires_injected_service(tmp_path: Path) -> None:
    """Feature on but no service injected: stable failure, no model call.
    特性开启但未注入服务：稳定失败，不调用模型。"""
    client = _RecordingClient()
    agent = _agent(client)
    with pytest.raises(AgentExecutionError) as exc_info:
        asyncio.run(
            agent.run(
                _evidence_sample(tmp_path),
                _context(
                    tmp_path,
                    visual_plan=_object_plan(),
                    visual_bindings=None,
                ),
            )
        )
    assert "vqa_evidence_service_unavailable" in str(exc_info.value)
    assert client.calls == []


def test_object_evidence_service_failure_maps_to_stable_code(tmp_path: Path) -> None:
    """An evidence-service exception becomes AgentExecutionError with a stable
    classification, never a raw traceback escape.
    证据服务异常转为携带稳定分类的 AgentExecutionError，绝不外泄原始
    traceback。"""
    class _BoomService:
        def execute(self, plan, images, *, fallback_image_id):
            raise RuntimeError("raw internal detail")

    bindings = VisualPlanBindings(vqa_evidence=_BoomService())
    with pytest.raises(AgentExecutionError) as exc_info:
        asyncio.run(
            _agent().run(
                _evidence_sample(tmp_path),
                _context(
                    tmp_path,
                    visual_plan=_object_plan(),
                    visual_bindings=bindings,
                ),
            )
        )
    assert "vqa_evidence_failed:RuntimeError" in str(exc_info.value)


# ── flag-off legacy parity / flag-off 旧路径一致性 ─────────────────────────


def test_direct_plan_keeps_legacy_path_byte_identical(tmp_path: Path) -> None:
    """A direct_vqa plan runs the legacy direct path: no evidence service,
    no additional results, no workflow key in the trace.
    direct_vqa 计划运行旧直接路径：无证据服务、无附加结果、trace 无
    workflow 键。"""
    client = _RecordingClient()
    budget = _FakeBudget()
    service = _FakeVqaEvidenceService(execution=_execution())
    bindings = VisualPlanBindings(vqa_evidence=service)
    execution = asyncio.run(
        _agent(client).run(
            _evidence_sample(tmp_path),
            _context(
                tmp_path,
                budget,
                visual_plan=_direct_plan(),
                visual_bindings=bindings,
            ),
        )
    )
    assert len(client.calls) == 1
    assert budget.qwen_calls == 1
    assert service.calls == []
    assert execution.additional_results == {}
    assert "workflow" not in execution.trace
    # Legacy user content: one image + one text block only.
    # 旧用户内容：仅一张图 + 一个文本块。
    content = _content(client)
    assert [block["type"] for block in content] == ["image_url", "text"]


def test_no_plan_keeps_legacy_path_byte_identical(tmp_path: Path) -> None:
    """Feature off entirely (no plan in context) keeps the legacy run intact.
    特性完全关闭（context 无 plan）保持旧运行不变。"""
    client = _RecordingClient()
    budget = _FakeBudget()
    execution = asyncio.run(
        _agent(client).run(_evidence_sample(tmp_path), _context(tmp_path, budget))
    )
    assert len(client.calls) == 1
    assert budget.qwen_calls == 1
    assert execution.additional_results == {}
    assert "workflow" not in execution.trace
    assert isinstance(execution.payload, AgentResult)
    assert execution.payload.answer == "yes"
