"""Offline tests for direct and v2 evidence-backed GeneralVQAAgent paths.

通用 VQA Agent 离线测试：直接路径继续只做一次模型调用，证据路径只接受
VisualTaskPlan 与 MaterializedVisualView，不再构造旧视觉计划或候选回退。
"""

from __future__ import annotations

import array
import asyncio
import base64
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from agents.base import AgentContext, AgentExecution, VisualPlanBindings
from agents.errors import AgentExecutionError, AgentTaskMismatchError
from agents.general_vqa import GeneralVQAAgent
from agents.general_vqa.agent import _match_choice, _normalize_choice_answer
from agents.general_vqa.evidence.executor import (
    EvidenceExecution,
    SegFormerPreviewEvidence,
)
from agents.general_vqa.evidence.rendering import (
    class_id_grid_from_any,
    leaf_boolean_grid,
    make_preview,
    render_pure_mask,
    segformer_palette,
)
from agents.general_vqa.evidence.schema import (
    LayerStateRecord,
    RoiEvidenceRecord,
    SegFormerEvidenceRecord,
    VqaEvidenceBundle,
    YoloDetectionRecord,
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
            choices=["yes", "no"],
            allow_multiple=False,
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


def _plan(
    categories: list[str] | None = None,
    *,
    task: str = "general_vqa",
    assistance: bool = True,
) -> VisualTaskPlan:
    return VisualTaskPlan(
        version="visual-task-plan-v5",
        task=task,  # type: ignore[arg-type]
        needs_visual_assistance=assistance,
        object_categories=categories or (["small-vehicle"] if assistance else []),
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


def _preview_evidence_from_masks(
    roi_id: str,
    masks: dict[tuple[str, str], Image.Image],
    *,
    binding: str = "seg_001",
) -> tuple[SegFormerPreviewEvidence, ...]:
    """Build a preview-space evidence wrapper from full-size per-leaf boolean
    masks, for agent-level rendering tests: each leaf gets one class id, the
    grid marks every masked pixel with that id, and the preview equals the
    mask size (test ROIs are <= 1080). 从全尺寸逐叶子 boolean mask 构造
    preview 空间证据包装，供 agent 级渲染测试：每个叶子一个 class id，grid
    在掩膜像素处标记该 id，preview 与 mask 尺寸一致（测试 ROI 均 <= 1080）。"""
    leaves = sorted({leaf for (roi, leaf) in masks if roi == roi_id})
    if not leaves:
        return ()
    width, height = next(iter(masks.values())).size
    leaf_class_ids = {leaf: frozenset({index + 1}) for index, leaf in enumerate(leaves)}
    grid = Image.new("I", (width, height), 0)
    for leaf in leaves:
        mask = masks[(roi_id, leaf)]
        values = array.array("i", grid.tobytes())
        class_id = next(iter(leaf_class_ids[leaf]))
        combined = array.array(
            "i",
            (
                class_id if mask_value else value
                for value, mask_value in zip(values, mask.tobytes())
            ),
        )
        grid = Image.frombytes("I", (width, height), combined.tobytes())
    return (
        SegFormerPreviewEvidence(
            roi_id=roi_id,
            binding=binding,
            preview_size=(width, height),
            class_id_grid=grid,
            leaf_class_ids=leaf_class_ids,
        ),
    )


class _FakeVqaEvidenceService:
    def __init__(
        self,
        bundle: VqaEvidenceBundle,
        *,
        masks: dict[tuple[str, str], Image.Image] | None = None,
        palette: dict[str, tuple[int, int, int]] | None = None,
    ) -> None:
        self.bundle = bundle
        self.masks = masks or {}
        self.palette = palette or {}
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
        preview_evidence = []
        roi_ids = {record.roi_id for record in self.bundle.rois}
        for roi_id in roi_ids:
            preview_evidence.extend(
                _preview_evidence_from_masks(roi_id, self.masks)
            )
        return EvidenceExecution(
            bundle=self.bundle,
            layer_states=(),
            outcomes=(),
            preview_evidence=tuple(preview_evidence),
            palette=self.palette,
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
    assert payload["allow_multiple"] is False
    assert "answer_constraints" not in payload
    assert execution.payload.answer == "yes"


@pytest.mark.parametrize("answer", ["B", "B.", "(B)", "Water", "(B) Water"])
def test_parenthesized_choice_answers_are_accepted(answer: str) -> None:
    choices = ["(A) Road", "(B) Water"]
    assert _normalize_choice_answer(answer, choices, False) == "B"


def test_letter_maps_by_position_for_unlabeled_choices() -> None:
    choices = ["Road", "Airport", "Water"]
    assert _match_choice("A", choices) == "Road"
    assert _match_choice("B", choices) == "Airport"


def test_compact_multiple_choice_letters_are_accepted() -> None:
    choices = ["(A) Road", "(B) Water", "(C) Forest"]
    assert _normalize_choice_answer("AC", choices, True) == "A, C"


def test_unmatched_multiple_choice_text_is_preserved() -> None:
    choices = ["(A) Road", "(B) Water"]
    assert _normalize_choice_answer(
        "It appears to be water", choices, False
    ) == "It appears to be water"


@pytest.mark.parametrize(
    ("task", "subtype"),
    [
        ("general_vqa", "attribute"),
        ("scene_classification", "scene_classification"),
        ("spatial_relation", "spatial_relation"),
    ],
)
def test_direct_payload_is_task_aware(
    tmp_path: Path, task: str, subtype: str
) -> None:
    client = _RecordingClient()
    sample = _sample(tmp_path, task=task)
    sample = sample.model_copy(
        update={
            "normalization": TaskNormalization(
                source_task="source",
                normalized_task=task,  # type: ignore[arg-type]
                semantic_subtype=subtype,
                normalizer="test",
                version="1",
            )
        }
    )
    asyncio.run(GeneralVQAAgent(client).run(sample, _context(tmp_path, client)))
    payload = json.loads(client.calls[0]["messages"][1]["content"][-1]["text"])
    assert payload == {
        "question": "What is in the image?",
        "task": task,
        "semantic_subtype": subtype,
        "coordinate_frame": "normalized_0_999_top_left",
        "box_format": "integer_xyxy_json",
    }


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


def _roi(roi_id: str, box: tuple[int, int, int, int]) -> RoiEvidenceRecord:
    x0, y0, x1, y1 = box
    return RoiEvidenceRecord(
        roi_id=roi_id,
        image_id="img1",
        source_size=(8, 6),
        core_xyxy=box,
        expanded_xyxy=box,
        crop_size=(x1 - x0, y1 - y0),
    )


def _four_branch_bundle() -> VqaEvidenceBundle:
    """ROIs covering all four branches: yolo-only, seg-only, both, neither.
    覆盖全部四个分支的 ROI：仅 YOLO、仅 SegFormer、两者、均无。"""
    return VqaEvidenceBundle(
        catalog_version="test-catalog-v1",
        preprocessing_version="yolo-v1-segformer-pad-v1",
        rois=[
            _roi("r_yolo", (0, 0, 4, 3)),
            _roi("r_seg", (4, 0, 8, 3)),
            _roi("r_both", (0, 3, 4, 6)),
            _roi("r_none", (4, 3, 8, 6)),
        ],
        detections=[
            YoloDetectionRecord(
                leaf_category="small_vehicle",
                roi_id="r_yolo",
                local_xyxy=(1.0, 1.0, 3.0, 2.0),
                local_roi_size=(4, 3),
                global_xyxy=(1.0, 1.0, 3.0, 2.0),
                global_image_size=(8, 6),
            ),
            YoloDetectionRecord(
                leaf_category="water",
                roi_id="r_both",
                local_xyxy=(0.5, 0.5, 2.5, 2.5),
                local_roi_size=(4, 3),
                global_xyxy=(0.5, 3.5, 2.5, 5.5),
                global_image_size=(8, 6),
            ),
        ],
        segments=[
            SegFormerEvidenceRecord(leaf_category="building", roi_id="r_seg"),
            SegFormerEvidenceRecord(leaf_category="water", roi_id="r_both"),
        ],
        missing_leaves=[],
        leaf_states={
            "small_vehicle": "hit",
            "building": "hit",
            "water": "hit",
        },
    )


def _four_branch_masks() -> dict[tuple[str, str], Image.Image]:
    building = Image.new("L", (4, 3), 0)
    building.putpixel((1, 1), 255)
    water = Image.new("L", (4, 3), 0)
    water.putpixel((2, 1), 255)
    return {("r_seg", "building"): building, ("r_both", "water"): water}


def _run_evidence(
    root: Path,
    client: _RecordingClient,
    service: _FakeVqaEvidenceService,
    *,
    plan: VisualTaskPlan | None = None,
    views: tuple[MaterializedVisualView, ...] = (),
    sample: UnifiedSample | None = None,
) -> tuple[AgentExecution, dict[str, Any]]:
    execution = asyncio.run(
        GeneralVQAAgent(client).run(
            sample or _sample(root),
            _context(
                root,
                client,
                plan=plan,
                views=views,
                bindings=VisualPlanBindings(vqa_evidence=service),
            ),
        )
    )
    return execution, client.calls[0]["request_meta"]


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


def test_v2_assistance_consumes_evidence_for_all_supported_tasks(
    tmp_path: Path,
) -> None:
    """Every GeneralVQAAgent-supported task consumes the VQA evidence service
    when the planner asks for assistance: one evidence call, exactly one final
    Qwen call, and the safe basename vqa_evidence.json persisted.
    当 planner 请求辅助时，GeneralVQAAgent 支持的每个 task 都消费 VQA evidence
    服务：一次 evidence 调用、恰好一次 final Qwen 调用，并持久化安全 basename
    vqa_evidence.json。"""
    for task in (
        "general_vqa",
        "scene_classification",
        "multiple_choice_vqa",
        "spatial_relation",
    ):
        client = _RecordingClient()
        service = _FakeVqaEvidenceService(_bundle())
        plan = _plan(task=task)
        execution = asyncio.run(
            GeneralVQAAgent(client).run(
                _sample(tmp_path, task=task),
                _context(
                    tmp_path,
                    client,
                    plan=plan,
                    views=(_view(),),
                    bindings=VisualPlanBindings(vqa_evidence=service),
                ),
            )
        )
        assert len(service.calls) == 1, task
        assert service.calls[0]["plan"] is plan
        assert len(client.calls) == 1, task
        assert execution.additional_results["vqa_evidence.json"]["rois"][0][
            "expanded_xyxy"
        ] == [0, 0, 8, 6]


def test_direct_path_never_calls_evidence_service_for_any_supported_task(
    tmp_path: Path,
) -> None:
    """assistance=false keeps every supported task on the direct path with
    zero evidence-service calls. assistance=false 时所有支持 task 保持 direct
    路径，evidence 服务零调用。"""
    for task in (
        "general_vqa",
        "scene_classification",
        "multiple_choice_vqa",
        "spatial_relation",
    ):
        client = _RecordingClient()
        service = _FakeVqaEvidenceService(_bundle())
        execution = asyncio.run(
            GeneralVQAAgent(client).run(
                _sample(tmp_path, task=task),
                _context(
                    tmp_path,
                    client,
                    plan=_plan(task=task, assistance=False),
                    views=(_view(),),
                    bindings=VisualPlanBindings(vqa_evidence=service),
                ),
            )
        )
        assert service.calls == [], task
        assert len(client.calls) == 1, task
        assert execution.payload.answer == "yes"


@pytest.mark.parametrize(
    ("source_size", "expected_size"),
    [
        ((2048, 1024), (1080, 540)),
        ((1024, 2048), (540, 1080)),
        ((800, 600), (800, 600)),
    ],
)
def test_direct_path_shrinks_materialized_view_for_final_qwen(
    tmp_path: Path,
    source_size: tuple[int, int],
    expected_size: tuple[int, int],
) -> None:
    """Direct VQA sends a shrink-only <=1080 preview while keeping the
    materialized source geometry authoritative. direct VQA 向最终 Qwen 发送
    只缩不放的 <=1080 预览，同时保留已物化源几何的权威性。
    """
    sample = _sample(tmp_path)
    Image.new("RGB", source_size, (17, 18, 19)).save(tmp_path / "img.png")
    view = MaterializedVisualView(
        image_id="img1",
        view_mode="full_image",
        source_size=source_size,
        crop_xyxy=(0, 0, *source_size),
        crop_size=source_size,
    )
    client = _RecordingClient()
    execution = asyncio.run(
        GeneralVQAAgent(client).run(
            sample,
            _context(
                tmp_path,
                client,
                plan=_plan(assistance=False),
                views=(view,),
            ),
        )
    )

    image_block = client.calls[0]["messages"][1]["content"][0]
    encoded = image_block["image_url"]["url"].split(",", 1)[1]
    received = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert received.size == expected_size
    assert execution.trace["visual_view_modes"][0]["crop_size"] == list(source_size)


def test_assistance_without_service_fails_stably_for_all_supported_tasks(
    tmp_path: Path,
) -> None:
    """assistance=true without an injected VqaEvidenceService fails stably for
    every supported task and never calls the final Qwen. assistance=true 但未
    注入 VqaEvidenceService 时，所有支持 task 都稳定失败且绝不调用 final Qwen。"""
    for task in (
        "general_vqa",
        "scene_classification",
        "multiple_choice_vqa",
        "spatial_relation",
    ):
        client = _RecordingClient()
        with pytest.raises(
            AgentExecutionError, match="vqa_evidence_service_unavailable"
        ):
            asyncio.run(
                GeneralVQAAgent(client).run(
                    _sample(tmp_path, task=task),
                    _context(
                        tmp_path,
                        client,
                        plan=_plan(task=task),
                        views=(_view(),),
                    ),
                )
            )
        assert client.calls == [], task


def test_multiple_choice_evidence_path_preserves_unmatched_answer(
    tmp_path: Path,
) -> None:
    """Unmatched free text remains a completed model answer for evaluation.
    无法匹配的自由文本仍是供评测的 completed 模型答案。"""
    client = _RecordingClient(answer="maybe")
    service = _FakeVqaEvidenceService(_bundle())
    execution = asyncio.run(
        GeneralVQAAgent(client).run(
            _sample(tmp_path, task="multiple_choice_vqa"),
            _context(
                tmp_path,
                client,
                plan=_plan(task="multiple_choice_vqa"),
                views=(_view(),),
                bindings=VisualPlanBindings(vqa_evidence=service),
            ),
        )
    )
    assert len(service.calls) == 1
    assert len(client.calls) == 1
    assert execution.payload.status == "completed"
    assert execution.payload.answer == "maybe"
    assert "answer_constraint_violation" not in execution.payload.geometry
    assert "vqa_evidence.json" in execution.additional_results


def test_three_branch_content_renders_frozen_protocol(tmp_path: Path) -> None:
    """Four ROIs cover the frozen branches (14.12.3): yolo-only -> annotated
    ROI; seg-only -> pure mask + clean ROI; both -> YOLO-on-mask + clean ROI;
    neither -> clean ROI. The text payload carries the exact rendered-leaf split,
    visual-input roles, and the palette is folded into mask_legend and
    evidence_identity.
    四个 ROI 覆盖冻结分支（14.12.3）：仅 YOLO -> 标注 ROI；仅 SegFormer -> 纯色
    mask + 干净 ROI；两者 -> YOLO-on-mask + 干净 ROI；均无 -> 干净 ROI。文本载荷
    携带精确的 rendered-leaf 划分、图像角色，调色表进入 mask_legend 与
    evidence_identity。"""
    client = _RecordingClient()
    masks = _four_branch_masks()
    palette = segformer_palette(["building", "water"])
    service = _FakeVqaEvidenceService(
        _four_branch_bundle(), masks=masks, palette=palette
    )
    execution, request_meta = _run_evidence(
        tmp_path,
        client,
        service,
        plan=_plan(["small-vehicle", "building", "water"]),
        views=(_view(),),
    )
    assert len(client.calls) == 1
    assert len(service.calls) == 1
    messages = client.calls[0]["messages"]
    content = messages[1]["content"]
    images = [block for block in content if block["type"] == "image_url"]
    text = next(block for block in content if block["type"] == "text")
    # 6 image blocks: 1 (yolo) + 2 (seg) + 2 (both) + 1 (neither).
    # 6 个图像块：1（YOLO）+ 2（Seg）+ 2（两者）+ 1（均无）。
    assert len(images) == 6
    assert all("image_url" in block and "url" in block["image_url"] for block in images)
    payload = json.loads(text["text"])
    evidence = payload["evidence"]
    assert evidence["requested_categories"] == ["small-vehicle", "building", "water"]
    assert evidence["missing_categories"] == []
    assert evidence["mask_legend"] == [
        {"category": "building", "color_rgb": list(palette["building"])},
        {"category": "water", "color_rgb": list(palette["water"])},
    ]
    assert "evidence_identity" not in payload
    assert all("source_size" not in roi and "crop_xyxy" not in roi for roi in evidence["rois"])
    assert evidence["visual_inputs"] == [
        {"content_image_index": 0, "roi_id": "r_yolo", "role": "annotated_roi"},
        {
            "content_image_index": 1,
            "roi_id": "r_seg",
            "role": "segformer_pure_mask",
        },
        {"content_image_index": 2, "roi_id": "r_seg", "role": "clean_roi"},
        {
            "content_image_index": 3,
            "roi_id": "r_both",
            "role": "yolo_on_segformer_pure_mask",
        },
        {"content_image_index": 4, "roi_id": "r_both", "role": "clean_roi"},
        {"content_image_index": 5, "roi_id": "r_none", "role": "clean_roi"},
    ]
    assert execution.trace["visual_content_version"] == "v2"
    assert len(evidence["detections"]) == 2
    assert len(evidence["segmentation_hits"]) == 2
    # The bundle is persisted as additional results, unchanged by rendering.
    # bundle 作为附加结果持久化，不因渲染而改变。
    persisted = execution.additional_results["vqa_evidence.json"]
    assert persisted["leaf_states"] == {
        "small_vehicle": "hit",
        "building": "hit",
        "water": "hit",
    }
    assert request_meta.request_hash


def test_segformer_only_agent_image_shrinks_large_mask_with_nearest(
    tmp_path: Path,
) -> None:
    """The Agent's SegFormer-only final image is <=1080, not just the helper.
    直接验证 Agent 的 SegFormer-only 最终图像最长边 <=1080，而不只验证 helper。"""
    size = (2000, 1200)
    sample = _sample(tmp_path)
    Image.new("RGB", size, (1, 2, 3)).save(tmp_path / "img.png", format="PNG")
    # Keep the sample contract while replacing its source with the intentionally
    # large image. 保持样本契约，同时将源文件替换为有意设置的大图。
    roi = RoiEvidenceRecord(
        roi_id="full",
        image_id="img1",
        source_size=size,
        core_xyxy=(0, 0, *size),
        expanded_xyxy=(0, 0, *size),
        crop_size=size,
    )
    mask = Image.new("L", size, 0)
    mask.paste(255, (300, 300, 1700, 900))
    bundle = VqaEvidenceBundle(
        catalog_version="test-catalog-v1",
        preprocessing_version="yolo-v1-segformer-pad-v1",
        rois=[roi],
        segments=[SegFormerEvidenceRecord(leaf_category="building", roi_id="full")],
        leaf_states={"building": "hit"},
    )
    client = _RecordingClient()
    palette = segformer_palette(["building"])
    execution, _ = _run_evidence(
        tmp_path,
        client,
        _FakeVqaEvidenceService(
            bundle,
            masks={("full", "building"): mask},
            palette=palette,
        ),
        plan=_plan(["building"]),
        views=(
            MaterializedVisualView(
                image_id="img1",
                view_mode="full_image",
                source_size=size,
                crop_xyxy=(0, 0, *size),
                crop_size=size,
            ),
        ),
        sample=sample,
    )
    assert execution.additional_results["vqa_evidence.json"]["segments"]
    image_blocks = [
        block
        for block in client.calls[0]["messages"][1]["content"]
        if block["type"] == "image_url"
    ]
    assert len(image_blocks) == 2
    decoded = [
        Image.open(io.BytesIO(base64.b64decode(block["image_url"]["url"].split(",", 1)[1])))
        for block in image_blocks
    ]
    assert decoded[0].size == (1080, 648)
    assert set(decoded[0].getdata()) <= {(0, 0, 0), palette["building"]}
    assert decoded[1].size == (1080, 648)
    assert decoded[1].getpixel((0, 0)) == (1, 2, 3)
    assert decoded[0].getpixel((0, 0)) != decoded[1].getpixel((0, 0))


def test_segformer_only_clean_roi_uses_exact_expanded_roi(tmp_path: Path) -> None:
    """The second SegFormer-only image is the matching expanded ROI, not the
    full source image or a planner preview.
    仅 SegFormer 的第二张图必须是对应 expanded ROI，而非整图或 planner 预览。"""
    sample = _sample(tmp_path)
    source = Image.new("RGB", (8, 6))
    for y in range(6):
        for x in range(8):
            source.putpixel((x, y), (x * 10, y * 20, 100))
    source.save(tmp_path / "img.png", format="PNG")
    roi = _roi("seg", (2, 1, 6, 5))
    mask = Image.new("L", (4, 4), 0)
    mask.putpixel((1, 1), 255)
    bundle = VqaEvidenceBundle(
        catalog_version="test-catalog-v1",
        preprocessing_version="yolo-v1-segformer-pad-v1",
        rois=[roi],
        segments=[SegFormerEvidenceRecord(leaf_category="building", roi_id="seg")],
        leaf_states={"building": "hit"},
    )
    client = _RecordingClient()
    palette = segformer_palette(["building"])
    _run_evidence(
        tmp_path,
        client,
        _FakeVqaEvidenceService(
            bundle,
            masks={("seg", "building"): mask},
            palette=palette,
        ),
        plan=_plan(["building"]),
        views=(_view(),),
        sample=sample,
    )
    image_blocks = [
        block
        for block in client.calls[0]["messages"][1]["content"]
        if block["type"] == "image_url"
    ]
    assert len(image_blocks) == 2
    clean = Image.open(
        io.BytesIO(
            base64.b64decode(image_blocks[1]["image_url"]["url"].split(",", 1)[1])
        )
    )
    assert clean.size == (4, 4)
    assert clean.getpixel((0, 0)) == source.getpixel((2, 1))
    assert clean.getpixel((3, 3)) == source.getpixel((5, 4))


def test_evidence_request_hash_is_stable_for_identical_inputs(tmp_path: Path) -> None:
    client_a = _RecordingClient()
    client_b = _RecordingClient()
    palette = segformer_palette(["building", "water"])
    _, first = _run_evidence(
        tmp_path,
        client_a,
        _FakeVqaEvidenceService(
            _four_branch_bundle(),
            masks=_four_branch_masks(),
            palette=palette,
        ),
        views=(_view(),),
    )
    _, second = _run_evidence(
        tmp_path,
        client_b,
        _FakeVqaEvidenceService(
            _four_branch_bundle(),
            masks=_four_branch_masks(),
            palette=palette,
        ),
        views=(_view(),),
    )
    assert first.request_hash == second.request_hash


def test_evidence_request_hash_changes_with_each_semantic_input(
    tmp_path: Path, monkeypatch
) -> None:
    """14.13: at least each of tile partition geometry / remainder
    interpolation result / YOLO box / mask pixels / palette-catalog version /
    image order / question must change the final-Qwen request hash. Partition
    geometry and remainder interpolation flow through the executor into the
    rendered crop/mask pixels, so their agent-level representatives are the
    ROI crop geometry and the mask pixels.
    14.13：至少 tile partition 几何、remainder 插值结果、YOLO 框、mask 像素、
    调色表/catalog 版本、图像顺序、问题中任意一项变化都必须改变最终 Qwen
    请求 hash。partition 几何与 remainder 插值经 executor 流入渲染 crop/mask
    像素，因此它们在 agent 层的代表分别是 ROI crop 几何与 mask 像素。"""
    palette = segformer_palette(["building", "water"])

    def run(
        *, bundle=None, masks=None, question="What is in the image?", clean_pixel=False
    ):
        client = _RecordingClient()
        sample = _sample(tmp_path)
        if clean_pixel:
            source = Image.open(tmp_path / "img.png")
            source.putpixel((4, 3), (255, 254, 253))
            source.save(tmp_path / "img.png", format="PNG")
        if question != "What is in the image?":
            sample = sample.model_copy(update={"question": question})
        _, meta = _run_evidence(
            tmp_path,
            client,
            _FakeVqaEvidenceService(
                bundle or _four_branch_bundle(),
                masks=masks if masks is not None else _four_branch_masks(),
                palette=palette,
            ),
            plan=_plan(["small-vehicle", "building", "water"]),
            views=(_view(),),
            sample=sample,
        )
        return meta.request_hash

    baseline = run()

    def assert_differs(description: str, **kwargs) -> None:
        changed = run(**kwargs)
        assert changed != baseline, f"{description} did not change the request hash"

    # Question. 问题。
    assert_differs("question", question="Are there any vehicles?")
    # YOLO box: the rendered annotation and the text payload change.
    # YOLO 框：渲染标注与文本载荷变化。
    bundle = _four_branch_bundle()
    bundle.detections[0] = bundle.detections[0].model_copy(
        update={"local_xyxy": (2.0, 1.0, 3.5, 2.5)}
    )
    assert_differs("yolo box", bundle=bundle)
    # Mask pixels (remainder interpolation representative): one flipped pixel
    # changes the mask digest.  mask 像素（remainder 插值代表）：翻转一个像素
    # 改变 mask 摘要。
    masks = _four_branch_masks()
    masks[("r_seg", "building")].putpixel((3, 2), 255)
    assert_differs("mask pixels", masks=masks)
    # Clean ROI pixels are also part of the final message and digest.
    # 干净 ROI 像素同样属于最终消息并进入摘要。
    assert_differs("clean ROI pixels", clean_pixel=True)
    # Catalog version is model-invisible but remains in the request-hash
    # target spec. catalog 版本对模型不可见，但继续进入请求哈希 target spec。
    bundle = _four_branch_bundle()
    bundle.catalog_version = "test-catalog-v2"
    assert_differs("catalog version", bundle=bundle)
    # Preprocessing identity: switching the combined version changes the
    # request hash, so old v1 caches never hit new pad-protocol requests.
    # 预处理身份：切换组合版本改变请求 hash，使旧 v1 cache 绝不命中新 pad
    # 协议请求。
    bundle = _four_branch_bundle()
    bundle.preprocessing_version = "greedy-1024-stretch-v1"
    assert_differs("preprocessing version", bundle=bundle)
    # Image order: swapping the two ROIs reorders the rendered image blocks
    # and the geometry text.  图像顺序：交换两个 ROI 重排渲染图像块与几何文本。
    bundle = _four_branch_bundle()
    bundle.rois = [bundle.rois[1], bundle.rois[0], bundle.rois[2], bundle.rois[3]]
    assert_differs("image order", bundle=bundle)
    # Tile partition geometry representative: a different quantized ROI crop
    # changes the rendered pixels and the geometry text.
    # tile partition 几何代表：不同量化 ROI 裁切改变渲染像素与几何文本。
    bundle = _four_branch_bundle()
    bundle.rois[0] = _roi("r_yolo", (0, 0, 5, 3))
    assert_differs("roi geometry", bundle=bundle)
    # Palette version is model-invisible but remains in the request-hash
    # target spec; keep it last so monkeypatch does not leak.
    # 调色表版本对模型不可见但继续进入请求哈希 target spec；保持最后执行，
    # 避免 monkeypatch 泄漏。
    monkeypatch.setattr("agents.general_vqa.agent.PALETTE_VERSION", "v2")
    assert_differs("palette version")


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




# ── end-to-end legacy-parity oracle (26 §11.1 / Gate 4) ──────────────────
# The final model-visible mask PNG produced by the real executor + agent must
# be byte-identical to the legacy restore-then-shrink pipeline, for a
# downscaled preview (medium ROI) — this is the evidence behind the
# version/cache decision (visual content version stays v2, no cache
# invalidation).
# 真实 executor + agent 产出的最终模型可见 mask PNG 必须与旧“恢复后缩小”
# 管线字节级一致（针对缩小的 preview / 中型 ROI）——这是 version/cache 决策
# （视觉内容版本保持 v2、不使缓存失效）的依据。


def _legacy_restore_oracle(
    model_mask: Image.Image,
    source_size: tuple[int, int],
) -> Image.Image:
    """Legacy full-resolution restore oracle: NEAREST to the padded canvas,
    then crop [0:W, 0:H] (test-only). 旧整分辨率恢复 oracle：NEAREST 到
    padded canvas 后裁切 [0:W, 0:H]（仅测试）。"""
    width, height = source_size
    padded_width = ((width + 1023) // 1024) * 1024
    padded_height = ((height + 1023) // 1024) * 1024
    restored = model_mask
    if (padded_width, padded_height) != model_mask.size:
        restored = model_mask.resize(
            (padded_width, padded_height), resample=Image.Resampling.NEAREST
        )
    return restored.crop((0, 0, width, height))


def _red_dominant_grid(image: Image.Image) -> list[list[int]]:
    """Deterministic class grid from RGB pixels: red >= 100 becomes class 1.
    由 RGB 像素构造的确定性 class grid：red >= 100 为 class 1。"""
    width, height = image.size
    data = image.tobytes()
    grid: list[list[int]] = []
    for y in range(height):
        row = [0] * width
        offset = y * width * 3
        for pixel in range(0, width * 3, 3):
            if data[offset + pixel] >= 100:
                row[pixel // 3] = 1
        grid.append(row)
    return grid


def test_end_to_end_mask_png_is_byte_identical_to_legacy_pipeline(
    tmp_path: Path,
) -> None:
    """26 §11.1 / Gate 4: for a medium ROI (1500x800 -> preview 1080x576) the
    real executor's preview-space evidence renders the exact same pure-mask
    PNG bytes as the legacy restore-then-shrink pipeline, with the same leaf
    hit decision and the same palette. 26 §11.1 / Gate 4：中型 ROI
    （1500x800 -> preview 1080x576）下，真实 executor 的 preview 空间证据渲染
    出的纯色 mask PNG 与旧“恢复后缩小”管线字节级一致，命中判定与调色表也
    一致。"""
    import io as _io

    from agents.base import VisualPlanBindings
    from agents.evidence_catalog import EvidenceCatalog
    from agents.general_vqa.evidence.executor import ObjectEvidenceExecutor
    from agents.general_vqa.evidence.schema import EvidencePreprocessing
    from models.base import SemanticMaskOutput
    from models.images import open_image_region_source

    roi_size = (1500, 800)
    source_image = Image.new("RGB", roi_size, (30, 40, 50))
    blob = Image.new("RGB", (400, 300), (200, 30, 40))
    source_image.paste(blob, (100, 80))

    catalog = EvidenceCatalog(
        {
            "catalog_version": "test-e2e-catalog-v1",
            "aliases": {},
            "parents": {},
            "leaves": {
                "building": {
                    "yolo_labels": [],
                    "yolo_enabled": False,
                    "segformer_labels": ["building"],
                    "segformer_binding": "seg_001",
                    "segformer_enabled": True,
                },
            },
            "task_capabilities": {
                "counting": ["building"],
                "fine_grained_counting": ["building"],
                "general_vqa": ["building"],
                "grounding": ["building"],
            },
        }
    )

    class _FixedMaskSegmenter:
        """Deterministic fake segmenter: red-dominant model-input pixels
        become class 1. 确定性假分割器：模型输入中红色主导像素为 class 1。"""

        id_to_label = {1: "building"}

        @property
        def cache_identity(self) -> ModelCacheIdentity:
            return ModelCacheIdentity(
                model="seg-e2e-v1",
                generation={"weights_sha256": "c" * 64},
                client_version="test",
            )

        def segment(self, image: Image.Image) -> SemanticMaskOutput:
            return SemanticMaskOutput(
                class_id_map=_red_dominant_grid(image),
                id_to_label=self.id_to_label,
                original_size=(1024, 1024),
                weights_sha256="c" * 64,
                diagnostics={"logical_model_id": "seg-e2e-v1"},
            )

    executor = ObjectEvidenceExecutor(
        catalog=catalog,
        policy=None,
        yolo_client=None,
        yolo_device=None,
        yolo_image_size=None,
        segmenter_clients={"seg_001": _FixedMaskSegmenter()},
        preprocessing=EvidencePreprocessing(max_tile_concurrency=2),
    )
    plan = _plan(["building"], assistance=True)
    view = MaterializedVisualView(
        image_id="img1",
        view_mode="full_image",
        source_size=roi_size,
        crop_xyxy=(0, 0, *roi_size),
        crop_size=roi_size,
    )
    source_path = tmp_path / "e2e.png"
    source_image.save(source_path, format="PNG")
    # Create the sample first, then overwrite its image file with the
    # intentionally large source so the view geometry (1500x800) matches the
    # decoded image. 先创建样本，再用有意设置的大图覆盖其图像文件，使视图
    # 几何（1500x800）与解码图像一致。
    sample = _sample(tmp_path)
    source_image.save(tmp_path / "img.png", format="PNG")
    client = _RecordingClient()
    palette = segformer_palette(["building"])
    agent = GeneralVQAAgent(client)

    # New pipeline: real executor (preview space) + real agent rendering.
    # 新管线：真实 executor（preview 空间）+ 真实 agent 渲染。
    asyncio.run(
        agent.run(
            sample,
            _context(
                tmp_path,
                client,
                plan=plan,
                views=(view,),
                bindings=VisualPlanBindings(vqa_evidence=executor),
            ),
        )
    )
    image_blocks = [
        block
        for block in client.calls[0]["messages"][1]["content"]
        if block["type"] == "image_url"
    ]
    # SegFormer-only branch: pure mask first, clean ROI second.
    # 仅 SegFormer 分支：纯色 mask 在前，干净 ROI 在后。
    assert len(image_blocks) == 2
    new_mask_png = _io.BytesIO(
        base64.b64decode(image_blocks[0]["image_url"]["url"].split(",", 1)[1])
    )
    assert Image.open(new_mask_png).size == (1080, 576)

    # Legacy oracle: build the model input exactly like prepare_segformer_roi,
    # derive the same model mask, restore full-res, extract the leaf boolean,
    # compose the WxH pure mask and NEAREST-shrink it to the preview.
    # 旧 oracle：与 prepare_segformer_roi 完全一致地构造模型输入、派生同一
    # model mask、整分辨率恢复、提取叶子 boolean、合成 WxH 纯色 mask 并
    # NEAREST 缩小到 preview。
    source = open_image_region_source(source_path)
    try:
        roi_crop = source.read_box((0, 0, *roi_size))
    finally:
        source.close()
    padded_width = ((1500 + 1023) // 1024) * 1024
    padded_height = ((800 + 1023) // 1024) * 1024
    canvas = Image.new("RGB", (padded_width, padded_height), (0, 0, 0))
    canvas.paste(roi_crop.convert("RGB"), (0, 0))
    model_input = canvas.resize((1024, 1024), resample=Image.Resampling.LANCZOS)
    model_mask = Image.frombytes(
        "I",
        (1024, 1024),
        array.array(
            "i",
            [
                value
                for row in _red_dominant_grid(model_input)
                for value in row
            ],
        ).tobytes(),
    )
    restored = _legacy_restore_oracle(model_mask, roi_size)
    legacy_boolean = Image.new("L", roi_size, 0)
    for y in range(roi_size[1]):
        for x in range(roi_size[0]):
            if restored.getpixel((x, y)) == 1:
                legacy_boolean.putpixel((x, y), 255)
    legacy_pure = render_pure_mask(roi_size, [("building", legacy_boolean)], palette)
    legacy_preview = make_preview(legacy_pure, resample=Image.Resampling.NEAREST)
    legacy_png = _io.BytesIO()
    legacy_preview.save(legacy_png, format="PNG")

    assert Image.open(new_mask_png).size == legacy_preview.size
    assert new_mask_png.getvalue() == legacy_png.getvalue()
