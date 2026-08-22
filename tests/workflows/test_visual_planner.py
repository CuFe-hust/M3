"""Contract tests for the v5 visual-only planner.

v5 纯视觉规划器契约测试：验证单次调用、请求身份、能力闭集、确定性视图
物化以及不会把 GT 或物理路径带入规划请求。
"""

from __future__ import annotations

import asyncio
import base64
import io
import inspect
import json
from pathlib import Path

import pytest
from PIL import Image

from agents.evidence_catalog import EvidenceCatalog
from agents.schema import VisualTaskPlan
from application.prompts import PromptCatalog
from application.settings import VisualPlannerSettings
from data.schema import ImageRef, SampleDraft, UnifiedSample
from models.base import ModelCacheIdentity
from models.images import image_sha256, materialize_quantized_roi
from workflows.call_budget import CallBudget
from workflows.visual_planner import VisualTaskPlanError, VisualTaskPlanner

REPO_ROOT = Path(__file__).resolve().parents[2]

_CATALOG_DATA = {
    "catalog_version": "first-qwen-evidence-catalog-v1",
    "aliases": {"airplane": "plane"},
    "parents": {
        "vehicle": ["small-vehicle", "large-vehicle"],
        "building": ["building-outline"],
    },
    "leaves": {
        "small-vehicle": {"yolo_labels": ["small vehicle"], "yolo_enabled": True},
        "large-vehicle": {"yolo_labels": ["large vehicle"], "yolo_enabled": True},
        "building-outline": {"yolo_labels": ["building"], "yolo_enabled": True},
        "plane": {"yolo_labels": ["plane"], "yolo_enabled": True},
    },
    "task_capabilities": {
        task: ["small-vehicle", "large-vehicle", "building-outline", "plane"]
        for task in ("counting", "fine_grained_counting", "general_vqa", "grounding")
    },
}
_EXECUTABLE_BY_TASK = {
    task: ("small-vehicle", "large-vehicle")
    for task in (
        "counting",
        "fine_grained_counting",
        "general_vqa",
        "scene_classification",
        "multiple_choice_vqa",
        "spatial_relation",
        "grounding",
    )
}


def _catalog(version: str = "first-qwen-evidence-catalog-v1") -> EvidenceCatalog:
    data = dict(_CATALOG_DATA)
    data["catalog_version"] = version
    return EvidenceCatalog(data)


def _identity(
    model: str = "qwen-demo",
    generation: dict | None = None,
    client_version: str = "fake-client-v1",
    revision: str = "rev-1",
) -> ModelCacheIdentity:
    return ModelCacheIdentity(
        model=model,
        generation=generation if generation is not None else {"temperature": 0.0},
        client_version=client_version,
        revision=revision,
    )


class _FakeClient:
    """Record calls and validate responses through the requested schema.
    记录调用，并通过传入的 schema 校验响应。"""

    def __init__(self, *, identity: ModelCacheIdentity, response: dict | None = None) -> None:
        self._identity = identity
        self._response = response
        self.calls: list[dict] = []

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return self._identity

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls.append(
            {
                "messages": messages,
                "request_meta": request_meta,
                "max_tokens": max_tokens,
            }
        )
        if self._response is None:
            raise AssertionError("fake client configured without a response")
        return response_model.model_validate(self._response)


def _make_image(
    tmp_path: Path,
    name: str = "img1.png",
    size: tuple[int, int] = (64, 48),
    fill: int = 7,
) -> Path:
    path = tmp_path / name
    Image.new("RGB", size, (fill, fill + 1, fill + 2)).save(path, format="PNG")
    return path


def _sample(
    tmp_path: Path,
    *,
    question: str = "Are there any vehicles?",
    image_size: tuple[int, int] = (64, 48),
    ground_truth: dict | None = None,
) -> UnifiedSample:
    image_path = _make_image(tmp_path, size=image_size)
    return UnifiedSample(
        sample_id="s1",
        dataset="demo",
        split="val",
        task="general_vqa",
        images=[ImageRef(image_id="img1", path=image_path.name, role="image")],
        question=question,
        ground_truth=ground_truth,
        metadata={},
        normalization=None,
    )


def _draft(tmp_path: Path, *, image_size: tuple[int, int] = (64, 48)) -> SampleDraft:
    image_path = _make_image(tmp_path, size=image_size)
    return SampleDraft(
        sample_id="draft-1",
        dataset="demo",
        split="val",
        images=[ImageRef(image_id="img1", path=image_path.name, role="image")],
        question="Are there any vehicles?",
        metadata={},
    )


def _response(
    *,
    task: str = "general_vqa",
    assistance: bool = False,
    categories: tuple[str, ...] = (),
    explicit_region: bool = False,
    image_index: int | None = None,
    roi: tuple[int, int, int, int] | None = None,
    count_target: str | None = None,
    **overrides,
) -> dict:
    data = {
        "version": "visual-task-plan-v5",
        "task": task,
        "needs_visual_assistance": assistance,
        "object_categories": list(categories),
        "count_target": count_target,
        "region_request": {
            "explicit": explicit_region,
            "image_index": image_index,
            "roi_xyxy": roi,
        },
        "reason_codes": ["test"],
    }
    data.update(overrides)
    return data


def _planner(client: _FakeClient, **kwargs) -> VisualTaskPlanner:
    return VisualTaskPlanner(
        client,
        system_prompt="You are the visual-only planner.",
        prompt_version="v5",
        catalog=kwargs.pop("catalog", _catalog()),
        executable_categories_by_task=kwargs.pop(
            "executable_categories_by_task", _EXECUTABLE_BY_TASK
        ),
        **kwargs,
    )


def _run(planner: VisualTaskPlanner, view, tmp_path: Path, budget=None):
    return asyncio.run(
        planner.plan_with_views(
            view,
            data_root=tmp_path,
            artifact_dir=tmp_path / "artifacts",
            budget=budget,
        )
    )


def _preview_bytes(call: dict) -> bytes:
    image_url = next(
        item["image_url"]["url"]
        for item in call["messages"][1]["content"]
        if item.get("type") == "image_url"
    )
    return base64.b64decode(image_url.split(";base64,", 1)[1])


def test_planner_calls_once_with_ordered_previews_and_raw_question(tmp_path: Path) -> None:
    client = _FakeClient(identity=_identity(), response=_response())
    plan, views = _run(_planner(client), _sample(tmp_path, image_size=(2000, 1500)), tmp_path)

    assert isinstance(plan, VisualTaskPlan)
    assert len(client.calls) == 1
    content = client.calls[0]["messages"][1]["content"]
    assert [item["type"] for item in content] == ["image_url", "text"]
    assert content[-1]["text"] == "Are there any vehicles?"
    assert len(views) == 1
    assert Image.open(io.BytesIO(_preview_bytes(client.calls[0]))).size == (1080, 810)
    assert client.calls[0]["request_meta"].request_id == "s1:visual_task_plan"


def test_planner_request_excludes_gt_paths_and_runtime_choices(tmp_path: Path) -> None:
    client = _FakeClient(identity=_identity(), response=_response())
    sample = _sample(
        tmp_path,
        ground_truth={"raw": {"answer": "two vehicles"}},
    )
    _run(_planner(client), sample, tmp_path)

    payload = str(client.calls[0]["messages"])
    assert "ground_truth" not in payload
    assert "two vehicles" not in payload
    assert "img1.png" not in payload
    assert "checkpoint" not in payload
    assert "data:image/" in payload

    text_payload = next(
        item["text"]
        for item in client.calls[0]["messages"][1]["content"]
        if item.get("type") == "text"
    )
    assert text_payload == "Are there any vehicles?"


def test_planner_budget_and_request_identity_are_deterministic(tmp_path: Path) -> None:
    budget = CallBudget(max_qwen_calls=1)
    client = _FakeClient(identity=_identity(), response=_response())
    _run(_planner(client), _sample(tmp_path), tmp_path, budget=budget)
    assert budget.qwen_calls_used == 1
    meta = client.calls[0]["request_meta"]
    assert meta.prompt_version == "v5"
    assert meta.image_sha256 == image_sha256(_preview_bytes(client.calls[0]))
    assert len(meta.request_hash) == 64


def test_planner_materializes_quantized_roi_from_v5_box(tmp_path: Path) -> None:
    client = _FakeClient(
        identity=_identity(),
        response=_response(
            assistance=True,
            categories=("small-vehicle",),
            explicit_region=True,
            image_index=0,
            roi=(500, 500, 999, 999),
        ),
    )
    plan, views = _run(
        _planner(client),
        _sample(tmp_path, image_size=(2048, 1536)),
        tmp_path,
    )

    assert plan.needs_visual_assistance is True
    assert views[0].view_mode == "quantized_roi"
    assert views[0].requested_roi_xyxy_0_999 == (500, 500, 999, 999)
    assert views[0].requested_pixel_xyxy == (1025, 768, 2048, 1536)
    assert views[0].quantized_side == 1024
    assert views[0].ideal_square_xyxy == (1024, 640, 2048, 1664)
    assert views[0].crop_xyxy == (1024, 640, 2048, 1536)
    assert views[0].crop_size == (1024, 896)
    assert views[0].was_clipped is True


def test_planner_materializes_one_roi_and_keeps_other_images_full_image(
    tmp_path: Path,
) -> None:
    first_path = _make_image(tmp_path, "first.png", size=(1000, 700))
    second_path = _make_image(tmp_path, "second.png", size=(2048, 1536))
    sample = UnifiedSample(
        sample_id="multi",
        dataset="demo",
        split="val",
        task="general_vqa",
        images=[
            ImageRef(image_id="first", path=first_path.name, role="image"),
            ImageRef(image_id="second", path=second_path.name, role="context"),
        ],
        question="Look at the lower-right area in the second image.",
    )
    client = _FakeClient(
        identity=_identity(),
        response=_response(
            explicit_region=True,
            image_index=1,
            roi=(500, 500, 999, 999),
        ),
    )
    _plan, views = _run(_planner(client), sample, tmp_path)
    assert len(views) == 2
    assert views[0].view_mode == "full_image"
    assert views[0].crop_xyxy == (0, 0, 1000, 700)
    assert views[1].view_mode == "quantized_roi"
    assert views[1].crop_size == (1024, 896)


@pytest.mark.parametrize("size", [(1024, 2048), (1025, 2048), (2048, 1024)])
def test_planner_materializes_explicit_roi_for_any_source_size(
    tmp_path: Path,
    size: tuple[int, int],
) -> None:
    client = _FakeClient(
        identity=_identity(),
        response=_response(explicit_region=True, image_index=0, roi=(400, 400, 600, 600)),
    )
    _plan, views = _run(_planner(client), _sample(tmp_path, image_size=size), tmp_path)
    assert views[0].view_mode == "quantized_roi"
    assert views[0].roi_quantum == 1024


@pytest.mark.parametrize(
    ("roi_xyxy", "expected_side"),
    [
        ((0, 0, 333, 333), 1024),
        ((0, 0, 666, 666), 2048),
        ((0, 0, 999, 999), 3072),
    ],
)
def test_quantized_roi_rounds_longest_side_up_to_quantum(
    roi_xyxy: tuple[int, int, int, int],
    expected_side: int,
) -> None:
    geometry = materialize_quantized_roi((3072, 3072), roi_xyxy)
    assert geometry.quantized_side == expected_side
    assert geometry.quantized_side % 1024 == 0


@pytest.mark.parametrize(
    ("source_size", "roi_xyxy", "expected_crop"),
    [
        ((3000, 3000), (0, 0, 100, 100), (0, 0, 662, 662)),
        ((3000, 3000), (899, 0, 999, 100), (2337, 0, 3000, 662)),
        ((3000, 3000), (0, 899, 100, 999), (0, 2337, 662, 3000)),
        ((3000, 3000), (899, 899, 999, 999), (2337, 2337, 3000, 3000)),
        ((3000, 1000), (400, 0, 600, 999), (989, 0, 2013, 1000)),
        ((1000, 3000), (0, 400, 999, 600), (0, 989, 1000, 2013)),
    ],
)
def test_quantized_roi_clips_ideal_square_without_shifting(
    source_size: tuple[int, int],
    roi_xyxy: tuple[int, int, int, int],
    expected_crop: tuple[int, int, int, int],
) -> None:
    geometry = materialize_quantized_roi(source_size, roi_xyxy)
    assert geometry.crop_xyxy == expected_crop
    assert geometry.crop_size == (
        expected_crop[2] - expected_crop[0],
        expected_crop[3] - expected_crop[1],
    )
    assert geometry.was_clipped is True


def test_quantized_roi_preserves_audit_when_crop_is_the_full_source() -> None:
    geometry = materialize_quantized_roi((2048, 2048), (0, 0, 999, 999))
    assert geometry.requested_pixel_xyxy == (0, 0, 2048, 2048)
    assert geometry.quantized_side == 2048
    assert geometry.ideal_square_xyxy == (0, 0, 2048, 2048)
    assert geometry.crop_xyxy == (0, 0, 2048, 2048)
    assert geometry.crop_size == (2048, 2048)
    assert geometry.was_clipped is False


def test_quantized_roi_maps_source_edges_and_is_repeatable() -> None:
    first = materialize_quantized_roi((1000, 800), (0, 0, 999, 999))
    second = materialize_quantized_roi((1000, 800), (0, 0, 999, 999))
    assert first == second
    assert first.requested_pixel_xyxy == (0, 0, 1000, 800)
    assert first.crop_xyxy == (0, 0, 1000, 800)


def test_planner_uses_exif_transposed_rgb_size_as_geometry_authority(
    tmp_path: Path,
) -> None:
    path = tmp_path / "oriented.jpg"
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90° clockwise when displayed. 文件显示时顺时针旋转 90°。
    # The planner must use the normalized display dimensions, not the encoded
    # storage dimensions. 规划器必须使用规范化显示尺寸，而不是文件存储尺寸。
    Image.new("RGB", (1536, 2048), (5, 6, 7)).save(path, format="JPEG", exif=exif)
    sample = UnifiedSample(
        sample_id="exif",
        dataset="demo",
        split="val",
        task="general_vqa",
        images=[ImageRef(image_id="img1", path=path.name, role="image")],
        question="Look at the whole image.",
    )
    client = _FakeClient(
        identity=_identity(),
        response=_response(
            explicit_region=True,
            image_index=0,
            roi=(0, 0, 999, 999),
        ),
    )
    _plan, views = _run(_planner(client), sample, tmp_path)
    assert views[0].source_size == (2048, 1536)
    assert views[0].requested_pixel_xyxy == (0, 0, 2048, 1536)


def test_planner_accepts_schema_valid_plan_without_subjective_score(tmp_path: Path) -> None:
    client = _FakeClient(identity=_identity(), response=_response())
    plan, _views = _run(_planner(client), _sample(tmp_path), tmp_path)
    assert plan.version == "visual-task-plan-v5"
    with pytest.raises(TypeError):
        _planner(client, confidence_threshold=0.7)

    unavailable = _FakeClient(
        identity=_identity(),
        response=_response(assistance=True, categories=("small-vehicle",)),
    )
    with pytest.raises(VisualTaskPlanError, match="CAPABILITY_UNAVAILABLE"):
        _run(
            _planner(
                unavailable,
                executable_categories_by_task={
                    task: () for task in _EXECUTABLE_BY_TASK
                },
            ),
            _sample(tmp_path),
            tmp_path,
        )


def test_planner_schema_and_image_index_fail_closed(tmp_path: Path) -> None:
    invalid = _FakeClient(identity=_identity(), response=_response(task="unknown_task"))
    with pytest.raises(VisualTaskPlanError, match="SCHEMA_INVALID"):
        _run(_planner(invalid), _sample(tmp_path), tmp_path)

    bad_index = _FakeClient(
        identity=_identity(),
        response=_response(explicit_region=True, image_index=9, roi=(400, 400, 600, 600)),
    )
    with pytest.raises(VisualTaskPlanError, match="IMAGE_INDEX_INVALID"):
        _run(_planner(bad_index), _sample(tmp_path), tmp_path)


def test_planner_prompt_snapshot_and_artifact_payload_are_v5_only() -> None:
    client = _FakeClient(identity=_identity(), response=_response())
    planner = _planner(client)
    assert planner.prompt_snapshot_filename == "visual_task_plan_v5.runtime.md"
    assert planner.planning_parameters["planning_mode"] == "visual-task-plan-v5"
    assert planner.planning_parameters["roi_quantum"] == 1024
    assert PromptCatalog(REPO_ROOT / "prompts").version("visual_task_plan") == "v5"


def test_artifact_payload_preserves_quantized_roi_audit(tmp_path: Path) -> None:
    client = _FakeClient(
        identity=_identity(),
        response=_response(
            explicit_region=True,
            image_index=0,
            roi=(0, 0, 999, 999),
        ),
    )
    planner = _planner(client)
    plan, views = _run(
        planner,
        _sample(tmp_path, image_size=(2048, 2048)),
        tmp_path,
    )
    payload = planner.artifact_payload(plan, views)
    view_payload = payload["materialized_views"][0]
    assert view_payload["requested_roi_xyxy_0_999"] == [0, 0, 999, 999]
    assert view_payload["requested_pixel_xyxy"] == [0, 0, 2048, 2048]
    assert view_payload["roi_quantum"] == 1024
    assert view_payload["quantized_side"] == 2048
    assert view_payload["ideal_square_xyxy"] == [0, 0, 2048, 2048]
    assert view_payload["crop_xyxy"] == [0, 0, 2048, 2048]
    assert view_payload["crop_size"] == [2048, 2048]
    assert view_payload["was_clipped"] is False
    json.dumps(payload)


def test_counting_plan_requires_target_and_exact_leaf_expansion(tmp_path: Path) -> None:
    client = _FakeClient(
        identity=_identity(),
        response=_response(
            task="counting",
            assistance=True,
            categories=("small-vehicle", "large-vehicle"),
            count_target="vehicle",
        ),
    )
    plan, _views = _run(_planner(client), _sample(tmp_path), tmp_path)
    assert plan.count_target == "vehicle"
    assert plan.object_categories == ["small-vehicle", "large-vehicle"]

    incomplete = _FakeClient(
        identity=_identity(),
        response=_response(
            task="counting",
            assistance=True,
            categories=("small-vehicle",),
            count_target="vehicle",
        ),
    )
    with pytest.raises(VisualTaskPlanError, match="SCHEMA_INVALID"):
        _run(_planner(incomplete), _sample(tmp_path), tmp_path)


def test_counting_unknown_target_uses_generic_backend_contract(tmp_path: Path) -> None:
    client = _FakeClient(
        identity=_identity(),
        response=_response(task="counting", count_target="building"),
    )
    plan, _views = _run(_planner(client), _sample(tmp_path), tmp_path)
    assert plan.count_target == "building"
    assert plan.needs_visual_assistance is False
    assert plan.object_categories == []


def test_planner_binding_and_post_validation_are_leaf_only(tmp_path: Path) -> None:
    client = _FakeClient(
        identity=_identity(),
        response=_response(assistance=True, categories=("vehicle",)),
    )
    planner = _planner(client)
    binding = json.loads(planner.system_prompt.split("planner_binding=", 1)[1])
    assert binding["planning_mode"] == "visual-task-plan-v5"
    assert binding["canonical_leaf_categories"] == [
        "small-vehicle",
        "large-vehicle",
        "building-outline",
        "plane",
    ]
    assert binding["parent_expansions"]["vehicle"] == [
        "small-vehicle",
        "large-vehicle",
    ]
    assert "vehicle" not in binding["task_executable_categories"]["counting"]
    with pytest.raises(VisualTaskPlanError, match="SCHEMA_INVALID"):
        _run(planner, _sample(tmp_path), tmp_path)


@pytest.mark.parametrize(
    "task",
    ["general_vqa", "scene_classification", "multiple_choice_vqa", "spatial_relation"],
)
def test_planner_binds_four_vqa_tasks_to_shared_capability_owner(
    tmp_path: Path, task: str
) -> None:
    """The four GeneralVQAAgent tasks share one VQA capability owner: the
    system binding exposes the identical runtime executable leaves for every
    one of them, and a legal assistance plan passes post-validation for each.
    四个 GeneralVQAAgent task 共享同一 VQA capability owner：system binding 对
    它们暴露完全相同的运行时可执行叶子，且合法 assistance 计划在 post-validate
    时对每个 task 都通过。"""
    client = _FakeClient(identity=_identity(), response=_response())
    planner = _planner(client)
    binding = json.loads(planner.system_prompt.split("planner_binding=", 1)[1])
    shared = binding["task_executable_categories"]["general_vqa"]
    for other in ("scene_classification", "multiple_choice_vqa", "spatial_relation"):
        assert binding["task_executable_categories"][other] == shared
    assert binding["task_executable_categories"]["grounding"] == list(
        _EXECUTABLE_BY_TASK["grounding"]
    )
    plan_client = _FakeClient(
        identity=_identity(),
        response=_response(task=task, assistance=True, categories=("small-vehicle",)),
    )
    plan, _views = _run(_planner(plan_client), _sample(tmp_path), tmp_path)
    assert plan.task == task
    assert plan.needs_visual_assistance is True
    assert plan.object_categories == ["small-vehicle"]


@pytest.mark.parametrize(
    "task",
    ["general_vqa", "scene_classification", "multiple_choice_vqa", "spatial_relation"],
)
def test_planner_vqa_tasks_fail_closed_on_unknown_or_unavailable_leaf(
    tmp_path: Path, task: str
) -> None:
    """Unknown, non-executable, or runtime-unavailable leaves fail closed for
    every VQA task exactly like general_vqa. 未知、不可执行或运行时不可用叶子对
    每个 VQA task 都与 general_vqa 一样严格失败。"""
    # A non-canonical leaf fails schema validation. 非 canonical leaf 在 schema
    # 校验失败。
    unknown = _FakeClient(
        identity=_identity(),
        response=_response(task=task, assistance=True, categories=("not-a-leaf",)),
    )
    with pytest.raises(VisualTaskPlanError, match="SCHEMA_INVALID"):
        _run(_planner(unknown), _sample(tmp_path), tmp_path)
    # A canonical catalog leaf that is not in the runtime executable set.
    # 属于 catalog 但不是运行时可执行集合的 canonical leaf。
    not_executable = _FakeClient(
        identity=_identity(),
        response=_response(task=task, assistance=True, categories=("plane",)),
    )
    with pytest.raises(VisualTaskPlanError, match="CAPABILITY_UNAVAILABLE"):
        _run(_planner(not_executable), _sample(tmp_path), tmp_path)
    # A runtime-unavailable leaf: empty executable binding fails closed with
    # CAPABILITY_UNAVAILABLE. 运行时不可用叶子：空可执行绑定以
    # CAPABILITY_UNAVAILABLE 严格失败。
    unavailable = _FakeClient(
        identity=_identity(),
        response=_response(task=task, assistance=True, categories=("small-vehicle",)),
    )
    with pytest.raises(VisualTaskPlanError, match="CAPABILITY_UNAVAILABLE"):
        _run(
            _planner(
                unavailable,
                executable_categories_by_task={
                    key: () for key in _EXECUTABLE_BY_TASK
                },
            ),
            _sample(tmp_path),
            tmp_path,
        )


def test_planner_vqa_scope_is_frozen_into_planning_parameters() -> None:
    """The frozen VQA assistance scope travels in planning_parameters (and
    therefore the system prompt binding) when the composition root binds it.
    组合根绑定 scope 时，冻结的 VQA assistance scope 进入 planning_parameters
    （进而进入 system prompt binding）。"""
    client = _FakeClient(identity=_identity(), response=_response())
    scoped = _planner(
        client,
        vqa_assistance_scope="general-vqa-agent-tasks-v1",
    )
    assert (
        scoped.planning_parameters["vqa_assistance_scope"]
        == "general-vqa-agent-tasks-v1"
    )
    binding = json.loads(scoped.system_prompt.split("planner_binding=", 1)[1])
    assert binding["vqa_assistance_scope"] == "general-vqa-agent-tasks-v1"
    unscoped = _planner(client)
    assert "vqa_assistance_scope" not in unscoped.planning_parameters
    with pytest.raises(ValueError, match="scope"):
        _planner(client, vqa_assistance_scope="../bad")
    with pytest.raises(ValueError, match="scope"):
        _planner(client, vqa_assistance_scope="bad scope with spaces")


def test_planner_scope_has_no_subjective_gate() -> None:
    """Guard only the planner contract; detector/counting confidence remains separate.
    只守护 planner 契约；detector/counting 的 confidence 保持独立。"""
    assert "confidence" not in VisualTaskPlan.model_fields
    assert "confidence_threshold" not in VisualPlannerSettings.model_fields
    assert "confidence_threshold" not in inspect.signature(VisualTaskPlanner).parameters
    source = (REPO_ROOT / "workflows" / "visual_planner.py").read_text(encoding="utf-8")
    assert "plan." + "confidence" not in source
    assert "LOW_" + "CONFIDENCE" not in source
