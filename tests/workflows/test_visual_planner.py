"""Contract tests for the v4 visual-only planner.

v4 纯视觉规划器契约测试：验证单次调用、请求身份、能力闭集、确定性视图
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
from models.images import image_sha256
from workflows.call_budget import CallBudget
from workflows.visual_planner import VisualTaskPlanError, VisualTaskPlanner

REPO_ROOT = Path(__file__).resolve().parents[2]

_CATALOG_DATA = {
    "catalog_version": "first-qwen-evidence-catalog-v1",
    "composites": {
        "vehicle": ["small_vehicle", "large_vehicle"],
        "building": ["building_outline"],
    },
    "leaves": {
        "small_vehicle": {"yolo_labels": ["small-vehicle"], "yolo_enabled": True},
        "large_vehicle": {"yolo_labels": ["large-vehicle"], "yolo_enabled": True},
        "building_outline": {"yolo_labels": ["building"], "yolo_enabled": True},
    },
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
    focus: tuple[float, float] | None = None,
    count_target: str | None = None,
    **overrides,
) -> dict:
    data = {
        "version": "visual-task-plan-v4",
        "task": task,
        "needs_visual_assistance": assistance,
        "object_categories": list(categories),
        "count_target": count_target,
        "region_request": {
            "explicit": explicit_region,
            "image_index": image_index,
            "focus_xy_norm": focus,
        },
        "reason_codes": ["test"],
    }
    data.update(overrides)
    return data


def _planner(client: _FakeClient, **kwargs) -> VisualTaskPlanner:
    return VisualTaskPlanner(
        client,
        system_prompt="You are the visual-only planner.",
        prompt_version="v4",
        catalog=kwargs.pop("catalog", _catalog()),
        executable_categories=kwargs.pop("executable_categories", ("vehicle",)),
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
    assert meta.prompt_version == "v4"
    assert meta.image_sha256 == image_sha256(_preview_bytes(client.calls[0]))
    assert len(meta.request_hash) == 64


def test_planner_materializes_fixed_roi_from_v4_focus(tmp_path: Path) -> None:
    client = _FakeClient(
        identity=_identity(),
        response=_response(
            assistance=True,
            categories=("small-vehicle",),
            explicit_region=True,
            image_index=0,
            focus=(0.9, 0.8),
        ),
    )
    plan, views = _run(
        _planner(client),
        _sample(tmp_path, image_size=(2048, 1536)),
        tmp_path,
    )

    assert plan.needs_visual_assistance is True
    assert views[0].view_mode == "fixed_roi"
    assert views[0].crop_xyxy == (1024, 512, 2048, 1536)
    assert views[0].crop_size == (1024, 1024)


@pytest.mark.parametrize(
    ("size", "expected_mode"),
    [((1024, 2048), "full_image"), ((1025, 2048), "fixed_roi"), ((2048, 1024), "full_image")],
)
def test_planner_requires_both_source_dimensions_above_roi_size(
    tmp_path: Path,
    size: tuple[int, int],
    expected_mode: str,
) -> None:
    client = _FakeClient(
        identity=_identity(),
        response=_response(explicit_region=True, image_index=0, focus=(0.5, 0.5)),
    )
    _plan, views = _run(_planner(client), _sample(tmp_path, image_size=size), tmp_path)
    assert views[0].view_mode == expected_mode


def test_planner_accepts_schema_valid_plan_without_subjective_score(tmp_path: Path) -> None:
    client = _FakeClient(identity=_identity(), response=_response())
    plan, _views = _run(_planner(client), _sample(tmp_path), tmp_path)
    assert plan.version == "visual-task-plan-v4"
    with pytest.raises(TypeError):
        _planner(client, confidence_threshold=0.7)

    unavailable = _FakeClient(
        identity=_identity(),
        response=_response(assistance=True, categories=("small-vehicle",)),
    )
    with pytest.raises(VisualTaskPlanError, match="CAPABILITY_UNAVAILABLE"):
        _run(_planner(unavailable, executable_categories=()), _sample(tmp_path), tmp_path)


def test_planner_schema_and_image_index_fail_closed(tmp_path: Path) -> None:
    invalid = _FakeClient(identity=_identity(), response=_response(task="unknown_task"))
    with pytest.raises(VisualTaskPlanError, match="SCHEMA_INVALID"):
        _run(_planner(invalid), _sample(tmp_path), tmp_path)

    bad_index = _FakeClient(
        identity=_identity(),
        response=_response(explicit_region=True, image_index=9, focus=(0.5, 0.5)),
    )
    with pytest.raises(VisualTaskPlanError, match="IMAGE_INDEX_INVALID"):
        _run(_planner(bad_index), _sample(tmp_path), tmp_path)


def test_planner_prompt_snapshot_and_artifact_payload_are_v4_only() -> None:
    client = _FakeClient(identity=_identity(), response=_response())
    planner = _planner(client)
    assert planner.prompt_snapshot_filename == "visual_task_plan_v4.runtime.md"
    assert planner.planning_parameters["planning_mode"] == "visual-task-plan-v4"
    assert PromptCatalog(REPO_ROOT / "prompts").version("visual_task_plan") == "v4"


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
    assert binding["planning_mode"] == "visual-task-plan-v4"
    assert binding["canonical_leaf_categories"] == [
        "building-outline",
        "large-vehicle",
        "small-vehicle",
    ]
    assert binding["parent_expansions"]["vehicle"] == [
        "small-vehicle",
        "large-vehicle",
    ]
    assert "vehicle" not in binding["task_executable_categories"]["counting"]
    with pytest.raises(VisualTaskPlanError, match="SCHEMA_INVALID"):
        _run(planner, _sample(tmp_path), tmp_path)


def test_planner_scope_has_no_subjective_gate() -> None:
    """Guard only the planner contract; detector/counting confidence remains separate.
    只守护 planner 契约；detector/counting 的 confidence 保持独立。"""
    assert "confidence" not in VisualTaskPlan.model_fields
    assert "confidence_threshold" not in VisualPlannerSettings.model_fields
    assert "confidence_threshold" not in inspect.signature(VisualTaskPlanner).parameters
    source = (REPO_ROOT / "workflows" / "visual_planner.py").read_text(encoding="utf-8")
    assert "plan." + "confidence" not in source
    assert "LOW_" + "CONFIDENCE" not in source
