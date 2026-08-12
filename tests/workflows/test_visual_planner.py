"""Contract tests for the isolated VisualPlanner (C4, 14A1).

C4 孤立 VisualPlanner 契约测试：恰好一次 schema 校验的 Qwen 调用、真实
ModelCacheIdentity 前置校验、request hash 覆盖全部语义输入、预算恰好消费
一次、5 条未冻结策略的 typed failure seam（无生产默认值）、严格拒绝额外
字段/目录外类别/退化 ROI/错误 image id/非 finite 值、预览只缩小不放大、
请求/产物不泄漏 GT、Base64 或绝对路径。
"""

from __future__ import annotations

import asyncio
import base64
import io
from pathlib import Path

import pytest
from PIL import Image

from agents.evidence_catalog import EvidenceCatalog
from agents.schema import FirstQwenVisualPlan
from application.prompts import PromptCatalog
from data.schema import ImageRef, UnifiedSample
from models.base import ModelCacheIdentity
from models.images import image_sha256
from workflows.call_budget import CallBudget
from workflows.visual_planner import VisualPlanError, VisualPlanner

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
    """Records every call; validates the configured response through the
    response model exactly like the runtime client, then returns it.
    记录每次调用；与运行时客户端一样通过 response model 校验配置的响应，
    然后返回。"""

    def __init__(self, *, identity, response=None, raise_error=None) -> None:
        self._identity = identity
        self._response = response
        self._raise_error = raise_error
        self.calls: list[dict] = []

    @property
    def cache_identity(self):
        return self._identity

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls.append(
            {"messages": messages, "request_meta": request_meta, "max_tokens": max_tokens}
        )
        if self._raise_error is not None:
            raise self._raise_error
        if self._response is None:
            raise AssertionError("fake client configured without a response")
        return response_model.model_validate(self._response)


def _make_image(tmp_path: Path, name: str = "img1.png", size=(64, 48), fill: int = 7) -> Path:
    image = Image.new("RGB", size, (fill, fill + 1, fill + 2))
    path = tmp_path / name
    image.save(path, format="PNG")
    return path


def _sample(
    tmp_path: Path,
    *,
    question: str = "Are there any vehicles?",
    image_fill: int = 7,
    image_size=(64, 48),
) -> UnifiedSample:
    image_path = _make_image(tmp_path, fill=image_fill, size=image_size)
    return UnifiedSample(
        sample_id="s1",
        dataset="demo",
        split="val",
        task="general_vqa",
        images=[ImageRef(image_id="img1", path=image_path.name, role="image")],
        question=question,
        ground_truth=None,
        metadata={},
        normalization=None,
    )


def _plan_response(
    *,
    confidence: float = 0.9,
    family: str = "object_evidence_vqa",
    categories: tuple[str, ...] = ("vehicle",),
    rois: list[dict] | None = None,
    **overrides,
) -> dict:
    data = {
        "version": "first-qwen-plan-v1",
        "execution_family": family,
        "confidence": confidence,
        "roi_plan": {"rois": rois or []},
        "evidence_request": (
            None
            if family == "direct_vqa"
            else {"composite_categories": list(categories)}
        ),
        "reason_codes": ["plan"],
    }
    data.update(overrides)
    return data


def _planner(client, *, catalog=None, prompt_version="v1", **kwargs) -> VisualPlanner:
    return VisualPlanner(
        client,
        system_prompt="You are the planner.",
        prompt_version=prompt_version,
        catalog=catalog or _catalog(),
        **kwargs,
    )


def _run(planner: VisualPlanner, sample: UnifiedSample, tmp_path: Path, budget=None):
    return asyncio.run(
        planner.plan(
            sample,
            data_root=tmp_path,
            artifact_dir=tmp_path / "artifacts",
            budget=budget,
        )
    )


# ── happy path / 正常路径 ───────────────────────────────────────────────


def test_plan_returns_plan_and_calls_client_exactly_once(tmp_path: Path) -> None:
    client = _FakeClient(
        identity=_identity(), response=_plan_response(categories=("vehicle",))
    )
    planner = _planner(client)
    plan = _run(planner, _sample(tmp_path), tmp_path)
    assert isinstance(plan, FirstQwenVisualPlan)
    assert plan.execution_family == "object_evidence_vqa"
    assert plan.evidence_request.composite_categories == ["vehicle"]
    assert len(client.calls) == 1


def test_plan_direct_vqa_carries_no_evidence_request(tmp_path: Path) -> None:
    client = _FakeClient(
        identity=_identity(), response=_plan_response(family="direct_vqa")
    )
    plan = _run(_planner(client), _sample(tmp_path), tmp_path)
    assert plan.execution_family == "direct_vqa"
    assert plan.evidence_request is None


def test_plan_dedupes_composite_categories_stably(tmp_path: Path) -> None:
    client = _FakeClient(
        identity=_identity(),
        response=_plan_response(categories=("vehicle", "vehicle", "building")),
    )
    plan = _run(_planner(client), _sample(tmp_path), tmp_path)
    assert plan.evidence_request.composite_categories == ["vehicle", "building"]


def test_plan_never_changes_sample(tmp_path: Path) -> None:
    client = _FakeClient(
        identity=_identity(), response=_plan_response(categories=("vehicle",))
    )
    sample = _sample(tmp_path)
    _run(_planner(client), sample, tmp_path)
    assert sample.task == "general_vqa"
    assert [ref.image_id for ref in sample.images] == ["img1"]


# ── request metadata / 请求元数据 ──────────────────────────────────────


def test_plan_request_meta_contract(tmp_path: Path) -> None:
    client = _FakeClient(
        identity=_identity(), response=_plan_response(categories=("vehicle",))
    )
    _run(_planner(client), _sample(tmp_path), tmp_path)
    (meta,) = [call["request_meta"] for call in client.calls]
    assert meta.request_id == "s1:visual_plan"
    assert meta.prompt_version == "v1"
    assert meta.sample_id == "s1"
    assert meta.image_sha256 == image_sha256(_preview_bytes(client.calls[0]))
    assert meta.artifact_dir == tmp_path / "artifacts" / "visual_plan"


def _preview_bytes(call: dict) -> bytes:
    (image_url,) = [
        item["image_url"]["url"]
        for item in call["messages"][1]["content"]
        if item.get("type") == "image_url"
    ]
    encoded = image_url.split(";base64,", 1)[1]
    return base64.b64decode(encoded)


def test_plan_user_payload_carries_question_and_closed_catalog(tmp_path: Path) -> None:
    client = _FakeClient(
        identity=_identity(), response=_plan_response(categories=("vehicle",))
    )
    sample = _sample(tmp_path, question="Any large vehicles?")
    _run(_planner(client), sample, tmp_path)
    (text,) = [
        item["text"]
        for item in client.calls[0]["messages"][1]["content"]
        if item.get("type") == "text"
    ]
    import json

    payload = json.loads(text)
    assert payload["question"] == "Any large vehicles?"
    assert payload["images"] == [{"image_id": "img1", "role": "image"}]
    assert payload["catalog_version"] == "first-qwen-evidence-catalog-v1"
    assert payload["composite_categories"] == ["vehicle", "building"]
    assert payload["answer_constraints"] == {}


def test_plan_never_leaks_ground_truth(tmp_path: Path) -> None:
    client = _FakeClient(
        identity=_identity(), response=_plan_response(categories=("vehicle",))
    )
    sample = _sample(tmp_path)
    sample.model_copy(update={"ground_truth": {"raw": {"answer": "two vehicles"}}})
    _run(_planner(client), sample, tmp_path)
    payload_text = str(client.calls[0]["messages"][1]["content"])
    assert "ground_truth" not in payload_text
    assert "two vehicles" not in payload_text


def test_plan_request_meta_never_holds_base64(tmp_path: Path) -> None:
    client = _FakeClient(
        identity=_identity(), response=_plan_response(categories=("vehicle",))
    )
    _run(_planner(client), _sample(tmp_path), tmp_path)
    dumped = client.calls[0]["request_meta"].model_dump(mode="json")
    joined = str(dumped)
    assert "base64" not in joined
    assert "data:image/" not in joined
    # The request hash is a digest, never the raw payload; the plan itself is
    # the only other artifact, and it holds no path or image content.
    # request hash 是摘要而非原始载荷；计划是唯一其他产物，不含路径或图像内容。
    plan = client.calls[0]["request_meta"].request_hash
    assert "/" not in plan and "\\" not in plan
    assert plan.isalnum()


# ── budget / 预算 ──────────────────────────────────────────────────────


def test_plan_consumes_exactly_one_budget_entry(tmp_path: Path) -> None:
    client = _FakeClient(
        identity=_identity(), response=_plan_response(categories=("vehicle",))
    )
    budget = CallBudget(max_qwen_calls=5)
    _run(_planner(client), _sample(tmp_path), tmp_path, budget=budget)
    assert budget.qwen_calls_used == 1


def test_plan_budget_exhausted_is_typed_failure(tmp_path: Path) -> None:
    client = _FakeClient(
        identity=_identity(), response=_plan_response(categories=("vehicle",))
    )
    budget = CallBudget(max_qwen_calls=0)
    with pytest.raises(VisualPlanError, match="BUDGET_EXHAUSTED"):
        _run(_planner(client), _sample(tmp_path), tmp_path, budget=budget)
    assert client.calls == []
    assert budget.qwen_calls_used == 0


# ── identity / 身份 ────────────────────────────────────────────────────


def test_plan_requires_real_model_cache_identity(tmp_path: Path) -> None:
    class _DuckClient:
        @property
        def cache_identity(self):
            return {"model": "fake"}

        async def complete_json(self, **kwargs):
            raise AssertionError("model must not be called without identity")

    budget = CallBudget(max_qwen_calls=5)
    with pytest.raises(VisualPlanError, match="CLIENT_UNAVAILABLE"):
        _run(_planner(_DuckClient()), _sample(tmp_path), tmp_path, budget=budget)
    assert budget.qwen_calls_used == 0


# ── client failures / 客户端失败 ───────────────────────────────────────


def test_plan_client_error_is_typed_failure(tmp_path: Path) -> None:
    client = _FakeClient(
        identity=_identity(), response=None, raise_error=RuntimeError("boom")
    )
    budget = CallBudget(max_qwen_calls=5)
    with pytest.raises(VisualPlanError, match="CLIENT_ERROR"):
        _run(_planner(client), _sample(tmp_path), tmp_path, budget=budget)
    # The call was attempted, so the budget entry was consumed.
    # 调用已尝试，因此预算条目已被消费。
    assert budget.qwen_calls_used == 1


# ── strict rejection / 严格拒绝 ────────────────────────────────────────


@pytest.mark.parametrize(
    "overrides",
    [
        {"execution_family": "hack", "evidence_request": None},  # wrong family
        {"version": "other-version"},  # frozen version mismatch
        {"confidence": 1.5},  # out-of-range confidence
        {"extra_field": True},  # unknown field
        {
            "roi_plan": {
                "rois": [
                    {"roi_id": "r1", "image_id": "img1", "xyxy": (0.3, 0.3, 0.3, 0.3)}
                ]
            }
        },  # degenerate box
        {
            "roi_plan": {
                "rois": [
                    {"roi_id": "r1", "image_id": "img1", "xyxy": (0.0, 0.0, 2.0, 1.0)}
                ]
            }
        },  # outside [0,1]
        {
            "roi_plan": {
                "rois": [
                    {
                        "roi_id": "r1",
                        "image_id": "img1",
                        "xyxy": (0.0, 0.0, float("nan"), 1.0),
                    }
                ]
            }
        },  # non-finite
        {"evidence_request": None},  # object_evidence_vqa missing request
        {"evidence_request": {"composite_categories": []}},  # empty categories
    ],
)
def test_plan_schema_invalid_responses_fail_typed(tmp_path: Path, overrides: dict) -> None:
    response = _plan_response(**overrides)
    client = _FakeClient(identity=_identity(), response=response)
    budget = CallBudget(max_qwen_calls=5)
    with pytest.raises(VisualPlanError, match="SCHEMA_INVALID"):
        _run(_planner(client), _sample(tmp_path), tmp_path, budget=budget)
    assert budget.qwen_calls_used == 1  # the call was attempted
    assert len(client.calls) == 1


def test_plan_out_of_catalog_category_fails_typed(tmp_path: Path) -> None:
    client = _FakeClient(
        identity=_identity(), response=_plan_response(categories=("flying_car",))
    )
    with pytest.raises(VisualPlanError, match="SCHEMA_INVALID"):
        _run(_planner(client), _sample(tmp_path), tmp_path)


def test_plan_roi_with_unknown_image_id_fails_typed(tmp_path: Path) -> None:
    client = _FakeClient(
        identity=_identity(),
        response=_plan_response(
            rois=[{"roi_id": "r1", "image_id": "ghost", "xyxy": (0.1, 0.1, 0.5, 0.5)}]
        ),
    )
    with pytest.raises(VisualPlanError, match="SCHEMA_INVALID"):
        _run(_planner(client), _sample(tmp_path), tmp_path)


def test_plan_low_confidence_is_typed_failure(tmp_path: Path) -> None:
    client = _FakeClient(
        identity=_identity(), response=_plan_response(confidence=0.4)
    )
    budget = CallBudget(max_qwen_calls=5)
    with pytest.raises(VisualPlanError, match="LOW_CONFIDENCE"):
        _run(_planner(client), _sample(tmp_path), tmp_path, budget=budget)
    assert budget.qwen_calls_used == 1


# ── preview decode / 预览解码 ──────────────────────────────────────────


def test_plan_missing_image_is_typed_failure(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    sample = sample.model_copy(
        update={"images": [sample.images[0].model_copy(update={"path": Path("nope.png")})]}
    )
    client = _FakeClient(
        identity=_identity(), response=_plan_response(categories=("vehicle",))
    )
    budget = CallBudget(max_qwen_calls=5)
    with pytest.raises(VisualPlanError, match="PREVIEW_DECODE_FAILED"):
        _run(_planner(client), sample, tmp_path, budget=budget)
    assert budget.qwen_calls_used == 0


def test_plan_corrupt_image_is_typed_failure(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"not an image at all")
    sample = UnifiedSample(
        sample_id="s1",
        dataset="demo",
        split="val",
        task="general_vqa",
        images=[ImageRef(image_id="img1", path=path.name, role="image")],
        question="Any vehicles?",
        ground_truth=None,
        metadata={},
        normalization=None,
    )
    client = _FakeClient(
        identity=_identity(), response=_plan_response(categories=("vehicle",))
    )
    with pytest.raises(VisualPlanError, match="PREVIEW_DECODE_FAILED"):
        _run(_planner(client), sample, tmp_path)


def test_plan_preview_shrinks_only_when_above_cap(tmp_path: Path) -> None:
    sample = _sample(tmp_path, image_size=(2000, 1000))
    client = _FakeClient(
        identity=_identity(), response=_plan_response(categories=("vehicle",))
    )
    _run(_planner(client), sample, tmp_path)
    preview = Image.open(io.BytesIO(_preview_bytes(client.calls[0])))
    assert preview.size == (1080, 540)
    # A small image is never upscaled. / 小图绝不放大。
    sample_small = _sample(tmp_path, image_size=(64, 48))
    client_small = _FakeClient(
        identity=_identity(), response=_plan_response(categories=("vehicle",))
    )
    _run(_planner(client_small), sample_small, tmp_path)
    small = Image.open(io.BytesIO(_preview_bytes(client_small.calls[0])))
    assert small.size == (64, 48)


# ── request hash coverage / request hash 覆盖 ──────────────────────────


def _run_and_hash(tmp_path: Path, planner: VisualPlanner, fill: int = 7):
    client = planner._client
    sample = _sample(tmp_path, image_fill=fill)
    _run(planner, sample, tmp_path)
    return client.calls[0]["request_meta"].request_hash


def test_plan_hash_covers_prompt_version(tmp_path: Path) -> None:
    response = _plan_response(categories=("vehicle",))
    a = _run_and_hash(
        tmp_path, _planner(_FakeClient(identity=_identity(), response=response), prompt_version="v1")
    )
    b = _run_and_hash(
        tmp_path, _planner(_FakeClient(identity=_identity(), response=response), prompt_version="v2")
    )
    assert a != b


def test_plan_hash_covers_catalog_version(tmp_path: Path) -> None:
    response = _plan_response(categories=("vehicle",))
    a = _run_and_hash(
        tmp_path,
        _planner(_FakeClient(identity=_identity(), response=response), catalog=_catalog("first-qwen-evidence-catalog-v1")),
    )
    b = _run_and_hash(
        tmp_path,
        _planner(_FakeClient(identity=_identity(), response=response), catalog=_catalog("first-qwen-evidence-catalog-v2")),
    )
    assert a != b


def test_plan_hash_covers_image_digest(tmp_path: Path) -> None:
    response = _plan_response(categories=("vehicle",))
    a = _run_and_hash(tmp_path, _planner(_FakeClient(identity=_identity(), response=response)), fill=7)
    b = _run_and_hash(tmp_path, _planner(_FakeClient(identity=_identity(), response=response)), fill=200)
    assert a != b


def test_plan_hash_covers_generation_and_identity(tmp_path: Path) -> None:
    response = _plan_response(categories=("vehicle",))
    base = _identity()
    variants = [
        _identity(generation={"temperature": 0.7}),
        _identity(client_version="fake-client-v2"),
        _identity(revision="rev-2"),
        _identity(model="qwen-demo-2"),
    ]
    hashes = [_run_and_hash(tmp_path, _planner(_FakeClient(identity=base, response=response)))]
    for identity in variants:
        hashes.append(
            _run_and_hash(
                tmp_path, _planner(_FakeClient(identity=identity, response=response))
            )
        )
    assert len(set(hashes)) == len(hashes)


def test_plan_hash_is_deterministic_for_identical_inputs(tmp_path: Path) -> None:
    response = _plan_response(categories=("vehicle",))
    a = _run_and_hash(tmp_path, _planner(_FakeClient(identity=_identity(), response=response)))
    b = _run_and_hash(tmp_path, _planner(_FakeClient(identity=_identity(), response=response)))
    assert a == b


# ── prompt catalog binding / PromptCatalog 绑定 ─────────────────────────


def test_visual_plan_prompt_bound_in_catalog() -> None:
    catalog = PromptCatalog(REPO_ROOT / "prompts")
    assert catalog.version("visual_plan") == "v1"
    assert catalog["visual_plan"].strip()
    assert (REPO_ROOT / "prompts" / "first_qwen_visual_plan_v1.md").is_file()
