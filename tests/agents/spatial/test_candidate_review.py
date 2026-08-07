"""Contract tests for the spatial candidate reviewer.

空间候选复核契约测试：compact 契约与 JSON 恢复、grid/普通 prompt 选择、
失败不丢初次结果、合并 audit 字段、budget 消费、cache identity。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from agents.schema import AgentResult, VisualEvidence
from agents.spatial.candidate_review import (
    SpatialCandidateReviewResult,
    SpatialCandidateReviewer,
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


class _ReviewClient:
    def __init__(self, review_result: SpatialCandidateReviewResult | None = None) -> None:
        self.calls: list[Any] = []
        self.review_result = review_result or SpatialCandidateReviewResult(
            boxes=[("small-vehicle", 100, 100, 200, 200)], complete=True
        )
        self.failure: BaseException | None = None

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="fake-model",
            generation={"temperature": 0.0, "do_sample": False, "max_tokens": 128},
            client_version="1",
        )

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls.append({"messages": messages, "request_meta": request_meta})
        if self.failure is not None:
            raise self.failure
        return response_model.model_validate(
            self.review_result.model_dump(mode="json")
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


def _first_result(items: list[VisualEvidence] | None = None) -> AgentResult:
    return AgentResult(
        agent_name="spatial_agent",
        answer="small-vehicle",
        evidence_items=items or [],
        status="completed",
    )


def _reviewer(client: _ReviewClient, **overrides) -> SpatialCandidateReviewer:
    values = dict(
        review_prompt="Enumerate candidates.",
        review_prompt_version="candidate-review-v1",
        grid_review_prompt="Enumerate grid candidates.",
        grid_review_prompt_version="candidate-review-grid-v1",
    )
    values.update(overrides)
    return SpatialCandidateReviewer(client, **values)


# ── compact 契约 / compact contract ───────────────────────────────────────


def test_review_result_validates_boxes() -> None:
    with pytest.raises(Exception):
        SpatialCandidateReviewResult(boxes=[("x", 100, 100, 50, 200)])  # reversed / 反向


def test_review_result_recover_json_payload() -> None:
    recovered = SpatialCandidateReviewResult.recover_json_payload(
        '{"boxes": ["small-vehicle", 100, 100, 200, 200], "complete": true}'
    )
    assert recovered is not None
    assert recovered["boxes"] == [["small-vehicle", 100, 100, 200, 200]]
    assert recovered["complete"] is True
    assert SpatialCandidateReviewResult.recover_json_payload("garbage") is None


def test_review_result_clamps_coordinate_drift() -> None:
    result = SpatialCandidateReviewResult(boxes=[("x", -1, 100, 1000, 200)])
    assert result.boxes[0] == ("x", 0, 100, 999, 200)


# ── 执行 / execution ──────────────────────────────────────────────────────


def test_review_merges_evidence_and_audits(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _image(root)
    client = _ReviewClient()
    budget = _FakeBudget()
    first = _first_result([VisualEvidence(label="small-vehicle", box=[100, 100, 200, 200])])
    reviewed = asyncio.run(
        _reviewer(client).review(
            _sample(root, spatial_query={"operation": "extreme_category", "target_label": "small-vehicle"}),
            first,
            tmp_path / "run",
            operation="extreme_category",
            target_label="small-vehicle",
            data_root=root,
            budget=budget,
        )
    )
    assert reviewed.geometry["candidate_review_used"] is True
    assert reviewed.geometry["candidate_review_added"] == 0
    assert reviewed.geometry["candidate_review_replaced"] == 0
    assert "candidate_review_geometry" in reviewed.geometry
    assert budget.qwen_calls == 1
    assert len(client.calls) == 1


def test_review_failure_keeps_first_result(tmp_path: Path) -> None:
    """Review failure must never lose the first result.
    复核失败绝不丢失初次结果。"""
    root = tmp_path / "data"
    _image(root)
    client = _ReviewClient()
    client.failure = RuntimeError("review crashed")
    first = _first_result([VisualEvidence(label="small-vehicle", box=[100, 100, 200, 200])])
    reviewed = asyncio.run(
        _reviewer(client).review(
            _sample(root),
            first,
            tmp_path / "run",
            operation="extreme_category",
            target_label="small-vehicle",
            data_root=root,
            budget=_FakeBudget(),
        )
    )
    assert reviewed is not first  # geometry updated / 几何已更新
    assert reviewed.geometry["candidate_review_used"] is True
    assert reviewed.geometry["candidate_review_error_type"] == "RuntimeError"
    assert reviewed.status == "partial"
    assert len(reviewed.evidence_items) == 1  # first evidence kept / 初次证据保留


def test_review_skipped_when_not_needed(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _image(root)
    client = _ReviewClient()
    first = _first_result()
    reviewed = asyncio.run(
        _reviewer(client).review(
            _sample(root),
            first,
            tmp_path / "run",
            operation="box_gap",
            target_label=None,
            data_root=root,
            budget=_FakeBudget(),
        )
    )
    assert reviewed is first  # untouched / 未触碰
    assert client.calls == []


def test_grid_review_uses_grid_prompt(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _image(root)
    client = _ReviewClient()
    asyncio.run(
        _reviewer(client).review(
            _sample(root, spatial_query={"operation": "grid_position", "target_label": "small-vehicle"}),
            _first_result(),
            tmp_path / "run",
            operation="grid_position",
            target_label="small-vehicle",
            data_root=root,
            budget=_FakeBudget(),
        )
    )
    meta = client.calls[0]["request_meta"]
    assert meta.prompt_version == "candidate-review-grid-v1"


def test_review_uses_full_cache_identity(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _image(root)
    client = _ReviewClient()
    asyncio.run(
        _reviewer(client).review(
            _sample(root),
            _first_result(),
            tmp_path / "run",
            operation="extreme_category",
            target_label="small-vehicle",
            data_root=root,
            budget=_FakeBudget(),
        )
    )
    meta = client.calls[0]["request_meta"]
    assert meta.request_hash
    assert meta.image_sha256
    assert meta.prompt_version == "candidate-review-v1"


def test_review_missing_identity_fails_before_budget(tmp_path: Path) -> None:
    from agents.counting.backends.base import MissingModelCacheIdentityError

    class _BareClient:
        async def complete_json(self, **kwargs):
            raise AssertionError("must not be called")

    root = tmp_path / "data"
    _image(root)
    budget = _FakeBudget()
    reviewer = _reviewer(_BareClient())  # type: ignore[arg-type]
    with pytest.raises(MissingModelCacheIdentityError, match="ModelCacheIdentity"):
        asyncio.run(
            reviewer.review(
                _sample(root),
                _first_result(),
                tmp_path / "run",
                operation="extreme_category",
                target_label="small-vehicle",
                data_root=root,
                budget=budget,
            )
        )
    assert budget.qwen_calls == 0


# ── 边界 / boundaries ─────────────────────────────────────────────────────


def test_candidate_review_has_no_dataset_branch() -> None:
    source = (Path(__file__).resolve().parents[3] / "agents" / "spatial" / "candidate_review.py").read_text(
        encoding="utf-8"
    )
    assert "VRSBench" not in source
    assert "vrsbench" not in source
    assert "spacers_agent" not in source
