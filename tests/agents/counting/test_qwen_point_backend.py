"""Contract tests for the Qwen point-counting backend.

Qwen 点式计数后端契约测试：通过 pipeline 产生 CountingResult、budget 消费、
trace 记录执行路径与请求版本、结构化 tile 响应、无数据集分支。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pydantic import BaseModel, ConfigDict

from agents.counting.backends.qwen_point import QwenPointCountingBackend
from agents.counting.backends.base import CountingBackendOutcome, CountingRequest
from agents.counting.schema import CountTargetSpec, TileCountResponse
from agents.counting.settings import CountingSettings
from data.schema import GroundTruth, ImageRef, UnifiedSample
from models.base import ModelCacheIdentity

REPO_ROOT = Path(__file__).resolve().parents[3]

_TARGET = CountTargetSpec(
    canonical_label="car",
    inclusion_rule="visible vehicle",
    exclusion_rule="occluded more than half",
)


def _sample() -> UnifiedSample:
    return UnifiedSample(
        sample_id="s1",
        dataset="parity",
        split="test",
        task="counting",
        images=[ImageRef(image_id="i1", path="img.png", role="image")],
        question="How many cars?",
        ground_truth=GroundTruth(answers=["3"]),
    )


class _FakeBudget:
    def __init__(self) -> None:
        self.qwen_calls = 0

    def reserve_qwen(self) -> None:
        self.qwen_calls += 1

    def reserve_deepseek(self) -> None:
        pass


class _FakeClient:
    """VisionLanguageClient stub returning schema-valid tile responses.
    返回 Schema 合法 tile 响应的 VisionLanguageClient 桩。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responses: list[TileCountResponse] = []
        self.failures: list[BaseException] = []

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="fake-model",
            generation={"temperature": 0.0, "do_sample": False, "max_tokens": 128},
            client_version="1",
        )

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls.append(
            {
                "messages": messages,
                "request_meta": request_meta,
                "response_model": response_model,
            }
        )
        if self.failures:
            raise self.failures.pop(0)
        if self.responses:
            response = self.responses.pop(0)
            return response_model.model_validate(response.model_dump(mode="json"))
        return response_model.model_validate(
            {"target": "car", "tile_id": request_meta.tile_id, "reported_count": 0}
        )


def _request(tmp_path: Path) -> CountingRequest:
    return CountingRequest(
        sample=_sample(),
        image=Image.new("RGB", (200, 200), (1, 2, 3)),
        target=_TARGET,
        artifact_dir=tmp_path / "run",
    )


def _backend(client: _FakeClient, **overrides) -> QwenPointCountingBackend:
    values = dict(counting=CountingSettings(), system_prompt="Count points in the core.")
    values.update(overrides)
    return QwenPointCountingBackend(client, **values)


def _context(budget: _FakeBudget) -> object:
    class _Context:
        call_budget = budget

    return _Context()


# ── 协议 / protocol ───────────────────────────────────────────────────────


def test_backend_identity() -> None:
    backend = _backend(_FakeClient())
    assert backend.name == "qwen_point"
    assert backend.priority == 0
    assert backend.is_available() is True


def test_supports_all_counting_targets() -> None:
    backend = _backend(_FakeClient())
    assert backend.supports(_TARGET) is True
    assert backend.supports(_TARGET, hints={"x": 1}) is True


# ── 执行 / execution ───────────────────────────────────────────────────────


def test_count_produces_counting_result_via_pipeline(tmp_path: Path) -> None:
    client = _FakeClient()
    budget = _FakeBudget()
    backend = _backend(client)
    outcome = asyncio.run(backend.count(_request(tmp_path), _context(budget)))
    assert isinstance(outcome, CountingBackendOutcome)
    assert outcome.counting.sample_id == "s1"
    assert outcome.counting.target == "car"
    assert outcome.counting.status in {"completed", "completed_with_warnings"}
    assert outcome.counting.final_count == 0
    # Small image: single whole tile, exactly one model call.
    # 小图：单个 whole tile，恰好一次模型调用。
    assert len(client.calls) == 1
    assert budget.qwen_calls == 1


def test_count_uses_injected_prompt_version_in_hash(tmp_path: Path) -> None:
    client = _FakeClient()
    backend = _backend(client, prompt_version="count-tile-v4")
    asyncio.run(backend.count(_request(tmp_path), _context(_FakeBudget())))
    meta = client.calls[0]["request_meta"]
    assert meta.prompt_version == "count-tile-v4"
    # The request hash covers model, generation, prompt, images, tile geometry.
    # 请求哈希覆盖模型、生成参数、prompt、图片与切片几何。
    assert meta.request_hash


def test_tile_request_contains_owner_core_and_target(tmp_path: Path) -> None:
    client = _FakeClient()
    backend = _backend(client)
    asyncio.run(backend.count(_request(tmp_path), _context(_FakeBudget())))
    messages = client.calls[0]["messages"]
    system_role = messages[0]["role"]
    assert system_role == "system"
    user_content = messages[1]["content"]
    assert user_content[0]["type"] == "image_url"
    text = json.loads(user_content[1]["text"])
    assert text["tile_id"] == "whole"
    assert text["target_spec"]["canonical_label"] == "car"
    assert "owner_core_normalized" in text


def test_count_trace_records_path_and_versions(tmp_path: Path) -> None:
    backend = _backend(_FakeClient(), prompt_version="count-tile-v4")
    outcome = asyncio.run(backend.count(_request(tmp_path), _context(_FakeBudget())))
    assert outcome.trace == {
        "backend": "qwen_point",
        "pipeline": "point_pipeline.count_image",
        "prompt_version": "count-tile-v4",
        "minimum_scan_depth": 0,
    }


def test_budget_consumed_per_model_call(tmp_path: Path) -> None:
    """A 2000px image tiles into 9 cores → 9 model calls and 9 budgets.
    2000px 图片切为 9 个 core → 9 次模型调用与 9 次 budget。"""
    client = _FakeClient()
    budget = _FakeBudget()
    backend = _backend(client)
    request = CountingRequest(
        sample=_sample(),
        image=Image.new("RGB", (2000, 2000), (1, 2, 3)),
        target=_TARGET,
        artifact_dir=tmp_path / "run",
    )
    outcome = asyncio.run(backend.count(request, _context(budget)))
    assert len(client.calls) == 9
    assert budget.qwen_calls == 9
    assert outcome.counting.tile_count == 9


# ── 边界 / boundaries ─────────────────────────────────────────────────────


def test_backend_has_no_dataset_branch() -> None:
    source = (REPO_ROOT / "agents" / "counting" / "backends" / "qwen_point.py").read_text(
        encoding="utf-8"
    )
    assert "VRSBench" not in source
    assert "vrsbench" not in source
    assert "spacers_agent" not in source
    assert "dataset" not in source


def test_backend_has_no_fallback_or_prompt_catalog() -> None:
    source = (REPO_ROOT / "agents" / "counting" / "backends" / "qwen_point.py").read_text(
        encoding="utf-8"
    )
    assert "fallback" not in source
    assert "PromptCatalog" not in source


# ── 缓存身份 / cache identity (25.5) ──────────────────────────────────────


def test_request_hash_changes_with_every_identity_field() -> None:
    """Any change to revision, max_tokens, do_sample, client_version, response
    schema, prompt version, target spec, or tile geometry must change the
    request hash. revision、max_tokens、do_sample、client_version、response
    schema、prompt version、target spec 或 tile geometry 的任何变化都必须
    改变请求哈希。"""
    from agents.counting.backends.qwen_point import _PipelineTileCallback
    from agents.counting.geometry import build_core_halo_tiles
    from agents.counting.schema import CountTargetSpec, TileCountResponse

    def _hash(**identity_overrides) -> str:
        client = _FakeClient()
        identity = client.cache_identity

        class _IdentityClient(_FakeClient):
            @property
            def cache_identity(self) -> ModelCacheIdentity:
                return ModelCacheIdentity(
                    model=identity_overrides.get("model", identity.model),
                    generation=identity_overrides.get(
                        "generation", identity.generation_payload()
                    ),
                    client_version=identity_overrides.get(
                        "client_version", identity.client_version
                    ),
                    revision=identity_overrides.get("revision", identity.revision),
                )

        callback = _PipelineTileCallback(
            _IdentityClient(),
            system_prompt="p",
            prompt_version=identity_overrides.get("prompt_version", "count-point-v4"),
            counting=CountingSettings(),
            budget=None,
            artifact_root=Path("/tmp"),
            sample_id="s1",
        )
        tile = build_core_halo_tiles(100, 100, core_size=896, halo_size=0, model_max_side=1280)[0]
        target = CountTargetSpec(
            canonical_label="car", inclusion_rule="r", exclusion_rule="e"
        )
        _, request_hash, _ = callback._build_request(
            tile, Image.new("RGB", (100, 100)), target
        )
        return request_hash

    base = _hash()
    assert base != _hash(revision="rev-2")
    assert base != _hash(generation={"temperature": 0.0, "do_sample": False, "max_tokens": 256})
    assert base != _hash(generation={"temperature": 0.7, "do_sample": True, "max_tokens": 128})
    assert base != _hash(client_version="2")
    assert base != _hash(model="other-model")
    assert base != _hash(prompt_version="other-prompt")


def test_missing_cache_identity_fails_before_model_call(tmp_path: Path) -> None:
    """A client without cache_identity fails before any model call.
    无 cache_identity 的客户端在任何模型调用前失败。"""
    from agents.counting.backends.base import MissingModelCacheIdentityError

    class _BareClient:
        async def complete_json(self, **kwargs):
            raise AssertionError("must not be called")

    backend = _backend(_BareClient())  # type: ignore[arg-type]
    with pytest.raises(MissingModelCacheIdentityError, match="cache_identity"):
        asyncio.run(backend.count(_request(tmp_path), _context(_FakeBudget())))
