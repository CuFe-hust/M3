"""Contract tests for the counting agent.

计数 Agent 契约测试：显式计划执行、unavailable/runtime 回退、zero review、
trace 完整性、AgentResult 模式、data_root 图片解析。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from agents.base import AgentContext, AgentExecution
from agents.counting.agent import CountingAgent
from agents.counting.backends.base import CountingBackendOutcome, CountingRequest
from agents.counting.backends.registry import BackendRegistry
from agents.counting.backends.qwen_point import QwenPointCountingBackend
from agents.counting.schema import CountTargetSpec, CountingResult, TileCountResponse
from agents.counting.settings import CountingSettings
from agents.errors import DetectorWeightsMissingError
from agents.schema import AgentResult
from data.schema import GroundTruth, ImageRef, UnifiedSample
from models.base import ModelCacheIdentity

REPO_ROOT = Path(__file__).resolve().parents[3]


class _FakeBudget:
    def __init__(self) -> None:
        self.qwen_calls = 0

    def reserve_qwen(self) -> None:
        self.qwen_calls += 1

    def reserve_deepseek(self) -> None:
        pass


class _FakeClient:
    """Handles target parsing and tile-count responses for the Qwen backend.
    处理 Qwen 后端的 target 解析与 tile 计数响应。"""

    def __init__(self, tile_points: list[tuple[int, int]] | None = None) -> None:
        self.calls: list[Any] = []
        self._tile_points = tile_points or []

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="fake-model",
            generation={"temperature": 0.0, "do_sample": False, "max_tokens": 128},
            client_version="1",
        )

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls.append(request_meta.request_id)
        if response_model is CountTargetSpec:
            return response_model.model_validate(
                {
                    "canonical_label": "car",
                    "inclusion_rule": "visible vehicle",
                    "exclusion_rule": "occluded more than half",
                }
            )
        if response_model is TileCountResponse:
            from agents.counting.schema import LocalPointObservation

            return response_model.model_validate(
                {
                    "target": "car",
                    "tile_id": request_meta.tile_id,
                    "reported_count": len(self._tile_points),
                    "points": [
                        {
                            "local_id": f"p{index}",
                            "x": x,
                            "y": y,
                            "confidence": 0.9,
                            "short_evidence": "e",
                        }
                        for index, (x, y) in enumerate(self._tile_points)
                    ],
                }
            )
        raise AssertionError(f"unexpected response_model {response_model}")


class _FakeYoloBackend:
    """Detector backend whose count behaviour is controlled by the test.
    行为由测试控制的检测器后端。"""

    name = "det-a"
    priority = 100

    def __init__(self, error: Exception | None = None, final_count: int = 1) -> None:
        self._error = error
        self._final_count = final_count

    def is_available(self) -> bool:
        return True

    def supports(self, target: CountTargetSpec, hints: Any | None = None) -> bool:
        return True

    def trace_profile(self) -> dict[str, object]:
        return {"detector_name": self.name, "model_id": "m1"}

    async def count(self, request: CountingRequest, context: object) -> CountingBackendOutcome:
        if self._error is not None:
            raise self._error
        return CountingBackendOutcome(
            counting=CountingResult(
                sample_id=request.sample.sample_id,
                target=request.target.canonical_label,
                question=request.sample.question,
                source_width=request.image.width,
                source_height=request.image.height,
                tile_count=1,
                succeeded_tiles=["r000_c000"],
                final_count=self._final_count,
                status="completed" if self._final_count > 0 else "completed_with_warnings",
            ),
            trace={"detector_note": "fake"},
        )


def _sample(root: Path, **metadata) -> UnifiedSample:
    return UnifiedSample(
        sample_id="s1",
        dataset="parity",
        split="test",
        task="counting",
        images=[ImageRef(image_id="i1", path="img.png", role="image")],
        question="How many cars?",
        ground_truth=GroundTruth(answers=["1"]),
        metadata=metadata,
    )


def _image(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "img.png"
    Image.new("RGB", (100, 100), (1, 2, 3)).save(path, format="PNG")
    return path


def _context(root: Path, budget: _FakeBudget | None = None) -> AgentContext:
    return AgentContext(
        artifact_dir=root / "artifacts",
        qwen_client=None,
        call_budget=budget or _FakeBudget(),
        data_root=root,
    )


def _registry(*backends) -> BackendRegistry:
    registry = BackendRegistry()
    for backend in backends:
        registry.register(backend)
    return registry


def _agent(client: _FakeClient, registry: BackendRegistry, **overrides) -> CountingAgent:
    values = dict(target_prompt="Parse the target.", backend_registry=registry)
    values.update(overrides)
    return CountingAgent(client, **values)


def _qwen_backend(client: _FakeClient) -> QwenPointCountingBackend:
    return QwenPointCountingBackend(
        client, counting=CountingSettings(), system_prompt="Count points."
    )


# ── 主流程 / primary execution ────────────────────────────────────────────


def test_run_returns_counting_result_payload(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _image(root)
    client = _FakeClient()
    registry = _registry(_qwen_backend(client))
    execution = asyncio.run(
        _agent(client, registry).run(_sample(root), _context(root))
    )
    assert isinstance(execution, AgentExecution)
    assert isinstance(execution.payload, CountingResult)
    assert execution.result_filename == "counting_result.json"
    assert execution.payload.final_count == 0
    assert execution.trace["primary_backend"] == "qwen_point"
    assert execution.trace["executed_backend"] == "qwen_point"
    assert execution.trace["fallback_triggered"] is False
    assert execution.trace["yolo"] == {"attempted": False, "used_for_final": False}
    assert "target_classes" in execution.trace


def test_run_agent_result_mode_uses_additional_counting_result(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _image(root)
    client = _FakeClient()
    registry = _registry(_qwen_backend(client))
    sample = _sample(root, answer_as_agent_result=True)
    execution = asyncio.run(_agent(client, registry).run(sample, _context(root)))
    assert isinstance(execution.payload, AgentResult)
    assert execution.result_filename == "agent_result.json"
    assert execution.payload.agent_name == "counting_agent"
    assert execution.payload.answer == "0"
    assert "counting_result.json" in execution.additional_results
    payload = execution.additional_results["counting_result.json"]
    assert isinstance(payload, dict)  # JSON-safe serialization / JSON 安全序列化
    assert payload["final_count"] == 0


def test_run_requires_data_root(tmp_path: Path) -> None:
    client = _FakeClient()
    registry = _registry(_qwen_backend(client))
    sample = _sample(tmp_path / "data")
    context = AgentContext(
        artifact_dir=tmp_path / "artifacts",
        qwen_client=None,
        call_budget=_FakeBudget(),
        data_root=None,
    )
    with pytest.raises(RuntimeError, match="data_root"):
        asyncio.run(_agent(client, registry).run(sample, context))


# ── 回退 / fallback ────────────────────────────────────────────────────────


def test_unavailable_detector_falls_back_to_qwen(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _image(root)
    client = _FakeClient()
    yolo = _FakeYoloBackend(error=DetectorWeightsMissingError("det-a", "det.pt"))
    registry = _registry(_qwen_backend(client), yolo)
    execution = asyncio.run(_agent(client, registry).run(_sample(root), _context(root)))
    assert execution.trace["primary_backend"] == "det-a"
    assert execution.trace["executed_backend"] == "qwen_point"
    assert execution.trace["attempted_backends"] == ["det-a", "qwen_point"]
    assert execution.trace["fallback_triggered"] is True
    assert execution.trace["fallback_kind"] == "unavailable"
    assert "DetectorWeightsMissingError" in execution.trace["fallback_reason"]
    assert execution.trace["yolo"]["attempted"] is True
    assert execution.trace["yolo"]["used_for_final"] is False


def test_runtime_error_on_detector_falls_back_to_qwen(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _image(root)
    client = _FakeClient()
    yolo = _FakeYoloBackend(error=RuntimeError("inference boom"))
    registry = _registry(_qwen_backend(client), yolo)
    execution = asyncio.run(_agent(client, registry).run(_sample(root), _context(root)))
    assert execution.trace["fallback_kind"] == "runtime_error"
    assert "RuntimeError" in execution.trace["fallback_reason"]


def test_fallback_disabled_raises_original_error(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _image(root)
    client = _FakeClient()
    yolo = _FakeYoloBackend(error=DetectorWeightsMissingError("det-a", "det.pt"))
    registry = _registry(_qwen_backend(client), yolo)
    agent = _agent(client, registry, fallback_to_qwen_on_unavailable=False)
    with pytest.raises(DetectorWeightsMissingError):
        asyncio.run(agent.run(_sample(root), _context(root)))


def test_qwen_primary_runtime_error_is_not_swallowed(tmp_path: Path) -> None:
    """Non-detector primaries never fall back silently; errors propagate.
    非检测器主后端绝不静默回退；错误向上传播。"""
    root = tmp_path / "data"
    _image(root)
    client = _FakeClient()
    registry = _registry(_qwen_backend(client))

    class _BrokenQwen(_FakeClient):
        async def complete_json(self, **kwargs):
            raise RuntimeError("qwen boom")

    agent = _agent(_BrokenQwen(), registry)
    with pytest.raises(RuntimeError, match="qwen boom"):
        asyncio.run(agent.run(_sample(root), _context(root)))


# ── zero review / 零计数复核 ──────────────────────────────────────────────


def test_zero_review_overrides_detector_zero(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _image(root)
    client = _FakeClient(tile_points=[(500, 500)])  # review finds one / 复核发现 1 个
    yolo = _FakeYoloBackend(final_count=0)
    registry = _registry(_qwen_backend(client), yolo)
    execution = asyncio.run(_agent(client, registry).run(_sample(root), _context(root)))
    yolo_trace = execution.trace["yolo"]
    assert yolo_trace["zero_review_triggered"] is True
    assert yolo_trace["zero_review_backend"] == "qwen_point"
    assert yolo_trace["zero_overridden"] is True
    assert execution.trace["fallback_kind"] == "zero_review"
    assert execution.trace["executed_backend"] == "qwen_point"
    assert execution.payload.final_count == 1


def test_zero_review_confirms_zero_without_override(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _image(root)
    client = _FakeClient()
    yolo = _FakeYoloBackend(final_count=0)
    registry = _registry(_qwen_backend(client), yolo)
    agent = _agent(client, registry, verify_empty_with_qwen=True)
    execution = asyncio.run(agent.run(_sample(root), _context(root)))
    assert execution.trace["yolo"]["zero_overridden"] is False
    # The qwen review also returns zero → the detector outcome stays final,
    # although qwen_point was the last attempted backend.
    # qwen 复核同样为零 → 检测器结果保持为最终结果，qwen_point 仍是最后尝试
    # 的后端。
    assert execution.trace["fallback_triggered"] is False
    assert execution.trace["executed_backend"] == "qwen_point"
    assert execution.payload.final_count == 0


def test_zero_review_disabled_skips_verification(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _image(root)
    client = _FakeClient()
    yolo = _FakeYoloBackend(final_count=0)
    registry = _registry(_qwen_backend(client), yolo)
    agent = _agent(client, registry, verify_empty_with_qwen=False)
    execution = asyncio.run(agent.run(_sample(root), _context(root)))
    assert "zero_review_triggered" not in execution.trace["yolo"]
    assert execution.trace["executed_backend"] == "det-a"


# ── 边界 / boundaries ──────────────────────────────────────────────────────


def test_agent_has_no_dataset_branch_or_judge() -> None:
    source = (REPO_ROOT / "agents" / "counting" / "agent.py").read_text(encoding="utf-8")
    assert "VRSBench" not in source
    assert "vrsbench" not in source
    assert "spacers_agent" not in source
    assert "Judge" not in source
    assert "judge" not in source
    assert "report" not in source.casefold()
