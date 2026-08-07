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
from agents.counting.backends.base import (
    CountingBackendOutcome,
    CountingBackendUnavailableError,
    CountingRequest,
)
from agents.counting.backends.registry import BackendRegistry
from agents.counting.backends.qwen_point import QwenPointCountingBackend
from agents.counting.schema import CountTargetSpec, CountingResult, TileCountResponse
from agents.counting.settings import CountingSettings
from agents.errors import AgentTaskMismatchError, DetectorWeightsMissingError
from agents.schema import AgentResult
from data.schema import (
    GroundTruth,
    ImageRef,
    TaskNormalization,
    UnifiedSample,
)
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
        from agents.counting.schema import GlobalPointObservation, PointProvenance

        points: list[GlobalPointObservation] = []
        if self._final_count > 0:
            points = [
                GlobalPointObservation(
                    global_id=f"{request.sample.sample_id}:det:p{i}",
                    target=request.target.canonical_label,
                    source_tile_id="r000_c000",
                    local_id=f"p{i}",
                    local_x_norm=500,
                    local_y_norm=500,
                    local_radius_norm=0,
                    global_x_px=request.image.width // 2,
                    global_y_px=request.image.height // 2,
                    global_x_norm=500,
                    global_y_norm=500,
                    radius_px=4.0,
                    confidence=0.9,
                    ownership_valid=True,
                    near_core_boundary=False,
                    accepted=True,
                    short_evidence="e",
                    provenance=PointProvenance(source="yolo_obb_center", source_class="car"),
                )
                for i in range(self._final_count)
            ]
        return CountingBackendOutcome(
            counting=CountingResult(
                sample_id=request.sample.sample_id,
                target=request.target.canonical_label,
                question=request.sample.question,
                source_width=request.image.width,
                source_height=request.image.height,
                tile_count=1,
                succeeded_tiles=["r000_c000"],
                global_points=points,
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


def test_run_agent_result_mode_uses_additional_agent_result(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _image(root)
    client = _FakeClient()
    registry = _registry(_qwen_backend(client))
    sample = _sample(root, answer_as_agent_result=True)
    execution = asyncio.run(_agent(client, registry).run(sample, _context(root)))
    # The primary payload and filename never change. / 主载荷与文件名绝不变。
    assert isinstance(execution.payload, CountingResult)
    assert execution.result_filename == "counting_result.json"
    assert "agent_result.json" in execution.additional_results
    payload = execution.additional_results["agent_result.json"]
    assert isinstance(payload, dict)  # JSON-safe serialization / JSON 安全序列化
    assert payload["agent_name"] == "counting_agent"
    assert payload["answer"] == "0"


@pytest.mark.parametrize("with_agent_answer", [False, True])
def test_primary_payload_is_always_counting_result(
    tmp_path: Path, with_agent_answer: bool
) -> None:
    """Every execution path returns CountingResult as the primary payload with
    the fixed filename. 所有执行路径都以 CountingResult 为主载荷并固定文件名。"""
    root = tmp_path / "data"
    _image(root)
    client = _FakeClient()
    registry = _registry(_qwen_backend(client))
    metadata = {"answer_as_agent_result": True} if with_agent_answer else {}
    execution = asyncio.run(_agent(client, registry).run(_sample(root, **metadata), _context(root)))
    assert isinstance(execution.payload, CountingResult)
    assert execution.result_filename == "counting_result.json"
    if with_agent_answer:
        assert "agent_result.json" in execution.additional_results


def test_task_mismatch_fails_before_any_work(tmp_path: Path) -> None:
    """Unsupported tasks fail before images, target parsing, budget, or model
    calls. 不支持的 task 在读图、解析 target、消费 budget 或调用模型前失败。"""
    root = tmp_path / "data"
    _image(root)
    client = _FakeClient()
    budget = _FakeBudget()
    registry = _registry(_qwen_backend(client))
    sample = UnifiedSample(
        sample_id="s1",
        dataset="parity",
        split="test",
        task="general_vqa",
        images=[ImageRef(image_id="i1", path="img.png", role="image")],
        question="Q",
        ground_truth=GroundTruth(answers=["x"]),
    )
    with pytest.raises(AgentTaskMismatchError):
        asyncio.run(_agent(client, registry).run(sample, _context(root, budget)))
    assert budget.qwen_calls == 0
    assert client.calls == []


def test_normalization_hint_takes_priority(tmp_path: Path) -> None:
    """normalization.count_target_hint beats the legacy metadata field.
    normalization.count_target_hint 优先于 legacy metadata 字段。"""
    root = tmp_path / "data"
    _image(root)
    client = _FakeClient()
    registry = _registry(_qwen_backend(client))
    sample = UnifiedSample(
        sample_id="s1",
        dataset="parity",
        split="test",
        task="counting",
        images=[ImageRef(image_id="i1", path="img.png", role="image")],
        question="How many ships?",
        ground_truth=GroundTruth(answers=["2"]),
        metadata={"count_target_hint": {"canonical_label": "old",
                                        "inclusion_rule": "r", "exclusion_rule": "e"}},
        normalization=TaskNormalization(
            source_task="counting",
            normalized_task="counting",
            normalizer="test", version="1",
            count_target_hint={
                "canonical_label": "ship",
                "inclusion_rule": "visible ship",
                "exclusion_rule": "occluded",
            },
        ),
    )
    execution = asyncio.run(_agent(client, registry).run(sample, _context(root)))
    assert execution.trace["target"] == "ship"
    assert execution.trace["target_classes"] == ["ship"]
    # Hint hit → no Qwen target call; only the tile count call remains.
    # hint 命中 → 无 Qwen target 调用；仅剩 tile 计数调用。
    assert len(client.calls) == 1


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


def test_fallback_disabled_raises_stable_error(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _image(root)
    client = _FakeClient()
    yolo = _FakeYoloBackend(error=DetectorWeightsMissingError("det-a", "det.pt"))
    registry = _registry(_qwen_backend(client), yolo)
    agent = _agent(client, registry, fallback_to_qwen_on_unavailable=False)
    with pytest.raises(CountingBackendUnavailableError, match="unavailable and no fallback"):
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
    # The qwen review also returns zero → the detector outcome stays final.
    # qwen 复核同样为零 → 检测器结果保持为最终结果。
    assert execution.trace["fallback_triggered"] is False
    assert execution.trace["review_backend"] == "qwen_point"
    assert execution.trace["final_backend"] == "det-a"
    assert execution.trace["yolo"]["used_for_final"] is True
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


# ── 全路径主输出契约 / primary payload across all paths (25.5) ───────────


def test_primary_payload_is_counting_result_on_all_execution_paths(tmp_path: Path) -> None:
    """Qwen point, detector, detector fallback, and zero-review override all
    return CountingResult with the fixed filename.
    Qwen point、检测器、检测器回退与 zero review 覆盖路径都以 CountingResult
    为主载荷并固定文件名。"""
    root = tmp_path / "data"
    _image(root)
    client = _FakeClient()

    # 1. qwen primary / qwen 主路径
    execution = asyncio.run(
        _agent(client, _registry(_qwen_backend(client))).run(_sample(root), _context(root))
    )
    assert isinstance(execution.payload, CountingResult)
    assert execution.result_filename == "counting_result.json"

    # 2. yolo primary / yolo 主路径
    yolo = _FakeYoloBackend(final_count=1)
    execution = asyncio.run(
        _agent(client, _registry(_qwen_backend(client), yolo)).run(_sample(root), _context(root))
    )
    assert isinstance(execution.payload, CountingResult)
    assert execution.result_filename == "counting_result.json"
    assert execution.trace["final_backend"] == "det-a"

    # 3. yolo unavailable fallback / yolo 不可用回退
    missing = _FakeYoloBackend(error=DetectorWeightsMissingError("det-a", "det.pt"))
    execution = asyncio.run(
        _agent(client, _registry(_qwen_backend(client), missing)).run(_sample(root), _context(root))
    )
    assert isinstance(execution.payload, CountingResult)
    assert execution.result_filename == "counting_result.json"
    assert execution.trace["final_backend"] == "qwen_point"

    # 4. zero review override / zero review 覆盖
    zero = _FakeYoloBackend(final_count=0)
    review_client = _FakeClient(tile_points=[(500, 500)])
    execution = asyncio.run(
        _agent(review_client, _registry(_qwen_backend(review_client), zero)).run(
            _sample(root), _context(root)
        )
    )
    assert isinstance(execution.payload, CountingResult)
    assert execution.result_filename == "counting_result.json"
    assert execution.trace["final_backend"] == "qwen_point"
    assert execution.trace["yolo"]["zero_overridden"] is True
