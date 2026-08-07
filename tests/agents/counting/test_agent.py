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
from agents.errors import (
    AgentExecutionError,
    AgentTaskMismatchError,
    DetectorWeightsMissingError,
)
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
    kind = "yolo_obb"
    priority = 100

    def __init__(self, error: Exception | None = None, final_count: int = 1) -> None:
        self._error = error
        self._final_count = final_count

    def is_enabled(self) -> bool:
        return True

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
    with pytest.raises(AgentExecutionError, match="DATA_ROOT_REQUIRED"):
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
    assert execution.trace["fallback_reason_code"] == "PRIMARY_BACKEND_UNAVAILABLE"
    assert execution.trace["fallback_error_type"] == "DetectorWeightsMissingError"
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
    assert execution.trace["fallback_reason_code"] == "PRIMARY_BACKEND_FAILED"
    assert execution.trace["fallback_error_type"] == "RuntimeError"


def test_fallback_disabled_raises_stable_error(tmp_path: Path) -> None:
    root = tmp_path / "data"
    _image(root)
    client = _FakeClient()
    yolo = _FakeYoloBackend(error=DetectorWeightsMissingError("det-a", "det.pt"))
    registry = _registry(_qwen_backend(client), yolo)
    agent = _agent(client, registry, fallback_to_qwen_on_unavailable=False)
    with pytest.raises(CountingBackendUnavailableError, match="PRIMARY_BACKEND_UNAVAILABLE"):
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
    with pytest.raises(AgentExecutionError, match="TARGET_PARSE_FAILED"):
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


# ── 25.6 收尾契约 / 25.6 finalization ─────────────────────────────────────


def test_public_error_identity_is_single_class() -> None:
    """agents top-level, agents.errors, and backends.base all expose the same
    CountingBackendUnavailableError class. agents 顶层、agents.errors 与
    backends.base 暴露同一个 CountingBackendUnavailableError 类。"""
    from agents import CountingBackendUnavailableError as PublicError
    from agents.counting.backends.base import (
        CountingBackendUnavailableError as BackendError,
    )
    from agents.errors import CountingBackendUnavailableError as CoreError

    assert PublicError is CoreError
    assert BackendError is CoreError


def test_public_error_is_raised_for_missing_plan(tmp_path: Path) -> None:
    """A missing backend plan surfaces as the public error class.
    缺失后端计划以公共错误类呈现。"""
    from agents import CountingBackendUnavailableError as PublicError
    from agents.errors import CountingBackendUnavailableError as CoreError

    root = tmp_path / "data"
    _image(root)
    client = _FakeClient()
    registry = BackendRegistry()  # empty → no plan / 空注册表 → 无计划
    with pytest.raises(CoreError) as info:
        asyncio.run(_agent(client, registry).run(_sample(root), _context(root)))
    assert isinstance(info.value, PublicError)
    assert info.value.reason_code == "NO_BACKEND_PLAN"


class _FakeQuantityProposalBackend:
    """Minimal quantity-proposal backend for agent integration.
    用于 Agent 集成的极简数量提议后端。"""

    name = "quantity_proposal"
    kind = "quantity_proposal"
    priority = 5

    def __init__(self, final_count: int = 0, error: Exception | None = None) -> None:
        self._final_count = final_count
        self._error = error

    def is_enabled(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def supports(self, target: CountTargetSpec, hints: Any | None = None) -> bool:
        return True

    async def count(self, request: CountingRequest, context: object) -> CountingBackendOutcome:
        if self._error is not None:
            raise self._error
        from agents.counting.schema import GlobalPointObservation, PointProvenance

        points: list[GlobalPointObservation] = []
        if self._final_count > 0:
            points = [
                GlobalPointObservation(
                    global_id=f"{request.sample.sample_id}:q:p{i}",
                    target=request.target.canonical_label,
                    source_tile_id="whole_image_overview",
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
                    provenance=PointProvenance(source="qwen_point", source_class="car"),
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
                succeeded_tiles=["whole_image_overview"],
                global_points=points,
                final_count=self._final_count,
                status="completed" if self._final_count > 0 else "completed_with_warnings",
            ),
            trace={"backend_kind": "quantity_proposal"},
        )


def test_quantity_proposal_zero_does_not_trigger_zero_review(tmp_path: Path) -> None:
    """Quantity proposal is not a detector: a zero result must not trigger the
    detector zero-review path. 数量提议不是检测器：零结果不得触发检测器
    zero review 路径。"""
    root = tmp_path / "data"
    _image(root)
    client = _FakeClient()
    quantity = _FakeQuantityProposalBackend(final_count=0)
    registry = _registry(_qwen_backend(client), quantity)
    execution = asyncio.run(_agent(client, registry).run(_sample(root), _context(root)))
    assert execution.trace["primary_backend"] == "quantity_proposal"
    assert execution.trace["primary_backend_kind"] == "quantity_proposal"
    assert execution.trace["final_backend"] == "quantity_proposal"
    assert execution.trace["review_backend"] is None
    assert execution.trace["fallback_triggered"] is False
    assert execution.trace["yolo"] == {"attempted": False, "used_for_final": False}
    # Only the Qwen target parse call happens; no review or tile calls.
    # 仅发生 Qwen target 解析调用；绝无复核或 tile 调用。
    assert client.calls == ["s1:target"]


def test_quantity_proposal_positive_result_payload(tmp_path: Path) -> None:
    """Quantity proposal positive result keeps CountingResult primary and
    CountingResult filename. 数量提议正数结果保持 CountingResult 主载荷与
    固定文件名。"""
    root = tmp_path / "data"
    _image(root)
    client = _FakeClient()
    quantity = _FakeQuantityProposalBackend(final_count=3)
    registry = _registry(_qwen_backend(client), quantity)
    execution = asyncio.run(_agent(client, registry).run(_sample(root), _context(root)))
    assert isinstance(execution.payload, CountingResult)
    assert execution.result_filename == "counting_result.json"
    assert execution.trace["yolo"]["attempted"] is False
    # answer_as_agent_result appends AgentResult as additional only.
    # answer_as_agent_result 仅把 AgentResult 追加为附加结果。
    sample = _sample(root, answer_as_agent_result=True)
    execution2 = asyncio.run(_agent(client, registry).run(sample, _context(root)))
    assert isinstance(execution2.payload, CountingResult)
    assert "agent_result.json" in execution2.additional_results


def test_quantity_proposal_runtime_error_is_agent_error(tmp_path: Path) -> None:
    """Quantity proposal runtime errors become AgentExecutionError and never
    take the detector fallback path. 数量提议运行时错误转换为
    AgentExecutionError，绝不走检测器回退路径。"""
    root = tmp_path / "data"
    _image(root)
    client = _FakeClient()
    quantity = _FakeQuantityProposalBackend(error=RuntimeError("proposal boom"))
    registry = _registry(_qwen_backend(client), quantity)
    with pytest.raises(AgentExecutionError, match="PRIMARY_BACKEND_FAILED"):
        asyncio.run(_agent(client, registry).run(_sample(root), _context(root)))


_SENSITIVE_ERROR_TEXT = (
    "/home/user/private/model.pt "
    "C:\\secret\\models\\det.onnx "
    "sk-test-secret "
    "Bearer abcdef "
    "data:image/png;base64,AAAA"
)


def test_public_error_and_trace_are_sanitized(tmp_path: Path) -> None:
    """Raw exception text with paths, secrets, and Base64 must never reach
    public errors, traces, warnings, or additional results.
    含路径、密钥与 Base64 的原始异常文本绝不进入公共错误、trace、warnings
    或附加结果。"""
    root = tmp_path / "data"
    _image(root)
    client = _FakeClient()
    quantity = _FakeQuantityProposalBackend(error=RuntimeError(_SENSITIVE_ERROR_TEXT))
    registry = _registry(_qwen_backend(client), quantity)
    with pytest.raises(AgentExecutionError) as info:
        asyncio.run(_agent(client, registry).run(_sample(root), _context(root)))
    # The public message carries only the stable cause code; the raw exception
    # text stays in __cause__ (standard chaining) but never in the message.
    # 公共消息只携带稳定 cause code；原始异常文本保留在 __cause__（标准
    # chaining），绝不进入消息。
    public_text = str(info.value)
    assert public_text == (
        "Agent 'counting_agent' failed on sample 's1': PRIMARY_BACKEND_FAILED"
    )
    for token in ("/home/user/private", "sk-test-secret", "Bearer abcdef", "base64,AAAA"):
        assert token not in public_text, token

    # A successful run with a backend trace must also stay clean.
    # 成功运行的后端 trace 同样保持干净。
    ok_quantity = _FakeQuantityProposalBackend(final_count=0)
    registry2 = _registry(_qwen_backend(client), ok_quantity)
    execution = asyncio.run(_agent(client, registry2).run(_sample(root), _context(root)))
    dump = str(execution.trace) + str(execution.additional_results)
    for token in ("/home/user/private", "sk-test-secret", "Bearer abcdef", "base64,AAAA"):
        assert token not in dump, token


# ── 25.7 真实 YOLO 全失败传播 / real YOLO all-tile failure propagation ───


def _real_yolo_backend(tmp_path: Path, *, exploding: bool) -> Any:
    """A real YoloOBBCountingBackend over a fake runtime model.
    基于假运行时模型的真实 YoloOBBCountingBackend。"""
    from agents.counting.backends.yolo_model_store import YoloModelStore
    from agents.counting.backends.yolo_obb import YoloOBBCountingBackend
    from agents.counting.settings import YoloDetectorSettings

    class _FakePredictModel:
        task = "obb"
        names = {0: "car"}
        providers = ("CPUExecutionProvider",)
        requested_provider = "CPUExecutionProvider"
        requested_device = "cpu"
        resolved_provider = "CPUExecutionProvider"
        resolved_device = "cpu"
        cpu_fallback_used = False

        def predict(self, **kwargs):
            raise RuntimeError("gpu driver crashed")

    import hashlib

    detector = YoloDetectorSettings(
        name="det-a",
        enabled=True,
        weights=tmp_path / "det.pt",
        model_id="m1",
        sha256=hashlib.sha256(b"fake").hexdigest(),
        classes=["car"],
        device="cpu",
        require_cuda=False,
        allow_cpu_fallback=False,
    )
    (tmp_path / "det.pt").write_bytes(b"fake")
    store = YoloModelStore(loader=lambda path: _FakePredictModel())
    return YoloOBBCountingBackend(detector, counting=CountingSettings(), model_store=store)


def test_real_yolo_all_tiles_failed_falls_back_to_qwen(tmp_path: Path) -> None:
    """Real YOLO backend with every tile failing triggers the Qwen fallback
    through the CountingAgent with a fully audited trace.
    真实 YOLO 后端全部 tile 失败时经 CountingAgent 触发 Qwen 回退，trace
    完整可审计。"""
    root = tmp_path / "data"
    _image(root)
    client = _FakeClient()
    yolo = _real_yolo_backend(tmp_path, exploding=True)
    registry = _registry(_qwen_backend(client), yolo)
    execution = asyncio.run(_agent(client, registry).run(_sample(root), _context(root)))
    assert isinstance(execution.payload, CountingResult)
    assert execution.trace["primary_backend"] == "det-a"
    assert execution.trace["primary_backend_kind"] == "yolo_obb"
    assert execution.trace["final_backend"] == "qwen_point"
    assert execution.trace["final_backend_kind"] == "qwen_point"
    assert execution.trace["fallback_triggered"] is True
    assert execution.trace["fallback_kind"] == "runtime_error"
    assert execution.trace["fallback_reason_code"] == "PRIMARY_BACKEND_FAILED"
    assert execution.trace["fallback_error_type"] == "DetectorInferenceError"
    assert execution.trace["attempted_backends"] == ["det-a", "qwen_point"]
    assert execution.trace["yolo"]["used_for_final"] is False
    # No raw exception text anywhere. / 任何位置都不含原始异常文本。
    dump = str(execution.trace) + str(execution.additional_results)
    assert "gpu driver crashed" not in dump


def test_real_yolo_all_tiles_failed_without_fallback(tmp_path: Path) -> None:
    """Fallback disabled: the all-tiles-failed YOLO surfaces as a stable
    AgentExecutionError. 回退禁用：全失败 YOLO 以稳定 AgentExecutionError
    呈现。"""
    root = tmp_path / "data"
    _image(root)
    client = _FakeClient()
    yolo = _real_yolo_backend(tmp_path, exploding=True)
    registry = _registry(_qwen_backend(client), yolo)
    agent = _agent(client, registry, fallback_to_qwen_on_error=False)
    with pytest.raises(AgentExecutionError, match="PRIMARY_BACKEND_FAILED"):
        asyncio.run(agent.run(_sample(root), _context(root)))


# ── 契约签名 / contract signatures ─────────────────────────────────────────


def test_counting_agent_public_signatures_avoid_any() -> None:
    """Counting public entry signatures use domain contracts, never Any.
    计数公共入口签名使用领域契约类型而非 Any。"""
    import typing

    from PIL import Image

    from agents.counting.agent import _resolve_sample_image
    from data.schema import UnifiedSample
    from models.base import VisionLanguageClient

    run_hints = typing.get_type_hints(CountingAgent.run)
    assert run_hints["sample"] is UnifiedSample
    assert run_hints["context"] is AgentContext
    assert run_hints["return"] is AgentExecution

    init_hints = typing.get_type_hints(CountingAgent.__init__)
    assert init_hints["client"] is VisionLanguageClient

    target_hints = typing.get_type_hints(CountingAgent._target)
    assert target_hints["sample"] is UnifiedSample

    resolve_hints = typing.get_type_hints(_resolve_sample_image)
    assert resolve_hints["sample"] is UnifiedSample
    assert resolve_hints["context"] is AgentContext
    assert resolve_hints["return"] is Image.Image
