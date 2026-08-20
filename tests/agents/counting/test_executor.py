"""Contract tests for CountingPlanExecutor.

CountingPlanExecutor 契约测试：primary 执行、detector unavailable/runtime
回退、fallback 缺失/失败、zero review 覆盖/保留/失败、invalid kind、
attempted 顺序与原始异常文本隔离。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from agents.base import AgentContext
from agents.counting.backends.base import (
    BackendPlan,
    CountingBackendOutcome,
    CountingRequest,
)
from agents.counting.backends.qwen_point import QwenPointCountingBackend
from agents.counting.backends.registry import BackendRegistry
from agents.counting.backends.selector import BackendSelector
from agents.counting.executor import (
    CountingExecutionPolicy,
    CountingExecutionResult,
    CountingPlanExecutor,
    _apply_disagreement_review,
)
from agents.counting.schema import (
    CountTargetSpec,
    CountingResult,
    DisagreementReview,
    GlobalPointObservation,
    PointProvenance,
    TileCountResponse,
)
from agents.counting.settings import CountingSettings
from agents.errors import (
    AgentExecutionError,
    CountingBackendUnavailableError,
    DetectorWeightsMissingError,
)
from data.schema import (
    GroundTruth,
    ImageRef,
    UnifiedSample,
)
from models.base import ModelCacheIdentity

_TARGET = CountTargetSpec(
    canonical_label="car",
    inclusion_rule="visible vehicle",
    exclusion_rule="occluded more than half",
)

_SENSITIVE_ERROR_TEXT = (
    "/home/user/private/model.pt "
    "C:\\secret\\models\\det.onnx "
    "sk-test-secret "
    "Bearer abcdef "
    "data:image/png;base64,AAAA"
)


def _review_point(global_id: str, confidence: float) -> GlobalPointObservation:
    return GlobalPointObservation(
        global_id=global_id,
        target="car",
        source_tile_id="r000_c000",
        local_id=global_id,
        local_x_norm=500,
        local_y_norm=500,
        local_radius_norm=0,
        global_x_px=32,
        global_y_px=32,
        global_x_norm=500,
        global_y_norm=500,
        radius_px=4.0,
        confidence=confidence,
        ownership_valid=True,
        near_core_boundary=False,
        accepted=True,
        short_evidence="e",
        provenance=PointProvenance(source="yolo_obb_center", source_class="car"),
    )


def _review_fixture() -> SimpleNamespace:
    return SimpleNamespace(
        points=[
            _review_point("candidate-a", 0.99),
            _review_point("candidate-b", 0.20),
            _review_point("fused-consensus", 0.95),
        ],
        review_candidates=[
            {
                "conflict_id": "candidate-a|candidate-b",
                "candidate_ids": ["candidate-a", "candidate-b"],
                "candidate_points": [],
            }
        ],
        unresolved_conflicts=["candidate-a|candidate-b"],
        warnings=[],
    )


class _FakeBudget:
    def __init__(self) -> None:
        self.qwen_calls = 0

    def reserve_qwen(self) -> None:
        self.qwen_calls += 1

    def reserve_deepseek(self) -> None:
        pass


class _FakeClient:
    """Handles tile-count responses for the Qwen backend.
    处理 Qwen 后端的 tile 计数响应。"""

    def __init__(
        self,
        tile_points: list[tuple[int, int]] | None = None,
        *,
        target_label: str = "car",
    ) -> None:
        self.calls: list[Any] = []
        self._tile_points = tile_points or []
        self._target_label = target_label

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="fake-model",
            generation={"temperature": 0.0, "do_sample": False, "max_tokens": 128},
            client_version="1",
        )

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls.append(request_meta.request_id)
        if response_model is TileCountResponse:
            from agents.counting.schema import LocalPointObservation

            return response_model.model_validate(
                {
                    "target": self._target_label,
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

    def __init__(
        self,
        error: Exception | None = None,
        final_count: int = 1,
        *,
        available: bool = True,
        point_confidence: float = 0.9,
    ) -> None:
        self._error = error
        self._final_count = final_count
        self._available = available
        self._point_confidence = point_confidence

    def is_enabled(self) -> bool:
        return True

    def is_available(self) -> bool:
        return self._available

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
                    global_id=f"{request.sample.sample_id}:{self.name}:p{i}",
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
                    confidence=self._point_confidence,
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


class _FakeDisagreementReviewer:
    name = "qwen_point"
    kind = "qwen_point"
    priority = 0

    def __init__(self, review: DisagreementReview) -> None:
        self.review = review
        self.calls = 0

    def is_enabled(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def supports(self, target: CountTargetSpec, hints: Any | None = None) -> bool:
        return True

    async def count(self, request: CountingRequest, context: object) -> CountingBackendOutcome:
        return CountingBackendOutcome(
            counting=CountingResult(
                sample_id=request.sample.sample_id,
                target=request.target.canonical_label,
                question=request.sample.question,
                source_width=request.image.width,
                source_height=request.image.height,
                tile_count=1,
                final_count=0,
                status="completed",
            )
        )

    async def review_disagreements(self, *, request, conflicts, context) -> DisagreementReview:
        self.calls += 1
        return self.review


class _FakeQuantityProposalBackend:
    """Quantity proposal is never treated as a detector.
    数量提议后端；绝不被当作检测器。"""

    name = "quantity_proposal"
    kind = "quantity_proposal"
    priority = 5

    def __init__(
        self,
        final_count: int = 0,
        error: Exception | None = None,
        *,
        available: bool = True,
    ) -> None:
        self._final_count = final_count
        self._error = error
        self._available = available

    def is_enabled(self) -> bool:
        return True

    def is_available(self) -> bool:
        return self._available

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


class _UnknownKindBackend(_FakeYoloBackend):
    name = "mystery"
    kind = "mystery_kind"


class _FakeSemanticBackend(_FakeQuantityProposalBackend):
    name = "segmenter-a"
    kind = "semantic_segmentation"
    priority = 100


class _FakeRaisingQwenBackend:
    """Qwen-kind backend that always raises; used to force genuine fallback
    and review failures. 始终抛错的 qwen 类后端；用于制造真实回退/复核失败。"""

    name = "qwen_point"
    kind = "qwen_point"
    priority = 0

    def __init__(self, error: Exception) -> None:
        self._error = error

    def is_enabled(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def supports(self, target: CountTargetSpec, hints: Any | None = None) -> bool:
        return True

    async def count(self, request: CountingRequest, context: object) -> CountingBackendOutcome:
        raise self._error


def _sample() -> UnifiedSample:
    return UnifiedSample(
        sample_id="s1",
        dataset="parity",
        split="test",
        task="counting",
        images=[ImageRef(image_id="i1", path="img.png", role="image")],
        question="How many cars?",
        ground_truth=GroundTruth(answers=["1"]),
    )


def _context(root: Path) -> AgentContext:
    return AgentContext(
        artifact_dir=root / "artifacts",
        qwen_client=None,
        call_budget=_FakeBudget(),
        data_root=None,
    )


def _request(root: Path, target: CountTargetSpec = _TARGET) -> CountingRequest:
    return CountingRequest(
        sample=_sample(),
        image=Image.new("RGB", (100, 100), (1, 2, 3)),
        target=target,
        executable_leaf_categories=(target.canonical_label,),
        artifact_dir=root / "artifacts",
    )


def _registry(*backends) -> BackendRegistry:
    registry = BackendRegistry()
    for backend in backends:
        registry.register(backend)
    return registry


def _qwen_backend(client: _FakeClient) -> QwenPointCountingBackend:
    return QwenPointCountingBackend(
        client, counting=CountingSettings(), system_prompt="Count points."
    )


def _executor(
    *backends,
    fallback_on_backend_unavailable: bool = True,
    fallback_on_backend_error: bool = True,
    verify_empty_detection: bool = True,
    trust_empty_detection: bool = False,
    verify_empty_semantic: bool = False,
) -> CountingPlanExecutor:
    selector = BackendSelector(_registry(*backends))
    return CountingPlanExecutor(
        selector,
        policy=CountingExecutionPolicy(
            fallback_on_backend_unavailable=fallback_on_backend_unavailable,
            fallback_on_backend_error=fallback_on_backend_error,
            verify_empty_detection=verify_empty_detection,
            trust_empty_detection=trust_empty_detection,
            verify_empty_semantic=verify_empty_semantic,
        ),
    )


def _run(
    executor: CountingPlanExecutor,
    plan: BackendPlan,
    root: Path,
    target: CountTargetSpec = _TARGET,
) -> CountingExecutionResult:
    return asyncio.run(executor.execute(plan=plan, request=_request(root, target), context=_context(root), agent_name="counting_agent"))


# ── 主流程 / primary execution ─────────────────────────────────────────────


def test_qwen_primary_executes_normally(tmp_path: Path) -> None:
    """Non-detector primary runs without any fallback state.
    非检测器主后端正常执行，无任何回退状态。"""
    client = _FakeClient()
    result = _run(_executor(_qwen_backend(client)), BackendPlan("qwen_point"), tmp_path)
    assert isinstance(result, CountingExecutionResult)
    assert result.primary_backend == "qwen_point"
    assert result.primary_kind == "qwen_point"
    assert result.final_backend == "qwen_point"
    assert result.final_kind == "qwen_point"
    assert result.attempted_backends == ("qwen_point",)
    assert result.review_backend is None
    assert result.fallback_triggered is False
    assert result.fallback_kind is None
    assert result.fallback_reason_code is None
    assert result.fallback_error_type is None
    assert result.yolo_trace is None
    assert result.outcome.counting.final_count == 0


def test_yolo_primary_executes_normally(tmp_path: Path) -> None:
    """YOLO primary keeps its own yolo trace namespace.
    YOLO 主后端保留专属 yolo trace 命名空间。"""
    client = _FakeClient()
    yolo = _FakeYoloBackend(final_count=1)
    result = _run(_executor(_qwen_backend(client), yolo), BackendPlan("det-a", ("qwen_point",)), tmp_path)
    assert result.final_backend == "det-a"
    assert result.final_kind == "yolo_obb"
    assert result.fallback_triggered is False
    assert result.attempted_backends == ("det-a",)
    assert result.yolo_trace is not None
    assert result.yolo_trace["detector_name"] == "det-a"
    # outcome trace merges into the yolo namespace. / outcome trace 合入 yolo 命名空间。
    assert result.yolo_trace["detector_note"] == "fake"
    assert result.outcome.counting.final_count == 1
    assert [item.backend_name for item in result.attempt_audits] == ["det-a"]
    assert result.attempt_audits[0].status == "succeeded"


def test_detector_ensemble_fuses_same_instance_once(tmp_path: Path) -> None:
    first = _FakeYoloBackend(final_count=1)
    second = _FakeYoloBackend(final_count=1)
    second.name = "det-b"

    result = _run(
        _executor(_qwen_backend(_FakeClient()), first, second),
        BackendPlan(
            "det-a",
            ("qwen_point",),
            ensemble_backend_names=("det-b",),
            selected_detector_expert_names=("det-a", "det-b"),
        ),
        tmp_path,
    )

    assert result.outcome.counting.final_count == 1
    assert result.outcome.counting.global_points[0].provenance is not None
    assert result.outcome.counting.global_points[0].provenance.source == "fused"
    assert result.outcome.counting.global_points[0].provenance.consensus_size == 2
    assert result.attempted_backends == ("det-a", "det-b")


def test_detector_ensemble_keeps_successful_peer_when_one_fails(tmp_path: Path) -> None:
    failed = _FakeYoloBackend(error=RuntimeError("first failed"))
    successful = _FakeYoloBackend(final_count=1)
    successful.name = "det-b"

    result = _run(
        _executor(_qwen_backend(_FakeClient()), failed, successful),
        BackendPlan(
            "det-a",
            ("qwen_point",),
            ensemble_backend_names=("det-b",),
            selected_detector_expert_names=("det-a", "det-b"),
        ),
        tmp_path,
    )

    assert result.outcome.counting.final_count == 1
    assert result.fallback_triggered is True
    assert result.fallback_reason_code == "DETECTOR_ENSEMBLE_PARTIAL"
    assert result.fallback_history[0].backend == "det-a"


def test_detector_ensemble_calls_disagreement_reviewer_once_for_low_singleton(
    tmp_path: Path,
) -> None:
    first = _FakeYoloBackend(final_count=1, point_confidence=0.4)
    second = _FakeYoloBackend(final_count=0)
    second.name = "det-b"
    reviewer = _FakeDisagreementReviewer(
        DisagreementReview(
            decisions=[
                {
                    "conflict_id": "s1:det-a:p0",
                    "decision": "reject_all",
                    "instance_count": 0,
                }
            ]
        )
    )

    result = _run(
        _executor(reviewer, first, second),
        BackendPlan(
            "det-a",
            ("qwen_point",),
            ensemble_backend_names=("det-b",),
            selected_detector_expert_names=("det-a", "det-b"),
        ),
        tmp_path,
    )

    assert reviewer.calls == 1
    assert result.outcome.counting.final_count == 0
    assert result.outcome.trace["ensemble"]["disagreement_review"]["disagreement_review_triggered"] is True


def test_disagreement_accept_one_uses_exact_requested_candidate() -> None:
    fused = _review_fixture()

    points, unresolved, _warnings = _apply_disagreement_review(
        fused,
        DisagreementReview(
            decisions=[
                {
                    "conflict_id": "candidate-a|candidate-b",
                    "decision": "accept_one",
                    "accepted_candidate_ids": ["candidate-b"],
                    "instance_count": 1,
                }
            ]
        ),
    )

    by_id = {point.global_id: point for point in points}
    assert by_id["candidate-b"].accepted is True
    assert by_id["candidate-a"].accepted is False
    assert by_id["fused-consensus"].accepted is True
    assert unresolved == []


def test_disagreement_accept_multiple_preserves_exact_candidate_set() -> None:
    fused = _review_fixture()
    fused.review_candidates[0]["candidate_ids"] = [
        "candidate-a",
        "candidate-b",
        "candidate-c",
    ]
    fused.points.append(_review_point("candidate-c", 0.10))

    points, unresolved, _warnings = _apply_disagreement_review(
        fused,
        DisagreementReview(
            decisions=[
                {
                    "conflict_id": "candidate-a|candidate-b",
                    "decision": "accept_multiple",
                    "accepted_candidate_ids": ["candidate-a", "candidate-c"],
                    "instance_count": 2,
                }
            ]
        ),
    )

    by_id = {point.global_id: point for point in points}
    assert by_id["candidate-a"].accepted is True
    assert by_id["candidate-b"].accepted is False
    assert by_id["candidate-c"].accepted is True
    assert by_id["fused-consensus"].accepted is True
    assert unresolved == []


@pytest.mark.parametrize(
    ("decision", "accepted_candidate_ids", "instance_count"),
    [
        ("accept_one", ["candidate-a", "candidate-b"], 1),
        ("accept_multiple", ["candidate-a"], 2),
        ("accept_multiple", ["candidate-a", "candidate-a"], 2),
        ("accept_one", ["unknown"], 1),
        ("reject_all", ["candidate-a"], 0),
    ],
)
def test_disagreement_invalid_decision_is_rejected(
    decision: str,
    accepted_candidate_ids: list[str],
    instance_count: int,
) -> None:
    with pytest.raises(ValueError):
        _apply_disagreement_review(
            _review_fixture(),
            DisagreementReview(
                decisions=[
                    {
                        "conflict_id": "candidate-a|candidate-b",
                        "decision": decision,
                        "accepted_candidate_ids": accepted_candidate_ids,
                        "instance_count": instance_count,
                    }
                ]
            ),
        )


def test_disagreement_uncertain_defers_to_unresolved_policy() -> None:
    fused = _review_fixture()

    points, unresolved, warnings = _apply_disagreement_review(
        fused,
        DisagreementReview(
            decisions=[
                {
                    "conflict_id": "candidate-a|candidate-b",
                    "decision": "uncertain",
                    "instance_count": 0,
                }
            ]
        ),
    )

    assert all(point.accepted for point in points)
    assert unresolved == ["candidate-a|candidate-b"]
    assert warnings[-1].code == "DETECTOR_DISAGREEMENT_REVIEW_UNCERTAIN"


def test_detector_ensemble_does_not_review_consensus(tmp_path: Path) -> None:
    first = _FakeYoloBackend(final_count=1)
    second = _FakeYoloBackend(final_count=1)
    second.name = "det-b"
    reviewer = _FakeDisagreementReviewer(DisagreementReview())

    result = _run(
        _executor(reviewer, first, second),
        BackendPlan(
            "det-a",
            ("qwen_point",),
            ensemble_backend_names=("det-b",),
            selected_detector_expert_names=("det-a", "det-b"),
        ),
        tmp_path,
    )

    assert reviewer.calls == 0
    assert result.outcome.counting.final_count == 1


def test_all_detector_experts_fail_then_use_fallback(tmp_path: Path) -> None:
    first = _FakeYoloBackend(error=RuntimeError("first failed"))
    second = _FakeYoloBackend(error=RuntimeError("second failed"))
    second.name = "det-b"

    result = _run(
        _executor(_qwen_backend(_FakeClient()), first, second),
        BackendPlan(
            "det-a",
            ("qwen_point",),
            ensemble_backend_names=("det-b",),
            selected_detector_expert_names=("det-a", "det-b"),
        ),
        tmp_path,
    )

    assert result.final_backend == "qwen_point"
    assert result.attempted_backends == ("det-a", "det-b", "qwen_point")


# ── 回退 / fallback ────────────────────────────────────────────────────────


def test_detector_unavailable_falls_back_to_qwen(tmp_path: Path) -> None:
    client = _FakeClient()
    yolo = _FakeYoloBackend(error=DetectorWeightsMissingError("det-a", "det.pt"))
    result = _run(
        _executor(_qwen_backend(client), yolo),
        BackendPlan("det-a", ("qwen_point",)),
        tmp_path,
    )
    assert result.primary_backend == "det-a"
    assert result.primary_kind == "yolo_obb"
    assert result.final_backend == "qwen_point"
    assert result.final_kind == "qwen_point"
    assert result.attempted_backends == ("det-a", "qwen_point")
    assert result.fallback_triggered is True
    assert result.fallback_kind == "unavailable"
    assert result.fallback_reason_code == "PRIMARY_BACKEND_UNAVAILABLE"
    assert result.fallback_error_type == "DetectorWeightsMissingError"
    # The initial detector profile stays visible in the yolo namespace.
    # 初始检测器 profile 仍在 yolo 命名空间可见。
    assert result.yolo_trace == {"detector_name": "det-a", "model_id": "m1"}


def test_detector_runtime_error_falls_back_to_qwen(tmp_path: Path) -> None:
    client = _FakeClient()
    yolo = _FakeYoloBackend(error=RuntimeError("inference boom"))
    result = _run(
        _executor(_qwen_backend(client), yolo),
        BackendPlan("det-a", ("qwen_point",)),
        tmp_path,
    )
    assert result.fallback_kind == "runtime_error"
    assert result.fallback_reason_code == "PRIMARY_BACKEND_FAILED"
    assert result.fallback_error_type == "RuntimeError"
    assert result.final_backend == "qwen_point"
    assert result.outcome.counting.final_count == 0
    assert [item.status for item in result.attempt_audits] == ["failed", "succeeded"]
    assert result.attempt_audits[0].counting is None
    assert result.attempt_audits[0].reason_code == "BACKEND_RUNTIME_ERROR"


def test_unavailable_fallback_disabled_raises_stable_error(tmp_path: Path) -> None:
    yolo = _FakeYoloBackend(error=DetectorWeightsMissingError("det-a", "det.pt"))
    executor = _executor(_qwen_backend(_FakeClient()), yolo, fallback_on_backend_unavailable=False)
    with pytest.raises(CountingBackendUnavailableError, match="PRIMARY_BACKEND_UNAVAILABLE"):
        _run(executor, BackendPlan("det-a", ("qwen_point",)), tmp_path)


def test_runtime_fallback_disabled_raises_stable_error(tmp_path: Path) -> None:
    yolo = _FakeYoloBackend(error=RuntimeError("inference boom"))
    executor = _executor(_qwen_backend(_FakeClient()), yolo, fallback_on_backend_error=False)
    with pytest.raises(AgentExecutionError, match="PRIMARY_BACKEND_FAILED"):
        _run(executor, BackendPlan("det-a", ("qwen_point",)), tmp_path)


def test_missing_fallback_backend_raises_fallback_backend_missing(tmp_path: Path) -> None:
    """A fallback name absent from the registry fails with a stable code.
    注册表中不存在的回退名以稳定错误码失败。"""
    client = _FakeClient()
    yolo = _FakeYoloBackend(error=DetectorWeightsMissingError("det-a", "det.pt"))
    executor = _executor(_qwen_backend(client), yolo)
    with pytest.raises(CountingBackendUnavailableError, match="FALLBACK_BACKEND_MISSING"):
        _run(executor, BackendPlan("det-a", ("ghost",)), tmp_path)


def test_failing_fallback_backend_raises_fallback_backend_failed(tmp_path: Path) -> None:
    """A fallback backend that itself fails surfaces as FALLBACK_BACKEND_FAILED.
    回退后端自身失败时以 FALLBACK_BACKEND_FAILED 呈现。"""
    yolo = _FakeYoloBackend(error=DetectorWeightsMissingError("det-a", "det.pt"))
    executor = _executor(
        _FakeRaisingQwenBackend(RuntimeError("qwen boom")), yolo
    )
    with pytest.raises(AgentExecutionError, match="FALLBACK_BACKEND_FAILED"):
        _run(executor, BackendPlan("det-a", ("qwen_point",)), tmp_path)


def test_yolo_unavailable_advances_to_semantic(tmp_path: Path) -> None:
    semantic = _FakeSemanticBackend(final_count=2)
    result = _run(
        _executor(_qwen_backend(_FakeClient()), semantic, _FakeYoloBackend(available=False)),
        BackendPlan("det-a", ("segmenter-a", "qwen_point")),
        tmp_path,
    )

    assert result.final_backend == "segmenter-a"
    assert result.attempted_backends == ("det-a", "segmenter-a")
    assert [item.to_trace() for item in result.fallback_history] == [
        {
            "backend": "det-a",
            "kind": "yolo_obb",
            "reason_code": "BACKEND_UNAVAILABLE",
            "error_type": "BackendUnavailable",
        }
    ]
    assert [item.status for item in result.attempt_audits] == [
        "unavailable",
        "succeeded",
    ]


def test_yolo_runtime_error_advances_to_semantic(tmp_path: Path) -> None:
    result = _run(
        _executor(
            _qwen_backend(_FakeClient()),
            _FakeSemanticBackend(final_count=1),
            _FakeYoloBackend(error=RuntimeError("detector failed")),
        ),
        BackendPlan("det-a", ("segmenter-a", "qwen_point")),
        tmp_path,
    )

    assert result.final_backend == "segmenter-a"
    assert result.fallback_history[0].error_type == "RuntimeError"


@pytest.mark.parametrize(
    "semantic",
    [
        _FakeSemanticBackend(available=False),
        _FakeSemanticBackend(error=RuntimeError("semantic failed")),
    ],
)
def test_semantic_failure_advances_to_quantity(
    semantic: _FakeSemanticBackend,
    tmp_path: Path,
) -> None:
    result = _run(
        _executor(
            _qwen_backend(_FakeClient()),
            _FakeQuantityProposalBackend(final_count=3),
            semantic,
        ),
        BackendPlan("segmenter-a", ("quantity_proposal", "qwen_point")),
        tmp_path,
    )

    assert result.final_backend == "quantity_proposal"
    assert result.attempted_backends == ("segmenter-a", "quantity_proposal")


def test_all_specialists_fail_before_qwen(tmp_path: Path) -> None:
    result = _run(
        _executor(
            _qwen_backend(_FakeClient()),
            _FakeYoloBackend(error=RuntimeError("yolo")),
            _FakeSemanticBackend(error=RuntimeError("semantic")),
            _FakeQuantityProposalBackend(error=RuntimeError("quantity")),
        ),
        BackendPlan(
            "det-a",
            ("segmenter-a", "quantity_proposal", "qwen_point"),
        ),
        tmp_path,
    )

    assert result.final_backend == "qwen_point"
    assert result.attempted_backends == (
        "det-a",
        "segmenter-a",
        "quantity_proposal",
        "qwen_point",
    )
    assert [item.backend for item in result.fallback_history] == [
        "det-a",
        "segmenter-a",
        "quantity_proposal",
    ]


def test_runtime_failures_then_quantity_success_are_fully_audited(tmp_path: Path) -> None:
    result = _run(
        _executor(
            _qwen_backend(_FakeClient()),
            _FakeYoloBackend(error=RuntimeError("yolo")),
            _FakeSemanticBackend(error=RuntimeError("semantic")),
            _FakeQuantityProposalBackend(final_count=3),
        ),
        BackendPlan("det-a", ("segmenter-a", "quantity_proposal", "qwen_point")),
        tmp_path,
    )
    assert result.final_backend == "quantity_proposal"
    assert [item.backend_name for item in result.attempt_audits] == [
        "det-a",
        "segmenter-a",
        "quantity_proposal",
    ]
    assert [item.status for item in result.attempt_audits] == [
        "failed",
        "failed",
        "succeeded",
    ]
    assert all(item.counting is None for item in result.attempt_audits[:2])
    assert result.attempt_audits[2].counting.final_count == 3


@pytest.mark.parametrize(
    ("semantic_available", "quantity_available", "expected_final"),
    [
        (True, True, "segmenter_mitb2_001"),
        (False, True, "quantity_proposal"),
        (False, False, "qwen_point"),
    ],
)
def test_vehicle_runtime_failure_chain_uses_every_declared_specialist(
    tmp_path: Path,
    semantic_available: bool,
    quantity_available: bool,
    expected_final: str,
) -> None:
    vehicle = _TARGET.model_copy(update={"canonical_label": "vehicle"})
    detector = _FakeYoloBackend(available=False)
    detector.name = "detector_obb_csl_001"
    semantic = _FakeSemanticBackend(
        final_count=2,
        available=semantic_available,
    )
    semantic.name = "segmenter_mitb2_001"
    quantity = _FakeQuantityProposalBackend(
        final_count=3,
        available=quantity_available,
    )
    qwen = _qwen_backend(_FakeClient(target_label="vehicle"))
    executor = _executor(qwen, detector, semantic, quantity)
    plan = BackendPlan(
        "detector_obb_csl_001",
        ("segmenter_mitb2_001", "quantity_proposal", "qwen_point"),
    )

    result = _run(executor, plan, tmp_path, vehicle)

    assert result.final_backend == expected_final
    assert result.attempted_backends == (
        plan.primary_backend_name,
        *plan.fallback_backend_names[:
            plan.fallback_backend_names.index(expected_final) + 1
        ],
    )


def test_qwen_failure_after_all_specialists_is_terminal(tmp_path: Path) -> None:
    executor = _executor(
        _FakeRaisingQwenBackend(RuntimeError("qwen")),
        _FakeYoloBackend(error=RuntimeError("yolo")),
        _FakeSemanticBackend(error=RuntimeError("semantic")),
        _FakeQuantityProposalBackend(error=RuntimeError("quantity")),
    )

    with pytest.raises(AgentExecutionError, match="FALLBACK_BACKEND_FAILED"):
        _run(
            executor,
            BackendPlan(
                "det-a",
                ("segmenter-a", "quantity_proposal", "qwen_point"),
            ),
            tmp_path,
        )


# ── zero review / 零计数复核 ──────────────────────────────────────────────


def test_zero_review_overrides_detector_zero(tmp_path: Path) -> None:
    client = _FakeClient(tile_points=[(500, 500)])  # review finds one / 复核发现 1 个
    yolo = _FakeYoloBackend(final_count=0)
    result = _run(
        _executor(_qwen_backend(client), yolo),
        BackendPlan("det-a", ("qwen_point",)),
        tmp_path,
    )
    assert result.yolo_trace is not None
    assert result.yolo_trace["zero_review_triggered"] is True
    assert result.yolo_trace["zero_review_backend"] == "qwen_point"
    assert result.yolo_trace["zero_review_status"] == "completed"
    assert result.yolo_trace["zero_review_result_count"] == 1
    assert result.yolo_trace["zero_overridden"] is True
    assert result.review_backend == "qwen_point"
    assert result.final_backend == "qwen_point"
    assert result.final_kind == "qwen_point"
    assert result.fallback_triggered is True
    assert result.fallback_kind == "zero_review"
    assert result.fallback_reason_code == "DETECTOR_ZERO_OVERRIDDEN_BY_REVIEW"
    assert result.attempted_backends == ("det-a", "qwen_point")
    assert result.outcome.counting.final_count == 1


def test_yolo_zero_uses_next_semantic_expert_for_review(tmp_path: Path) -> None:
    result = _run(
        _executor(
            _qwen_backend(_FakeClient()),
            _FakeSemanticBackend(final_count=2),
            _FakeYoloBackend(final_count=0),
        ),
        BackendPlan("det-a", ("segmenter-a", "qwen_point")),
        tmp_path,
    )

    assert result.review_backend == "segmenter-a"
    assert result.final_backend == "segmenter-a"
    assert result.outcome.counting.final_count == 2
    assert result.fallback_kind == "zero_review"
    assert [item.phase for item in result.attempt_audits] == ["primary", "zero_review"]
    assert [item.counting.final_count for item in result.attempt_audits] == [0, 2]


def test_semantic_zero_review_requires_explicit_policy(tmp_path: Path) -> None:
    semantic = _FakeSemanticBackend(final_count=0)
    quantity = _FakeQuantityProposalBackend(final_count=2)
    plan = BackendPlan("segmenter-a", ("quantity_proposal", "qwen_point"))

    retained = _run(
        _executor(_qwen_backend(_FakeClient()), quantity, semantic),
        plan,
        tmp_path,
    )
    reviewed = _run(
        _executor(
            _qwen_backend(_FakeClient()),
            quantity,
            semantic,
            verify_empty_semantic=True,
        ),
        plan,
        tmp_path,
    )

    assert retained.final_backend == "segmenter-a"
    assert retained.outcome.counting.final_count == 0
    assert reviewed.review_backend == "quantity_proposal"
    assert reviewed.final_backend == "quantity_proposal"
    assert reviewed.outcome.counting.final_count == 2


def test_zero_review_confirms_zero_retains_detector(tmp_path: Path) -> None:
    client = _FakeClient()
    yolo = _FakeYoloBackend(final_count=0)
    result = _run(
        _executor(_qwen_backend(client), yolo),
        BackendPlan("det-a", ("qwen_point",)),
        tmp_path,
    )
    assert result.yolo_trace is not None
    assert result.yolo_trace["zero_overridden"] is False
    assert result.review_backend == "qwen_point"
    assert result.final_backend == "det-a"
    assert result.final_kind == "yolo_obb"
    assert result.fallback_triggered is False
    assert result.outcome.counting.final_count == 0


def test_zero_review_failure_keeps_detector_with_warning(tmp_path: Path) -> None:
    """A failing review retains the detector zero and injects a stable warning
    without raw exception text. 复核失败保留检测器零结果并注入稳定 warning，
    不泄漏原始异常文本。"""
    yolo = _FakeYoloBackend(final_count=0)
    result = _run(
        _executor(
            _FakeRaisingQwenBackend(RuntimeError(_SENSITIVE_ERROR_TEXT)), yolo
        ),
        BackendPlan("det-a", ("qwen_point",)),
        tmp_path,
    )
    assert result.yolo_trace is not None
    assert result.yolo_trace["zero_review_status"] == "failed"
    assert result.yolo_trace["zero_review_result_count"] is None
    assert result.yolo_trace["zero_overridden"] is False
    assert result.final_backend == "det-a"
    assert result.fallback_triggered is False
    counting = result.outcome.counting
    assert counting.status == "completed_with_warnings"
    assert counting.final_count == 0
    assert counting.warnings[0].code == "DETECTOR_ZERO_REVIEW_FAILED"
    assert "retained" in counting.warnings[0].message
    for token in ("/home/user/private", "sk-test-secret", "Bearer abcdef", "base64,AAAA"):
        assert token not in str(result), token


def test_zero_review_disabled_skips_verification(tmp_path: Path) -> None:
    client = _FakeClient()
    yolo = _FakeYoloBackend(final_count=0)
    executor = _executor(_qwen_backend(client), yolo, verify_empty_detection=False)
    result = _run(executor, BackendPlan("det-a", ("qwen_point",)), tmp_path)
    assert result.yolo_trace is not None
    assert "zero_review_triggered" not in result.yolo_trace
    assert result.final_backend == "det-a"
    assert result.review_backend is None


# ── 边界 / boundaries ──────────────────────────────────────────────────────


def test_quantity_unavailable_falls_back_to_qwen(tmp_path: Path) -> None:
    """Every specialist may advance to Qwen when unavailable.
    每种 specialist 不可用时都可以继续到 Qwen。"""
    quantity = _FakeQuantityProposalBackend(
        error=DetectorWeightsMissingError("quantity_proposal", "det.pt")
    )
    executor = _executor(_qwen_backend(_FakeClient()), quantity)
    result = _run(executor, BackendPlan("quantity_proposal", ("qwen_point",)), tmp_path)
    assert result.final_backend == "qwen_point"
    assert result.fallback_history[0].reason_code == "BACKEND_UNAVAILABLE"


def test_quantity_runtime_error_falls_back_to_qwen(tmp_path: Path) -> None:
    quantity = _FakeQuantityProposalBackend(error=RuntimeError("proposal boom"))
    executor = _executor(_qwen_backend(_FakeClient()), quantity)
    result = _run(executor, BackendPlan("quantity_proposal", ("qwen_point",)), tmp_path)
    assert result.final_backend == "qwen_point"
    assert result.fallback_history[0].reason_code == "BACKEND_RUNTIME_ERROR"


def test_invalid_backend_kind_raises_stable_error(tmp_path: Path) -> None:
    """Unknown kinds fail with a fixed public error, never echoing raw values.
    未知 kind 以固定公共错误失败，绝不回显原始值。"""
    client = _FakeClient()
    executor = _executor(_qwen_backend(client), _UnknownKindBackend())
    with pytest.raises(CountingBackendUnavailableError, match="INVALID_BACKEND_KIND") as info:
        _run(executor, BackendPlan("mystery", ()), tmp_path)
    for token in ("/home/user", "Bearer abcdef"):
        assert token not in str(info.value), token


def test_attempted_backends_records_execution_order(tmp_path: Path) -> None:
    """attempted_backends lists primary first, then fallback/review.
    attempted_backends 先列主后端，再列回退/复核后端。"""
    # unavailable fallback / 不可用回退
    client = _FakeClient()
    yolo = _FakeYoloBackend(error=DetectorWeightsMissingError("det-a", "det.pt"))
    result = _run(
        _executor(_qwen_backend(client), yolo),
        BackendPlan("det-a", ("qwen_point",)),
        tmp_path,
    )
    assert result.attempted_backends == ("det-a", "qwen_point")
    # zero review override / zero review 覆盖
    client2 = _FakeClient(tile_points=[(500, 500)])
    yolo2 = _FakeYoloBackend(final_count=0)
    result2 = _run(
        _executor(_qwen_backend(client2), yolo2),
        BackendPlan("det-a", ("qwen_point",)),
        tmp_path,
    )
    assert result2.attempted_backends == ("det-a", "qwen_point")


def test_raw_exception_text_never_enters_result(tmp_path: Path) -> None:
    """Raw exception text with paths, secrets, and Base64 must never reach the
    structured result. 含路径、密钥与 Base64 的原始异常文本绝不进入结构化结果。"""
    client = _FakeClient()
    yolo = _FakeYoloBackend(error=RuntimeError(_SENSITIVE_ERROR_TEXT))
    result = _run(
        _executor(_qwen_backend(client), yolo),
        BackendPlan("det-a", ("qwen_point",)),
        tmp_path,
    )
    assert result.fallback_error_type == "RuntimeError"
    dump = str(result) + str(result.outcome) + str(result.yolo_trace)
    for token in ("/home/user/private", "sk-test-secret", "Bearer abcdef", "base64,AAAA"):
        assert token not in dump, token


def test_public_fields_are_all_derived_from_result(tmp_path: Path) -> None:
    """Every public trace-relevant field is read off the result object; no
    tuple-based fallback state anywhere. 所有公开 trace 相关字段都来自结果
    对象；任何位置都不存在 tuple 形式的回退状态。"""
    client = _FakeClient(tile_points=[(500, 500)])
    yolo = _FakeYoloBackend(final_count=0)
    result = _run(
        _executor(_qwen_backend(client), yolo),
        BackendPlan("det-a", ("qwen_point",)),
        tmp_path,
    )
    assert result.outcome.counting.final_count == 1
    assert result.primary_backend == "det-a"
    assert result.primary_kind == "yolo_obb"
    assert result.final_backend == "qwen_point"
    assert result.final_kind == "qwen_point"
    assert result.attempted_backends == ("det-a", "qwen_point")
    assert result.review_backend == "qwen_point"
    assert result.fallback_triggered is True
    assert result.fallback_kind == "zero_review"
    assert result.fallback_reason_code == "DETECTOR_ZERO_OVERRIDDEN_BY_REVIEW"
    # Only unavailable/runtime fallbacks carry an error type; zero review does
    # not. 只有 unavailable/runtime 回退携带 error type；zero review 不携带。
    assert result.fallback_error_type is None
    assert result.yolo_trace is not None
    assert result.primary_backend != result.final_backend  # override happened / 发生了覆盖


def test_executor_entry_signatures_use_domain_contracts() -> None:
    """The executor entry uses BackendPlan/CountingRequest/AgentContext and
    returns CountingExecutionResult; no plan/sample Any.
    Executor 入口使用 BackendPlan/CountingRequest/AgentContext 并返回
    CountingExecutionResult；无 plan/sample Any。"""
    import typing

    hints = typing.get_type_hints(CountingPlanExecutor.execute)
    assert hints["plan"] is BackendPlan
    assert hints["request"] is CountingRequest
    assert hints["context"] is AgentContext
    assert hints["return"] is CountingExecutionResult
