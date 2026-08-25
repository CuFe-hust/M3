"""Contract tests for the quantity proposal backend.

数量提议后端契约测试：无可靠 hint 时拒绝 supports、hint 驱动能力判断、
proposal→localize 混合路径、证据不一致与恢复、结构化响应、trace。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from agents.counting.backends.base import CountingBackendOutcome, CountingRequest
from agents.counting.backends.quantity_proposal import QuantityProposalBackend
from agents.counting.schema import CountTargetSpec, CountingResult
from agents.counting.settings import CountingSettings
from agents.schema import AgentResult
from data.schema import GroundTruth, ImageRef, UnifiedSample
from models.base import ModelCacheIdentity

REPO_ROOT = Path(__file__).resolve().parents[3]

_CAR = CountTargetSpec(
    canonical_label="car",
    inclusion_rule="visible vehicle",
    exclusion_rule="occluded more than half",
)
_SHIP = CountTargetSpec(
    canonical_label="ship",
    inclusion_rule="visible ship",
    exclusion_rule="occluded",
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
    """Stub responding with a proposal (with boxes) and then a localizer
    result. 依次返回提议（带框）与定位结果的桩。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.proposal_boxes: list[list[float]] = [[100, 100, 200, 200]]
        self.proposal_answer = "1"
        self.localizer_answer: str | None = None
        self.localizer_points: list[list[int]] | None = None
        self.fail_proposal: BaseException | None = None
        self.raw_response_text: str | None = None

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="fake-model",
            generation={"temperature": 0.0, "do_sample": False, "max_tokens": 128},
            client_version="1",
        )

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        stage = "proposal" if "count-proposal" in request_meta.request_id else "localizer"
        self.calls.append((stage, request_meta.prompt_version))
        if stage == "proposal":
            if self.fail_proposal is not None:
                raise self.fail_proposal
            return response_model.model_validate(
                {
                    "agent_name": "counting_agent",
                    "answer": self.proposal_answer,
                    "boxes": self.proposal_boxes,
                    "status": "completed",
                }
            )
        payload: dict[str, Any] = {
            "agent_name": "counting_agent",
            "answer": self.localizer_answer or self.proposal_answer,
            "boxes": [],
            "status": "completed",
        }
        if self.localizer_points is not None:
            payload["evidence_items"] = [
                {"label": "car", "point": point}
                for point in self.localizer_points
            ]
        return response_model.model_validate(payload)


def _request(tmp_path: Path) -> CountingRequest:
    return CountingRequest(
        sample=_sample(),
        image=Image.new("RGB", (1000, 1000), (1, 2, 3)),
        target=_CAR,
        executable_leaf_categories=("car",),
        artifact_dir=tmp_path / "run",
    )


def _backend(client: _FakeClient, **overrides) -> QuantityProposalBackend:
    values = dict(
        counting=CountingSettings(),
        proposal_prompt="Propose a count.",
        localizer_prompt="Localize each instance.",
        proposal_prompt_version="count-proposal-v1",
        localizer_prompt_version="count-localize-v1",
    )
    values.update(overrides)
    return QuantityProposalBackend(client, **values)


def _context(budget: _FakeBudget) -> object:
    class _Context:
        call_budget = budget

    return _Context()


# ── 能力判断 / capability gating ──────────────────────────────────────────


def test_supports_refuses_without_reliable_hint() -> None:
    """Without a reliable hint the backend refuses instead of guessing.
    没有可靠 hint 时后端拒绝，而不是猜测。"""
    backend = _backend(_FakeClient())
    assert backend.supports(_CAR) is False
    assert backend.supports(_CAR, hints={}) is False
    assert backend.supports(_CAR, hints={"something_else": True}) is False


def test_supports_accepts_with_quantity_estimation_hint() -> None:
    backend = _backend(_FakeClient(), supported_targets=("car",))
    assert backend.supports(_CAR, hints={"quantity_estimation": True}) is True
    # The default target set is explicit rather than inferred from a broad label.
    assert _backend(_FakeClient()).supports(
        CountTargetSpec(canonical_label="car", inclusion_rule="r", exclusion_rule="e"),
        hints={
            "quantity_estimation": True,
            "canonical_label": "small-vehicle",
        },
    ) is True


def test_supports_checks_supported_targets() -> None:
    backend = _backend(_FakeClient(), supported_targets=("small-vehicle", "large-vehicle"))
    assert backend.supports(_CAR, hints={"quantity_estimation": True}) is False
    assert (
        backend.supports(
            CountTargetSpec(canonical_label="small-vehicle", inclusion_rule="r", exclusion_rule="e"),
            hints={"quantity_estimation": True},
        )
        is True
    )


def test_quantity_proposal_supports_vehicle_by_default() -> None:
    backend = _backend(_FakeClient())
    vehicle = CountTargetSpec(
        canonical_label="vehicle",
        inclusion_rule="visible vehicles",
        exclusion_rule="non-vehicles",
    )

    assert backend.supports(
        vehicle,
        hints={"quantity_estimation": True, "canonical_label": "vehicle"},
    ) is True


def test_supports_uses_explicit_catalog_canonical_label() -> None:
    backend = _backend(_FakeClient())
    assert backend.supports(
        _CAR,
        hints={
            "quantity_estimation": True,
            "canonical_label": "small-vehicle",
            "countable": True,
            "hints": ["small_object"],
        },
    ) is True
    assert backend.supports(
        _CAR,
        hints={
            "quantity_estimation": True,
            "canonical_label": "small-vehicle",
            "countable": False,
        },
    ) is False


def test_backend_identity() -> None:
    backend = _backend(_FakeClient())
    assert backend.name == "quantity_proposal"
    assert backend.priority == 5
    assert backend.is_available() is True


# ── 执行 / execution ───────────────────────────────────────────────────────


def test_count_matching_proposal_and_evidence(tmp_path: Path) -> None:
    """Proposal boxes localize exactly the proposed count → no localizer call.
    提议框恰好定位提议数量 → 不调用定位器。"""
    client = _FakeClient()
    budget = _FakeBudget()
    backend = _backend(client)
    outcome = asyncio.run(backend.count(_request(tmp_path), _context(budget)))
    assert isinstance(outcome, CountingBackendOutcome)
    assert outcome.counting.final_count == 1
    assert outcome.counting.status == "completed"
    assert [stage for stage, _ in client.calls] == ["proposal"]
    assert budget.qwen_calls == 1
    assert outcome.agent_result is not None
    assert outcome.agent_result.answer == "1"
    assert isinstance(outcome.agent_result, AgentResult)


def test_count_evidence_mismatch_triggers_localizer(tmp_path: Path) -> None:
    client = _FakeClient()
    client.proposal_answer = "3"
    client.proposal_boxes = [[100, 100, 200, 200]]  # only one localized
    client.localizer_points = [[150, 150], [400, 400], [600, 600]]
    budget = _FakeBudget()
    backend = _backend(client)
    outcome = asyncio.run(backend.count(_request(tmp_path), _context(budget)))
    assert [stage for stage, _ in client.calls] == ["proposal", "localizer"]
    assert budget.qwen_calls == 2
    assert outcome.counting.final_count == 3
    codes = {record.code for record in outcome.counting.warnings}
    assert "COUNT_PROPOSAL_EVIDENCE_MISMATCH" in codes
    assert outcome.agent_result.answer == "3"
    assert outcome.trace["localization_used"] is True


def test_zero_proposal_positive_localizer_is_complete_with_warning(
    tmp_path: Path,
) -> None:
    client = _FakeClient()
    client.proposal_answer = "0"
    client.proposal_boxes = []
    client.localizer_answer = "3"
    client.localizer_points = [[150, 150], [400, 400], [600, 600]]

    outcome = asyncio.run(
        _backend(client).count(_request(tmp_path), _context(_FakeBudget()))
    )

    assert outcome.counting.final_count == 3
    assert outcome.counting.status == "completed_with_warnings"
    assert outcome.agent_result.status == "completed"
    assert {warning.code for warning in outcome.counting.warnings} == {
        "COUNT_PROPOSAL_EVIDENCE_MISMATCH"
    }


def test_self_consistent_localizer_overrides_bad_proposal(tmp_path: Path) -> None:
    client = _FakeClient()
    client.proposal_answer = "5"
    client.proposal_boxes = []
    client.localizer_answer = "3"
    client.localizer_points = [[150, 150], [400, 400], [600, 600]]

    outcome = asyncio.run(
        _backend(client).count(_request(tmp_path), _context(_FakeBudget()))
    )

    assert outcome.counting.final_count == 3
    assert outcome.counting.status == "completed_with_warnings"
    assert "COUNT_LOCALIZATION_EVIDENCE_MISMATCH" not in {
        warning.code for warning in outcome.counting.warnings
    }


def test_inconsistent_localizer_answer_is_partial(tmp_path: Path) -> None:
    client = _FakeClient()
    client.proposal_answer = "5"
    client.proposal_boxes = []
    client.localizer_answer = "4"
    client.localizer_points = [[150, 150], [400, 400], [600, 600]]

    outcome = asyncio.run(
        _backend(client).count(_request(tmp_path), _context(_FakeBudget()))
    )

    assert outcome.counting.final_count == 3
    assert outcome.counting.status == "partial"
    assert "COUNT_LOCALIZATION_EVIDENCE_MISMATCH" in {
        warning.code for warning in outcome.counting.warnings
    }


def test_unparseable_localizer_answer_with_points_is_partial(tmp_path: Path) -> None:
    client = _FakeClient()
    client.proposal_answer = "5"
    client.proposal_boxes = []
    client.localizer_answer = "unknown"
    client.localizer_points = [[150, 150], [400, 400], [600, 600]]

    outcome = asyncio.run(
        _backend(client).count(_request(tmp_path), _context(_FakeBudget()))
    )

    assert outcome.counting.final_count == 3
    assert outcome.counting.status == "partial"


def test_count_incomplete_mismatch_is_partial(tmp_path: Path) -> None:
    client = _FakeClient()
    client.proposal_answer = "5"
    client.proposal_boxes = [[100, 100, 200, 200]]
    client.localizer_points = [[150, 150]]
    backend = _backend(client)
    outcome = asyncio.run(backend.count(_request(tmp_path), _context(_FakeBudget())))
    assert outcome.counting.status == "partial"
    assert outcome.counting.final_count == 1
    assert "incomplete" in outcome.agent_result.answer
    codes = {record.code for record in outcome.counting.warnings}
    assert "COUNT_LOCALIZATION_EVIDENCE_MISMATCH" in codes


def test_count_recovers_proposal_header(tmp_path: Path) -> None:
    """A parse failure recovers the integer answer header from the current
    request's hash-scoped artifact directory. 解析失败时从当前请求的 hash
    作用域产物目录恢复整数答案。"""
    client = _FakeClient()
    client.fail_proposal = ValueError("malformed JSON")
    client.proposal_answer = "2"
    client.localizer_points = [[150, 150], [400, 400]]
    request = _request(tmp_path)
    # Compute the hash-scoped directory exactly as the backend does.
    # 与后端完全一致地计算 hash 作用域目录。
    identity = client.cache_identity
    import hashlib

    from agents.counting.backends.quantity_proposal import _encode_image
    from agents.counting.evidence import box_evidence  # noqa: F401
    from models.base import build_request_hash
    from models.images import image_to_data_url  # noqa: F401

    image_bytes = _encode_image(request.image)
    system_prompt = "Propose a count." + (
        "\n\nReturn valid JSON only. Set agent_name to 'counting_agent'; put the "
            "concise final answer in answer, use empty boxes when they are not "
        "needed, and set status to 'completed'."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_bytes, "image/png")}},
                {"type": "text", "text": request.sample.question},
            ],
        },
    ]
    request_hash = build_request_hash(
        model=identity.model,
        generation=identity.generation_payload(),
        prompt_version="count-proposal-v1",
        messages=messages,
        image_sha256=hashlib.sha256(image_bytes).hexdigest(),
        response_schema=__import__(
            "agents.counting.backends.quantity_proposal", fromlist=["_CountProposalResult"]
        )._CountProposalResult.model_json_schema(),
        client_version=identity.client_version,
        model_revision=identity.revision,
    )
    raw_dir = tmp_path / "run" / "counting_agent" / "count_proposal" / request_hash
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "raw_response.txt").write_text('{"answer": "2"', encoding="utf-8")
    backend = _backend(client)
    outcome = asyncio.run(backend.count(request, _context(_FakeBudget())))
    assert outcome.counting.final_count == 2
    codes = {record.code for record in outcome.counting.warnings}
    assert "COUNT_PROPOSAL_HEADER_RECOVERED" in codes


def test_stale_raw_response_is_never_reused(tmp_path: Path) -> None:
    """A raw response left by a previous request (different hash directory)
    must never be recovered. 旧请求遗留（不同 hash 目录）的 raw response 绝不
    被恢复。"""
    client = _FakeClient()
    client.fail_proposal = ValueError("malformed JSON")
    stale_dir = tmp_path / "run" / "counting_agent" / "count_proposal" / ("a" * 64)
    stale_dir.mkdir(parents=True, exist_ok=True)
    (stale_dir / "raw_response.txt").write_text('{"answer": "99"', encoding="utf-8")
    backend = _backend(client)
    with pytest.raises(ValueError, match="malformed JSON"):
        asyncio.run(backend.count(_request(tmp_path), _context(_FakeBudget())))


def test_non_parse_failure_never_recovers(tmp_path: Path) -> None:
    """Network/filesystem-style failures must propagate unchanged without
    attempting raw-response recovery. 网络/文件系统类失败必须原样传播，不得
    尝试 raw response 恢复。"""
    client = _FakeClient()
    client.fail_proposal = RuntimeError("connection refused")
    raw_dir = tmp_path / "run" / "counting_agent" / "count_proposal"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "raw_response.txt").write_text('{"answer": "5"', encoding="utf-8")
    backend = _backend(client)
    with pytest.raises(RuntimeError, match="connection refused"):
        asyncio.run(backend.count(_request(tmp_path), _context(_FakeBudget())))


def test_count_trace_records_versions_and_path(tmp_path: Path) -> None:
    client = _FakeClient()
    backend = _backend(client)
    outcome = asyncio.run(backend.count(_request(tmp_path), _context(_FakeBudget())))
    assert outcome.trace["backend"] == "quantity_proposal"
    assert outcome.trace["pipeline"] == "quantity_proposal_then_grounded_localization"
    assert outcome.trace["proposal_prompt_version"] == "count-proposal-v1"
    assert outcome.trace["localizer_prompt_version"] == "count-localize-v1"


# ── 边界 / boundaries ─────────────────────────────────────────────────────


def test_backend_has_no_dataset_references() -> None:
    source = (
        REPO_ROOT / "agents" / "counting" / "backends" / "quantity_proposal.py"
    ).read_text(encoding="utf-8")
    assert "VRSBench" not in source
    assert "vrsbench" not in source
    assert "spacers_agent" not in source
    assert "dataset" not in source


def test_backend_has_no_fallback_or_prompt_catalog() -> None:
    source = (
        REPO_ROOT / "agents" / "counting" / "backends" / "quantity_proposal.py"
    ).read_text(encoding="utf-8")
    assert "fallback" not in source
    assert "PromptCatalog" not in source


def test_duck_typed_identity_is_rejected(tmp_path: Path) -> None:
    from agents.counting.backends.base import MissingModelCacheIdentityError

    class _DuckIdentity:
        model = "fake-model"
        client_version = "1"
        revision = None

        def generation_payload(self):
            return {"temperature": 0.0}

    class _DuckClient(_FakeClient):
        cache_identity = _DuckIdentity()

    backend = _backend(_DuckClient())  # type: ignore[arg-type]
    with pytest.raises(MissingModelCacheIdentityError, match="ModelCacheIdentity"):
        asyncio.run(backend.count(_request(tmp_path), _context(_FakeBudget())))
