"""Contract tests for the text-only judge schemas, payload/hash builders,
and the DeepSeek judge client recovery/cache/artifact behavior.

仅文本 judge 的 Schema、载荷/哈希构建与 DeepSeek judge 客户端
恢复/缓存/产物行为的契约测试。所有测试离线：传输层使用注入的 fake
transport，绝不发起真实网络请求。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agents.counting.schema import CountTargetSpec, CountingResult, GlobalPointObservation
from data.schema import GroundTruth
from evaluation.judges.base import (
    DeepSeekJudgeResult,
    VQAAnswerJudgeResult,
    build_count_judge_payload,
    build_judge_request_hash,
    build_vqa_judge_payload,
    stable_error_label,
)
from evaluation.judges.deepseek import (
    DeepSeekJudgeClient,
    DeepSeekJudgeError,
    JudgeTransportError,
    _assert_text_only_payload,
)
from models.base import RequestMeta
from models.cache import JsonResponseCache
from models.settings import DeepSeekSettings


# ── helpers / 测试辅助 ──────────────────────────────────────────────────────


def _settings(max_retries: int = 1) -> DeepSeekSettings:
    return DeepSeekSettings(
        base_url="https://example.invalid",
        model="deepseek-test-model",
        max_retries=max_retries,
        timeout_seconds=5,
    )


def _request_meta(artifact_dir: Path | None = None) -> RequestMeta:
    return RequestMeta(
        request_id="s1:deepseek-vqa",
        request_hash="a" * 64,
        prompt_version="deepseek-vqa-judge-v1",
        sample_id="s1",
        artifact_dir=artifact_dir,
    )


def _vqa_payload() -> dict:
    return build_vqa_judge_payload(
        question="Is there a road?",
        reference_answers=["yes"],
        candidate_answer="yes",
    )


def _client(transport, *, cache=None, max_retries: int = 1) -> DeepSeekJudgeClient:
    return DeepSeekJudgeClient(
        _settings(max_retries=max_retries),
        api_key="test-api-key",
        judge_prompt="judge system prompt",
        repair_prompt="repair system prompt",
        cache=cache,
        transport=transport,
        retry_base_seconds=0.0,
    )


class _QueuedTransport:
    """Transport fed by an explicit queue of strings or exceptions; records
    every call with its keyword arguments. 由字符串/异常显式队列驱动的传输；
    记录每次调用及其关键字参数。"""

    def __init__(self, *items: object) -> None:
        self.items = list(items)
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not self.items:
            raise AssertionError("transport called more times than configured")
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _point(gid: str = "p0", *, accepted: bool = True, confidence: float = 0.9) -> GlobalPointObservation:
    return GlobalPointObservation(
        global_id=gid,
        target="car",
        source_tile_id="t0",
        local_id=gid,
        local_x_norm=100,
        local_y_norm=100,
        local_radius_norm=5,
        global_x_px=100,
        global_y_px=100,
        global_x_norm=100,
        global_y_norm=100,
        radius_px=5.0,
        confidence=confidence,
        ownership_valid=True,
        near_core_boundary=False,
        accepted=accepted,
        short_evidence="visible",
    )


def _counting(points: tuple = (), final_count: int = 0) -> CountingResult:
    return CountingResult(
        sample_id="s1",
        target="car",
        question="How many cars?",
        source_width=1000,
        source_height=1000,
        tile_count=1,
        succeeded_tiles=["t0"],
        failed_tiles=[],
        global_points=list(points),
        merged_groups=[],
        unresolved_conflicts=[],
        final_count=final_count,
        status="completed",
    )


def _target() -> CountTargetSpec:
    return CountTargetSpec(
        canonical_label="car",
        inclusion_rule="visible cars",
        exclusion_rule="parked",
        aliases=["vehicle"],
    )


def _deepseek_result_json(verdict: str = "correct") -> str:
    return json.dumps(
        {
            "judge_scope": "text_and_structured_evidence_only",
            "can_verify_visual_truth": False,
            "semantic_correctness": 1.0,
            "answer_evidence_consistency": 1.0,
            "constraint_following": 1.0,
            "clarity": 1.0,
            "verdict": verdict,
            "issues": [],
            "concise_rationale": "ok",
        }
    )


# ── payload builders / 载荷构建 ─────────────────────────────────────────────


def test_vqa_payload_shape_and_text_only() -> None:
    payload = build_vqa_judge_payload(
        question="Is there a road?",
        reference_answers=["yes", "Yes."],
        candidate_answer="yes",
    )
    assert payload["task"] == "general_vqa_answer_validation"
    assert payload["prediction"] == {"answer": "yes"}
    assert payload["ground_truth"] == {"answers": ["yes", "Yes."]}
    assert payload["deterministic_metrics"] == {"exact_match": 1}
    _assert_text_only_payload(payload)  # no image markers / 无图像标记


def test_vqa_payload_exact_match_mismatch() -> None:
    payload = build_vqa_judge_payload(
        question="Is there a road?",
        reference_answers=["yes"],
        candidate_answer="no",
    )
    assert payload["deterministic_metrics"] == {"exact_match": 0}


def test_counting_payload_shape() -> None:
    points = (
        _point("p0", accepted=True, confidence=0.9),
        _point("p1", accepted=True, confidence=0.1),
        _point("p2", accepted=False, confidence=0.5),
    )
    payload = build_count_judge_payload(
        question="How many cars?",
        target=_target(),
        display_answer="2",
        counting=_counting(points, final_count=2),
        ground_truth=GroundTruth(answers=["2"], count=2),
        min_confidence=0.2,
    )
    assert payload["task"] == "counting"
    assert payload["target_spec"] == {
        "canonical_label": "car",
        "inclusion_rule": "visible cars",
        "exclusion_rule": "parked",
    }
    assert payload["prediction"] == {
        "display_answer": "2",
        "final_count": 2,
        "point_count": 2,
        "failed_tiles": [],
        "unresolved_conflicts": [],
    }
    assert payload["ground_truth"] == {"count": 2, "answers": ["2"]}
    assert payload["deterministic_metrics"]["exact_match"] == 1
    assert payload["evidence_summary"] == {
        "tile_count": 1,
        "succeeded_tiles": 1,
        "low_confidence_points": 1,  # only p1 is accepted below 0.2 / 仅 p1 低于 0.2 且被接受
        "seam_merges": 0,
    }
    _assert_text_only_payload(payload)


def test_counting_payload_without_ground_truth() -> None:
    payload = build_count_judge_payload(
        question="How many cars?",
        target=_target(),
        display_answer="0",
        counting=_counting(),
        ground_truth=None,
        min_confidence=0.2,
    )
    assert payload["ground_truth"] is None
    assert payload["deterministic_metrics"] is None


def test_counting_payload_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError, match="min_confidence"):
        build_count_judge_payload(
            question="q",
            target=_target(),
            display_answer="0",
            counting=_counting(),
            ground_truth=None,
            min_confidence=1.5,
        )
    with pytest.raises(ValueError, match="min_confidence"):
        build_count_judge_payload(
            question="q",
            target=_target(),
            display_answer="0",
            counting=_counting(),
            ground_truth=None,
            min_confidence=-0.1,
        )


def test_text_only_guard_rejects_image_markers() -> None:
    for payload in ({"image_path": "x"}, {"nested": {"base64": "y"}}, {"pixel_array": [1]}):
        with pytest.raises(ValueError, match="text and structured evidence"):
            _assert_text_only_payload(payload)


# ── request hashes / 请求哈希 ───────────────────────────────────────────────


def _vqa_hash(**overrides: object) -> str:
    values = dict(
        model="m",
        prompt_text="p",
        prompt_version="deepseek-vqa-judge-v1",
        sample_id="s1",
        payload=_vqa_payload(),
        response_schema=VQAAnswerJudgeResult.model_json_schema(),
    )
    values.update(overrides)
    return build_judge_request_hash(**values)  # type: ignore[arg-type]


def _counting_hash(**overrides: object) -> str:
    points = (_point("p0", accepted=True), _point("p1", accepted=True))
    payload = build_count_judge_payload(
        question="How many cars?",
        target=_target(),
        display_answer="2",
        counting=_counting(points, final_count=2),
        ground_truth=GroundTruth(count=2),
        min_confidence=0.2,
    )
    values = dict(
        model="m",
        prompt_text="p",
        prompt_version="deepseek-judge-v1",
        sample_id="s1",
        payload=payload,
        response_schema=DeepSeekJudgeResult.model_json_schema(),
    )
    values.update(overrides)
    return build_judge_request_hash(**values)  # type: ignore[arg-type]


def test_hashes_are_deterministic_and_hex() -> None:
    first = _vqa_hash()
    second = _vqa_hash()
    assert first == second
    assert len(first) == 64
    assert all(character in "0123456789abcdef" for character in first)


def test_vqa_hash_sensitive_to_every_input() -> None:
    base = _vqa_hash()
    assert _vqa_hash(model="x") != base
    assert _vqa_hash(prompt_text="x") != base
    assert _vqa_hash(prompt_version="other-version") != base
    assert _vqa_hash(sample_id="s2") != base
    other_payload = dict(_vqa_payload())
    other_payload["question"] = "Different question?"
    assert _vqa_hash(payload=other_payload) != base
    other_schema = dict(VQAAnswerJudgeResult.model_json_schema())
    other_schema["title"] = "ChangedSchema"
    assert _vqa_hash(response_schema=other_schema) != base


def test_counting_hash_sensitive_to_every_payload_part() -> None:
    base = _counting_hash()
    payload = dict(base_payload := _counting_payload_for_hash())
    payload["prediction"] = {"final_count": 3}
    assert _counting_hash(payload=payload) != base
    payload = dict(base_payload)
    payload["question"] = "How many trucks?"
    assert _counting_hash(payload=payload) != base
    payload = dict(base_payload)
    payload["target_spec"] = {"canonical_label": "truck"}
    assert _counting_hash(payload=payload) != base
    payload = dict(base_payload)
    payload["evidence_summary"] = {"tile_count": 9}
    assert _counting_hash(payload=payload) != base
    payload = dict(base_payload)
    payload["ground_truth"] = {"count": 5, "answers": []}
    assert _counting_hash(payload=payload) != base
    other_schema = dict(DeepSeekJudgeResult.model_json_schema())
    other_schema["title"] = "ChangedSchema"
    assert _counting_hash(response_schema=other_schema) != base
    assert _counting_hash(model="x") != base
    assert _counting_hash(prompt_text="x") != base
    assert _counting_hash(prompt_version="other") != base
    assert _counting_hash(sample_id="s2") != base


def _counting_payload_for_hash() -> dict:
    points = (_point("p0", accepted=True), _point("p1", accepted=True))
    return build_count_judge_payload(
        question="How many cars?",
        target=_target(),
        display_answer="2",
        counting=_counting(points, final_count=2),
        ground_truth=GroundTruth(count=2),
        min_confidence=0.2,
    )


# ── judge result schemas / judge 结果 Schema ────────────────────────────────


def test_deepseek_judge_result_contract() -> None:
    result = DeepSeekJudgeResult.model_validate_json(_deepseek_result_json())
    assert result.verdict == "correct"
    assert result.judge_scope == "text_and_structured_evidence_only"
    with pytest.raises(ValidationError):
        DeepSeekJudgeResult.model_validate_json(_deepseek_result_json(verdict="maybe"))
    with pytest.raises(ValidationError):
        DeepSeekJudgeResult.model_validate_json(
            _deepseek_result_json().replace('"can_verify_visual_truth": false', '"can_verify_visual_truth": true')
        )
    with pytest.raises(ValidationError):
        DeepSeekJudgeResult.model_validate_json(
            _deepseek_result_json().replace('"issues": []', '"issues": [], "extra": 1')
        )


def test_vqa_answer_judge_result_contract() -> None:
    result = VQAAnswerJudgeResult(score=1)
    assert result.judge_scope == "text_and_structured_evidence_only"
    assert result.can_verify_visual_truth is False
    with pytest.raises(ValidationError):
        VQAAnswerJudgeResult(score=2)
    with pytest.raises(ValidationError):
        VQAAnswerJudgeResult(score=0, can_verify_visual_truth=True)


def test_stable_error_label_is_content_free() -> None:
    error = RuntimeError("secret-raw-detail")
    assert stable_error_label(error) == "RuntimeError"
    assert "secret-raw-detail" not in stable_error_label(error)


# ── client: success / cache / artifacts ─────────────────────────────────────


def test_judge_json_success_writes_artifacts_and_cache(tmp_path: Path) -> None:
    transport = _QueuedTransport('{"score": 1, "concise_rationale": "ok"}')
    cache = JsonResponseCache(tmp_path / "cache")
    client = _client(transport, cache=cache)
    meta = _request_meta(tmp_path / "artifacts")
    result = client.judge_json(
        _vqa_payload(), response_model=VQAAnswerJudgeResult, request_meta=meta
    )
    assert result.score == 1
    assert len(transport.calls) == 1
    assert transport.calls[0]["model"] == "deepseek-test-model"
    assert transport.calls[0]["api_key"] == "test-api-key"
    artifacts = tmp_path / "artifacts"
    assert (artifacts / "request_meta.json").is_file()
    assert (artifacts / "raw_response.txt").is_file()
    assert (artifacts / "parsed.json").is_file()
    validation = json.loads((artifacts / "validation.json").read_text(encoding="utf-8"))
    assert validation == {
        "cache_hit": False,
        "attempt_errors": [],
        "response_metadata": {"latency_seconds": validation["response_metadata"]["latency_seconds"]},
        "valid": True,
    }
    cached = cache.load(meta.request_hash)
    assert cached is not None
    assert cached.parsed == result.model_dump(mode="json")


def test_cache_hit_skips_transport(tmp_path: Path) -> None:
    cache = JsonResponseCache(tmp_path / "cache")
    transport = _QueuedTransport('{"score": 1, "concise_rationale": "ok"}')
    meta = _request_meta(tmp_path / "artifacts")
    _client(transport, cache=cache).judge_json(
        _vqa_payload(), response_model=VQAAnswerJudgeResult, request_meta=meta
    )

    def unexpected_transport(**kwargs):
        raise AssertionError("transport must not be called on cache hit")

    result = _client(unexpected_transport, cache=cache).judge_json(
        _vqa_payload(), response_model=VQAAnswerJudgeResult, request_meta=meta
    )
    assert result.score == 1
    validation = json.loads(
        (tmp_path / "artifacts" / "validation.json").read_text(encoding="utf-8")
    )
    assert validation["cache_hit"] is True


def test_judge_convenience_method_returns_counting_result(tmp_path: Path) -> None:
    transport = _QueuedTransport(_deepseek_result_json())
    client = _client(transport)
    result = client.judge(_vqa_payload(), request_meta=_request_meta())
    assert isinstance(result, DeepSeekJudgeResult)
    assert result.verdict == "correct"


def test_json_fence_is_stripped(tmp_path: Path) -> None:
    transport = _QueuedTransport('```json\n{"score": 1, "concise_rationale": "ok"}\n```')
    client = _client(transport)
    result = client.judge_json(
        _vqa_payload(), response_model=VQAAnswerJudgeResult, request_meta=_request_meta()
    )
    assert result.score == 1


# ── client: bounded recovery / 受限恢复 ─────────────────────────────────────


def test_empty_response_retries_then_fails(tmp_path: Path) -> None:
    transport = _QueuedTransport("", "")
    client = _client(transport, max_retries=1)
    with pytest.raises(DeepSeekJudgeError, match="DEEPSEEK_JUDGE_EMPTY_RESPONSE"):
        client.judge_json(
            _vqa_payload(),
            response_model=VQAAnswerJudgeResult,
            request_meta=_request_meta(tmp_path / "artifacts"),
        )
    assert len(transport.calls) == 2  # initial + one retry / 初次 + 一次重试
    validation = json.loads(
        (tmp_path / "artifacts" / "validation.json").read_text(encoding="utf-8")
    )
    assert validation["valid"] is False
    assert [entry["error_type"] for entry in validation["attempt_errors"]] == [
        "EmptyJudgeResponseError",
        "EmptyJudgeResponseError",
    ]
    assert all(entry["retryable"] for entry in validation["attempt_errors"])
    # Error records carry type names only, never raw text. / 错误记录只有类型名。
    assert all("error" not in entry for entry in validation["attempt_errors"])


def test_empty_response_then_success(tmp_path: Path) -> None:
    transport = _QueuedTransport("", '{"score": 1, "concise_rationale": "ok"}')
    client = _client(transport, max_retries=3)
    result = client.judge_json(
        _vqa_payload(), response_model=VQAAnswerJudgeResult, request_meta=_request_meta()
    )
    assert result.score == 1
    assert len(transport.calls) == 2


def test_invalid_json_repaired_once(tmp_path: Path) -> None:
    transport = _QueuedTransport("{not json", '{"score": 0, "concise_rationale": "fixed"}')
    client = _client(transport)
    result = client.judge_json(
        _vqa_payload(), response_model=VQAAnswerJudgeResult, request_meta=_request_meta()
    )
    assert result.score == 0
    assert len(transport.calls) == 2
    repair_messages = transport.calls[1]["messages"]
    assert repair_messages[0]["content"] == "repair system prompt"
    repair_user = json.loads(repair_messages[1]["content"])
    assert "raw_output" in repair_user and "validation_error" in repair_user
    assert repair_user["validation_error"] == "JSONDecodeError"


def test_invalid_json_twice_fails(tmp_path: Path) -> None:
    transport = _QueuedTransport("{bad", "{bad")
    client = _client(transport)
    with pytest.raises(DeepSeekJudgeError, match="DEEPSEEK_JUDGE_INVALID_JSON"):
        client.judge_json(
            _vqa_payload(),
            response_model=VQAAnswerJudgeResult,
            request_meta=_request_meta(tmp_path / "artifacts"),
        )
    assert len(transport.calls) == 2  # initial + one repair / 初次 + 一次修复


def test_transport_500_retries_then_success(tmp_path: Path) -> None:
    transport = _QueuedTransport(
        JudgeTransportError("boom", status_code=500),
        '{"score": 1, "concise_rationale": "ok"}',
    )
    client = _client(transport, max_retries=1)
    result = client.judge_json(
        _vqa_payload(), response_model=VQAAnswerJudgeResult, request_meta=_request_meta()
    )
    assert result.score == 1
    assert len(transport.calls) == 2


def test_transport_401_fails_immediately(tmp_path: Path) -> None:
    transport = _QueuedTransport(JudgeTransportError("unauthorized", status_code=401))
    client = _client(transport, max_retries=3)
    with pytest.raises(DeepSeekJudgeError, match="DEEPSEEK_JUDGE_TRANSPORT_FAILED"):
        client.judge_json(
            _vqa_payload(),
            response_model=VQAAnswerJudgeResult,
            request_meta=_request_meta(tmp_path / "artifacts"),
        )
    assert len(transport.calls) == 1


def test_unexpected_exception_fails_without_raw_text(tmp_path: Path) -> None:
    transport = _QueuedTransport(RuntimeError("boom-secret-detail"))
    client = _client(transport)
    with pytest.raises(DeepSeekJudgeError, match="DEEPSEEK_JUDGE_TRANSPORT_FAILED") as error:
        client.judge_json(
            _vqa_payload(),
            response_model=VQAAnswerJudgeResult,
            request_meta=_request_meta(tmp_path / "artifacts"),
        )
    assert "boom-secret-detail" not in str(error.value)
    validation_text = (tmp_path / "artifacts" / "validation.json").read_text(encoding="utf-8")
    assert "boom-secret-detail" not in validation_text
    validation = json.loads(validation_text)
    assert validation["attempt_errors"][0]["error_type"] == "RuntimeError"
    assert "error" not in validation["attempt_errors"][0]


def test_text_only_guard_fails_before_transport() -> None:
    transport = _QueuedTransport()
    client = _client(transport)
    with pytest.raises(ValueError, match="text and structured evidence"):
        client.judge_json(
            {"image_path": "x"},
            response_model=VQAAnswerJudgeResult,
            request_meta=_request_meta(),
        )
    assert transport.calls == []


def test_missing_api_key_rejected() -> None:
    with pytest.raises(DeepSeekJudgeError, match="api key"):
        DeepSeekJudgeClient(
            _settings(),
            api_key="",
            judge_prompt="p",
            repair_prompt="r",
        )
