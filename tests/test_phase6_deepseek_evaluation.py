from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from spacers_agent.clients.base import JsonResponseCache, RequestMeta
from spacers_agent.clients.deepseek import DeepSeekJudgeClient
from spacers_agent.evaluation import (
    DeepSeekJudgeResult,
    build_count_judge_payload,
    build_judge_request_hash,
    count_deterministic_metrics,
    merge_count_evaluation,
)
from spacers_agent.schemas import CountTargetSpec, CountingResult, GlobalPointObservation, GroundTruth
from spacers_agent.settings import DeepSeekSettings


def _response(content: str) -> SimpleNamespace:
    usage = SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=usage)


def _meta(tmp_path: Path, request_id: str = "sample:judge") -> RequestMeta:
    return RequestMeta(
        request_id=request_id,
        request_hash="b" * 64,
        prompt_version="deepseek-judge-v1",
        sample_id="sample",
        artifact_dir=tmp_path / "judge",
    )


def _counting() -> CountingResult:
    point = GlobalPointObservation(
        global_id="sample:r000_c000:p1",
        target="building",
        source_tile_id="r000_c000",
        local_id="p1",
        local_x_norm=500,
        local_y_norm=500,
        local_radius_norm=0,
        global_x_px=10,
        global_y_px=10,
        global_x_norm=100,
        global_y_norm=100,
        radius_px=0,
        confidence=0.9,
        ownership_valid=True,
        near_core_boundary=False,
        accepted=True,
        short_evidence="roof",
    )
    return CountingResult(
        sample_id="sample",
        target="building",
        question="count buildings",
        source_width=100,
        source_height=100,
        tile_count=1,
        succeeded_tiles=["r000_c000"],
        global_points=[point],
        final_count=1,
        status="completed",
    )


def _target() -> CountTargetSpec:
    return CountTargetSpec(
        canonical_label="building",
        inclusion_rule="Count buildings.",
        exclusion_rule="Exclude shadows.",
    )


def _judge(verdict: str = "incorrect") -> DeepSeekJudgeResult:
    return DeepSeekJudgeResult(
        judge_scope="text_and_structured_evidence_only",
        can_verify_visual_truth=False,
        semantic_correctness=0.4,
        answer_evidence_consistency=0.8,
        constraint_following=0.9,
        clarity=0.9,
        verdict=verdict,
        issues=["count differs"],
        concise_rationale="The structured count differs from gold.",
    )


def test_deterministic_metrics_and_payload_exclude_visual_payloads() -> None:
    metrics = count_deterministic_metrics(37, 36)
    payload = build_count_judge_payload(
        question="count buildings",
        target=_target(),
        display_answer="37 buildings",
        counting=_counting(),
        ground_truth=GroundTruth(count=1, answers=["1"]),
        min_confidence=0.2,
    )

    assert metrics.exact_match == 0
    assert metrics.absolute_error == 1
    assert payload["prediction"]["point_count"] == 1
    assert "global_points" not in str(payload)
    assert "image" not in str(payload).casefold()
    assert len(build_judge_request_hash(model="deepseek", prompt_text="prompt", sample_id="sample", payload=payload)) == 64


def test_judge_cannot_silently_override_known_count_mismatch() -> None:
    record = merge_count_evaluation(
        sample_id="sample",
        counting=_counting(),
        ground_truth=GroundTruth(count=2),
        judge_raw='{"verdict":"correct"}',
        judge_parsed=_judge("correct"),
    )

    assert record.deterministic_metrics is not None
    assert record.deterministic_metrics.exact_match == 0
    assert record.judge_inconsistency
    assert record.judge_parsed is not None and record.judge_parsed.verdict == "correct"


@pytest.mark.asyncio
async def test_deepseek_client_repairs_invalid_json_and_writes_text_only_artifacts(tmp_path: Path) -> None:
    invalid = '{"judge_scope":"wrong","can_verify_visual_truth":false}'
    valid = _judge().model_dump_json()
    responses = iter([_response(invalid), _response(valid)])
    seen_messages: list[list[dict[str, object]]] = []

    async def complete(**kwargs: object) -> SimpleNamespace:
        seen_messages.append(kwargs["messages"])  # type: ignore[arg-type]
        return next(responses)

    client = DeepSeekJudgeClient(
        DeepSeekSettings(max_retries=0),
        judge_prompt="return json",
        repair_prompt="repair json",
        completion_create=complete,
        retry_base_seconds=0,
    )
    result = await client.judge({"task": "counting", "prediction": {"final_count": 1}}, request_meta=_meta(tmp_path))

    assert result.judge_scope == "text_and_structured_evidence_only"
    assert len(seen_messages) == 2
    assert all("image" not in str(messages).casefold() for messages in seen_messages)
    assert "ValidationError" in (tmp_path / "judge" / "validation.json").read_text(encoding="utf-8")
    assert "total_tokens" in (tmp_path / "judge" / "validation.json").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_deepseek_client_retries_empty_response_and_uses_cache(tmp_path: Path) -> None:
    calls = 0
    responses = iter([_response(""), _response(_judge().model_dump_json())])

    async def complete(**_: object) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return next(responses)

    cache = JsonResponseCache(tmp_path / "cache")
    client = DeepSeekJudgeClient(
        DeepSeekSettings(max_retries=1),
        judge_prompt="return json",
        repair_prompt="repair json",
        cache=cache,
        completion_create=complete,
        retry_base_seconds=0,
    )
    first = await client.judge({"task": "counting"}, request_meta=_meta(tmp_path, "first"))
    second = await client.judge({"task": "counting"}, request_meta=_meta(tmp_path, "second"))

    assert first == second
    assert calls == 2
