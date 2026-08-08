"""Contract tests for JudgeService: policy, budget, merge, and resume.

JudgeService 契约测试：策略（none/errors-only/all）、预算（仅真正发起
Judge 时 reserve_deepseek）、确定性指标不可被 judge 覆盖、失败保留
deterministic、resume 不重复/可补。所有测试离线：使用注入的 fake client。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.counting.schema import CountTargetSpec, CountingResult, GlobalPointObservation
from data.schema import GroundTruth, ImageRef, UnifiedSample
from evaluation.judges.base import DeepSeekJudgeResult, VQAAnswerJudgeResult
from evaluation.records import EvaluationRecord, VQADeterministicMetrics, VQAEvaluationRecord
from workflows.call_budget import CallBudget
from workflows.judge_service import JudgeService


# ── helpers / 测试辅助 ──────────────────────────────────────────────────────


class _FakeJudgeClient:
    """Protocol-compatible fake that records calls and returns or raises a
    configured outcome. 记录调用并返回/抛出配置结果的协议兼容 fake。"""

    def __init__(self, verdict=None, error: Exception | None = None) -> None:
        self.verdict = verdict
        self.error = error
        self.calls: list[tuple[dict, object, str | None]] = []

    def judge(self, payload, *, request_meta):
        return self.judge_json(
            payload,
            response_model=DeepSeekJudgeResult,
            request_meta=request_meta,
        )

    def judge_json(self, payload, *, response_model, request_meta, system_prompt=None):
        self.calls.append((payload, request_meta, system_prompt))
        if self.error is not None:
            raise self.error
        if self.verdict is None:
            raise AssertionError("fake client configured without a verdict")
        return response_model.model_validate(self.verdict.model_dump())


def _service(client=None, *, model_id: str = "deepseek-model") -> JudgeService:
    return JudgeService(
        judge_prompt="counting judge prompt",
        vqa_judge_prompt="vqa judge prompt",
        judge_client=client,
        model_id=model_id,
        counting_min_confidence=0.2,
    )


def _sample(question: str = "Is there a road?", answers: list[str] | None = None) -> UnifiedSample:
    return UnifiedSample(
        sample_id="s1",
        dataset="parity",
        split="test",
        task="general_vqa",
        images=[ImageRef(image_id="i0", path=Path("img.png"), role="image")],
        question=question,
        ground_truth=GroundTruth(answers=answers or ["yes"]),
    )


def _budget(max_deepseek: int = 2) -> CallBudget:
    return CallBudget(max_qwen_calls=10, max_deepseek_calls=max_deepseek)


def _vqa_verdict(score: int = 1) -> VQAAnswerJudgeResult:
    return VQAAnswerJudgeResult(score=score, concise_rationale="ok")


def _point(gid: str = "p0", *, accepted: bool = True) -> GlobalPointObservation:
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
        confidence=0.9,
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
    )


def _counting_verdict() -> DeepSeekJudgeResult:
    return DeepSeekJudgeResult(
        judge_scope="text_and_structured_evidence_only",
        can_verify_visual_truth=False,
        semantic_correctness=1.0,
        answer_evidence_consistency=1.0,
        constraint_following=1.0,
        clarity=1.0,
        verdict="correct",
        issues=[],
        concise_rationale="ok",
    )


def _write_evaluation(sample_dir: Path, record: dict) -> None:
    (sample_dir / "vqa_evaluation.json").write_text(
        json.dumps(record, ensure_ascii=False), encoding="utf-8"
    )


# ── policy / budget / merge ─────────────────────────────────────────────────


def test_no_client_is_deterministic_only(tmp_path: Path) -> None:
    record = _service(None).judge_vqa(
        sample=_sample(),
        candidate_answer="yes",
        sample_dir=tmp_path,
        judge_policy="all",
    )
    assert isinstance(record, EvaluationRecord)
    assert record.task == "general_vqa"
    assert record.judge_status == "not_requested"
    assert record.judge_parsed is None
    assert record.deterministic_metrics is not None
    assert record.deterministic_metrics.exact_match is True


def test_policy_none_never_calls(tmp_path: Path) -> None:
    client = _FakeJudgeClient(verdict=_vqa_verdict())
    budget = _budget()
    record = _service(client).judge_vqa(
        sample=_sample(),
        candidate_answer="yes",
        sample_dir=tmp_path,
        judge_policy="none",
        call_budget=budget,
    )
    assert record.judge_status == "not_requested"
    assert client.calls == []
    assert budget.deepseek_calls_used == 0


def test_errors_only_exact_match_never_calls(tmp_path: Path) -> None:
    client = _FakeJudgeClient(verdict=_vqa_verdict())
    budget = _budget()
    record = _service(client).judge_vqa(
        sample=_sample(),
        candidate_answer="yes",
        sample_dir=tmp_path,
        judge_policy="errors-only",
        call_budget=budget,
    )
    assert record.judge_status == "not_requested"
    assert client.calls == []
    assert budget.deepseek_calls_used == 0


def test_errors_only_mismatch_calls_once_and_preserves_deterministic(tmp_path: Path) -> None:
    client = _FakeJudgeClient(verdict=_vqa_verdict())
    budget = _budget()
    record = _service(client).judge_vqa(
        sample=_sample(),
        candidate_answer="no",
        sample_dir=tmp_path,
        judge_policy="errors-only",
        call_budget=budget,
    )
    assert len(client.calls) == 1
    assert budget.deepseek_calls_used == 1
    assert record.judge_status == "succeeded"
    assert record.judge_parsed is not None
    assert record.judge_parsed.score == 1
    # Judge output never overrides the deterministic mismatch. / judge 绝不覆盖确定性失配。
    assert record.deterministic_metrics is not None
    assert record.deterministic_metrics.exact_match is False


def test_policy_all_calls_even_on_exact_match(tmp_path: Path) -> None:
    client = _FakeJudgeClient(verdict=_vqa_verdict(score=0))
    budget = _budget()
    record = _service(client).judge_vqa(
        sample=_sample(),
        candidate_answer="yes",
        sample_dir=tmp_path,
        judge_policy="all",
        call_budget=budget,
    )
    assert len(client.calls) == 1
    assert budget.deepseek_calls_used == 1
    assert record.judge_status == "succeeded"
    assert record.judge_parsed.score == 0
    assert record.deterministic_metrics is not None
    assert record.deterministic_metrics.exact_match is True


def test_budget_exhausted_records_failure_without_calling(tmp_path: Path) -> None:
    client = _FakeJudgeClient(verdict=_vqa_verdict())
    record = _service(client).judge_vqa(
        sample=_sample(),
        candidate_answer="yes",
        sample_dir=tmp_path,
        judge_policy="all",
        call_budget=_budget(max_deepseek=0),
    )
    assert client.calls == []
    assert record.judge_status == "failed"
    assert record.judge_error == "CallBudgetExceeded"
    assert record.deterministic_metrics is not None
    assert record.deterministic_metrics.exact_match is True


def test_judge_failure_keeps_deterministic_without_raw_text(tmp_path: Path) -> None:
    client = _FakeJudgeClient(error=RuntimeError("secret-raw-detail"))
    record = _service(client).judge_vqa(
        sample=_sample(),
        candidate_answer="yes",
        sample_dir=tmp_path,
        judge_policy="all",
    )
    assert record.judge_status == "failed"
    assert record.judge_error == "RuntimeError"
    assert "secret-raw-detail" not in record.model_dump_json()
    assert record.deterministic_metrics is not None
    assert record.deterministic_metrics.exact_match is True


def test_vqa_request_meta_and_text_only_payload(tmp_path: Path) -> None:
    client = _FakeJudgeClient(verdict=_vqa_verdict())
    _service(client).judge_vqa(
        sample=_sample(question="Is there a road?", answers=["yes"]),
        candidate_answer="yes",
        sample_dir=tmp_path,
        judge_policy="all",
    )
    payload, meta, system_prompt = client.calls[0]
    assert meta.request_id == "s1:deepseek-vqa"
    assert meta.prompt_version == "deepseek-vqa-judge-v1"
    assert meta.sample_id == "s1"
    assert meta.artifact_dir == tmp_path / "deepseek_vqa_judge"
    assert system_prompt == "vqa judge prompt"
    assert payload["prediction"] == {"answer": "yes"}
    assert payload["deterministic_metrics"] == {"exact_match": 1}
    serialized = json.dumps(payload).casefold()
    assert "image" not in serialized


# ── counting post-hoc judge / 计数事后 judge ────────────────────────────────


def test_judge_counting_deterministic_only(tmp_path: Path) -> None:
    points = (_point("p0"), _point("p1"))
    record = _service(None).judge_counting(
        sample_id="s1",
        question="How many cars?",
        target=_target(),
        display_answer="2",
        counting=_counting(points, final_count=2),
        ground_truth=GroundTruth(count=2),
        artifact_dir=tmp_path,
    )
    assert record.task == "counting"
    assert record.judge_status == "not_requested"
    assert record.deterministic_metrics is not None
    assert record.deterministic_metrics.exact_match == 1


def test_judge_counting_with_client(tmp_path: Path) -> None:
    points = (_point("p0"), _point("p1"))
    client = _FakeJudgeClient(verdict=_counting_verdict())
    record = _service(client).judge_counting(
        sample_id="s1",
        question="How many cars?",
        target=_target(),
        display_answer="2",
        counting=_counting(points, final_count=2),
        ground_truth=GroundTruth(count=2),
        artifact_dir=tmp_path,
    )
    assert record.judge_status == "succeeded"
    assert record.judge_parsed is not None
    assert record.judge_parsed.verdict == "correct"
    assert record.deterministic_metrics is not None
    assert record.deterministic_metrics.exact_match == 1
    payload, meta, _ = client.calls[0]
    assert payload["task"] == "counting"
    assert payload["target_spec"]["canonical_label"] == "car"
    assert payload["prediction"]["final_count"] == 2
    assert meta.request_id == "s1:deepseek"
    assert meta.prompt_version == "deepseek-judge-v1"
    assert meta.artifact_dir == tmp_path


def test_judge_counting_client_failure_keeps_deterministic(tmp_path: Path) -> None:
    points = (_point("p0"), _point("p1"))
    client = _FakeJudgeClient(error=RuntimeError("raw-counting-secret"))
    record = _service(client).judge_counting(
        sample_id="s1",
        question="q",
        target=_target(),
        display_answer="2",
        counting=_counting(points, final_count=2),
        ground_truth=GroundTruth(count=2),
        artifact_dir=tmp_path,
    )
    assert record.judge_status == "failed"
    assert record.judge_error == "RuntimeError"
    assert "raw-counting-secret" not in record.model_dump_json()
    assert record.deterministic_metrics is not None
    assert record.deterministic_metrics.exact_match == 1


def test_judge_counting_request_hash_stable_and_model_sensitive(tmp_path: Path) -> None:
    points = (_point("p0"), _point("p1"))
    counting = _counting(points, final_count=2)
    ground_truth = GroundTruth(count=2)
    client = _FakeJudgeClient(verdict=_counting_verdict())
    kwargs = dict(
        sample_id="s1",
        question="q",
        target=_target(),
        display_answer="2",
        counting=counting,
        ground_truth=ground_truth,
        artifact_dir=tmp_path,
    )
    _service(client).judge_counting(**kwargs)
    _service(client).judge_counting(**kwargs)
    first_hash = client.calls[0][1].request_hash
    second_hash = client.calls[1][1].request_hash
    assert first_hash == second_hash
    _service(client, model_id="another-model").judge_counting(**kwargs)
    assert client.calls[2][1].request_hash != first_hash


# ── resume / 续跑补判 ───────────────────────────────────────────────────────


def test_resume_succeeded_unified_shape_is_not_rejudged(tmp_path: Path) -> None:
    client = _FakeJudgeClient(verdict=_vqa_verdict())
    _write_evaluation(
        tmp_path,
        EvaluationRecord(
            sample_id="s1",
            task="general_vqa",
            deterministic_metrics=VQADeterministicMetrics(exact_match=True),
            judge_status="succeeded",
            judge_parsed=_vqa_verdict().model_dump(mode="json"),
        ).model_dump(mode="json"),
    )
    record = _service(client).judge_vqa_resume(
        sample=_sample(),
        candidate_answer="yes",
        sample_dir=tmp_path,
        judge_policy="all",
    )
    assert record.judge_status == "succeeded"
    assert client.calls == []


def test_resume_succeeded_legacy_shape_is_converted_not_rejudged(tmp_path: Path) -> None:
    client = _FakeJudgeClient(verdict=_vqa_verdict())
    _write_evaluation(
        tmp_path,
        VQAEvaluationRecord(
            sample_id="s1",
            question="Is there a road?",
            reference_answers=["yes"],
            candidate_answer="yes",
            exact_match=True,
            judge_status="succeeded",
            judge_score=1,
            judge_parsed=_vqa_verdict().model_dump(mode="json"),
        ).model_dump(mode="json"),
    )
    record = _service(client).judge_vqa_resume(
        sample=_sample(),
        candidate_answer="yes",
        sample_dir=tmp_path,
        judge_policy="all",
    )
    assert record.judge_status == "succeeded"
    assert record.task == "general_vqa"
    assert client.calls == []


def test_resume_failed_evaluation_is_rejudged(tmp_path: Path) -> None:
    client = _FakeJudgeClient(verdict=_vqa_verdict())
    (tmp_path / "agent_result.json").write_text(
        json.dumps({"answer": "yes"}), encoding="utf-8"
    )
    _write_evaluation(
        tmp_path,
        EvaluationRecord(
            sample_id="s1",
            task="general_vqa",
            deterministic_metrics=VQADeterministicMetrics(exact_match=False),
            judge_status="failed",
            judge_error="DeepSeekJudgeError",
        ).model_dump(mode="json"),
    )
    record = _service(client).judge_vqa_resume(
        sample=_sample(),
        candidate_answer="yes",
        sample_dir=tmp_path,
        judge_policy="all",
    )
    assert record.judge_status == "succeeded"
    assert len(client.calls) == 1


def test_resume_missing_evaluation_uses_saved_answer(tmp_path: Path) -> None:
    client = _FakeJudgeClient(verdict=_vqa_verdict())
    (tmp_path / "agent_result.json").write_text(
        json.dumps({"answer": "42"}), encoding="utf-8"
    )
    record = _service(client).judge_vqa_resume(
        sample=_sample(),
        candidate_answer="fallback",
        sample_dir=tmp_path,
        judge_policy="all",
    )
    assert record.judge_status == "succeeded"
    assert len(client.calls) == 1
    assert client.calls[0][0]["prediction"]["answer"] == "42"


def test_resume_corrupt_evaluation_is_rejudged(tmp_path: Path) -> None:
    client = _FakeJudgeClient(verdict=_vqa_verdict())
    (tmp_path / "agent_result.json").write_text(
        json.dumps({"answer": "yes"}), encoding="utf-8"
    )
    _write_evaluation(tmp_path, "{corrupt json")
    record = _service(client).judge_vqa_resume(
        sample=_sample(),
        candidate_answer="yes",
        sample_dir=tmp_path,
        judge_policy="all",
    )
    assert record.judge_status == "succeeded"
    assert len(client.calls) == 1


def test_resume_missing_agent_result_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="agent_result.json"):
        _service(_FakeJudgeClient(verdict=_vqa_verdict())).judge_vqa_resume(
            sample=_sample(),
            candidate_answer="yes",
            sample_dir=tmp_path,
            judge_policy="all",
        )


def test_resume_rejudge_still_respects_policy(tmp_path: Path) -> None:
    client = _FakeJudgeClient(verdict=_vqa_verdict())
    (tmp_path / "agent_result.json").write_text(
        json.dumps({"answer": "yes"}), encoding="utf-8"
    )
    _write_evaluation(
        tmp_path,
        EvaluationRecord(
            sample_id="s1",
            task="general_vqa",
            deterministic_metrics=VQADeterministicMetrics(exact_match=False),
            judge_status="failed",
            judge_error="DeepSeekJudgeError",
        ).model_dump(mode="json"),
    )
    record = _service(client).judge_vqa_resume(
        sample=_sample(),
        candidate_answer="yes",
        sample_dir=tmp_path,
        judge_policy="none",
    )
    assert record.judge_status == "not_requested"
    assert client.calls == []
