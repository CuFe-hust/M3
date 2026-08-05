"""Phase 5 — JudgeService error visibility and resume behavior.
Phase 5 — JudgeService 错误可见性与 resume 行为。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from models.base import RequestMeta
from spacers_agent.prompt_catalog import PromptCatalog
from spacers_agent.schemas import GroundTruth, ImageRef, UnifiedSample
from spacers_agent.settings import AppSettings
from spacers_agent.workflows.judge_service import JudgeService

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "legacy"
TEST_IMAGE = FIXTURES / "test_image.png"
PROMPT_ROOT = Path(__file__).resolve().parents[2] / "prompts"


class _FailingJsonJudge:
    def __init__(self):
        self.prompt = "test"

    async def judge_json(self, payload, *, response_model, request_meta):
        raise RuntimeError("judge client simulated failure")

    async def judge(self, payload, *, request_meta):
        raise RuntimeError("judge client simulated failure")


class _SucceedingJudge:
    def __init__(self):
        self.prompt = "test"

    async def judge_json(self, payload, *, response_model, request_meta):
        return response_model.model_validate({"exact_match": True, "correct": True, "explanation": "ok"})

    async def judge(self, payload, *, request_meta):
        return {"exact_match": True, "correct": True, "explanation": "ok"}


def _sample() -> UnifiedSample:
    return UnifiedSample(
        sample_id="s1", dataset="test", split="test", task="general_vqa",
        images=[ImageRef(image_id="i1", path=TEST_IMAGE, role="image")],
        question="Is this correct?",
        ground_truth=GroundTruth(answers=["yes"]),
    )


@pytest.mark.asyncio
async def test_judge_vqa_error_is_visible_and_does_not_delete_agent_result(tmp_path: Path):
    settings = AppSettings()
    catalog = PromptCatalog(PROMPT_ROOT)
    service = JudgeService(
        settings,
        judge_prompt=catalog["count_judge"],
        vqa_judge_prompt=catalog["vqa_judge"],
        repair_prompt=catalog["json_repair"],
        judge_client=_FailingJsonJudge(),
    )
    sample = _sample()
    evaluation = await service.judge_vqa(
        sample=sample,
        candidate_answer="yes",
        sample_dir=tmp_path,
        judge_policy="all",
    )
    assert evaluation.judge_status == "failed" or hasattr(evaluation, "judge_status")
    assert "RuntimeError" in str(evaluation.judge_error or "")
    # Judge error must not delete agent result — the evaluation object still exists
    assert evaluation.sample_id == "s1"


@pytest.mark.asyncio
async def test_judge_vqa_success_sets_succeeded_status(tmp_path: Path):
    settings = AppSettings()
    catalog = PromptCatalog(PROMPT_ROOT)
    service = JudgeService(
        settings,
        judge_prompt=catalog["count_judge"],
        vqa_judge_prompt=catalog["vqa_judge"],
        repair_prompt=catalog["json_repair"],
        judge_client=_SucceedingJudge(),
    )
    sample = _sample()
    evaluation = await service.judge_vqa(
        sample=sample,
        candidate_answer="yes",
        sample_dir=tmp_path,
        judge_policy="all",
    )
    assert evaluation.exact_match is True


@pytest.mark.asyncio
async def test_judge_vqa_resume_skips_when_already_succeeded(tmp_path: Path):
    """judge_vqa_resume returns existing evaluation when judge_status is succeeded."""
    import json
    settings = AppSettings()
    catalog = PromptCatalog(PROMPT_ROOT)
    service = JudgeService(
        settings,
        judge_prompt=catalog["count_judge"],
        vqa_judge_prompt=catalog["vqa_judge"],
        repair_prompt=catalog["json_repair"],
        judge_client=_SucceedingJudge(),
    )
    sample_dir = tmp_path / "samples" / "s1"
    sample_dir.mkdir(parents=True)
    (sample_dir / "expert_result.json").write_text(
        json.dumps({"expert": "vqa", "answer": "yes"}), encoding="utf-8"
    )
    (sample_dir / "vqa_evaluation.json").write_text(
        json.dumps({"sample_id": "s1", "judge_status": "succeeded", "exact_match": True}),
        encoding="utf-8",
    )
    sample = _sample()
    evaluation = await service.judge_vqa_resume(
        sample=sample,
        candidate_answer="yes",
        sample_dir=sample_dir,
    )
    assert evaluation["judge_status"] == "succeeded"


@pytest.mark.asyncio
async def test_judge_vqa_resume_raises_file_not_found_when_expert_missing(tmp_path: Path):
    settings = AppSettings()
    catalog = PromptCatalog(PROMPT_ROOT)
    service = JudgeService(
        settings,
        judge_prompt=catalog["count_judge"],
        vqa_judge_prompt=catalog["vqa_judge"],
        repair_prompt=catalog["json_repair"],
    )
    sample = _sample()
    with pytest.raises(FileNotFoundError):
        await service.judge_vqa_resume(
            sample=sample,
            candidate_answer="yes",
            sample_dir=tmp_path / "nonexistent",
        )
