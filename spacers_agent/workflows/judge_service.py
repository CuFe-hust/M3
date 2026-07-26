"""Text-only DeepSeek judge service for counting and VQA evaluation.
纯文本 DeepSeek 审核服务，用于计数与 VQA 评估。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spacers_agent.clients.base import RequestMeta
from spacers_agent.clients.deepseek import DeepSeekJudgeClient
from spacers_agent.evaluation import (
    VQAAnswerJudgeResult,
    build_count_judge_payload,
    build_judge_request_hash,
    build_vqa_judge_payload,
    build_vqa_judge_request_hash,
    merge_count_evaluation,
    merge_vqa_evaluation,
)
from spacers_agent.schemas import CountTargetSpec, CountingResult, GroundTruth, UnifiedSample
from spacers_agent.routing import CallBudget
from spacers_agent.settings import AppSettings


class JudgeService:
    """Wraps DeepSeekJudgeClient for counting and VQA judge workflows.
    封装 DeepSeekJudgeClient 用于计数与 VQA 审核工作流。
    """

    def __init__(
        self,
        settings: AppSettings,
        *,
        judge_prompt: str,
        vqa_judge_prompt: str,
        repair_prompt: str,
        judge_client: DeepSeekJudgeClient | None = None,
    ) -> None:
        self.settings = settings
        self.judge_client = judge_client
        self._judge_prompt = judge_prompt
        self._vqa_judge_prompt = vqa_judge_prompt
        self._repair_prompt = repair_prompt

    def _counting_client(self) -> DeepSeekJudgeClient | None:
        """Return the injected client; never create a hidden network client.
        返回注入的客户端；绝不创建隐藏的网络客户端。
        """

        return self.judge_client

    def _vqa_client(self) -> DeepSeekJudgeClient | None:
        """Return the injected client for VQA judging. / 返回注入的 VQA 审核客户端。"""

        return self.judge_client

    async def judge_counting(
        self,
        *,
        sample_id: str,
        question: str,
        target: CountTargetSpec,
        display_answer: str,
        counting: CountingResult,
        ground_truth: GroundTruth | None,
        artifact_dir: Path,
    ) -> Any:
        """Judge one counting result deterministically and via DeepSeek.
        对单个计数结果进行确定性 + DeepSeek 审核。
        """

        client = self._counting_client()
        if client is None:
            return merge_count_evaluation(
                sample_id=sample_id,
                counting=counting,
                ground_truth=ground_truth,
            )
        payload = build_count_judge_payload(
            question=question,
            target=target,
            display_answer=display_answer,
            counting=counting,
            ground_truth=ground_truth,
            min_confidence=self.settings.counting.min_confidence,
        )
        try:
            verdict = await client.judge(
                payload,
                request_meta=RequestMeta(
                    request_id=f"{sample_id}:deepseek",
                    request_hash=build_judge_request_hash(
                        model=self.settings.models.deepseek.model,
                        prompt_text=getattr(client, "judge_prompt", self._judge_prompt),
                        sample_id=sample_id,
                        payload=payload,
                    ),
                    prompt_version="deepseek-judge-v1",
                    sample_id=sample_id,
                    artifact_dir=artifact_dir,
                ),
            )
            return merge_count_evaluation(
                sample_id=sample_id,
                counting=counting,
                ground_truth=ground_truth,
                judge_parsed=verdict,
            )
        except Exception as error:
            return merge_count_evaluation(
                sample_id=sample_id,
                counting=counting,
                ground_truth=ground_truth,
                judge_error=f"{type(error).__name__}: {error}",
            )

    async def judge_vqa(
        self,
        *,
        sample: UnifiedSample,
        candidate_answer: str,
        sample_dir: Path,
        judge_policy: str = "all",
        call_budget: CallBudget | None = None,
    ) -> Any:
        """Judge one VQA answer deterministically and via DeepSeek.
        对单个 VQA 答案进行确定性 + DeepSeek 审核。
        """

        references = sample.ground_truth.answers if sample.ground_truth is not None else []
        initial = merge_vqa_evaluation(
            sample_id=sample.sample_id,
            question=sample.question,
            reference_answers=references,
            candidate_answer=candidate_answer,
        )
        client = self._vqa_client()
        should_judge = client is not None and (
            judge_policy == "all" or (judge_policy == "errors-only" and not initial.exact_match)
        )
        if not should_judge:
            return initial
        if call_budget is not None:
            call_budget.reserve_deepseek()
        payload = build_vqa_judge_payload(
            question=sample.question,
            reference_answers=references,
            candidate_answer=candidate_answer,
        )
        try:
            verdict = await client.judge_json(
                payload,
                response_model=VQAAnswerJudgeResult,
                request_meta=RequestMeta(
                    request_id=f"{sample.sample_id}:deepseek-vqa",
                    request_hash=build_vqa_judge_request_hash(
                        model=self.settings.models.deepseek.model,
                        prompt_text=getattr(client, "judge_prompt", self._vqa_judge_prompt),
                        sample_id=sample.sample_id,
                        payload=payload,
                    ),
                    prompt_version="deepseek-vqa-judge-v1",
                    sample_id=sample.sample_id,
                    artifact_dir=sample_dir / "deepseek_vqa_judge",
                ),
            )
            evaluation = merge_vqa_evaluation(
                sample_id=sample.sample_id,
                question=sample.question,
                reference_answers=references,
                candidate_answer=candidate_answer,
                judge_parsed=verdict,
            )
        except Exception as error:
            evaluation = merge_vqa_evaluation(
                sample_id=sample.sample_id,
                question=sample.question,
                reference_answers=references,
                candidate_answer=candidate_answer,
                judge_error=f"{type(error).__name__}: {error}",
            )
        return evaluation

    async def judge_vqa_resume(
        self,
        *,
        sample: UnifiedSample,
        candidate_answer: str,
        sample_dir: Path,
        judge_policy: str = "all",
        call_budget: CallBudget | None = None,
    ) -> Any:
        """Judge a VQA answer during resume, preserving existing evaluation.
        在 resume 期间审核 VQA 答案，保留已有评估。
        """

        evaluation_path = sample_dir / "vqa_evaluation.json"
        if evaluation_path.is_file():
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            if evaluation.get("judge_status") == "succeeded":
                return evaluation
        result_path = sample_dir / "expert_result.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"expert_result.json missing for resume judge: {sample_dir}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        saved_answer = str(result.get("answer", ""))
        return await self.judge_vqa(
            sample=sample,
            candidate_answer=saved_answer or candidate_answer,
            sample_dir=sample_dir,
            judge_policy=judge_policy,
            call_budget=call_budget,
        )
