"""Judge orchestration: policy, budget, and merge for the optional text-only
judge layer. 可选仅文本 judge 层的编排：策略、预算与合并。

JudgeService 只做策略（none/errors-only/all）、预算（仅真正发起 Judge 时
reserve_deepseek）与合并（judge 结果永不覆盖确定性指标）；网络与恢复细节
在 evaluation.judges.deepseek。judge 异常以稳定 judge_error 记录（类名），
绝不向调用方抛出原始异常。本模块不读取环境变量、不加载任何模型；client
与 prompt 由 composition root 注入。
"""

from __future__ import annotations

import json
from pathlib import Path

from agents.counting.schema import CountTargetSpec, CountingResult
from data.schema import GroundTruth, UnifiedSample
from evaluation.judges.base import (
    JudgeClient,
    DeepSeekJudgeResult,
    VQAAnswerJudgeResult,
    build_count_judge_payload,
    build_judge_request_hash,
    build_vqa_judge_payload,
    stable_error_label,
)
from evaluation.metrics.counting import merge_count_evaluation
from evaluation.metrics.caption import merge_caption_evaluation
from evaluation.metrics.vqa import merge_vqa_evaluation, to_evaluation_record
from evaluation.records import (
    EVALUATION_FILENAME_BY_TASK,
    EvaluationRecord,
    VQAEvaluationRecord,
)
from models.base import RequestMeta
from workflows.call_budget import CallBudget


class JudgeService:
    """Text-only judge orchestration for counting and VQA evaluation; judge
    output is recorded alongside deterministic metrics and can never replace
    them. 计数与 VQA 评估的仅文本 judge 编排；judge 输出与确定性指标并列
    记录，绝不替换它们。"""

    def __init__(
        self,
        *,
        judge_prompt: str,
        judge_prompt_version: str,
        vqa_judge_prompt: str,
        vqa_judge_prompt_version: str,
        judge_client: JudgeClient | None = None,
        model_id: str = "deepseek",
        counting_min_confidence: float = 0.2,
    ) -> None:
        self.judge_client = judge_client
        self._judge_prompt = judge_prompt
        self._judge_prompt_version = judge_prompt_version
        self._vqa_judge_prompt = vqa_judge_prompt
        self._vqa_judge_prompt_version = vqa_judge_prompt_version
        self._model_id = model_id
        self._counting_min_confidence = counting_min_confidence

    def judge_counting(
        self,
        *,
        sample_id: str,
        question: str,
        target: CountTargetSpec,
        display_answer: str,
        counting: CountingResult,
        ground_truth: GroundTruth | None,
        artifact_dir: Path,
    ) -> EvaluationRecord:
        """Judge one counting result deterministically and, when a client is
        injected, via DeepSeek. Post-hoc evaluation path without a per-sample
        call budget. 对单个计数结果进行确定性审核，注入客户端时追加 DeepSeek
        审核。事后评估路径，不设逐样本调用预算。"""

        client = self.judge_client
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
            min_confidence=self._counting_min_confidence,
        )
        try:
            verdict = client.judge(
                payload,
                request_meta=RequestMeta(
                    request_id=f"{sample_id}:deepseek",
                    request_hash=build_judge_request_hash(
                        model=self._model_id,
                        prompt_text=self._judge_prompt,
                        prompt_version=self._judge_prompt_version,
                        sample_id=sample_id,
                        payload=payload,
                        response_schema=DeepSeekJudgeResult.model_json_schema(),
                    ),
                    prompt_version=self._judge_prompt_version,
                    sample_id=sample_id,
                    artifact_dir=artifact_dir,
                ),
            )
        except Exception as error:
            return merge_count_evaluation(
                sample_id=sample_id,
                counting=counting,
                ground_truth=ground_truth,
                judge_error=stable_error_label(error),
            )
        return merge_count_evaluation(
            sample_id=sample_id,
            counting=counting,
            ground_truth=ground_truth,
            judge_parsed=verdict,
        )

    def judge_vqa(
        self,
        *,
        sample: UnifiedSample,
        candidate_answer: str,
        sample_dir: Path,
        judge_policy: str = "all",
        call_budget: CallBudget | None = None,
    ) -> EvaluationRecord:
        """Judge one VQA answer deterministically and, per policy, via
        DeepSeek. The DeepSeek call budget is reserved only when a judge call
        is actually attempted; failures are recorded as stable judge_error and
        never replace the deterministic match. 对单个 VQA 答案进行确定性审核，
        并按策略追加 DeepSeek 审核。只在真正尝试 Judge 调用时预留 DeepSeek
        预算；失败记录为稳定 judge_error，绝不替换确定性匹配。"""

        references = (
            list(sample.ground_truth.answers)
            if sample.ground_truth is not None
            else []
        )
        initial = merge_vqa_evaluation(
            sample_id=sample.sample_id,
            question=sample.question,
            reference_answers=references,
            candidate_answer=candidate_answer,
        )
        client = self.judge_client
        exact = (
            initial.deterministic_metrics is not None
            and initial.deterministic_metrics.exact_match
        )
        should_judge = client is not None and (
            judge_policy == "all"
            or (judge_policy == "errors-only" and not exact)
        )
        if not should_judge:
            return initial
        payload = build_vqa_judge_payload(
            question=sample.question,
            reference_answers=references,
            candidate_answer=candidate_answer,
        )
        try:
            if call_budget is not None:
                call_budget.reserve_deepseek()
            verdict = client.judge_json(
                payload,
                response_model=VQAAnswerJudgeResult,
                request_meta=RequestMeta(
                    request_id=f"{sample.sample_id}:deepseek-vqa",
                    request_hash=build_judge_request_hash(
                        model=self._model_id,
                        prompt_text=self._vqa_judge_prompt,
                        prompt_version=self._vqa_judge_prompt_version,
                        sample_id=sample.sample_id,
                        payload=payload,
                        response_schema=VQAAnswerJudgeResult.model_json_schema(),
                    ),
                    prompt_version=self._vqa_judge_prompt_version,
                    sample_id=sample.sample_id,
                    artifact_dir=sample_dir / "deepseek_vqa_judge",
                ),
                system_prompt=self._vqa_judge_prompt,
            )
        except Exception as error:
            return merge_vqa_evaluation(
                sample_id=sample.sample_id,
                question=sample.question,
                reference_answers=references,
                candidate_answer=candidate_answer,
                judge_error=stable_error_label(error),
            )
        return merge_vqa_evaluation(
            sample_id=sample.sample_id,
            question=sample.question,
            reference_answers=references,
            candidate_answer=candidate_answer,
            judge_parsed=verdict,
        )

    def judge_caption(
        self,
        *,
        sample: UnifiedSample,
        candidate_answer: str,
        sample_dir: Path,
        judge_policy: str = "all",
        call_budget: CallBudget | None = None,
    ) -> EvaluationRecord:
        """Judge a caption answer with the same text-only answer contract as
        VQA while preserving the caption evaluation task and metrics.
        使用与 VQA 相同的纯文本答案审核契约，但保留 caption 任务和指标。
        """

        references = (
            list(sample.ground_truth.answers)
            if sample.ground_truth is not None
            else []
        )
        initial = merge_caption_evaluation(
            sample_id=sample.sample_id,
            references=references,
            candidate=candidate_answer,
        )
        client = self.judge_client
        if client is None or judge_policy == "none":
            return initial
        payload = build_vqa_judge_payload(
            question=sample.question,
            reference_answers=references,
            candidate_answer=candidate_answer,
        )
        try:
            if call_budget is not None:
                call_budget.reserve_deepseek()
            verdict = client.judge_json(
                payload,
                response_model=VQAAnswerJudgeResult,
                request_meta=RequestMeta(
                    request_id=f"{sample.sample_id}:deepseek-caption",
                    request_hash=build_judge_request_hash(
                        model=self._model_id,
                        prompt_text=self._vqa_judge_prompt,
                        prompt_version=self._vqa_judge_prompt_version,
                        sample_id=sample.sample_id,
                        payload=payload,
                        response_schema=VQAAnswerJudgeResult.model_json_schema(),
                    ),
                    prompt_version=self._vqa_judge_prompt_version,
                    sample_id=sample.sample_id,
                    artifact_dir=sample_dir / "deepseek_caption_judge",
                ),
                system_prompt=self._vqa_judge_prompt,
            )
        except Exception as error:
            return merge_caption_evaluation(
                sample_id=sample.sample_id,
                references=references,
                candidate=candidate_answer,
                judge_error=stable_error_label(error),
            )
        return merge_caption_evaluation(
            sample_id=sample.sample_id,
            references=references,
            candidate=candidate_answer,
            judge_parsed=verdict,
        )

    def judge_vqa_resume(
        self,
        *,
        sample: UnifiedSample,
        candidate_answer: str,
        sample_dir: Path,
        judge_policy: str = "all",
        call_budget: CallBudget | None = None,
    ) -> EvaluationRecord:
        """Judge a VQA answer during resume: a persisted succeeded evaluation
        is returned unchanged (never re-judged); missing, corrupt, or failed
        evaluations are re-judged from the persisted agent answer. 在 resume
        期间审核 VQA 答案：已持久化的 succeeded 评估原样返回（绝不重判）；
        缺失/损坏/failed 的评估用持久化 agent 答案重新审核。"""

        evaluation_path = (
            sample_dir / EVALUATION_FILENAME_BY_TASK["general_vqa"]
        )
        if evaluation_path.is_file():
            existing = self._load_existing_evaluation(evaluation_path)
            if existing is not None and existing.judge_status == "succeeded":
                return existing
        result_path = sample_dir / "agent_result.json"
        if not result_path.is_file():
            raise FileNotFoundError(
                f"agent_result.json missing for resume judge: {sample_dir}"
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        saved_answer = str(result.get("answer", ""))
        return self.judge_vqa(
            sample=sample,
            candidate_answer=saved_answer or candidate_answer,
            sample_dir=sample_dir,
            judge_policy=judge_policy,
            call_budget=call_budget,
        )

    def judge_caption_resume(
        self,
        *,
        sample: UnifiedSample,
        candidate_answer: str,
        sample_dir: Path,
        judge_policy: str = "all",
        call_budget: CallBudget | None = None,
    ) -> EvaluationRecord:
        """Resume a caption judge only when its persisted record is missing
        or failed. 已持久化成功的 caption judge 不重复调用。"""

        evaluation_path = sample_dir / EVALUATION_FILENAME_BY_TASK["caption"]
        if evaluation_path.is_file():
            try:
                existing = EvaluationRecord.model_validate(
                    json.loads(evaluation_path.read_text(encoding="utf-8"))
                )
            except (json.JSONDecodeError, UnicodeDecodeError, OSError, ValueError):
                existing = None
            if (
                existing is not None
                and existing.task == "caption"
                and existing.judge_status == "succeeded"
            ):
                return existing
        result_path = sample_dir / "agent_result.json"
        if not result_path.is_file():
            raise FileNotFoundError(
                f"agent_result.json missing for resume judge: {sample_dir}"
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        saved_answer = str(result.get("answer", ""))
        return self.judge_caption(
            sample=sample,
            candidate_answer=saved_answer or candidate_answer,
            sample_dir=sample_dir,
            judge_policy=judge_policy,
            call_budget=call_budget,
        )

    def _load_existing_evaluation(
        self, path: Path
    ) -> EvaluationRecord | None:
        """Parse a persisted VQA evaluation in either the unified or the
        legacy wrapper shape; anything unparseable counts as missing.
        解析持久化 VQA 评估（统一形状或旧版包装形状）；无法解析一律视为
        缺失。"""

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            return None
        if not isinstance(raw, dict):
            return None
        try:
            return EvaluationRecord.model_validate(raw)
        except ValueError:
            pass
        try:
            return to_evaluation_record(VQAEvaluationRecord.model_validate(raw))
        except ValueError:
            return None
