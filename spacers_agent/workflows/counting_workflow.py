"""Top-level counting orchestration above tile geometry. / 位于 tile 几何之上的顶层计数编排。"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from spacers_agent.counting import BoundaryConflict, PointCountingOrchestrator, finalize_representatives
from spacers_agent.evaluation import EvaluationRecord, merge_count_evaluation
from spacers_agent.evaluation import build_count_judge_payload, build_judge_request_hash
from spacers_agent.clients.base import RequestMeta
from spacers_agent.clients.deepseek import DeepSeekJudgeClient
from spacers_agent.routing import CallBudget, CallBudgetExceeded
from spacers_agent.imaging import build_core_halo_tiles, read_normalized_image
from spacers_agent.run_store import RunStore
from spacers_agent.schemas import CountTargetSpec, CountingDraft, CountingResult, IssueRecord, UnifiedSample
from spacers_agent.visualization import render_counting_overlay
from spacers_agent.targeting import CountTargetParser
from spacers_agent.workflow import atomic_write_json


class SampleExecutionResult(BaseModel):
    """Persisted top-level sample outcome. / 已持久化的顶层样本结果。"""

    model_config = ConfigDict(extra="forbid")

    result: CountingResult
    evaluation: EvaluationRecord | None = None


class SeamVerifier:
    """Verify only explicit boundary candidates. / 仅核验显式边界候选。"""

    async def verify(self, orchestrator: PointCountingOrchestrator, image_path: Path, draft: CountingDraft) -> list[tuple[str, str]]:
        """Return only model-approved same-instance pairs. / 仅返回模型确认的同一实例对。"""

        conflicts = [BoundaryConflict.model_validate(value) for value in draft.boundary_conflicts]
        decisions, _ = await orchestrator._verify_seams(read_normalized_image(image_path), conflicts, draft.raw_global_points, draft.sample_id)
        return [(item.first_global_id, item.second_global_id) for item in conflicts if decisions.get(item.conflict_id) == "same_instance"]


class EvidenceReviewer:
    """Preserve low-confidence evidence for later review. / 保留低置信度证据供后续复核。"""

    def review(self, result: CountingResult) -> list[IssueRecord]:
        """Create visible review issues without changing accepted points. / 创建可见复核问题但不改变接受点。"""

        return [IssueRecord(code="LOW_CONFIDENCE_REVIEW", message="Point rejected by confidence policy.", tile_ids=[point.source_tile_id], point_ids=[point.global_id]) for point in result.global_points if point.rejection_reason == "LOW_CONFIDENCE"]


class EvaluationService:
    """Create deterministic evaluation records without overriding predictions. / 创建不覆盖预测的确定性评估记录。"""

    def __init__(self, judge: DeepSeekJudgeClient | None = None, judge_prompt: str = "") -> None:
        self.judge, self.judge_prompt = judge, judge_prompt

    async def evaluate_counting(self, sample: UnifiedSample, result: CountingResult, *, target: CountTargetSpec, call_deepseek: bool, artifact_dir: Path, budget: CallBudget) -> EvaluationRecord:
        """Persist deterministic metrics first and optionally add a text-only judge. / 先持久化确定性指标，再可选添加纯文本 Judge。"""

        if not call_deepseek or self.judge is None:
            return merge_count_evaluation(sample_id=sample.sample_id, counting=result, ground_truth=sample.ground_truth)
        payload = build_count_judge_payload(question=sample.question, target=target, display_answer=f"{result.final_count} accepted points", counting=result, ground_truth=sample.ground_truth, min_confidence=0.0)
        try:
            budget.reserve_deepseek()
            verdict = await self.judge.judge(payload, request_meta=RequestMeta(request_id=f"{sample.sample_id}:judge", request_hash=build_judge_request_hash(model=self.judge.settings.model, prompt_text=self.judge_prompt, sample_id=sample.sample_id, payload=payload), prompt_version="deepseek-judge-v1", sample_id=sample.sample_id, artifact_dir=artifact_dir / "deepseek"))
            return merge_count_evaluation(sample_id=sample.sample_id, counting=result, ground_truth=sample.ground_truth, judge_parsed=verdict)
        except (CallBudgetExceeded, Exception) as error:
            return merge_count_evaluation(sample_id=sample.sample_id, counting=result, ground_truth=sample.ground_truth, judge_error=f"{type(error).__name__}: {error}")

    def evaluate(self, sample: UnifiedSample, result: CountingResult) -> EvaluationRecord:
        """Evaluate from persisted ground truth when available. / 在可用时从持久真值评估。"""

        return merge_count_evaluation(sample_id=sample.sample_id, counting=result, ground_truth=sample.ground_truth)


class CountingWorkflow:
    """Orchestrate target, seam, review, persistence and evaluation. / 编排目标、seam、复核、持久化与评估。"""

    def __init__(self, *, target_parser: CountTargetParser, orchestrator_factory: Callable[[Path], PointCountingOrchestrator], seam_verifier: SeamVerifier, evidence_reviewer: EvidenceReviewer, evaluation_service: EvaluationService, run_store: RunStore) -> None:
        self.target_parser, self.orchestrator_factory = target_parser, orchestrator_factory
        self.seam_verifier, self.evidence_reviewer, self.evaluation_service, self.run_store = seam_verifier, evidence_reviewer, evaluation_service, run_store

    async def run(self, sample: UnifiedSample, *, run_dir: Path, target_override: CountTargetSpec | None, evaluate: bool, render: bool, resume: bool) -> SampleExecutionResult:
        """Run the full point-derived chain while tile recovery remains delegated. / 运行完整点导出链，同时将 tile 恢复委托给编排器。"""

        sample_dir = run_dir / "samples" / sample.sample_id
        result_path = sample_dir / "counting_result.json"
        if resume and result_path.is_file():
            result = CountingResult.model_validate_json(result_path.read_text(encoding="utf-8"))
            return SampleExecutionResult(result=result)
        image_path = sample.images[0].path
        atomic_write_json(sample_dir / "input.json", sample.model_dump(mode="json"))
        target = target_override or await self.target_parser.parse(sample.question, sample_id=sample.sample_id, artifact_dir=sample_dir)
        atomic_write_json(sample_dir / "target_spec.json", target.model_dump(mode="json"))
        orchestrator = self.orchestrator_factory(sample_dir)
        draft = await orchestrator.collect_points(read_normalized_image(image_path), sample_id=sample.sample_id, question=sample.question, target=target)
        atomic_write_json(sample_dir / "counting_draft.json", draft.model_dump(mode="json"))
        pairs = await self.seam_verifier.verify(orchestrator, image_path, draft) if draft.boundary_conflicts else []
        points, groups = finalize_representatives(draft.raw_global_points, pairs)
        unresolved = [item["conflict_id"] for item in draft.boundary_conflicts if (item["first_global_id"], item["second_global_id"]) not in pairs]
        status = "partial" if draft.failed_tiles and draft.succeeded_tiles else "failed" if draft.failed_tiles else "completed_with_warnings" if unresolved else "completed"
        result = CountingResult(sample_id=sample.sample_id, target=target.canonical_label, question=sample.question, source_width=draft.source_width, source_height=draft.source_height, tile_count=len(draft.processed_tiles), initial_tile_count=draft.initial_tile_count, leaf_tile_count=len(draft.processed_tiles), succeeded_tiles=draft.succeeded_tiles, failed_tiles=draft.failed_tiles, global_points=points, merged_groups=groups, unresolved_conflicts=unresolved, warnings=draft.warnings, final_count=sum(point.accepted for point in points), status=status)
        result = result.model_copy(update={"warnings": result.warnings + self.evidence_reviewer.review(result)})
        atomic_write_json(result_path, result.model_dump(mode="json"))
        record = await self.evaluation_service.evaluate_counting(sample, result, target=target, call_deepseek=evaluate, artifact_dir=sample_dir, budget=CallBudget(max_qwen_calls=0, max_deepseek_calls=1)) if evaluate else None
        if record is not None:
            atomic_write_json(sample_dir / "evaluation.json", record.model_dump(mode="json"))
            jsonl = run_dir / "evaluations.jsonl"
            with jsonl.open("a", encoding="utf-8", newline="\n") as file:
                file.write(record.model_dump_json() + "\n")
        if render:
            image = read_normalized_image(image_path)
            render_counting_overlay(image, result=result, tiles=build_core_halo_tiles(*image.size, core_size=orchestrator.counting.tile_core_size, halo_size=orchestrator.counting.halo_size, model_max_side=orchestrator.counting.model_max_side), output_path=sample_dir / "overlay.png")
        return SampleExecutionResult(result=result, evaluation=record)
