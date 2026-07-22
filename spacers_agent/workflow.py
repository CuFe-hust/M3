"""Runnable per-image and per-dataset multi-agent workflows.
可运行的单图和数据集多 Agent 工作流。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from spacers_agent.clients.base import RequestMeta, VisionLanguageClient, build_request_hash, image_to_data_url
from spacers_agent.clients.deepseek import DeepSeekJudgeClient
from spacers_agent.counting import PointCountingOrchestrator
from spacers_agent.dataset_adapters import DatasetAdapter
from spacers_agent.evaluation import (
    build_count_judge_payload,
    build_judge_request_hash,
    build_vqa_judge_payload,
    build_vqa_judge_request_hash,
    merge_count_evaluation,
    merge_vqa_evaluation,
    VQAAnswerJudgeResult,
)
from spacers_agent.imaging import read_normalized_image
from spacers_agent.routing import CallBudget, TaskRouter
from spacers_agent.routing import CountingExpert
from spacers_agent.settings import AppSettings
from spacers_agent.schemas import (
    CountTargetSpec,
    CountingResult,
    DatasetRunSummary,
    ExpertResult,
    GlobalPointObservation,
    IssueRecord,
    SampleRunStatus,
    UnifiedSample,
    VisualEvidence,
)
from spacers_agent.vqa_geometry import (
    apply_vrsbench_geometry,
    vrsbench_answer_vocabulary,
    vrsbench_count_target,
    vrsbench_question_subtype,
    vrsbench_vehicle_class,
)


class _CountProposalResult(BaseModel):
    """Compact whole-image count proposal matching the proven GeneralVQA contract.
    匹配已验证 GeneralVQA 契约的紧凑整图计数提议。"""

    model_config = ConfigDict(extra="forbid")

    expert: str
    answer: str
    boxes: list[list[float]] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    status: Literal["completed", "partial", "failed"] = "completed"


def atomic_write_json(path: Path, value: Any) -> None:
    """Atomically publish a JSON artifact after a complete temporary write. / 完整写入临时文件后原子发布 JSON 产物。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class CountTargetParser:
    """Parse a stable counting target using a structured Qwen request. / 使用结构化 Qwen 请求解析稳定的计数目标。"""

    def __init__(self, client: VisionLanguageClient, prompt: str, model: str) -> None:
        self.client = client
        self.prompt = prompt
        self.model = model

    async def parse(self, question: str, *, sample_id: str, artifact_dir: Path) -> CountTargetSpec:
        """Return a cached-schema target specification for one question. / 返回单个问题经缓存和 Schema 校验的目标规范。"""

        messages: list[dict[str, Any]] = [{"role": "system", "content": self.prompt}, {"role": "user", "content": question}]
        request_hash = build_request_hash(model=self.model, generation={"temperature": 0.0}, prompt_version="target-parse-v1", messages=messages, image_sha256=None)
        return await self.client.complete_json(
            messages=messages,
            response_model=CountTargetSpec,
            request_meta=RequestMeta(request_id=f"{sample_id}:target", request_hash=request_hash, prompt_version="target-parse-v1", sample_id=sample_id, artifact_dir=artifact_dir / "target_parse"),
        )


class VisualExpert:
    """Generic Qwen visual primitive for change, grounding, spatial and VQA. / 用于变化、定位、空间和问答的通用 Qwen 视觉原语。"""

    def __init__(
        self,
        client: VisionLanguageClient,
        prompt: str,
        model: str,
        name: str,
        prompt_version: str,
    ) -> None:
        self.client, self.prompt, self.model, self.name = client, prompt, model, name
        self.prompt_version = prompt_version

    def _prompt_for_sample(self, sample: UnifiedSample) -> tuple[str, str]:
        """Return the prompt asset and version selected for one sample.
        返回为单条样本选择的 Prompt 资源及版本。
        """

        return self.prompt, self.prompt_version

    async def run(self, sample: UnifiedSample, *, artifact_dir: Path) -> ExpertResult:
        """Use overview images as evidence without adding any detector. / 使用概览图作为证据且不引入检测器。"""

        content: list[dict[str, Any]] = []
        image_hashes: list[str] = []
        for image_ref in sample.images:
            data = image_ref.path.read_bytes()
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(data, "image/png")}})
            image_hashes.append(__import__("hashlib").sha256(data).hexdigest())
        user_payload: dict[str, Any] = {
            "question": sample.question,
            "dataset_question_type": sample.metadata.get("question_type"),
            "coordinate_frame": "normalized_0_999_top_left",
        }
        if sample.dataset == "VRSBench":
            subtype = vrsbench_question_subtype(
                sample.question,
                str(sample.metadata.get("question_type", "")),
            )
            answer_vocabulary = [] if subtype == "grid_position" else vrsbench_answer_vocabulary(subtype)
            user_payload.update(
                {
                    "semantic_subtype": subtype,
                    "answer_vocabulary": answer_vocabulary,
                }
            )
        content.append(
            {
                "type": "text",
                "text": json.dumps(user_payload, ensure_ascii=False),
            }
        )
        prompt, prompt_version = self._prompt_for_sample(sample)
        structured_prompt = (
            prompt
            + f"\n\nReturn valid JSON only. Set expert to {self.name!r}; put the concise final answer in answer, "
            "retain relevant labeled boxes or points in evidence_items, copy evidence boxes into boxes, "
            "use concise factual evidence strings, and set status to 'completed'."
        )
        messages: list[dict[str, Any]] = [{"role": "system", "content": structured_prompt}, {"role": "user", "content": content}]
        request_hash = build_request_hash(model=self.model, generation={"temperature": 0.0}, prompt_version=prompt_version, messages=messages, image_sha256="|".join(image_hashes))
        return await self.client.complete_json(
            messages=messages,
            response_model=ExpertResult,
            request_meta=RequestMeta(request_id=f"{sample.sample_id}:{self.name}", request_hash=request_hash, prompt_version=prompt_version, sample_id=sample.sample_id, artifact_dir=artifact_dir / self.name),
        )


# Preserve the original internal name while exposing the requested public contract.
# 保留原有内部名称，同时暴露所需的公开契约。
TargetParser = CountTargetParser


class ChangeExpert(VisualExpert):
    """Run the change visual primitive. / 运行变化视觉原语。"""

    def __init__(self, client: VisionLanguageClient, prompt: str, model: str) -> None:
        super().__init__(client, prompt, model, "change_expert", "change-expert-v1")


class GroundingExpert(VisualExpert):
    """Run the grounding visual primitive. / 运行定位视觉原语。"""

    def __init__(self, client: VisionLanguageClient, prompt: str, model: str) -> None:
        super().__init__(client, prompt, model, "grounding_expert", "general-vqa-v2")


class SpatialExpert(VisualExpert):
    """Run the spatial visual primitive. / 运行空间关系视觉原语。"""

    def __init__(
        self,
        client: VisionLanguageClient,
        prompt: str,
        model: str,
        review_prompt: str = "",
        grid_prompt: str = "",
        grid_review_prompt: str = "",
    ) -> None:
        super().__init__(client, prompt, model, "spatial_expert", "spatial-v4")
        self.review_prompt = review_prompt
        self.grid_prompt = grid_prompt
        self.grid_review_prompt = grid_review_prompt

    def _prompt_for_sample(self, sample: UnifiedSample) -> tuple[str, str]:
        """Use the grounded prompt only for grid-position questions.
        仅对九宫格位置问题使用实体定位 Prompt。
        """

        subtype = vrsbench_question_subtype(
            sample.question,
            str(sample.metadata.get("question_type", "")),
        )
        if subtype == "grid_position" and self.grid_prompt:
            return self.grid_prompt, "spatial-v5"
        return super()._prompt_for_sample(sample)

    def _review_prompt_for_sample(self, sample: UnifiedSample) -> tuple[str, str]:
        """Select the subtype-scoped independent review prompt.
        选择限定于语义子类型的独立复查 Prompt。
        """

        subtype = vrsbench_question_subtype(
            sample.question,
            str(sample.metadata.get("question_type", "")),
        )
        if subtype == "grid_position" and self.grid_review_prompt:
            return self.grid_review_prompt, "spatial-candidate-review-v3"
        return self.review_prompt, "spatial-candidate-review-v2"

    async def run(self, sample: UnifiedSample, *, artifact_dir: Path) -> ExpertResult:
        """Run one spatial pass and repair incomplete candidate enumeration once.
        执行一次空间推理，并对不完整候选枚举最多补全一次。
        """

        result = await super().run(sample, artifact_dir=artifact_dir)
        review_prompt, _ = self._review_prompt_for_sample(sample)
        if not review_prompt or not _needs_spatial_candidate_review(sample, result):
            return result
        try:
            review = await self._review_candidates(sample, artifact_dir)
        except Exception as error:
            geometry = dict(result.geometry)
            geometry.update(
                {
                    "candidate_review_used": True,
                    "candidate_review_added": 0,
                    "candidate_review_error": f"{type(error).__name__}: {error}",
                }
            )
            return result.model_copy(update={"geometry": geometry, "status": "partial"})
        first_evidence = result.evidence_items
        replaced_evidence = 0
        subtype = vrsbench_question_subtype(
            sample.question,
            str(sample.metadata.get("question_type", "")),
        )
        if subtype == "grid_position":
            first_evidence = [
                item
                for item in result.evidence_items
                if not (_matches_position_target(sample.question, item) and _is_corner_anchored_box(item))
            ]
            replaced_evidence = len(result.evidence_items) - len(first_evidence)
        review_evidence, labeled_review_boxes = _position_review_evidence(sample.question, subtype, review)
        merged = _merge_visual_evidence(first_evidence, review_evidence)
        geometry = dict(result.geometry)
        merged_quality = ["trusted_box" if item.box is not None else "trusted_point" for item in merged]
        geometry.update(
            {
                "candidate_review_used": True,
                "candidate_review_added": len(merged) - len(first_evidence),
                "candidate_review_replaced": replaced_evidence,
                "candidate_review_labeled_boxes": labeled_review_boxes,
                "candidate_review_geometry": review.geometry,
                "evidence_quality": merged_quality,
                "repair_severity": _maximum_repair_severity(
                    str(result.geometry.get("repair_severity", "none")),
                    str(review.geometry.get("repair_severity", "none")),
                ),
            }
        )
        reviewed_result = result.model_copy(update={"evidence_items": merged, "geometry": geometry})
        status = "partial" if _needs_spatial_candidate_review(sample, reviewed_result) else "completed"
        return result.model_copy(
            update={
                "boxes": [list(item.box) for item in merged if item.box is not None],
                "evidence_items": merged,
                "geometry": geometry,
                "status": status,
            }
        )

    async def _review_candidates(
        self,
        sample: UnifiedSample,
        artifact_dir: Path,
    ) -> ExpertResult:
        """Request a localization-only completeness pass over the same image.
        对同一图像请求一次仅定位的完整性复查。
        """

        content: list[dict[str, Any]] = []
        image_hashes: list[str] = []
        for image_ref in sample.images:
            data = image_ref.path.read_bytes()
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(data, "image/png")}})
            image_hashes.append(hashlib.sha256(data).hexdigest())
        subtype = vrsbench_question_subtype(
            sample.question,
            str(sample.metadata.get("question_type", "")),
        )
        answer_vocabulary = [] if subtype == "grid_position" else vrsbench_answer_vocabulary(subtype)
        content.append(
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "question": sample.question,
                        "dataset_question_type": sample.metadata.get("question_type"),
                        "semantic_subtype": subtype,
                        "answer_vocabulary": answer_vocabulary,
                        "coordinate_frame": "normalized_0_999_top_left",
                        "review_mode": "independent_candidate_enumeration",
                    },
                    ensure_ascii=False,
                ),
            }
        )
        review_prompt, review_prompt_version = self._review_prompt_for_sample(sample)
        system_prompt = (
            review_prompt
            + "\n\nReturn valid JSON only. Set expert to 'spatial_expert', keep answer concise, "
            "copy every evidence box into boxes, and set status to 'completed' only when enumeration is complete."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        request_hash = build_request_hash(
            model=self.model,
            generation={"temperature": 0.0},
            prompt_version=review_prompt_version,
            messages=messages,
            image_sha256="|".join(image_hashes),
        )
        return await self.client.complete_json(
            messages=messages,
            response_model=ExpertResult,
            request_meta=RequestMeta(
                request_id=f"{sample.sample_id}:spatial-candidate-review",
                request_hash=request_hash,
                prompt_version=review_prompt_version,
                sample_id=sample.sample_id,
                artifact_dir=artifact_dir / "spatial_expert_candidate_review",
            ),
        )


class GeneralVQAExpert(VisualExpert):
    """Run the general-VQA visual primitive. / 运行通用问答视觉原语。"""

    def __init__(self, client: VisionLanguageClient, prompt: str, model: str) -> None:
        super().__init__(client, prompt, model, "general_vqa_expert", "general-vqa-v2")


class WorkflowService:
    """Dispatch routed non-counting tasks to concrete experts. / 将已路由的非计数任务分派给具体专家。"""

    def __init__(self, client: VisionLanguageClient, prompts: dict[str, str], model: str) -> None:
        self.experts = {
            "change_expert": ChangeExpert(client, prompts["change"], model),
            "grounding_expert": GroundingExpert(client, prompts["general"], model),
            "spatial_expert": SpatialExpert(
                client,
                prompts["spatial"],
                model,
                prompts.get("spatial_review", ""),
                prompts.get("spatial_grid", ""),
                prompts.get("spatial_grid_review", ""),
            ),
            "general_vqa_expert": GeneralVQAExpert(client, prompts["general"], model),
        }

    async def execute(self, expert_name: str, sample: UnifiedSample, artifact_dir: Path) -> ExpertResult:
        """Execute one named expert or fail with a visible code. / 执行一个具名专家，或以可见代码失败。"""

        try:
            expert = self.experts[expert_name]
        except KeyError as error:
            raise ValueError(f"UNSUPPORTED_EXPERT:{expert_name}") from error
        return await expert.run(sample, artifact_dir=artifact_dir)


class DatasetRunner:
    """Sequential sample runner with durable status and resume semantics. / 具有持久状态和恢复语义的顺序样本运行器。"""

    def __init__(
        self,
        settings: AppSettings,
        adapter: DatasetAdapter,
        *,
        run_dir: Path,
        client: VisionLanguageClient,
        prompts: dict[str, str],
        judge_client: DeepSeekJudgeClient | None = None,
        judge_policy: str = "none",
    ) -> None:
        self.settings, self.adapter, self.run_dir, self.client, self.prompts = settings, adapter, run_dir, client, prompts
        self.judge_client = judge_client
        self.judge_policy = judge_policy

    async def run(self, *, split: str, task: str, resume: bool = False, limit: int | None = None, shard_index: int = 0, shard_count: int = 1, start_index: int = 0, sample_ids: set[str] | None = None, fail_fast: bool = False, sample_concurrency: int = 1) -> DatasetRunSummary:
        """Run selected samples sequentially and keep every failure visible. / 顺序运行选中样本并保持每个失败可见。"""

        if shard_count < 1 or not 0 <= shard_index < shard_count:
            raise ValueError("invalid shard selection")
        if sample_concurrency < 1:
            raise ValueError("sample_concurrency must be positive")
        probe = self.adapter.probe(self.settings.paths.dataset_root)
        manifest_path = self.run_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["dataset_probe"] = {
                "dataset": probe.dataset,
                "version": probe.version,
                "sample_file": str(probe.sample_file),
                "observed_fields": list(probe.observed_fields),
                "sample_count": probe.sample_count,
            }
            atomic_write_json(manifest_path, manifest)
        statuses: list[SampleRunStatus] = []
        pending: dict[asyncio.Task[SampleRunStatus], UnifiedSample] = {}

        async def collect_one() -> bool:
            """Persist one completed sample outcome before scheduling more work. / 在调度更多工作前持久化一个已完成样本结果。"""

            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            stop = False
            for future in done:
                sample = pending.pop(future)
                status = await future
                statuses.append(status)
                _append_jsonl(self.run_dir / "predictions.jsonl", {"sample_id": sample.sample_id, "task": sample.task, "status": status.state, "result_path": str(status.result_path) if status.result_path else None})
                stop = stop or (fail_fast and status.state == "failed")
            return stop

        selected = 0
        for index, sample in enumerate(self.adapter.iter_samples(self.settings.paths.dataset_root, split, task)):
            if index < start_index or _shard_for_sample(sample.sample_id, shard_count) != shard_index:
                continue
            if sample_ids is not None and sample.sample_id not in sample_ids:
                continue
            if limit is not None and selected >= limit:
                break
            selected += 1
            status_path = self.run_dir / "samples" / sample.sample_id / "status.json"
            if resume and status_path.is_file():
                previous = SampleRunStatus.model_validate_json(status_path.read_text(encoding="utf-8"))
                if previous.state == "succeeded":
                    if await self._resume_vqa_judge(sample, status_path.parent):
                        statuses.append(previous)
                        continue
                    statuses.append(previous.model_copy(update={"state": "skipped"}))
                    continue
            pending[asyncio.create_task(self._run_one(sample, status_path))] = sample
            if len(pending) >= sample_concurrency and await collect_one():
                break
        while pending:
            if await collect_one():
                for future in pending:
                    future.cancel()
                break
        summary = DatasetRunSummary(
            run_id=self.run_dir.name, dataset=getattr(self.adapter, "name"), split=split, task=task, total=len(statuses),
            succeeded=sum(item.state == "succeeded" for item in statuses), partial=sum(item.state == "partial" for item in statuses),
            failed=sum(item.state == "failed" for item in statuses), skipped=sum(item.state == "skipped" for item in statuses),
        )
        atomic_write_json(self.run_dir / "dataset_summary.json", summary.model_dump(mode="json"))
        return summary

    async def _run_one(self, sample: UnifiedSample, status_path: Path) -> SampleRunStatus:
        """Execute one routed sample and translate exceptions to visible status. / 执行一个路由样本并将异常转换为可见状态。"""

        sample_dir = status_path.parent
        atomic_write_json(sample_dir / "sample.json", sample.model_dump(mode="json"))
        started = _status(sample, "running")
        atomic_write_json(status_path, started.model_dump(mode="json"))
        try:
            high_resolution = any(
                (ref.width or 0) * (ref.height or 0) > self.settings.counting.max_pixels_without_tiling
                for ref in sample.images
            )
            question_type = str(sample.metadata.get("question_type", ""))
            if sample.dataset == "VRSBench" and sample.task == "general_vqa":
                decision = TaskRouter().route_vrsbench_vqa(
                    question_type,
                    high_resolution=high_resolution,
                )
            else:
                decision = TaskRouter().route_known(sample.task, high_resolution=high_resolution)
            atomic_write_json(sample_dir / "routing_decision.json", decision.model_dump(mode="json"))
            if sample.task in {"counting", "fine_grained_counting"}:
                target = await TargetParser(self.client, self.prompts["target"], self.settings.models.qwen.model).parse(sample.question, sample_id=sample.sample_id, artifact_dir=sample_dir)
                image = read_normalized_image(sample.images[0].path)
                result = await PointCountingOrchestrator(self.client, counting=self.settings.counting, qwen=self.settings.models.qwen, system_prompt=self.prompts["count"], run_dir=sample_dir, seam_prompt=self.prompts["seam"]).count_image(image, sample_id=sample.sample_id, question=sample.question, target=target)
                atomic_write_json(sample_dir / "counting_result.json", result.model_dump(mode="json"))
                from spacers_agent.evaluation import merge_count_evaluation
                atomic_write_json(sample_dir / "evaluation_record.json", merge_count_evaluation(sample_id=sample.sample_id, counting=result, ground_truth=sample.ground_truth).model_dump(mode="json"))
                state = "succeeded" if result.status in {"completed", "completed_with_warnings"} else "partial"
                final = _status(sample, state, result_path=sample_dir / "counting_result.json")
            else:
                expert_name = decision.experts[0].name
                inference_started = time.perf_counter()
                if expert_name == "counting_expert":
                    expert_class_name = "CountingExpert"
                    result = await self._run_vqa_counting(sample, sample_dir)
                    prompt_version = str(result.geometry.get("prompt_version", "vrsbench-count-hybrid-v1"))
                    route = (
                        f"{type(self.adapter).__name__} -> TaskRouter.route_vrsbench_vqa -> "
                        "vrsbench_count_target -> CountingExpert.run -> GeneralVQA count proposal -> "
                        f"{type(self.client).__name__}.complete_json -> accepted points"
                    )
                    if result.geometry.get("localization_used"):
                        route += f" -> CountEvidenceLocalizer -> {type(self.client).__name__}.complete_json"
                else:
                    service = WorkflowService(self.client, self.prompts, self.settings.models.qwen.model)
                    expert = service.experts[expert_name]
                    expert_class_name = type(expert).__name__
                    prompt_version = expert._prompt_for_sample(sample)[1]
                    result = await service.execute(expert_name, sample, sample_dir)
                    if sample.dataset == "VRSBench" and sample.task == "general_vqa":
                        result = apply_vrsbench_geometry(sample.question, question_type, result)
                    route_entry = (
                        "TaskRouter.route_vrsbench_vqa"
                        if sample.dataset == "VRSBench" and sample.task == "general_vqa"
                        else "TaskRouter.route_known"
                    )
                    route = (
                        f"{type(self.adapter).__name__} -> {route_entry} -> "
                        f"{expert_class_name}.run -> {type(self.client).__name__}.complete_json"
                    )
                    if result.geometry.get("candidate_review_used"):
                        route += " -> SpatialExpert.candidate_review"
                inference_seconds = round(time.perf_counter() - inference_started, 6)
                atomic_write_json(sample_dir / "expert_result.json", result.model_dump(mode="json"))
                judge_status = "not_requested"
                if sample.task == "general_vqa":
                    evaluation = await self._evaluate_vqa(sample, result, sample_dir)
                    judge_status = evaluation.judge_status
                if sample.task == "general_vqa" and judge_status != "not_requested":
                    route += " -> DeepSeekJudgeClient.judge"
                atomic_write_json(
                    sample_dir / "agent_trace.json",
                    {
                        "agent_class": f"spacers_agent.workflow.{expert_class_name}",
                        "entrypoint": "run",
                        "route": route,
                        "router_used": True,
                        "task_type": sample.task,
                        "qwen_backend": self.settings.models.qwen.backend,
                        "judge_status": judge_status,
                        "inference_seconds": inference_seconds,
                        "execution_task": decision.task,
                        "official_question_type": question_type or None,
                        "prompt_version": prompt_version,
                        "geometry": result.geometry,
                    },
                )
                state = {"completed": "succeeded", "partial": "partial", "failed": "failed"}[result.status]
                final = _status(sample, state, result_path=sample_dir / "expert_result.json")
        except Exception as error:
            final = _status(sample, "failed", error_code=type(error).__name__, error_message=str(error))
        atomic_write_json(status_path, final.model_dump(mode="json"))
        return final

    async def _run_vqa_counting(self, sample: UnifiedSample, sample_dir: Path) -> ExpertResult:
        """Run VRSBench quantity VQA through proposal-grounded accepted points.
        通过数量提议及其定位证据运行 VRSBench 接受点计数。
        """

        target = vrsbench_count_target(sample.question)
        image = read_normalized_image(sample.images[0].path)
        proposal, recovery = await self._run_vqa_count_proposal(sample, sample_dir)
        proposal_count = _parse_count_answer(proposal.answer)
        issues: list[IssueRecord] = []
        if recovery is not None:
            issues.append(
                IssueRecord(code="COUNT_PROPOSAL_HEADER_RECOVERED", message=recovery)
            )
        proposal_evidence = _box_evidence(
            proposal.boxes,
            target.canonical_label,
            sample.images[0].image_id,
        )
        points, supporting_boxes, dropped = _accepted_count_evidence(
            proposal_evidence,
            target.canonical_label,
            sample.images[0].image_id,
        )
        localization_used = proposal_count == 0 or len(points) != proposal_count
        localizer_answer: int | None = None
        if localization_used:
            issues.append(
                IssueRecord(
                    code="COUNT_PROPOSAL_EVIDENCE_MISMATCH",
                    message=f"proposal={proposal_count}, proposal_boxes={len(points)}",
                )
            )
            localized = await self._run_vqa_count_localizer(
                sample,
                sample_dir,
                target,
                proposal_count,
            )
            try:
                localizer_answer = _parse_count_answer(localized.answer)
            except ValueError:
                localizer_answer = None
            localized_evidence = localized.evidence_items or _box_evidence(
                localized.boxes,
                target.canonical_label,
                sample.images[0].image_id,
            )
            points, supporting_boxes, dropped = _accepted_count_evidence(
                localized_evidence,
                target.canonical_label,
                sample.images[0].image_id,
            )
        if dropped:
            issues.append(
                IssueRecord(
                    code="COUNT_BORDER_OR_DUPLICATE_EVIDENCE_DROPPED",
                    message=f"Dropped {dropped} duplicate or tiny border-fragment observations.",
                )
            )
        complete = len(points) == proposal_count and (
            localizer_answer is None or localizer_answer == len(points) or dropped > 0
        )
        if not complete:
            issues.append(
                IssueRecord(
                    code="COUNT_LOCALIZATION_EVIDENCE_MISMATCH",
                    message=(
                        f"proposal={proposal_count}, localizer={localizer_answer}, "
                        f"accepted_points={len(points)}"
                    ),
                )
            )
        global_points = [
            _global_count_point(
                sample.sample_id,
                target.canonical_label,
                item,
                index,
                image.width,
                image.height,
            )
            for index, item in enumerate(points, start=1)
        ]
        counting_status: Literal["completed", "completed_with_warnings", "partial"] = (
            "partial" if not complete else "completed_with_warnings" if issues else "completed"
        )
        counting = CountingResult(
            sample_id=sample.sample_id,
            target=target.canonical_label,
            question=sample.question,
            source_width=image.width,
            source_height=image.height,
            tile_count=1,
            initial_tile_count=1,
            leaf_tile_count=1,
            succeeded_tiles=["whole_image_overview"],
            failed_tiles=[],
            global_points=global_points,
            merged_groups=[],
            unresolved_conflicts=[],
            warnings=issues,
            final_count=len(global_points),
            status=counting_status,
        )
        atomic_write_json(sample_dir / "counting_result.json", counting.model_dump(mode="json"))
        geometry = {
            "version": "accepted-point-count-v3",
            "prompt_version": "vrsbench-count-hybrid-v1",
            "coordinate_frame": "normalized_0_999_top_left",
            "rule": "final_count_equals_accepted_points",
            "accepted_point_count": len(points),
            "final_count": counting.final_count,
            "proposal_count": proposal_count,
            "proposal_status": proposal.status,
            "proposal_recovery": recovery,
            "localization_used": localization_used,
            "localizer_answer": localizer_answer,
            "supporting_box_count": len(supporting_boxes),
            "pipeline": "general_vqa_v1_proposal_then_grounded_localization",
            "counting_status": counting.status,
            "warnings": [item.model_dump(mode="json") for item in counting.warnings],
        }
        return ExpertResult(
            expert="counting_expert",
            answer=(
                str(counting.final_count)
                if complete
                else f"Confirmed {counting.final_count} localized instances; the count is incomplete."
            ),
            boxes=[[float(value) for value in box] for box in supporting_boxes],
            evidence=[f"Accepted point {index + 1}: {item.point}" for index, item in enumerate(points)],
            evidence_items=points,
            geometry=geometry,
            status="completed" if complete else "partial",
        )

    async def _run_vqa_count_proposal(
        self,
        sample: UnifiedSample,
        sample_dir: Path,
    ) -> tuple[_CountProposalResult, str | None]:
        """Request an independent count and recover only a complete integer header.
        请求独立数量，并且只恢复完整的整数头部。
        """

        image_bytes = sample.images[0].path.read_bytes()
        system_prompt = self.prompts["count_proposal"] + (
            "\n\nReturn valid JSON only. Set expert to 'general_vqa_expert'; put the concise "
            "final answer in answer, use empty boxes/evidence when they are not needed, and set "
            "status to 'completed'."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image_bytes, "image/png")},
                    },
                    {"type": "text", "text": sample.question},
                ],
            },
        ]
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        request_hash = build_request_hash(
            model=self.settings.models.qwen.model,
            generation={"temperature": 0.0, "max_tokens": self.settings.models.qwen.max_tokens},
            prompt_version="general-vqa-v1-count-proposal",
            messages=messages,
            image_sha256=image_hash,
        )
        artifact_dir = sample_dir / "counting_expert" / "count_proposal"
        try:
            proposal = await self.client.complete_json(
                messages=messages,
                response_model=_CountProposalResult,
                request_meta=RequestMeta(
                    request_id=f"{sample.sample_id}:count-proposal",
                    request_hash=request_hash,
                    prompt_version="general-vqa-v1-count-proposal",
                    sample_id=sample.sample_id,
                    image_sha256=image_hash,
                    artifact_dir=artifact_dir,
                ),
            )
            return proposal, None
        except Exception:
            raw_path = artifact_dir / "raw_response.txt"
            recovered = _recover_count_proposal_header(
                raw_path.read_text(encoding="utf-8") if raw_path.is_file() else ""
            )
            if recovered is None:
                raise
            return (
                _CountProposalResult(
                    expert="general_vqa_expert",
                    answer=str(recovered),
                    boxes=[],
                    evidence=[],
                    status="partial",
                ),
                "Recovered a complete integer answer header; malformed geometry was discarded.",
            )

    async def _run_vqa_count_localizer(
        self,
        sample: UnifiedSample,
        sample_dir: Path,
        target: CountTargetSpec,
        proposal_count: int,
    ) -> ExpertResult:
        """Independently localize a count proposal with tight whole-image boxes.
        使用整图紧框独立定位数量提议。
        """

        image_bytes = sample.images[0].path.read_bytes()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.prompts["count_localize"]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image_bytes, "image/png")},
                    },
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "question": sample.question,
                                "target_spec": target.model_dump(mode="json"),
                                "independent_count_proposal": proposal_count,
                                "image_scope": "complete_image",
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
            },
        ]
        request_hash = build_request_hash(
            model=self.settings.models.qwen.model,
            generation={"temperature": 0.0, "max_tokens": self.settings.models.qwen.max_tokens},
            prompt_version="count-localize-v1",
            messages=messages,
            image_sha256=image_hash,
            target_spec=target.model_dump(mode="json"),
        )
        return await self.client.complete_json(
            messages=messages,
            response_model=ExpertResult,
            request_meta=RequestMeta(
                request_id=f"{sample.sample_id}:count-localizer",
                request_hash=request_hash,
                prompt_version="count-localize-v1",
                sample_id=sample.sample_id,
                image_sha256=image_hash,
                artifact_dir=sample_dir / "counting_expert" / "count_localizer",
            ),
        )

    async def _evaluate_vqa(
        self,
        sample: UnifiedSample,
        result: ExpertResult,
        sample_dir: Path,
    ) -> Any:
        """Compare answers deterministically and optionally call the text-only judge.
        确定性比较答案，并可选调用纯文本审核器。
        """

        references = sample.ground_truth.answers if sample.ground_truth is not None else []
        initial = merge_vqa_evaluation(
            sample_id=sample.sample_id,
            question=sample.question,
            reference_answers=references,
            candidate_answer=result.answer,
        )
        should_judge = self.judge_client is not None and (
            self.judge_policy == "all" or (self.judge_policy == "errors-only" and not initial.exact_match)
        )
        if not should_judge:
            atomic_write_json(sample_dir / "vqa_evaluation.json", initial.model_dump(mode="json"))
            return initial
        payload = build_vqa_judge_payload(
            question=sample.question,
            reference_answers=references,
            candidate_answer=result.answer,
        )
        try:
            verdict = await self.judge_client.judge_json(
                payload,
                response_model=VQAAnswerJudgeResult,
                request_meta=RequestMeta(
                    request_id=f"{sample.sample_id}:deepseek-vqa",
                    request_hash=build_vqa_judge_request_hash(
                        model=self.settings.models.deepseek.model,
                        prompt_text=self.judge_client.judge_prompt,
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
                candidate_answer=result.answer,
                judge_parsed=verdict,
            )
        except Exception as error:
            evaluation = merge_vqa_evaluation(
                sample_id=sample.sample_id,
                question=sample.question,
                reference_answers=references,
                candidate_answer=result.answer,
                judge_error=f"{type(error).__name__}: {error}",
            )
        atomic_write_json(sample_dir / "vqa_evaluation.json", evaluation.model_dump(mode="json"))
        return evaluation

    async def _resume_vqa_judge(self, sample: UnifiedSample, sample_dir: Path) -> bool:
        """Retry only a missing or failed VQA Judge while reusing saved Qwen output.
        仅重试缺失或失败的 VQA 审核，并复用已保存的 Qwen 输出。
        """

        if sample.task != "general_vqa" or self.judge_client is None or self.judge_policy == "none":
            return False
        evaluation_path = sample_dir / "vqa_evaluation.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8")) if evaluation_path.is_file() else {}
        if evaluation.get("judge_status") == "succeeded":
            return False
        result_path = sample_dir / "expert_result.json"
        if not result_path.is_file():
            return False
        result = ExpertResult.model_validate_json(result_path.read_text(encoding="utf-8"))
        refreshed = await self._evaluate_vqa(sample, result, sample_dir)
        trace_path = sample_dir / "agent_trace.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.is_file() else {}
        trace["judge_status"] = refreshed.judge_status
        route = str(trace.get("route", ""))
        if "DeepSeekJudgeClient.judge" not in route:
            trace["route"] = route + " -> DeepSeekJudgeClient.judge"
        atomic_write_json(trace_path, trace)
        return True


def _parse_count_answer(value: str) -> int:
    """Parse one non-negative integer without consulting reference answers.
    在不读取参考答案的前提下解析一个非负整数。
    """

    normalized = value.strip()
    if re.fullmatch(r"\d+", normalized) is None:
        raise ValueError(f"Count proposal is not a non-negative integer: {value!r}")
    return int(normalized)


def _recover_count_proposal_header(raw_response: str) -> int | None:
    """Recover only a syntactically complete integer answer before malformed geometry.
    仅恢复畸形几何之前语法完整的整数答案。
    """

    match = re.search(r'"answer"\s*:\s*"(\d+)"', raw_response)
    return int(match.group(1)) if match is not None else None


def _box_evidence(
    boxes: list[list[float]],
    target: str,
    image_id: str,
) -> list[VisualEvidence]:
    """Normalize legacy proposal boxes without inventing missing geometry.
    规范化旧版提议框，但不虚构缺失几何。
    """

    evidence: list[VisualEvidence] = []
    for raw_box in boxes:
        if len(raw_box) != 4 or any(not isinstance(value, (int, float)) for value in raw_box):
            continue
        x1, y1, x2, y2 = [max(0, min(999, round(value))) for value in raw_box]
        box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        if box[0] >= box[2] or box[1] >= box[3]:
            continue
        evidence.append(
            VisualEvidence(
                label=target,
                box=box,
                confidence=0.9,
                image_id=image_id,
            )
        )
    return evidence


def _accepted_count_evidence(
    evidence: list[VisualEvidence],
    target: str,
    image_id: str,
) -> tuple[list[VisualEvidence], list[list[int]], int]:
    """Deduplicate evidence, reject tiny border fragments, and emit accepted centres.
    去重证据、拒绝微小边界残片并输出接受中心点。
    """

    merged = _merge_count_evidence(evidence)
    raw_points: list[VisualEvidence] = []
    boxes: list[list[int]] = []
    dropped = len(evidence) - len(merged)
    for item in merged:
        if item.box is not None:
            if _is_tiny_border_fragment(item.box):
                dropped += 1
                continue
            boxes.append(list(item.box))
            point = [
                round((item.box[0] + item.box[2]) / 2),
                round((item.box[1] + item.box[3]) / 2),
            ]
        elif item.point is not None:
            point = list(item.point)
        else:
            dropped += 1
            continue
        raw_points.append(
            VisualEvidence(
                label=target,
                point=point,
                confidence=item.confidence,
                image_id=image_id,
            )
        )
    points = _merge_count_evidence(raw_points)
    dropped += len(raw_points) - len(points)
    return points, boxes, max(0, dropped)


def _merge_count_evidence(evidence: list[VisualEvidence]) -> list[VisualEvidence]:
    """Deduplicate count evidence without collapsing adjacent coarse vehicle boxes.
    对计数证据去重，同时避免合并相邻车辆的粗略框。
    """

    merged: list[VisualEvidence] = []
    for candidate in evidence:
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(merged)
                if _same_count_observation(candidate, existing)
            ),
            None,
        )
        if duplicate_index is None:
            merged.append(candidate)
        elif _prefer_candidate_evidence(candidate, merged[duplicate_index]):
            merged[duplicate_index] = candidate
    return merged


def _same_count_observation(first: VisualEvidence, second: VisualEvidence) -> bool:
    """Match only near-identical count evidence because adjacent vehicles may overlap.
    计数时只匹配近乎相同的证据，因为相邻车辆的粗略框可能重叠。
    """

    if vrsbench_vehicle_class(first.label) != vrsbench_vehicle_class(second.label):
        return False
    if first.box is not None and second.box is not None:
        return _box_iou(first.box, second.box) >= 0.9
    if first.point is not None and second.point is not None:
        return _point_distance(first.point, second.point) <= 12
    box_item, point_item = (first, second) if first.box is not None else (second, first)
    if box_item.box is None or point_item.point is None:
        return False
    box_center = [
        round((box_item.box[0] + box_item.box[2]) / 2),
        round((box_item.box[1] + box_item.box[3]) / 2),
    ]
    return _point_distance(box_center, point_item.point) <= 12


def _is_tiny_border_fragment(box: list[int]) -> bool:
    """Identify a border-clipped fragment whose visible centre remains at the edge.
    识别可见中心仍贴近边缘的边界截断残片。
    """

    center_x = (box[0] + box[2]) / 2
    center_y = (box[1] + box[3]) / 2
    return (
        (box[0] == 0 and center_x < 25)
        or (box[1] == 0 and center_y < 25)
        or (box[2] == 999 and center_x > 974)
        or (box[3] == 999 and center_y > 974)
    )


def _global_count_point(
    sample_id: str,
    target: str,
    evidence: VisualEvidence,
    index: int,
    width: int,
    height: int,
) -> GlobalPointObservation:
    """Convert one normalized accepted centre into durable whole-image provenance.
    将一个归一化接受中心转换为持久化的整图来源记录。
    """

    if evidence.point is None:
        raise ValueError("Accepted count evidence requires a point")
    x, y = evidence.point
    local_id = f"p{index:03d}"
    return GlobalPointObservation(
        global_id=f"{sample_id}:whole_image_overview:{local_id}",
        target=target,
        source_tile_id="whole_image_overview",
        local_id=local_id,
        local_x_norm=x,
        local_y_norm=y,
        local_radius_norm=0,
        global_x_px=round(x * (width - 1) / 999),
        global_y_px=round(y * (height - 1) / 999),
        global_x_norm=x,
        global_y_norm=y,
        radius_px=0.0,
        confidence=evidence.confidence,
        ownership_valid=True,
        near_core_boundary=False,
        accepted=True,
        short_evidence="whole-image localized instance centre",
    )


def _needs_spatial_candidate_review(sample: UnifiedSample, result: ExpertResult) -> bool:
    """Detect spatial questions whose instance evidence is not yet enumerable.
    检测实例证据尚不足以枚举的空间问题。
    """

    if sample.dataset != "VRSBench":
        return False
    subtype = vrsbench_question_subtype(
        sample.question,
        str(sample.metadata.get("question_type", "")),
    )
    if subtype not in {"extreme_category", "extreme_existence", "grid_position", "arrangement", "proximity"}:
        return False
    vehicles = [
        item
        for item in result.evidence_items
        if item.box is not None and vrsbench_vehicle_class(item.label) in {"small-vehicle", "large-vehicle"}
    ]
    if subtype == "grid_position":
        targets = [item for item in vehicles if _matches_position_target(sample.question, item)]
        if result.geometry.get("candidate_review_used"):
            return not targets
        return len(targets) != 1 or _is_corner_anchored_box(targets[0])
    if not result.geometry.get("candidate_review_used"):
        return True
    if subtype in {"extreme_category", "arrangement"}:
        return len(vehicles) < 2
    if subtype == "extreme_existence":
        return not vehicles
    return len(result.evidence_items) < 2


def _matches_position_target(question: str, item: VisualEvidence) -> bool:
    """Match evidence to the vehicle class named by a position question.
    将证据与位置问题指定的车辆类别进行匹配。
    """

    desired = _position_target_label(question)
    return desired is None or vrsbench_vehicle_class(item.label) == desired


def _position_target_label(question: str) -> str | None:
    """Return the explicit vehicle class named by a position question.
    返回位置问题中明确指定的车辆类别。
    """

    lowered = question.casefold()
    if "large vehicle" in lowered:
        return "large-vehicle"
    if "small vehicle" in lowered:
        return "small-vehicle"
    return None


def _position_review_evidence(
    question: str,
    subtype: str,
    review: ExpertResult,
) -> tuple[list[VisualEvidence], int]:
    """Label model-provided review boxes from an explicit singular target class.
    使用明确的单数目标类别标注模型已返回的复查框。
    """

    evidence = list(review.evidence_items)
    target_label = _position_target_label(question)
    if subtype != "grid_position" or target_label is None:
        return evidence, 0
    if any(_matches_position_target(question, item) and item.box is not None for item in evidence):
        return evidence, 0
    labeled = [
        VisualEvidence(
            label=target_label,
            box=[int(round(value)) for value in box],
            confidence=0.0,
        )
        for box in review.boxes
    ]
    return evidence + labeled, len(labeled)


def _is_corner_anchored_box(item: VisualEvidence, *, tolerance: int = 5) -> bool:
    """Flag boxes anchored to two image borders as likely answer-region placeholders.
    将同时贴住两条图像边界的框标记为可能的答案区域占位框。
    """

    if item.box is None:
        return False
    left, top, right, bottom = item.box
    touches_horizontal_border = left <= tolerance or right >= 999 - tolerance
    touches_vertical_border = top <= tolerance or bottom >= 999 - tolerance
    return touches_horizontal_border and touches_vertical_border


def _merge_visual_evidence(
    first: list[VisualEvidence],
    second: list[VisualEvidence],
) -> list[VisualEvidence]:
    """Merge two evidence passes while suppressing strongly overlapping duplicates.
    合并两轮视觉证据，并去除高度重叠的重复项。
    """

    merged = list(first)
    for candidate in second:
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(merged)
                if _same_visual_observation(candidate, existing)
            ),
            None,
        )
        if duplicate_index is None:
            merged.append(candidate)
        elif _prefer_candidate_evidence(candidate, merged[duplicate_index]):
            merged[duplicate_index] = candidate
    return merged


def _same_visual_observation(first: VisualEvidence, second: VisualEvidence) -> bool:
    """Match repeated boxes or points without merging distinct nearby vehicles.
    匹配重复框或点，同时避免合并相邻但不同的车辆。
    """

    if vrsbench_vehicle_class(first.label) != vrsbench_vehicle_class(second.label):
        return False
    if first.box is not None and second.box is not None:
        return _box_iou(first.box, second.box) >= 0.7
    if first.point is not None and second.point is not None:
        return _point_distance(first.point, second.point) <= 12
    box_item, point_item = (first, second) if first.box is not None else (second, first)
    if box_item.box is None or point_item.point is None:
        return False
    x, y = point_item.point
    return box_item.box[0] <= x <= box_item.box[2] and box_item.box[1] <= y <= box_item.box[3]


def _prefer_candidate_evidence(candidate: VisualEvidence, existing: VisualEvidence) -> bool:
    """Prefer a real box over a point, then retain the higher-confidence duplicate.
    重复证据中优先保留真实框，其次保留置信度更高的观测。
    """

    if candidate.box is not None and existing.box is None:
        return True
    if candidate.box is None and existing.box is not None:
        return False
    return candidate.confidence > existing.confidence


def _point_distance(first: list[int], second: list[int]) -> float:
    """Return Euclidean distance between normalized evidence points.
    返回归一化证据点之间的欧氏距离。
    """

    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def _maximum_repair_severity(first: str, second: str) -> str:
    """Retain the highest evidence-repair severity across independent passes.
    在独立复查轮次之间保留最高的证据修复严重度。
    """

    rank = {"none": 0, "low": 1, "high": 2}
    return max((first, second), key=lambda value: rank.get(value, 2))


def _box_iou(first: list[int], second: list[int]) -> float:
    """Return intersection over union for normalized axis-aligned boxes.
    返回归一化轴对齐框的交并比。
    """

    intersection_width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _status(sample: UnifiedSample, state: str, *, error_code: str | None = None, error_message: str | None = None, result_path: Path | None = None) -> SampleRunStatus:
    """Create a timestamped status with no traceback payload. / 创建不含 traceback 载荷的带时间戳状态。"""

    return SampleRunStatus(sample_id=sample.sample_id, task=sample.task, state=state, error_code=error_code, error_message=error_message, result_path=result_path, updated_at=datetime.now(timezone.utc).isoformat())  # type: ignore[arg-type]


def _shard_for_sample(sample_id: str, shard_count: int) -> int:
    """Map a stable sample ID to a stable shard. / 将稳定样本 ID 映射到稳定分片。"""

    return int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest(), 16) % shard_count


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    """Append one completed sample record after its atomic status write. / 在原子状态写入后追加一条完成样本记录。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
