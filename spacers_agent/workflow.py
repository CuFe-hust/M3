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
from typing import Any

from PIL import Image

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
from spacers_agent.schemas import (
    CountTargetSpec,
    CountingResult,
    DatasetRunSummary,
    ExpertResult,
    SampleRunStatus,
    UnifiedSample,
    VisualEvidence,
)
from spacers_agent.settings import AppSettings
from spacers_agent.vqa_geometry import (
    apply_vrsbench_geometry,
    vrsbench_count_target,
    vrsbench_vehicle_class,
)


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

    async def run(self, sample: UnifiedSample, *, artifact_dir: Path) -> ExpertResult:
        """Use overview images as evidence without adding any detector. / 使用概览图作为证据且不引入检测器。"""

        content: list[dict[str, Any]] = []
        image_hashes: list[str] = []
        for image_ref in sample.images:
            data = image_ref.path.read_bytes()
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(data, "image/png")}})
            image_hashes.append(__import__("hashlib").sha256(data).hexdigest())
        content.append(
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "question": sample.question,
                        "dataset_question_type": sample.metadata.get("question_type"),
                        "coordinate_frame": "normalized_0_999_top_left",
                    },
                    ensure_ascii=False,
                ),
            }
        )
        structured_prompt = (
            self.prompt
            + f"\n\nReturn valid JSON only. Set expert to {self.name!r}; put the concise final answer in answer, "
            "retain relevant labeled boxes or points in evidence_items, copy evidence boxes into boxes, "
            "use concise factual evidence strings, and set status to 'completed'."
        )
        messages: list[dict[str, Any]] = [{"role": "system", "content": structured_prompt}, {"role": "user", "content": content}]
        request_hash = build_request_hash(model=self.model, generation={"temperature": 0.0}, prompt_version=self.prompt_version, messages=messages, image_sha256="|".join(image_hashes))
        return await self.client.complete_json(
            messages=messages,
            response_model=ExpertResult,
            request_meta=RequestMeta(request_id=f"{sample.sample_id}:{self.name}", request_hash=request_hash, prompt_version=self.prompt_version, sample_id=sample.sample_id, artifact_dir=artifact_dir / self.name),
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
    ) -> None:
        super().__init__(client, prompt, model, "spatial_expert", "spatial-v3")
        self.review_prompt = review_prompt

    async def run(self, sample: UnifiedSample, *, artifact_dir: Path) -> ExpertResult:
        """Run one spatial pass and repair incomplete candidate enumeration once.
        执行一次空间推理，并对不完整候选枚举最多补全一次。
        """

        result = await super().run(sample, artifact_dir=artifact_dir)
        if not self.review_prompt or not _needs_spatial_candidate_review(sample, result):
            return result
        try:
            review = await self._review_candidates(sample, result, artifact_dir)
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
        merged = _merge_visual_evidence(result.evidence_items, review.evidence_items)
        geometry = dict(result.geometry)
        geometry.update(
            {
                "candidate_review_used": True,
                "candidate_review_added": len(merged) - len(result.evidence_items),
            }
        )
        reviewed_result = result.model_copy(update={"evidence_items": merged})
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
        first: ExpertResult,
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
        content.append(
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "question": sample.question,
                        "dataset_question_type": sample.metadata.get("question_type"),
                        "first_pass_evidence": [item.model_dump(mode="json") for item in first.evidence_items],
                        "coordinate_frame": "normalized_0_999_top_left",
                    },
                    ensure_ascii=False,
                ),
            }
        )
        system_prompt = (
            self.review_prompt
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
            prompt_version="spatial-candidate-review-v1",
            messages=messages,
            image_sha256="|".join(image_hashes),
        )
        return await self.client.complete_json(
            messages=messages,
            response_model=ExpertResult,
            request_meta=RequestMeta(
                request_id=f"{sample.sample_id}:spatial-candidate-review",
                request_hash=request_hash,
                prompt_version="spatial-candidate-review-v1",
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
                    prompt_version = self.settings.counting.prompt_version
                    result = await self._run_vqa_counting(sample, sample_dir)
                    route = (
                        f"{type(self.adapter).__name__} -> TaskRouter.route_vrsbench_vqa -> "
                        "vrsbench_count_target -> CountingExpert.answer -> "
                        f"PointCountingOrchestrator.count_image -> {type(self.client).__name__}.complete_json"
                    )
                else:
                    service = WorkflowService(self.client, self.prompts, self.settings.models.qwen.model)
                    expert = service.experts[expert_name]
                    expert_class_name = type(expert).__name__
                    prompt_version = expert.prompt_version
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
        """Run VRSBench quantity VQA through accepted-point counting.
        通过接受点计数运行 VRSBench 数量问答。
        """

        artifact_dir = sample_dir / "counting_expert"
        target = vrsbench_count_target(sample.question)
        image = read_normalized_image(sample.images[0].path)
        pipeline = PointCountingOrchestrator(
            self.client,
            counting=self.settings.counting,
            qwen=self.settings.models.qwen,
            system_prompt=self.prompts["count"],
            run_dir=artifact_dir,
            seam_prompt=self.prompts["seam"],
            empty_review_prompt=self.prompts.get("count_zero_review"),
        )
        counted = await CountingExpert(pipeline).answer(
            image,
            sample_id=sample.sample_id,
            question=sample.question,
            target=target,
            minimum_scan_depth=self.settings.counting.vrsbench_min_scan_depth,
            review_empty=self.settings.counting.vrsbench_zero_review,
            upscale_max_side=self.settings.counting.vrsbench_tile_upscale_max_side,
        )
        counting = counted.counting_result
        atomic_write_json(sample_dir / "counting_result.json", counting.model_dump(mode="json"))
        points = [
            VisualEvidence(
                label=point.target,
                point=[point.global_x_norm, point.global_y_norm],
                confidence=point.confidence,
                image_id=sample.images[0].image_id,
            )
            for point in counting.global_points
            if point.accepted
        ]
        geometry = {
            "version": "accepted-point-count-v2",
            "coordinate_frame": "normalized_0_999_top_left",
            "rule": "final_count_equals_accepted_points",
            "accepted_point_count": len(points),
            "final_count": counting.final_count,
            "minimum_scan_depth": self.settings.counting.vrsbench_min_scan_depth,
            "zero_review_enabled": self.settings.counting.vrsbench_zero_review,
            "tile_upscale_max_side": self.settings.counting.vrsbench_tile_upscale_max_side,
            "counting_status": counting.status,
            "warnings": [item.model_dump(mode="json") for item in counting.warnings],
        }
        return ExpertResult(
            expert="counting_expert",
            answer=str(counting.final_count) if counted.complete else counted.answer,
            evidence=[f"Accepted point {index + 1}: {item.point}" for index, item in enumerate(points)],
            evidence_items=points,
            geometry=geometry,
            status="completed" if counted.complete else "partial",
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


def _needs_spatial_candidate_review(sample: UnifiedSample, result: ExpertResult) -> bool:
    """Detect spatial questions whose instance evidence is not yet enumerable.
    检测实例证据尚不足以枚举的空间问题。
    """

    if sample.dataset != "VRSBench":
        return False
    lowered = sample.question.casefold()
    requires_instances = bool(
        re.search(r"\b(top|bottom)[ -]?most\b", lowered)
        or "predominantly arranged" in lowered
        or "arrangement" in lowered
    )
    if not requires_instances:
        return False
    vehicles = [
        item
        for item in result.evidence_items
        if item.box is not None and vrsbench_vehicle_class(item.label) in {"small-vehicle", "large-vehicle"}
    ]
    return len(vehicles) < 2


def _merge_visual_evidence(
    first: list[VisualEvidence],
    second: list[VisualEvidence],
) -> list[VisualEvidence]:
    """Merge two evidence passes while suppressing strongly overlapping duplicates.
    合并两轮视觉证据，并去除高度重叠的重复项。
    """

    merged = list(first)
    for candidate in second:
        duplicate = any(
            candidate.box is not None
            and existing.box is not None
            and vrsbench_vehicle_class(candidate.label) == vrsbench_vehicle_class(existing.label)
            and _box_iou(candidate.box, existing.box) >= 0.7
            for existing in merged
        )
        if not duplicate:
            merged.append(candidate)
    return merged


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
