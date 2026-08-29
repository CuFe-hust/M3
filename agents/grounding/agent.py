"""Grounding agent — thin agent over the grounding visual primitive.

定位 Agent — 定位视觉原语上的轻量 Agent。只支持 grounding 一个 task；
结果使用统一 0..999 归一化坐标证据（由基类 VisualEvidence 契约强制），
postprocess 强制 completed 必须携带合法定位证据，不在本模块计算任何指标。
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from agents.base import (
    AgentContext,
    AgentExecution,
    GroundingEvidenceService,
)
from agents.errors import AgentExecutionError
from agents.grounding.evidence import (
    GroundingEvidenceError,
    GroundingEvidenceResult,
)
from agents.schema import (
    AgentName,
    AgentResult,
    VisualEvidence,
    VisualTaskPlan,
)
from agents.visual_base import PromptBinding, VisualAgentBase
from data.schema import UnifiedSample
from models.base import VisionLanguageClient

# Grounding reuses the neutral VQA instruction (the baseline catalog mapped
# grounding to general_vqa_v3); the base class appends the shared JSON output
# contract and enforces 0..999 evidence coordinates.
# 定位复用中性 VQA 指令（基线目录将 grounding 映射到 general_vqa_v3）；
# JSON 输出契约由基类附加，0..999 证据坐标由基类强制。
_DEFAULT_PROMPT_TEXT = (
    "Answer the question concisely from the image. Preserve up to four "
    "representative relevant localized objects as labeled evidence_items; "
    "copy all evidence-item boxes into boxes in the same order. Coordinates "
    "are integer whole-image 0..999 raster coordinates in JSON with the origin at the "
    "top-left, positive x to the right, and positive y downward. A box is one "
    "flat array [x1,y1,x2,y2], never a pair of corner arrays. Do not include "
    "confidence values or hidden reasoning."
)

_DEFAULT_PROMPT_VERSION = "general_vqa_v3"

# Semantic placeholder for the persisted grounding evidence bundle (C7, 14A2
# §4.3); the final owned basename set is frozen by C8 §5.1.
# 持久化 Grounding 证据包的语义占位 basename（C7，14A2 §4.3）；最终 owned
# basename 集合由 C8 §5.1 冻结。
_GROUNDING_EVIDENCE_FILENAME = "grounding_evidence.json"


class GroundingAgent(VisualAgentBase):
    """Thin agent over the grounding visual primitive.
    定位视觉原语上的轻量 Agent。"""

    name: AgentName = "grounding_agent"
    supported_tasks: frozenset[str] = frozenset({"grounding"})

    def __init__(
        self,
        client: VisionLanguageClient,
        *,
        prompt: PromptBinding | None = None,
    ) -> None:
        super().__init__(
            client,
            agent_name=self.name,
            supported_tasks=self.supported_tasks,
            prompt=prompt
            or PromptBinding(text=_DEFAULT_PROMPT_TEXT, version=_DEFAULT_PROMPT_VERSION),
        )

    def build_user_payload(self, sample: UnifiedSample) -> dict[str, Any]:
        """Build the direct whole-image grounding payload.
        构造 direct 整图定位载荷。"""
        return {
            "task": sample.task,
            "question": sample.question,
            "coordinate_frame": "normalized_0_999_top_left",
            "box_format": "integer_xyxy_json",
        }

    async def run(self, sample: UnifiedSample, context: AgentContext) -> AgentExecution:
        """Protocol-owner entry for the v5 plan and injected grounding service.
        The direct path remains the explicit no-assistance branch. The trace is
        always enriched with a stable agent class and route; no request
        construction happens here.
        v5 计划与注入 Grounding 服务的协议 owner 入口。direct 路径是显式的无
        辅助分支。trace 始终补充稳定的 agent class 与 route；本处不做请求构造。"""
        task_plan = context.visual_task_plan
        bindings = context.visual_bindings
        if task_plan is not None:
            if not task_plan.needs_visual_assistance:
                execution = await super().run(sample, context)
                return replace(
                    execution,
                    trace={
                        **execution.trace,
                        "agent_class": f"{type(self).__module__}.{type(self).__qualname__}",
                        "route": f"{type(self).__name__}.run -> VisualAgentBase.run -> complete_json",
                    },
                )
            if bindings is None or bindings.grounding_evidence is None:
                raise AgentExecutionError(
                    self.name,
                    sample.sample_id,
                    cause="grounding_evidence_service_unavailable",
                )
            execution = await self._run_grounding_evidence(
                sample, context, task_plan, bindings.grounding_evidence
            )
            return replace(
                execution,
                trace={
                    **execution.trace,
                    "agent_class": f"{type(self).__module__}.{type(self).__qualname__}",
                    "route": f"{type(self).__name__}.run -> GroundingEvidenceExecutor.run",
                },
            )

        execution = await super().run(sample, context)
        return replace(
            execution,
            trace={
                **execution.trace,
                "agent_class": f"{type(self).__module__}.{type(self).__qualname__}",
                "route": f"{type(self).__name__}.run -> VisualAgentBase.run -> complete_json",
            },
        )

    async def _run_grounding_evidence(
        self,
        sample: UnifiedSample,
        context: AgentContext,
        plan: VisualTaskPlan,
        service: GroundingEvidenceService,
    ) -> AgentExecution:
        """Run the C6 seam (per-ROI YOLO + exactly one final Grounding Qwen
        call) and serialize the deterministic whole-image boxes into the
        existing AgentResult grounding contract; the evidence bundle persists
        under the protocol owner's additional results. The seam never emits
        confidence, never draws boxes, and never rewrites sample.task.
        运行 C6 seam（逐 ROI YOLO + 恰好一次最终 Grounding Qwen 调用），把
        确定性整图框序列化进现有 AgentResult grounding 契约；证据包在协议
        owner 的附加结果名下持久化。seam 绝不输出置信度、绝不绘制框、绝不
        改写 sample.task。"""
        if not sample.images:
            raise AgentExecutionError(
                self.name,
                sample.sample_id,
                cause="grounding evidence requires at least one image",
            )
        images = self._read_evidence_images(sample, context)
        try:
            result = await service.run(
                plan,
                sample,
                images,
                base_user_payload=self.build_user_payload(sample),
                fallback_image_id=sample.images[0].image_id,
                artifact_dir=context.artifact_dir,
                budget=context.call_budget,
                materialized_views=context.visual_views,
            )
        except GroundingEvidenceError as exc:
            raise AgentExecutionError(
                self.name,
                sample.sample_id,
                cause=f"grounding_evidence_failed:{exc.code}",
            ) from exc
        except Exception as exc:
            raise AgentExecutionError(
                self.name,
                sample.sample_id,
                cause=f"grounding_evidence_failed:{type(exc).__name__}",
            ) from exc
        payload = AgentResult(
            agent_name=self.name,
            # The public grounding contract carries the primary coordinate in
            # answer; boxes/evidence_items retain every returned localized box.
            answer=json.dumps(list(result.whole_image_boxes[0].box), separators=(",", ":")),
            boxes=[list(box.box) for box in result.whole_image_boxes],
            evidence_items=[
                VisualEvidence(label=item.label, box=list(item.box))
                for item in result.whole_image_boxes
            ],
            status="completed",
        )
        payload = await self.postprocess(sample, payload)
        return AgentExecution(
            agent_name=self.name,
            payload=payload,
            result_filename=self.result_filename(sample),
            trace={
                "workflow": "grounding_evidence",
                "catalog_version": result.bundle.catalog_version,
            },
            additional_results={
                _GROUNDING_EVIDENCE_FILENAME: result.bundle.model_dump(mode="json"),
            },
        )

    async def postprocess(
        self,
        sample: UnifiedSample,
        result: AgentResult,
    ) -> AgentResult:
        """A completed grounding result must carry valid localized evidence
        (evidence_items geometry or top-level boxes). Missing geometry
        downgrades to partial; an empty answer with no geometry becomes failed.
        completed 的定位结果必须携带合法定位证据（evidence_items 几何或顶层
        boxes）。缺失几何降级为 partial；answer 与几何均无效时为 failed。"""
        evidence = [
            item
            for item in result.evidence_items
            if item.box is not None or item.point is not None
        ]
        has_geometry = bool(evidence) or bool(result.boxes)
        if has_geometry:
            return result
        geometry = dict(result.geometry or {})
        geometry["grounding_constraint_violation"] = "no_localized_evidence"
        if result.answer.strip():
            return result.model_copy(
                update={"status": "partial", "geometry": geometry}
            )
        return result.model_copy(update={"status": "failed", "geometry": geometry})
