"""General VQA agent — thin VisualAgentBase subclass for open-ended QA.

通用 VQA Agent — 开放问答的轻量 VisualAgentBase 子类。覆盖 general_vqa、
scene_classification、multiple_choice_vqa 与 spatial_relation 四个 task；
选择题载荷包含 choices 与单/多选约束，输出在 postprocess 中按 choices 约束
校验。spatial_relation 与通用 VQA 共享同一条单次 Qwen 调用路径，不做任何
专用几何后处理，也不读取 Prompt 文件（提示文本以中性 PromptBinding 注入）。
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Mapping
from typing import Any

from PIL import Image

from agents.base import (
    AgentContext,
    AgentExecution,
    VqaEvidenceService,
)
from agents.errors import AgentExecutionError, AgentTaskMismatchError
from agents.general_vqa.evidence.executor import (
    EvidenceExecution,
    SegFormerPreviewEvidence,
    _pixel_xyxy_to_999,
)
from agents.general_vqa.evidence.rendering import (
    PALETTE_VERSION,
    leaf_boolean_grid,
    make_preview,
    render_pure_mask,
    render_yolo_annotation,
)
from agents.general_vqa.evidence.schema import RoiEvidenceRecord
from agents.schema import (
    AgentName,
    AgentResult,
    GENERAL_VQA_AGENT_TASKS,
    VisualTaskPlan,
)
from agents.visual_base import PromptBinding, VisualAgentBase
from data.schema import UnifiedSample
from models.base import (
    ModelCacheIdentity,
    RequestMeta,
    VisionLanguageClient,
    build_request_hash,
)
from models.images import (
    ImageRegionSource,
    image_to_data_url,
    image_sha256,
    open_image_region_source,
)

# Neutral default prompt text (English mirror of the baseline general_vqa_v3
# prompt). The repository prompt file is intentionally not read by agents;
# the version string stays aligned with the baseline asset name.
# 中性默认提示文本（基线 general_vqa_v3 prompt 的英文镜像）。Agent 有意不
# 读取仓库 Prompt 文件；版本字符串与基线资产名保持一致。
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

# Semantic placeholder for the persisted VQA evidence bundle (C7, 14A2 §4.3);
# the final owned basename set is frozen by C8 §5.1.
# VQA 证据包持久化的语义占位 basename（C7，14A2 §4.3）；最终 owned basename
# 集合由 C8 §5.1 冻结。
_VQA_EVIDENCE_FILENAME = "vqa_evidence.json"

# Closed final-content roles keep the image/text contract explicit and
# auditable; they do not participate in capability selection.
# 封闭的最终内容角色集合用于明确、可审计的图文契约；不参与能力选择。
VISUAL_INPUT_ROLES = frozenset(
    {
        "annotated_roi",
        "segformer_pure_mask",
        "yolo_on_segformer_pure_mask",
        "clean_roi",
    }
)
VISUAL_CONTENT_VERSION = "v2"


class GeneralVQAAgent(VisualAgentBase):
    """Open-ended / closed-vocabulary visual QA agent.
    开放/闭集词汇视觉问答 Agent。"""

    name: AgentName = "general_vqa_agent"
    supported_tasks: frozenset[str] = GENERAL_VQA_AGENT_TASKS

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

    @staticmethod
    def build_user_payload(sample: UnifiedSample) -> dict[str, Any]:
        """Build the canonical task-aware VQA payload without duplicate facts.
        构造规范的 task-aware VQA 载荷，不重复表达事实。"""
        payload: dict[str, Any] = {
            "question": sample.question,
            "task": sample.task,
        }
        if sample.task == "multiple_choice_vqa":
            choices, allow_multiple = _canonical_choices(sample)
            payload["choices"] = choices
            payload["allow_multiple"] = allow_multiple
        if (
            sample.normalization is not None
            and sample.normalization.semantic_subtype is not None
        ):
            payload["semantic_subtype"] = sample.normalization.semantic_subtype
        payload["coordinate_frame"] = "normalized_0_999_top_left"
        payload["box_format"] = "integer_xyxy_json"
        return payload

    async def postprocess(
        self,
        sample: UnifiedSample,
        result: AgentResult,
    ) -> AgentResult:
        """Enforce the multiple-choice output constraint: a single-choice
        answer must map to exactly one choice; a multi-choice answer must
        contain only choices, deduplicated in stable choice order. Violations
        downgrade status to partial and record answer_constraint_violation.
        强制选择题输出约束：单选答案必须唯一映射到一个选项；多选答案只能
        包含选项、去重并按选项稳定顺序排列。违规将状态降级为 partial 并记录
        answer_constraint_violation。"""
        if sample.task != "multiple_choice_vqa":
            return result
        choices, allow_multiple = _canonical_choices(sample)
        violation, normalized_answer = _validate_choice_answer(
            result.answer, choices, allow_multiple
        )
        if violation is None:
            if normalized_answer is not None and normalized_answer != result.answer.strip():
                result = result.model_copy(update={"answer": normalized_answer})
            return result
        geometry = dict(result.geometry or {})
        geometry["answer_constraint_violation"] = violation
        return result.model_copy(update={"status": "partial", "geometry": geometry})

    async def run(self, sample: UnifiedSample, context: AgentContext) -> AgentExecution:
        """Run direct VQA or the canonical v3 evidence path.
        运行直接 VQA 或规范 v3 证据路径。"""
        if sample.task not in self.supported_tasks:
            raise AgentTaskMismatchError(
                self.name, sample.task, supported=self.supported_tasks
            )
        # Validate canonical choice facts before identity lookup, image I/O,
        # budget consumption, evidence execution, or Qwen. 在身份读取、图像
        # I/O、预算消耗、证据执行或 Qwen 之前校验规范选项事实。
        if sample.task == "multiple_choice_vqa":
            _canonical_choices(sample)
        task_plan = context.visual_task_plan
        if task_plan is not None:
            if not task_plan.needs_visual_assistance:
                return await super().run(sample, context)
            if context.visual_bindings is None or context.visual_bindings.vqa_evidence is None:
                raise AgentExecutionError(
                    self.name,
                    sample.sample_id,
                    cause="vqa_evidence_service_unavailable",
                )
            return await self._run_object_evidence(
                sample,
                context,
                task_plan,
                context.visual_bindings.vqa_evidence,
            )

        return await super().run(sample, context)

    def prepare_materialized_model_image(self, image: Image.Image) -> Image.Image:
        """Shrink direct final-Qwen views to the canonical VQA preview size.

        Source/ROI geometry remains authoritative; this preview is transport
        only and therefore must not replace persisted detector coordinates.
        将 direct 最终 Qwen 视图只缩小到规范 VQA 预览尺寸。源图/ROI 几何仍是
        权威坐标；该预览仅用于传输，不得覆盖持久化检测坐标。
        """
        return make_preview(image)

    async def _run_object_evidence(
        self,
        sample: UnifiedSample,
        context: AgentContext,
        plan: VisualTaskPlan,
        service: VqaEvidenceService,
    ) -> AgentExecution:
        """One evidence pass plus exactly one final-Qwen call assembled per
        14B §10; the evidence bundle is persisted under the protocol owner's
        additional results, so it reaches ArtifactWriter through the existing
        safe-basename + atomic-write path.
        一次证据处理加上按 14B §10 组装恰好一次最终 Qwen 调用；证据包在协议
        owner 的附加结果名下持久化，经现有安全 basename + 原子写入路径到达
        ArtifactWriter。"""
        identity = getattr(self._client, "cache_identity", None)
        if not isinstance(identity, ModelCacheIdentity):
            raise AgentExecutionError(
                self.name,
                sample.sample_id,
                cause="model client returned an invalid cache_identity",
            )
        if not sample.images:
            raise AgentExecutionError(
                self.name,
                sample.sample_id,
                cause="object_evidence_vqa requires at least one image",
            )
        sources = self._open_evidence_sources(sample, context)
        try:
            execution = service.execute(
                plan,
                sources,
                fallback_image_id=sample.images[0].image_id,
                materialized_views=context.visual_views,
            )
            content, final_hashes = self._build_evidence_content(
                sample, plan, execution, sources
            )
        except Exception as exc:
            raise AgentExecutionError(
                self.name,
                sample.sample_id,
                cause=f"vqa_evidence_failed:{type(exc).__name__}",
            ) from exc
        finally:
            # The region sources are single-sample resources and are always
            # closed, success or failure. 区域 source 是单样本资源，无论成败
            # 都必须显式关闭。
            for source in sources.values():
                source.close()

        prompt_sel = self.select_prompt(sample)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _structured_prompt(prompt_sel, self.name)},
            {"role": "user", "content": content},
        ]
        # The request hash covers every semantic input of the actual call:
        # logical identity, generation, prompt version, messages, the honest
        # digest of exactly what the model receives, and the response schema.
        # request hash 覆盖实际调用的全部语义输入：逻辑身份、generation、prompt
        # version、messages、模型实际收到内容的真实摘要与响应 schema。
        request_hash = build_request_hash(
            model=identity.model,
            generation=identity.generation_payload(),
            prompt_version=prompt_sel.version,
            messages=messages,
            image_sha256="|".join(final_hashes),
            target_spec=self._evidence_hash_identity(plan, execution),
            response_schema=AgentResult.model_json_schema(),
            client_version=identity.client_version,
            model_revision=identity.revision,
        )
        # Exactly one final-Qwen budget entry per sample (14A2 §5.2); the
        # planner's call was a separate entry on the same shared budget.
        # 每条样本恰好一次最终 Qwen budget 条目（14A2 §5.2）；规划器调用在同一
        # 共享 budget 上是独立条目。
        context.call_budget.reserve_qwen()
        result = await self._client.complete_json(
            messages=messages,
            response_model=AgentResult,
            request_meta=RequestMeta(
                request_id=f"{sample.sample_id}:{self.name}",
                request_hash=request_hash,
                prompt_version=prompt_sel.version,
                sample_id=sample.sample_id,
                artifact_dir=context.artifact_dir / self.name,
            ),
        )
        if result.agent_name != self.name:
            raise AgentExecutionError(
                self.name,
                sample.sample_id,
                cause=f"model returned agent_name {result.agent_name!r}",
            )
        result = await self.postprocess(sample, result)
        return AgentExecution(
            agent_name=self.name,
            payload=result,
            result_filename=self.result_filename(sample),
            trace={
                "workflow": "object_evidence_vqa",
                "visual_content_version": VISUAL_CONTENT_VERSION,
                "prompt_version": prompt_sel.version,
                "request_hash": request_hash,
                "image_sha256": final_hashes,
                "model": identity.model,
            },
            additional_results={
                _VQA_EVIDENCE_FILENAME: execution.bundle.model_dump(mode="json"),
            },
        )

    def _open_evidence_sources(
        self,
        sample: UnifiedSample,
        context: AgentContext,
    ) -> dict[str, ImageRegionSource]:
        """Open one read-only region source per sample image through the
        escape-guarded seam, keyed by image_id; I/O failures map to stable
        codes and never leak machine paths. The Agent consumes only the
        generic seam — it never chooses a JPEG/TIFF backend. Sources are
        single-sample resources and must be closed by the caller.
        通过防逃逸 seam 为每条样本图像打开一个只读 region source，按 image_id
        索引；I/O 失败映射为稳定 code，绝不泄漏机器路径。Agent 只消费通用
        seam——绝不选择 JPEG/TIFF backend。source 是单样本资源，调用方必须
        关闭。"""
        sources: dict[str, ImageRegionSource] = {}
        try:
            for image_ref in sample.images:
                candidate_path, _ = self._read_image(
                    image_ref.path, context, sample_id=sample.sample_id
                )
                try:
                    sources[image_ref.image_id] = open_image_region_source(
                        candidate_path
                    )
                except (OSError, ValueError) as exc:
                    raise AgentExecutionError(
                        self.name,
                        sample.sample_id,
                        cause=f"image_decode_failed:{type(exc).__name__}",
                    ) from exc
        except Exception:
            for source in sources.values():
                source.close()
            raise
        return sources

    def _build_evidence_content(
        self,
        sample: UnifiedSample,
        plan: VisualTaskPlan,
        execution: EvidenceExecution,
        images: Mapping[str, ImageRegionSource],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Assemble the single final-Qwen user content per the frozen
        three-branch protocol (14.12-14.13). Per ROI: YOLO-only -> annotated
        ROI; SegFormer-only -> pure mask plus clean ROI; YOLO + SegFormer ->
        YOLO-on-pure-mask plus clean ROI; neither -> clean ROI. Only the executor's finished
        bundle/masks/palette are consumed — models are never rerun and
        capabilities are never re-decided here. Confidence never appears and
        the palette is carried in-memory only; both the rendered digests and
        evidence_identity fold the protocol identity into the request hash.
        按冻结三分支协议（14.12-14.13）组装唯一最终 Qwen 用户内容。逐 ROI：
        仅 YOLO -> 标注 ROI；仅 SegFormer -> 纯色 mask 加干净 ROI；YOLO + SegFormer ->
        YOLO-on-pure-mask 加干净 ROI；均无 -> 干净 ROI。只消费 executor 已完成
        bundle/masks/palette——此处绝不重跑模型或重新决定 capability。confidence
        绝不出现，调色表仅存内存；渲染摘要与 evidence_identity 共同把协议身份
        折叠进请求 hash。"""
        bundle = execution.bundle
        content: list[dict[str, Any]] = []
        final_hashes: list[str] = []
        visual_inputs: list[dict[str, Any]] = []
        roi_records: list[dict[str, Any]] = []
        yolo_detections: list[dict[str, Any]] = []
        segformer_hits: list[dict[str, Any]] = []

        yolo_by_roi: dict[str, list[tuple[str, tuple[float, float, float, float]]]] = {}
        roi_image_ids = {record.roi_id: record.image_id for record in bundle.rois}
        for record in bundle.detections:
            yolo_by_roi.setdefault(record.roi_id, []).append(
                (record.leaf_category, tuple(record.local_xyxy))
            )
            yolo_detections.append(
                {
                    "category": record.leaf_category,
                    "roi_id": record.roi_id,
                    "image_id": roi_image_ids[record.roi_id],
                    # The final Qwen sees an ROI crop, so expose only its local
                    # geometry in the same integer 0..999 JSON frame as SFT.
                    # 最终 Qwen 看到的是 ROI 裁切图，因此只暴露局部几何，并统一
                    # 为与 SFT 相同的 0..999 整数 JSON 坐标。
                    "box": _pixel_xyxy_to_999(
                        record.local_xyxy, record.local_roi_size
                    ),
                    # AgentResult evidence uses normalized whole-image geometry;
                    # expose the matching detector geometry so training and
                    # inference never relabel an ROI-local box as global.
                    # AgentResult 证据使用整图归一化几何；同时暴露对应的检测框，
                    # 避免训练或推理把 ROI 局部框误标为整图框。
                    "global_box": _pixel_xyxy_to_999(
                        record.global_xyxy, record.global_image_size
                    ),
                }
            )
        seg_by_roi: dict[str, list[str]] = {}
        for record in bundle.segments:
            seg_by_roi.setdefault(record.roi_id, []).append(record.leaf_category)
            segformer_hits.append(
                {"roi_id": record.roi_id, "category": record.leaf_category}
            )
        # Rendered sets follow the stable leaf order (14.12.1): the executor
        # already filtered hits to the requested plan categories and emitted
        # detections in plan leaf order, so deduplicating the detection list
        # preserves that order without the agent re-deciding capabilities.
        # rendered 集合遵循稳定 leaf 顺序（14.12.1）：executor 已把命中过滤到
        # 请求的 plan 类别并按 plan leaf 顺序输出检测，因此对检测列表去重即可
        # 保持该顺序，无需 agent 重新决定 capability。
        seg_hit_leaves = {record.leaf_category for record in bundle.segments}
        rendered_segformer_leaves = [
            leaf for leaf in bundle.leaf_states if leaf in seg_hit_leaves
        ]

        def append_visual(image: Image.Image, *, roi_id: str, role: str) -> None:
            if role not in VISUAL_INPUT_ROLES:
                raise ValueError(f"unsupported visual input role: {role}")
            content.append(self._image_block(image, final_hashes))
            visual_inputs.append(
                {
                    "content_image_index": len(visual_inputs),
                    "roi_id": roi_id,
                    "role": role,
                }
            )

        for record in bundle.rois:
            raw_crop = images[record.image_id].read_box(record.expanded_xyxy)
            clean = make_preview(raw_crop)
            roi_records.append(
                {
                    "roi_id": record.roi_id,
                    "image_id": record.image_id,
                    "crop_size": list(record.crop_size),
                }
            )
            yolo_boxes = yolo_by_roi.get(record.roi_id, [])
            seg_leaves = seg_by_roi.get(record.roi_id, [])
            if yolo_boxes and not seg_leaves:
                # YOLO only: annotated ROI. 仅 YOLO：标注 ROI。
                append_visual(
                    render_yolo_annotation(raw_crop, yolo_boxes),
                    roi_id=record.roi_id,
                    role="annotated_roi",
                )
            elif seg_leaves and not yolo_boxes:
                # SegFormer only: pure mask plus the matching clean ROI.
                # 仅 SegFormer：纯色 mask 加同一 ROI 的干净原图。
                append_visual(
                    make_preview(
                        self._pure_mask(record, seg_leaves, execution),
                        resample=Image.Resampling.NEAREST,
                    ),
                    roi_id=record.roi_id,
                    role="segformer_pure_mask",
                )
                append_visual(
                    clean,
                    roi_id=record.roi_id,
                    role="clean_roi",
                )
            elif yolo_boxes and seg_leaves:
                # Both: YOLO-on-pure-mask plus the clean ROI (14.12.3).
                # 两者：YOLO-on-pure-mask 加干净 ROI（14.12.3）。
                append_visual(
                    render_yolo_annotation(
                        make_preview(
                            self._pure_mask(record, seg_leaves, execution),
                            resample=Image.Resampling.NEAREST,
                        ),
                        yolo_boxes,
                        # The pure mask is already shrunk with NEAREST, but the
                        # boxes still live in the source ROI pixel frame: pass
                        # source_size so they scale onto the preview instead of
                        # being drawn at scale 1.0 on the smaller canvas.
                        # 纯色 mask 已用 NEAREST 缩小，但框仍位于源 ROI 像素帧：
                        # 传入 source_size 使框按预览缩放，而不是以 scale 1.0
                        # 画在更小的画布上。
                        source_size=record.crop_size,
                        # The pure mask is already shrunk with NEAREST;
                        # keep that discrete palette while drawing YOLO.
                        # 纯色 mask 已经用 NEAREST 缩小；绘制 YOLO 框时继续
                        # 保持离散调色表，禁止 LANCZOS 重新采样。
                        resample=Image.Resampling.NEAREST,
                    ),
                    roi_id=record.roi_id,
                    role="yolo_on_segformer_pure_mask",
                )
                append_visual(clean, roi_id=record.roi_id, role="clean_roi")
            else:
                # Neither branch: the clean ROI keeps visual context.
                # 均无分支：干净 ROI 保持视觉上下文。
                append_visual(clean, roi_id=record.roi_id, role="clean_roi")

        payload = dict(self.build_user_payload(sample))
        # The final Qwen receives ROI crops, so detection geometry is local to
        # those crops while retaining the SFT integer 0..999 JSON convention.
        # 最终 Qwen 接收的是 ROI 裁切图，因此检测框相对于裁切图局部定义，同时
        # 保持 SFT 的 0..999 整数 JSON 约定。
        payload["coordinate_frame"] = "roi_normalized_0_999_top_left"
        payload["box_format"] = "integer_xyxy_json"
        payload["evidence"] = {
            "visual_inputs": visual_inputs,
            "rois": roi_records,
            "requested_categories": list(plan.object_categories),
            "detections": yolo_detections,
            "segmentation_hits": segformer_hits,
            "missing_categories": list(bundle.missing_leaves),
            "mask_legend": [
                {"category": leaf, "color_rgb": list(execution.palette[leaf])}
                for leaf in rendered_segformer_leaves
            ],
        }
        content.append(
            {"type": "text", "text": json.dumps(payload, ensure_ascii=False)}
        )
        return content, final_hashes

    @staticmethod
    def _evidence_hash_identity(
        plan: VisualTaskPlan,
        execution: EvidenceExecution,
    ) -> dict[str, Any]:
        """Return non-model-visible evidence identity for request hashing.
        返回仅用于请求哈希、模型不可见的证据身份。"""
        bundle = execution.bundle
        return {
            "protocol": {
                "catalog_version": bundle.catalog_version,
                "preprocessing_version": bundle.preprocessing_version,
                "palette_version": PALETTE_VERSION,
                "visual_content_version": VISUAL_CONTENT_VERSION,
            },
            "source_geometry": [
                {
                    "roi_id": record.roi_id,
                    "image_id": record.image_id,
                    "source_size": list(record.source_size),
                    "core_xyxy": list(record.core_xyxy),
                    "expanded_xyxy": list(record.expanded_xyxy),
                    "crop_size": list(record.crop_size),
                }
                for record in bundle.rois
            ],
            "planned_categories": list(plan.object_categories),
        }

    def _pure_mask(
        self,
        record: RoiEvidenceRecord,
        seg_leaves: list[str],
        execution: EvidenceExecution,
    ) -> Image.Image:
        """Compose the per-ROI SegFormer pure mask in stable hit order,
        directly in preview space: every leaf's boolean mask is extracted
        from the ROI's preview class-id grid, so no WxH mask is ever created.
        The palette and the later-leaf-overwrites-earlier-leaf precedence are
        unchanged. 按稳定命中顺序直接在 preview 空间合成逐 ROI SegFormer 纯色
        mask：每个叶子的 boolean mask 都从该 ROI 的 preview class-id grid 提取，
        绝不创建 WxH mask。调色表与后叶覆盖前叶的优先级不变。"""
        evidences = [
            evidence
            for evidence in execution.preview_evidence
            if evidence.roi_id == record.roi_id
        ]
        if not evidences:
            raise ValueError(
                f"no preview evidence for hit ROI {record.roi_id!r}"
            )
        leaf_masks: list[tuple[str, Image.Image]] = []
        for leaf in seg_leaves:
            evidence = next(
                (
                    evidence
                    for evidence in evidences
                    if leaf in evidence.leaf_class_ids
                ),
                None,
            )
            if evidence is None:
                raise ValueError(
                    f"no preview evidence for hit leaf {leaf!r} of ROI {record.roi_id!r}"
                )
            leaf_masks.append(
                (
                    leaf,
                    leaf_boolean_grid(
                        evidence.class_id_grid, evidence.leaf_class_ids[leaf]
                    ),
                )
            )
        return render_pure_mask(
            evidences[0].preview_size,
            leaf_masks,
            execution.palette,
        )

    @staticmethod
    def _image_block(image: Image.Image, hashes: list[str]) -> dict[str, Any]:
        """Encode one in-memory image to a PNG data URL and record the honest
        digest of exactly what the model receives; in-memory transport only —
        persistence format/quality parameters stay unchosen.
        将一张内存图像编码为 PNG data URL 并记录模型实际收到内容的真实摘要；
        仅内存传输——持久化格式/质量参数保持未选择。"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        hashes.append(image_sha256(data))
        return {
            "type": "image_url",
            "image_url": {"url": image_to_data_url(data, "image/png")},
        }


def _structured_prompt(prompt: PromptBinding, agent_name: str) -> str:
    """Frozen structured-output suffix, verbatim mirror of
    VisualAgentBase.run() so the evidence path's final call stays
    format-identical with the direct path.
    冻结结构化输出后缀，逐字镜像 VisualAgentBase.run()，使证据路径的最终调用
    与 direct 路径格式一致。"""
    return (
        prompt.text
        + f"\n\nReturn valid JSON only. Set agent_name to {agent_name!r}; "
        "put the concise final answer in answer, retain relevant labeled boxes or points "
        "in evidence_items, copy evidence boxes into boxes, omit confidence values, "
        "and set status to 'completed'."
    )


def _canonical_choices(sample: UnifiedSample) -> tuple[list[str], bool]:
    """Read canonical multiple-choice facts or fail before model-side work.
    读取规范多选事实；缺失时在模型侧工作开始前稳定失败。"""
    normalization = sample.normalization
    if normalization is None or len(normalization.choices) < 2:
        raise AgentExecutionError(
            "general_vqa_agent",
            sample.sample_id,
            cause="multiple_choice_sample_without_choices",
        )
    return list(normalization.choices), normalization.allow_multiple


def _normalize_choice(value: str) -> str:
    return value.strip().casefold()


_OPTION_PREFIX = re.compile(
    r"^\s*(?:\(([A-Za-z])\)|([A-Za-z])(?:[.)、:-]|\s+))\s*(.*?)\s*$"
)


def _parse_choice(choice: str) -> tuple[str | None, str]:
    """Parse an optional choice label and its display text.
    解析可选的选项字母及其显示文本。"""
    match = _OPTION_PREFIX.match(choice)
    if match is None:
        return None, choice.strip()
    label = (match.group(1) or match.group(2)).upper()
    return label, match.group(3).strip()


def _match_choice(value: str, choices: list[str]) -> str | None:
    """Match an answer item to a choice: by normalized full text, by the
    prefix-stripped text, or by an explicit/positional option letter.
    将答案项匹配到选项：按归一化全文、去前缀文本或显式/位置选项字母。"""
    normalized = _normalize_choice(value)
    parsed = [_parse_choice(choice) for choice in choices]
    for choice in choices:
        if _normalize_choice(choice) == normalized:
            return choice
    for choice, (_, text) in zip(choices, parsed):
        if _normalize_choice(text) == normalized:
            return choice
    letter = value.strip().upper()
    if len(letter) == 1 and "A" <= letter <= "Z":
        for choice, (label, _) in zip(choices, parsed):
            if label == letter:
                return choice
        if all(label is None for label, _ in parsed):
            index = ord(letter) - ord("A")
            if index < len(choices):
                return choices[index]
    return None


def _validate_choice_answer(
    answer: str,
    choices: list[str],
    allow_multiple: bool,
) -> tuple[str | None, str | None]:
    """Return (violation, normalized_answer); both None when the answer
    satisfies the constraint. 返回（违规描述、规范化答案）；满足约束时两者
    均为 None。"""
    if not allow_multiple:
        if _match_choice(answer, choices) is not None:
            return None, None
        return f"answer {answer!r} does not map to a single choice", None
    parts = [part.strip() for part in re.split(r"[,;，；]", answer) if part.strip()]
    labels = {label for label, _ in map(_parse_choice, choices) if label is not None}
    compact = answer.strip().upper()
    if (
        len(parts) == 1
        and len(compact) > 1
        and compact.isalpha()
        and all(letter in labels for letter in compact)
    ):
        # Some datasets serialize multi-select labels compactly as "ABC".
        # 一些数据集将多选字母紧凑序列化为 "ABC"。
        parts = list(compact)
    if not parts:
        return "empty multiple-choice answer", None
    matched: list[str] = []
    for part in parts:
        choice = _match_choice(part, choices)
        if choice is None:
            return f"answer item {part!r} is not among the choices", None
        if choice not in matched:
            matched.append(choice)
    # Stable order follows the choice list. / 稳定顺序遵循选项列表。
    ordered = [choice for choice in choices if choice in matched]
    return None, ", ".join(ordered)
