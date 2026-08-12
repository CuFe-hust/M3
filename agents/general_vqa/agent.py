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
from agents.general_vqa.evidence.executor import EvidenceExecution
from agents.general_vqa.evidence.rendering import (
    make_preview,
    overlay_mask,
    render_roi_crop,
    stable_palette_color,
)
from agents.general_vqa.evidence.schema import RoiEvidenceRecord, VqaEvidenceBundle
from agents.schema import AgentName, AgentResult, FirstQwenVisualPlan, RoiRegion
from agents.visual_base import PromptBinding, VisualAgentBase
from data.schema import UnifiedSample
from models.base import (
    ModelCacheIdentity,
    RequestMeta,
    VisionLanguageClient,
    build_request_hash,
)
from models.images import (
    image_to_data_url,
    image_sha256,
)

# Neutral default prompt text (English mirror of the baseline general_vqa_v2
# prompt). The repository prompt file is intentionally not read by agents;
# the version string stays aligned with the baseline asset name.
# 中性默认提示文本（基线 general_vqa_v2 prompt 的英文镜像）。Agent 有意不
# 读取仓库 Prompt 文件；版本字符串与基线资产名保持一致。
_DEFAULT_PROMPT_TEXT = (
    "Answer the question concisely from the image. Preserve up to four "
    "representative relevant localized objects as labeled evidence_items; "
    "copy all evidence-item boxes into boxes in the same order. Coordinates "
    "are whole-image 0..999 raster coordinates with the origin at the "
    "top-left, positive x to the right, and positive y downward. A box is one "
    "flat array [x1,y1,x2,y2], never a pair of corner arrays. Use an empty "
    "evidence list only when the answer genuinely has no localizable visual "
    "support. Do not include hidden reasoning."
)

_DEFAULT_PROMPT_VERSION = "general_vqa_v2"

# Compatibility matrix (14A2 gate 1): only general_vqa may run the
# object_evidence_vqa family; the other VQA-family tasks are direct_vqa only
# and any object_evidence_vqa plan for them is a forbidden combination that
# fails stably without ever touching sample.task.
# 兼容矩阵（14A2 门禁 1）：只有 general_vqa 可运行 object_evidence_vqa 家族；
# 其余 VQA 家族任务只走 direct_vqa，为它们出现 object_evidence_vqa 计划属于
# 禁止组合，稳定失败且绝不触碰 sample.task。
_DIRECT_ONLY_TASKS = frozenset({
    "scene_classification",
    "multiple_choice_vqa",
    "spatial_relation",
})

# Semantic placeholder for the persisted VQA evidence bundle (C7, 14A2 §4.3);
# the final owned basename set is frozen by C8 §5.1.
# VQA 证据包持久化的语义占位 basename（C7，14A2 §4.3）；最终 owned basename
# 集合由 C8 §5.1 冻结。
_VQA_EVIDENCE_FILENAME = "vqa_evidence.json"


class GeneralVQAAgent(VisualAgentBase):
    """Open-ended / closed-vocabulary visual QA agent.
    开放/闭集词汇视觉问答 Agent。"""

    name: AgentName = "general_vqa_agent"
    supported_tasks: frozenset[str] = frozenset({
        "general_vqa",
        "scene_classification",
        "multiple_choice_vqa",
        "spatial_relation",
    })

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
        """Extend the neutral payload with choice constraints for
        multiple_choice_vqa; other tasks keep the neutral payload unchanged.
        为 multiple_choice_vqa 扩展中性载荷（加入选项与单/多选约束）；其他
        task 保持中性载荷不变。"""
        payload = super().build_user_payload(sample)
        if sample.task == "multiple_choice_vqa":
            constraints = _choice_constraints(sample)
            payload["choices"] = _extract_choices(constraints)
            payload["allow_multiple"] = bool(constraints.get("allow_multiple", False))
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
        constraints = _choice_constraints(sample)
        choices = _extract_choices(constraints)
        if not choices:
            raise AgentExecutionError(
                self.name,
                sample.sample_id,
                cause="multiple_choice_sample_without_choices",
            )
        allow_multiple = bool(constraints.get("allow_multiple", False))
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
        """Protocol-owner entry (14A2 §4.3): when the feature is on (typed
        plan present) and the plan selects object_evidence_vqa, run the
        object-evidence path; every other sample keeps the legacy direct path
        byte-identical. The plan never rewrites sample.task, and a plan that
        selects an unapproved task combination fails stably instead of
        silently degrading to the direct path.
        协议 owner 入口（14A2 §4.3）：特性开启（存在 typed plan）且计划选择
        object_evidence_vqa 时运行对象证据路径；其余样本保持旧直接路径逐字节
        一致。计划绝不改写 sample.task；选择未批准任务组合的计划稳定失败，而
        不静默降级到直接路径。"""
        if sample.task not in self.supported_tasks:
            raise AgentTaskMismatchError(
                self.name, sample.task, supported=self.supported_tasks
            )
        plan = context.visual_plan
        if plan is None or plan.execution_family != "object_evidence_vqa":
            # Feature off, or the planner chose the direct path — legacy run.
            # 特性关闭，或规划器选择直接路径——旧 run。
            return await super().run(sample, context)
        # Compatibility matrix (14A2 gate 1): object evidence is approved only
        # for general_vqa; scene_classification / multiple_choice_vqa /
        # spatial_relation plans must be direct_vqa.
        # 兼容矩阵（14A2 门禁 1）：对象证据只对 general_vqa 批准；
        # scene_classification / multiple_choice_vqa / spatial_relation 的计划
        # 必须是 direct_vqa。
        if sample.task != "general_vqa":
            raise AgentExecutionError(
                self.name,
                sample.sample_id,
                cause=f"object_evidence_plan_forbidden_for_task:{sample.task}",
            )
        if context.visual_bindings is None or context.visual_bindings.vqa_evidence is None:
            raise AgentExecutionError(
                self.name,
                sample.sample_id,
                cause="vqa_evidence_service_unavailable",
            )
        return await self._run_object_evidence(
            sample, context, plan, context.visual_bindings.vqa_evidence
        )

    async def _run_object_evidence(
        self,
        sample: UnifiedSample,
        context: AgentContext,
        plan: FirstQwenVisualPlan,
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
        images = self._read_evidence_images(sample, context)
        try:
            execution: EvidenceExecution = service.execute(
                plan,
                images,
                fallback_image_id=sample.images[0].image_id,
            )
            content, final_hashes = self._build_evidence_content(
                sample, plan, execution.bundle, execution.masks, images
            )
        except Exception as exc:
            raise AgentExecutionError(
                self.name,
                sample.sample_id,
                cause=f"vqa_evidence_failed:{type(exc).__name__}",
            ) from exc

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
                "prompt_version": prompt_sel.version,
                "request_hash": request_hash,
                "image_sha256": final_hashes,
                "model": identity.model,
            },
            additional_results={
                _VQA_EVIDENCE_FILENAME: execution.bundle.model_dump(mode="json"),
            },
        )

    def _build_evidence_content(
        self,
        sample: UnifiedSample,
        plan: FirstQwenVisualPlan,
        bundle: VqaEvidenceBundle,
        masks: Mapping[tuple[str, str], Any],
        images: Mapping[str, Image.Image],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Assemble the single final-Qwen user content per 14B §10: clean ROI
        images first, then per-ROI mask overlays, then text evidence (question,
        answer constraints, image sizes, ROI crop geometry, YOLO text records,
        SegFormer legend). Confidence never appears and no detection box is
        ever drawn.
        按 14B §10 组装唯一最终 Qwen 用户内容：先干净 ROI 图像、再逐 ROI 掩膜
        overlay、后文本证据（问题、答案约束、图像尺寸、ROI 裁切几何、YOLO
        文本记录、SegFormer 图例）。confidence 绝不出现，绝不绘制检测框。"""
        region_by_id = {region.roi_id: region for region in plan.roi_plan.rois}
        content: list[dict[str, Any]] = []
        final_hashes: list[str] = []
        roi_geometry: list[dict[str, Any]] = []
        for record in bundle.rois:
            region = region_by_id.get(record.roi_id)
            if region is None:
                # The bundle's unique full-image ROI when the plan carried
                # none (14B §10: 无可靠空间约束 -> 唯一整图 ROI).
                # 计划无 ROI 时 bundle 的唯一整图 ROI。
                if record.roi_id == "full" and not region_by_id:
                    region = RoiRegion(
                        roi_id="full",
                        image_id=record.image_id,
                        xyxy=(0.0, 0.0, 1.0, 1.0),
                    )
                else:
                    raise ValueError(
                        f"bundle references unknown roi_id {record.roi_id!r}"
                    )
            clean = make_preview(
                render_roi_crop(images[record.image_id], region, record)
            )
            content.append(self._image_block(clean, final_hashes))
            roi_geometry.append(
                {
                    "roi_id": record.roi_id,
                    "image_id": record.image_id,
                    "source_size": list(record.source_size),
                    "crop_xyxy": list(record.expanded_xyxy),
                    "crop_size": list(record.crop_size),
                }
            )
            # Per-ROI overlays follow the clean image, in catalog leaf order
            # (14B §10.2); masks never blend across ROIs.
            # 逐 ROI overlay 紧跟干净图，按目录叶子顺序（14B §10.2）；掩膜跨
            # ROI 绝不融合。
            for leaf in bundle.leaf_states:
                mask = masks.get((record.roi_id, leaf))
                if mask is None:
                    continue
                overlay = overlay_mask(
                    clean,
                    self._mask_to_image(mask, clean.size),
                    color=stable_palette_color(leaf),
                )
                content.append(self._image_block(overlay, final_hashes))
        payload = dict(self.build_user_payload(sample))
        payload["images"] = [
            {
                "image_id": image_ref.image_id,
                "width": images[image_ref.image_id].size[0],
                "height": images[image_ref.image_id].size[1],
            }
            for image_ref in sample.images
        ]
        payload["rois"] = roi_geometry
        payload["yolo_detections"] = [
            {
                "leaf_category": record.leaf_category,
                "roi_id": record.roi_id,
                "local_xyxy": list(record.local_xyxy),
                "global_xyxy": list(record.global_xyxy),
            }
            for record in bundle.detections
        ]
        payload["segformer_hits"] = [
            {"roi_id": record.roi_id, "leaf_category": record.leaf_category}
            for record in bundle.segments
        ]
        payload["segformer_legend"] = [
            {"leaf_category": leaf, "color_rgb": list(stable_palette_color(leaf))}
            for leaf in sorted({leaf for (_, leaf) in masks})
        ]
        payload["missing_leaves"] = list(bundle.missing_leaves)
        content.append(
            {"type": "text", "text": json.dumps(payload, ensure_ascii=False)}
        )
        return content, final_hashes

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

    @staticmethod
    def _mask_to_image(mask: Any, size: tuple[int, int]) -> Image.Image:
        """Convert an in-memory presence mask (duck-typed boolean array) into
        a grayscale image sized to the preview; NEAREST keeps presence pixels
        boolean-shaped when the preview shrank the crop. 将内存存在掩膜（鸭子
        类型布尔数组）转换为匹配预览尺寸的灰度图像；预览缩小裁切时用 NEAREST
        保持存在性像素接近布尔。"""
        shape = getattr(mask, "shape", None)
        if not isinstance(shape, tuple) or len(shape) != 2:
            raise ValueError("presence mask must expose a 2-D shape")
        height, width = shape
        raw = mask.tobytes()
        if len(raw) != width * height:
            raise ValueError("presence mask bytes do not match its shape")
        image = Image.frombytes("L", (width, height), raw)
        if image.size != size:
            image = image.resize(size, Image.Resampling.NEAREST)
        return image


def _structured_prompt(prompt: PromptBinding, agent_name: str) -> str:
    """Frozen structured-output suffix, verbatim mirror of
    VisualAgentBase.run() so the evidence path's final call stays
    format-identical with the legacy direct path.
    冻结结构化输出后缀，逐字镜像 VisualAgentBase.run()，使证据路径的最终调用
    与旧直接路径格式一致。"""
    return (
        prompt.text
        + f"\n\nReturn valid JSON only. Set agent_name to {agent_name!r}; "
        "put the concise final answer in answer, retain relevant labeled boxes or points "
        "in evidence_items, copy evidence boxes into boxes, use concise factual evidence "
        "strings, and set status to 'completed'."
    )


def _choice_constraints(sample: UnifiedSample) -> dict[str, Any]:
    """Answer constraints for a multiple-choice sample.
    多选题样本的答案约束。"""
    if sample.normalization is None:
        return {}
    return sample.normalization.answer_constraints


def _normalize_choice(value: str) -> str:
    return value.strip().casefold()


def _choice_text(choice: str) -> str:
    """Strip a leading option letter prefix (A., B), C - ...).
    去除选项首字母前缀（A.、B)、C - ...）。"""
    text = choice.strip()
    match = re.match(r"^([A-Za-z])[.)、\s-]\s*(.*)$", text)
    if match:
        return match.group(2).strip()
    return text


def _match_choice(value: str, choices: list[str]) -> str | None:
    """Match an answer item to a choice: by normalized full text, by the
    prefix-stripped text, or by the leading option letter (A/B/...).
    将答案项匹配到选项：按归一化全文、去前缀文本或选项首字母（A/B/...）。"""
    normalized = _normalize_choice(value)
    for choice in choices:
        if _normalize_choice(choice) == normalized:
            return choice
    for choice in choices:
        if _normalize_choice(_choice_text(choice)) == normalized:
            return choice
    letter = value.strip().upper()
    if len(letter) == 1 and "A" <= letter <= "Z":
        for choice in choices:
            if choice.strip().upper() == letter:
                return choice
            if _normalize_choice(choice).startswith(letter.lower()):
                return choice
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


def _extract_choices(constraints: dict[str, Any]) -> list[str]:
    """Extract string choices from answer constraints. The constraints are
    answer-domain restrictions (e.g. closed_vocabulary values), never the
    ground truth itself, so nothing is leaked.
    从答案约束提取字符串选项。约束是答案域限制（如 closed_vocabulary
    values），本身并非 ground truth，因此不泄漏任何内容。"""
    for key in ("choices", "values"):
        value = constraints.get(key)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return list(value)
    return []
