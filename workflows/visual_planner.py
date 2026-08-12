"""Isolated first-Qwen visual planner (C4, 14A1) — not wired to SampleRunner.

孤立的第一 Qwen 视觉规划器（C4，14A1）——尚未接入 SampleRunner。

Pipeline (order is immutable: SampleDraft -> TaskResolver -> UnifiedSample ->
VisualPlanner): 管线（顺序不可变：SampleDraft -> TaskResolver -> UnifiedSample
-> VisualPlanner）：

    UnifiedSample
      -> safe preview(s)
      -> question + answer-domain constraints + same-version closed catalog
      -> exactly one schema-validated Qwen call
      -> FirstQwenVisualPlan

Hard constraints / 硬约束：

- depends only on VisionLanguageClient, the injected prompt/version, and the
  shared plan schema + evidence catalog; 只依赖 VisionLanguageClient、注入的
  prompt/version、共享 plan schema 与证据目录；
- never reads Ground Truth or dataset-specific JSON, never selects a model,
  never changes sample.task; 绝不读取 Ground Truth 或 dataset-specific JSON，
  不选择具体模型，绝不改变 sample.task；
- a real ModelCacheIdentity is verified before the call and exactly one Qwen
  budget entry is consumed; 调用前验证真实 ModelCacheIdentity，且恰好消费
  一次 Qwen budget；
- the request hash covers prompt/schema/messages/image digest/generation/client
  version/logical model identity/revision/catalog version; request hash 覆盖
  prompt/schema/messages/图片摘要/generation/client version/logical model
  identity/revision/catalog version；
- artifacts/requests never carry Base64, secrets, raw model bodies, or absolute
  image paths; 产物/请求绝不携带 Base64、secret、原始模型正文或绝对图像路径；
- strict rejection of extra fields, out-of-catalog categories, degenerate ROIs,
  wrong image ids, and non-finite values. 严格拒绝额外字段、目录外类别、退化
  ROI、错误 image id 与非 finite 值。

Unfrozen policies (typed failure seam ONLY — no production defaults, no
"existing agent + full image" fallback): schema invalid, low confidence, client
unavailable/error, budget exhausted, preview decode failure. Each surfaces as a
stable VisualPlanError code; 14A2 must freeze the real policy before wiring the
SampleRunner. 未冻结策略（仅 typed failure seam——无生产默认值、无“现有
Agent + 整图”回退）：schema invalid、low confidence、client unavailable/error、
budget exhausted、preview decode failure。每条都以稳定 VisualPlanError code
呈现；接入 SampleRunner 前须由 14A2 冻结真实策略。
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from agents.base import CallBudget, VisualPlanBindings
from agents.evidence_catalog import CatalogCategoryError, EvidenceCatalog
from agents.general_vqa.evidence.geometry import MAX_MODEL_SIDE
from agents.general_vqa.evidence.rendering import preview_from_path
from agents.schema import FirstQwenVisualPlan, ObjectEvidenceRequest
from data.schema import UnifiedSample
from models.base import (
    MissingModelCacheIdentityError,
    RequestMeta,
    VisionLanguageClient,
    build_request_hash,
    require_model_cache_identity,
)


class VisualPlanError(ValueError):
    """Stable error for visual-planning failures; the public message carries
    only the stable code, never raw model text, image bytes, or paths.
    视觉规划失败的稳定错误；公共消息只携带稳定 code，绝不携带原始模型文本、
    图像字节或路径。"""

    def __init__(self, code: str) -> None:
        super().__init__(f"VISUAL_PLAN_FAILED:{code}")
        self.code = code


class VisualPlanner:
    """Produce one strict FirstQwenVisualPlan from a UnifiedSample with
    exactly one schema-validated model call. The planner is an orchestration
    service, not a business agent: it is not a member of the AgentRegistry
    and it never executes downstream evidence work.
    从 UnifiedSample 通过恰好一次 schema 校验的模型调用产出一条严格
    FirstQwenVisualPlan。规划器是编排服务而非业务 Agent：不是 AgentRegistry
    成员，绝不执行下游证据工作。"""

    def __init__(
        self,
        client: VisionLanguageClient,
        *,
        system_prompt: str,
        prompt_version: str = "v1",
        catalog: EvidenceCatalog,
        confidence_threshold: float = 0.70,
        max_side: int = MAX_MODEL_SIDE,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0.0, 1.0]")
        self._client = client
        self._system_prompt = system_prompt
        self._prompt_version = prompt_version
        self._catalog = catalog
        self._confidence_threshold = confidence_threshold
        self._max_side = max_side

    async def plan(
        self,
        sample: UnifiedSample,
        *,
        data_root: Path,
        artifact_dir: Path,
        budget: CallBudget | None = None,
    ) -> FirstQwenVisualPlan:
        """One schema-validated Qwen call producing the plan. The identity is
        required up front; the budget is consumed only when a model call is
        actually attempted. 一次 schema 校验的 Qwen 调用产出计划。身份必须
        前置验证；只在真正尝试模型调用时才消费 budget。"""
        try:
            identity = require_model_cache_identity(
                self._client, component="visual_planner"
            )
        except MissingModelCacheIdentityError as exc:
            raise VisualPlanError("CLIENT_UNAVAILABLE") from exc

        data_urls, image_hashes = self._safe_previews(sample, data_root)

        user_payload = {
            "question": sample.question,
            "images": [
                {"image_id": image_ref.image_id, "role": image_ref.role}
                for image_ref in sample.images
            ],
            "catalog_version": self._catalog.catalog_version,
            # The closed category list is data-driven (not baked into the
            # static prompt) so the request hash covers the catalog version
            # and its contents, and the model can only pick from it.
            # 封闭类别列表由数据驱动（不写死在静态 prompt 里），使 request
            # hash 覆盖目录版本及其内容，模型只能从中挑选。
            "composite_categories": list(self._catalog.composite_categories),
            "answer_constraints": (
                sample.normalization.answer_constraints
                if sample.normalization is not None
                else {}
            ),
        }
        content: list[dict[str, object]] = [
            *[
                {"type": "image_url", "image_url": {"url": url}}
                for url in data_urls
            ],
            {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)},
        ]
        messages: list[dict[str, object]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": content},
        ]
        image_digest = "|".join(image_hashes)
        request_hash = build_request_hash(
            model=identity.model,
            generation=identity.generation_payload(),
            prompt_version=self._prompt_version,
            messages=messages,
            image_sha256=image_digest,
            response_schema=FirstQwenVisualPlan.model_json_schema(),
            client_version=identity.client_version,
            model_revision=identity.revision,
        )
        # The budget is consumed only when a model call is actually attempted;
        # identity/preview/hash failures never reserve a call.
        # 只在真正尝试模型调用时才消费 budget；身份/预览/哈希失败绝不预留调用。
        if budget is not None:
            try:
                budget.reserve_qwen()
            except Exception as exc:
                raise VisualPlanError("BUDGET_EXHAUSTED") from exc
        try:
            plan = await self._client.complete_json(
                messages=messages,
                response_model=FirstQwenVisualPlan,
                request_meta=RequestMeta(
                    request_id=f"{sample.sample_id}:visual_plan",
                    request_hash=request_hash,
                    prompt_version=self._prompt_version,
                    sample_id=sample.sample_id,
                    image_sha256=image_digest,
                    artifact_dir=artifact_dir / "visual_plan",
                ),
            )
        except ValidationError as exc:
            # The model returned a plan the strict schema rejects: extra
            # fields, degenerate/non-finite ROIs, wrong family linkage, etc.
            # 模型返回的计划被严格 schema 拒绝：额外字段、退化/非有限 ROI、
            # 家族联动错误等。
            raise VisualPlanError("SCHEMA_INVALID") from exc
        except Exception as exc:
            raise VisualPlanError("CLIENT_ERROR") from exc
        return self._post_validate(plan, sample)

    # ── previews / 预览 ─────────────────────────────────────────────────

    def _safe_previews(
        self,
        sample: UnifiedSample,
        data_root: Path,
    ) -> tuple[list[str], list[str]]:
        """Read every sample image through data_root (escape-guarded),
        normalize to a safe in-memory preview through the agents-side rendering
        seam (which owns the models.images dependency), and return the model
        data URLs plus their honest digests. Failures map to the
        PREVIEW_DECODE_FAILED seam; absolute machine paths never enter the
        request or any artifact.
        通过 data_root（防逃逸）读取每条样本图像，经 agents 侧渲染 seam（该
        层持有 models.images 依赖）规范化为安全内存预览，返回模型 data URL
        及其真实摘要。失败映射到 PREVIEW_DECODE_FAILED seam；机器绝对路径
        绝不进入请求或任何产物。"""
        root = data_root.resolve()
        data_urls: list[str] = []
        hashes: list[str] = []
        for image_ref in sample.images:
            candidate = (root / image_ref.path).resolve()
            if not candidate.is_relative_to(root):
                raise VisualPlanError("PREVIEW_DECODE_FAILED")
            try:
                data_url, digest = preview_from_path(
                    candidate, max_side=self._max_side
                )
            except (OSError, ValueError) as exc:
                raise VisualPlanError("PREVIEW_DECODE_FAILED") from exc
            data_urls.append(data_url)
            hashes.append(digest)
        return data_urls, hashes

    # ── post-validation / 后校验 ────────────────────────────────────────

    def _post_validate(
        self,
        plan: FirstQwenVisualPlan,
        sample: UnifiedSample,
    ) -> FirstQwenVisualPlan:
        """Planner-side strict checks the schema cannot express: categories
        must belong to the same-version closed catalog, every ROI image_id
        must reference a sample image, duplicates are deduplicated stably, and
        low confidence is a typed failure until a policy is frozen.
        Schema 无法表达的规划器侧严格检查：类别必须属于同版本封闭目录、每个
        ROI 的 image_id 必须引用样本图像、重复类别稳定去重，且低置信度在策略
        冻结前作为 typed failure。"""
        if plan.evidence_request is not None:
            categories = plan.evidence_request.composite_categories
            try:
                self._catalog.validate_plan_categories(categories)
            except CatalogCategoryError as exc:
                raise VisualPlanError("SCHEMA_INVALID") from exc
            # Stable dedupe preserving first occurrence; the schema enforces
            # only the count, the planner enforces the closed-set membership.
            # 稳定去重并保留首次出现顺序；schema 只强制数量，封闭集合归属由
            # 规划器强制。
            if len(categories) != len(set(categories)):
                deduped: list[str] = []
                for category in categories:
                    if category not in deduped:
                        deduped.append(category)
                plan = plan.model_copy(
                    update={
                        "evidence_request": ObjectEvidenceRequest(
                            composite_categories=deduped
                        )
                    }
                )
        known_ids = {image_ref.image_id for image_ref in sample.images}
        for region in plan.roi_plan.rois:
            if region.image_id not in known_ids:
                raise VisualPlanError("SCHEMA_INVALID")
        if plan.confidence < self._confidence_threshold:
            # The low-confidence policy (retry / fallback / reject) is not
            # frozen; C4 only surfaces the typed failure.
            # 低置信度策略（重试 / 回退 / 拒绝）未冻结；C4 只暴露 typed failure。
            raise VisualPlanError("LOW_CONFIDENCE")
        return plan


class VisualPlanningGate:
    """Feature-flagged planning gate (C7, 14A2): exactly one planner call per
    sample for the approved compatibility-matrix tasks, strict
    VisualPlanError on any failure, and None for tasks outside the matrix —
    the gate never invents a plan for caption/change/counting tasks.
    特性开关式规划门（C7，14A2）：对兼容矩阵批准的规划任务每条样本恰好一次
    规划调用，任何失败严格抛 VisualPlanError；对矩阵之外的任务
    （caption/change/counting）返回 None——gate 绝不为它们杜撰计划。"""

    # Approved planning tasks of the compatibility matrix (14A2 gate 1): the
    # VQA family plus grounding; counting later receives only validated target
    # hints, caption/change receive no evidence at all.
    # 兼容矩阵批准的规划任务（14A2 门禁 1）：VQA 家族 + grounding；counting
    # 之后只收已校验 target hint，caption/change 完全无证据。
    PLANNING_TASKS = frozenset({
        "general_vqa",
        "scene_classification",
        "multiple_choice_vqa",
        "spatial_relation",
        "grounding",
    })

    def __init__(
        self,
        planner: VisualPlanner,
        *,
        bindings: VisualPlanBindings | None = None,
    ) -> None:
        self._planner = planner
        self.bindings = bindings

    async def plan_sample(
        self,
        sample: UnifiedSample,
        *,
        data_root: Path,
        artifact_dir: Path,
        budget: CallBudget | None = None,
    ) -> FirstQwenVisualPlan | None:
        """One planner call for approved planning tasks; None for every other
        task, so the sample keeps its legacy path untouched.
        对已批准规划任务执行一次规划调用；其他任务返回 None，样本保持旧路径
        不变。"""
        if sample.task not in self.PLANNING_TASKS:
            return None
        return await self._planner.plan(
            sample,
            data_root=data_root,
            artifact_dir=artifact_dir,
            budget=budget,
        )
