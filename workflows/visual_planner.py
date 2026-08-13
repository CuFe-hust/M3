"""Isolated first-Qwen visual planner (C4, 14A1) and its feature-flagged
SampleRunner gate (C7, 14A2). The planner is an orchestration service that
never executes downstream evidence work; the gate is wired only when the
composition root injects it (production assembly is deferred to 14A3).

孤立的第一 Qwen 视觉规划器（C4，14A1）及其特性开关式 SampleRunner 门
（C7，14A2）。规划器是绝不执行下游证据工作的编排服务；门只有在组合根注入时
才接入（生产组装延后到 14A3）。

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
- strict rejection of extra fields, out-of-catalog categories, wrong image
  ids, and non-finite values; per 14B §6.2 an over-limit, out-of-range, or
  degenerate ROI plan collapses to the unique full-image ROI (the category
  plan survives). 严格拒绝额外字段、目录外类别、错误 image id 与非 finite 值；
  按 14B §6.2，超限、越界或退化的 ROI 计划折叠为唯一整图 ROI（类别计划保留）。

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
from typing import get_args

from pydantic import ValidationError

from agents.base import CallBudget, VisualPlanBindings
from agents.evidence_catalog import CatalogCategoryError, EvidenceCatalog
from agents.general_vqa.evidence.geometry import MAX_MODEL_SIDE
from agents.general_vqa.evidence.rendering import preview_from_path
from agents.schema import (
    FirstQwenVisualPlan,
    JointQwenVisualPlan,
    ObjectEvidenceRequest,
    RoiPlan,
)
from data.schema import SampleDraft, TaskName, UnifiedSample
from models.base import (
    MissingModelCacheIdentityError,
    RequestMeta,
    VisionLanguageClient,
    build_request_hash,
    require_model_cache_identity,
)

# Closed set of legal task names; the model can only pick from this set.
# 合法任务名的封闭集合；模型只能从中挑选。
_ALL_TASK_NAMES = frozenset(get_args(TaskName))


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
        max_rois: int = 3,
        max_side: int = MAX_MODEL_SIDE,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0.0, 1.0]")
        if not 1 <= max_rois <= 3:
            raise ValueError("max_rois must be within [1, 3]")
        self._client = client
        self._system_prompt = system_prompt
        self._prompt_version = prompt_version
        self._catalog = catalog
        self._confidence_threshold = confidence_threshold
        self._max_rois = max_rois
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
            # fields, non-finite ROIs, wrong family linkage, etc. Finite but
            # invalid geometry is handled by planner-side full-image fallback.
            # 模型返回的计划被严格 schema 拒绝：额外字段、非有限 ROI、
            # 家族联动错误等。有限但几何无效的 ROI 由 Planner 回退整图。
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
        low confidence is a typed failure until a policy is frozen. Per 14B
        §6.2, an over-limit, out-of-range, or degenerate ROI plan collapses to
        the unique full-image ROI (empty plan = full image at the geometry
        layer); the already-valid category plan is preserved, nothing is
        truncated, and the planning Qwen is never re-called.
        Schema 无法表达的规划器侧严格检查：类别必须属于同版本封闭目录、每个
        ROI 的 image_id 必须引用样本图像、重复类别稳定去重，且低置信度在策略
        冻结前作为 typed failure。按 14B §6.2，超限、越界或退化的 ROI 计划折
        叠为唯一整图 ROI（空计划在几何层即整图）；保留已合法解析的类别计划，
        绝不截断，也绝不重调规划 Qwen。"""
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
        if self._needs_full_image_fallback(plan):
            # 14B §6.2: the whole ROI plan is void; the unique full-image ROI
            # replaces it (empty roi_plan), while the validated category plan
            # survives intact. 14B §6.2：整个 ROI 计划失效；唯一整图 ROI 取
            # 代之（空 roi_plan），已校验类别计划原样保留。
            plan = plan.model_copy(
                update={"roi_plan": RoiPlan(rois=[])}
            )
        if plan.confidence < self._confidence_threshold:
            # The low-confidence policy (retry / fallback / reject) is not
            # frozen; C4 only surfaces the typed failure.
            # 低置信度策略（重试 / 回退 / 拒绝）未冻结；C4 只暴露 typed failure。
            raise VisualPlanError("LOW_CONFIDENCE")
        return plan

    def _needs_full_image_fallback(self, plan: FirstQwenVisualPlan) -> bool:
        """14B §6.2 verdict on the ROI plan: over the configured cap, outside
        the normalized [0,1] frame, or degenerate (zero extent) — any of these
        voids the whole ROI plan and triggers the unique full-image fallback.
        14B §6.2 对 ROI 计划的判定：超过配置上限、越出归一化 [0,1] 制式或退化
        （零范围）——任一情况使整个 ROI 计划失效并触发唯一整图回退。"""
        if len(plan.roi_plan.rois) > self._max_rois:
            return True
        for region in plan.roi_plan.rois:
            x1, y1, x2, y2 = region.xyxy
            if not (0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0):
                return True
            if not (0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0):
                return True
            if x1 >= x2 or y1 >= y2:
                return True
        return False


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


# ── Joint task + visual planning (doc 15) ────────────────────────────────
# One schema-validated Qwen call replaces the former two-stage
# (text TaskResolver -> VisualPlanner) decision for every entry point. The
# planner accepts a pre-routing view (SampleDraft or UnifiedSample) and
# returns the authoritative execution task plus the reusable visual-plan
# substructure. The joint planner is an orchestration service: it never runs
# agents, detectors, segmenters, or final Q&A. 联合任务 + 视觉规划（doc 15）。
# 单次 schema 校验的 Qwen 调用取代所有入口先前的两阶段决策（文本
# TaskResolver -> VisualPlanner）。规划器接收物化前视图（SampleDraft 或
# UnifiedSample），返回权威执行 task 加可复用视觉计划子结构。联合规划器是
# 编排服务：绝不执行 Agent、detector、segmenter 或最终问答。

class JointPlanError(ValueError):
    """Stable error for joint planning failures; the public message carries
    only the stable code, never raw model text, image bytes, or paths.
    联合规划失败的稳定错误；公共消息只携带稳定 code，绝不携带原始模型文本、
    图像字节或路径。"""

    def __init__(self, code: str) -> None:
        super().__init__(f"JOINT_PLAN_FAILED:{code}")
        self.code = code


class JointVisualPlanner:
    """Produce one strict JointQwenVisualPlan from a pre-routing view with
    exactly one schema-validated model call: preview(s) + text -> task +
    visual plan. The returned task is authoritative for materialization,
    routing, and execution; a dataset-supplied source task is never sent to
    the model and never overrides it. The planner is an orchestration
    service, not a business agent: it never executes downstream evidence
    work. 从物化前视图通过恰好一次 schema 校验的模型调用产出一条严格
    JointQwenVisualPlan：缩略图 + 文本 -> task + 视觉计划。返回的 task 对
    物化、路由与执行权威；数据集提供的来源 task 绝不发给模型也绝不覆盖它。
    规划器是编排服务而非业务 Agent：绝不执行下游证据工作。"""

    def __init__(
        self,
        client: VisionLanguageClient,
        *,
        system_prompt: str,
        prompt_version: str = "v1",
        catalog: EvidenceCatalog,
        confidence_threshold: float = 0.70,
        max_rois: int = 3,
        max_side: int = MAX_MODEL_SIDE,
        bindings: VisualPlanBindings | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0.0, 1.0]")
        if not 1 <= max_rois <= 3:
            raise ValueError("max_rois must be within [1, 3]")
        self._client = client
        self._system_prompt = system_prompt
        self._prompt_version = prompt_version
        self._catalog = catalog
        self._confidence_threshold = confidence_threshold
        self._max_rois = max_rois
        self._max_side = max_side
        self.bindings = bindings

    async def plan(
        self,
        view: SampleDraft | UnifiedSample,
        *,
        data_root: Path,
        artifact_dir: Path,
        budget: CallBudget | None = None,
    ) -> JointQwenVisualPlan:
        """One schema-validated Qwen call producing the joint plan. The
        identity is required up front; the budget is consumed only when a
        model call is actually attempted. 一次 schema 校验的 Qwen 调用产出
        联合计划。身份必须前置验证；只在真正尝试模型调用时才消费 budget。"""
        try:
            identity = require_model_cache_identity(
                self._client, component="joint_visual_planner"
            )
        except MissingModelCacheIdentityError as exc:
            raise JointPlanError("CLIENT_UNAVAILABLE") from exc

        data_urls, image_hashes = self._safe_previews(view, data_root)

        answer_constraints = (
            view.normalization.answer_constraints
            if isinstance(view, UnifiedSample)
            and view.normalization is not None
            else {}
        )
        user_payload = {
            "question": view.question,
            "images": [
                {"image_id": image_ref.image_id, "role": image_ref.role}
                for image_ref in view.images
            ],
            "catalog_version": self._catalog.catalog_version,
            # The closed category list and the allowed task set are
            # data-driven (not baked into the static prompt) so the request
            # hash covers the catalog version, its contents, and the task
            # set; the model can only pick from them.
            # 封闭类别列表与允许任务集合由数据驱动（不写死在静态 prompt 里），
            # 使 request hash 覆盖目录版本、其内容与任务集合；模型只能从中挑选。
            "composite_categories": list(self._catalog.composite_categories),
            "allowed_tasks": sorted(_ALL_TASK_NAMES),
            "answer_constraints": answer_constraints,
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
            response_schema=JointQwenVisualPlan.model_json_schema(),
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
                raise JointPlanError("BUDGET_EXHAUSTED") from exc
        try:
            plan = await self._client.complete_json(
                messages=messages,
                response_model=JointQwenVisualPlan,
                request_meta=RequestMeta(
                    request_id=f"{view.sample_id}:joint_plan",
                    request_hash=request_hash,
                    prompt_version=self._prompt_version,
                    sample_id=view.sample_id,
                    image_sha256=image_digest,
                    artifact_dir=artifact_dir / "joint_plan",
                ),
            )
        except ValidationError as exc:
            # The model returned a plan the strict schema rejects: extra
            # fields, non-finite ROIs, wrong family linkage, an unknown task,
            # etc. Finite but invalid geometry is handled by planner-side
            # full-image fallback. 模型返回的计划被严格 schema 拒绝：额外
            # 字段、非有限 ROI、家族联动错误、未知 task 等。有限但几何无效
            # 的 ROI 由 Planner 回退整图。
            raise JointPlanError("SCHEMA_INVALID") from exc
        except Exception as exc:
            raise JointPlanError("CLIENT_ERROR") from exc
        return self._post_validate(plan, view)

    # ── previews / 预览 ─────────────────────────────────────────────────

    def _safe_previews(
        self,
        view: SampleDraft | UnifiedSample,
        data_root: Path,
    ) -> tuple[list[str], list[str]]:
        """Read every view image through data_root (escape-guarded), normalize
        to a safe in-memory preview through the agents-side rendering seam
        (which owns the models.images dependency), and return the model data
        URLs plus their honest digests. Failures map to the
        PREVIEW_DECODE_FAILED seam; absolute machine paths never enter the
        request or any artifact.
        通过 data_root（防逃逸）读取每条视图图像，经 agents 侧渲染 seam（该
        层持有 models.images 依赖）规范化为安全内存预览，返回模型 data URL
        及其真实摘要。失败映射到 PREVIEW_DECODE_FAILED seam；机器绝对路径
        绝不进入请求或任何产物。"""
        root = data_root.resolve()
        data_urls: list[str] = []
        hashes: list[str] = []
        for image_ref in view.images:
            candidate = (root / image_ref.path).resolve()
            if not candidate.is_relative_to(root):
                raise JointPlanError("PREVIEW_DECODE_FAILED")
            try:
                data_url, digest = preview_from_path(
                    candidate, max_side=self._max_side
                )
            except (OSError, ValueError) as exc:
                raise JointPlanError("PREVIEW_DECODE_FAILED") from exc
            data_urls.append(data_url)
            hashes.append(digest)
        return data_urls, hashes

    # ── post-validation / 后校验 ────────────────────────────────────────

    def _post_validate(
        self,
        plan: JointQwenVisualPlan,
        view: SampleDraft | UnifiedSample,
    ) -> JointQwenVisualPlan:
        """Planner-side strict checks the schema cannot express, mirroring the
        frozen 14B §6.2 rules on the visual-plan substructure: categories must
        belong to the same-version closed catalog, every ROI image_id must
        reference a view image, duplicates are deduplicated stably, and low
        confidence is a typed failure until a policy is frozen. An
        over-limit, out-of-range, or degenerate ROI plan collapses to the
        unique full-image ROI (empty plan = full image at the geometry
        layer); the already-valid category plan is preserved, nothing is
        truncated, and the planning Qwen is never re-called.
        Schema 无法表达的规划器侧严格检查，镜像冻结的 14B §6.2 规则作用于
        视觉计划子结构：类别必须属于同版本封闭目录、每个 ROI 的 image_id
        必须引用视图图像、重复类别稳定去重，且低置信度在策略冻结前作为
        typed failure。超限、越界或退化的 ROI 计划折叠为唯一整图 ROI（空
        计划在几何层即整图）；保留已合法解析的类别计划，绝不截断，也绝不
        重调规划 Qwen。"""
        visual_plan = plan.visual_plan
        if visual_plan.evidence_request is not None:
            categories = visual_plan.evidence_request.composite_categories
            try:
                self._catalog.validate_plan_categories(categories)
            except CatalogCategoryError as exc:
                raise JointPlanError("SCHEMA_INVALID") from exc
            if len(categories) != len(set(categories)):
                deduped: list[str] = []
                for category in categories:
                    if category not in deduped:
                        deduped.append(category)
                visual_plan = visual_plan.model_copy(
                    update={
                        "evidence_request": ObjectEvidenceRequest(
                            composite_categories=deduped
                        )
                    }
                )
        known_ids = {image_ref.image_id for image_ref in view.images}
        for region in visual_plan.roi_plan.rois:
            if region.image_id not in known_ids:
                raise JointPlanError("SCHEMA_INVALID")
        if self._needs_full_image_fallback(visual_plan):
            # 14B §6.2: the whole ROI plan is void; the unique full-image ROI
            # replaces it (empty roi_plan), while the validated category plan
            # survives intact. 14B §6.2：整个 ROI 计划失效；唯一整图 ROI 取
            # 代之（空 roi_plan），已校验类别计划原样保留。
            visual_plan = visual_plan.model_copy(
                update={"roi_plan": RoiPlan(rois=[])}
            )
        if visual_plan is not plan.visual_plan:
            plan = plan.model_copy(update={"visual_plan": visual_plan})
        if plan.visual_plan.confidence < self._confidence_threshold:
            # The low-confidence policy (retry / fallback / reject) is not
            # frozen; doc 15 only surfaces the typed failure.
            # 低置信度策略（重试 / 回退 / 拒绝）未冻结；doc 15 只暴露
            # typed failure。
            raise JointPlanError("LOW_CONFIDENCE")
        return plan

    def _needs_full_image_fallback(self, plan: FirstQwenVisualPlan) -> bool:
        """14B §6.2 verdict on the ROI plan, identical to the isolated
        planner: over the configured cap, outside the normalized [0,1] frame,
        or degenerate (zero extent) — any of these voids the whole ROI plan
        and triggers the unique full-image fallback.
        14B §6.2 对 ROI 计划的判定，与孤立规划器一致：超过配置上限、越出
        归一化 [0,1] 制式或退化（零范围）——任一情况使整个 ROI 计划失效并
        触发唯一整图回退。"""
        if len(plan.roi_plan.rois) > self._max_rois:
            return True
        for region in plan.roi_plan.rois:
            x1, y1, x2, y2 = region.xyxy
            if not (0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0):
                return True
            if not (0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0):
                return True
            if x1 >= x2 or y1 >= y2:
                return True
        return False
