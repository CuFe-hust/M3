"""Canonical visual-only task planner and deterministic view materialization.
规范纯视觉任务规划器与确定性视图物化。

The v5 planner is the only fresh-inference planning seam. It performs one
schema-validated call, then materializes exact full-image or quantized-ROI
views. v5 规划器是所有新鲜推理唯一的规划 seam：执行一次 schema 校验调用，
然后物化精确整图或量化 ROI 视图。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import get_args

from pydantic import ValidationError

from agents.base import CallBudget
from agents.evidence_catalog import CatalogCategoryError, EvidenceCatalog
from agents.general_vqa.evidence.rendering import (
    materialize_quantized_roi,
    normalized_image_size,
    preview_from_path,
)
from agents.schema import (
    COUNTING_TASKS,
    GENERAL_VQA_AGENT_TASKS,
    MaterializedVisualView,
    VisualTaskPlan,
)
from data.schema import SampleDraft, TaskName, UnifiedSample
from models.base import (
    MissingModelCacheIdentityError,
    RequestMeta,
    VisionLanguageClient,
    build_request_hash,
    require_model_cache_identity,
)

_ALL_TASK_NAMES = frozenset(get_args(TaskName))
# Tasks whose plans may request the shared VQA evidence capability. The four
# GeneralVQAAgent tasks map to one catalog capability owner below; this list
# only declares that these tasks participate in VQA capability planning.
# 可以请求共享 VQA evidence 能力的 task 集合。四个 GeneralVQAAgent task 在
# 下方映射到同一个 catalog capability owner；本列表只声明这些 task 参与 VQA
# 能力规划。
_VISUAL_CAPABILITY_TASKS = (
    "counting",
    "fine_grained_counting",
    *sorted(GENERAL_VQA_AGENT_TASKS),
    "grounding",
)
# Single shared capability owner for the GeneralVQAAgent tasks: the catalog
# declares one VQA capability family (general_vqa) and the planner resolves
# every GeneralVQAAgent task to it for leaf validation. This is a catalog
# lookup mapping only — never an assistance true/false decision table.
# GeneralVQAAgent task 共享的单一 capability owner：目录只声明一个 VQA 能力
# 族（general_vqa），planner 把所有 GeneralVQAAgent task 解析到它做 leaf 校验。
# 这仅仅是目录查找映射——绝不是 assistance true/false 决策表。
_VQA_CAPABILITY_OWNER = "general_vqa"
_ROI_COORDINATE_FRAME = "normalized_0_999_top_left"
_ROI_MATERIALIZATION_POLICY = "longest-side-ceil-quantum-center-clip"
# Stable scope identifiers use the same conservative pattern as frozen
# catalog versions. / 稳定 scope 标识符与冻结 catalog 版本使用同一保守模式。
_SCOPE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def _vqa_capability_owner(task: str) -> str:
    """Resolve the catalog capability owner for one task: every GeneralVQAAgent
    task shares the single VQA capability family; other tasks keep their own
    catalog entry. This is a catalog lookup mapping only — it never decides
    whether assistance is needed.
    解析一个 task 的 catalog capability owner：全部 GeneralVQAAgent task 共享
    唯一 VQA 能力族；其他 task 使用自己的目录条目。这只是目录查找映射——
    绝不决定是否需要辅助。"""
    if task in GENERAL_VQA_AGENT_TASKS:
        return _VQA_CAPABILITY_OWNER
    return task


class VisualTaskPlanError(ValueError):
    """Stable failure for the visual-only task planner.
    纯视觉任务规划器的稳定失败类型。"""

    def __init__(self, code: str) -> None:
        super().__init__(f"VISUAL_TASK_PLAN_FAILED:{code}")
        self.code = code


class VisualTaskPlanner:
    """Plan task and optional visual assistance from previews plus raw text.
    仅依据预览图像与原始文本规划任务及可选视觉辅助。"""

    def __init__(
        self,
        client: VisionLanguageClient,
        *,
        system_prompt: str,
        prompt_version: str = "v5",
        catalog: EvidenceCatalog,
        executable_categories_by_task: Mapping[str, tuple[str, ...]] | None = None,
        max_side: int = 1080,
        roi_quantum: int = 1024,
        roi_coordinate_frame: str = _ROI_COORDINATE_FRAME,
        roi_materialization_policy: str = _ROI_MATERIALIZATION_POLICY,
        large_image_policy: str = "both-dimensions-strictly-greater-than-1024",
        vqa_assistance_scope: str | None = None,
    ) -> None:
        if max_side <= 0 or roi_quantum <= 0:
            raise ValueError("preview and ROI sizes must be positive")
        if prompt_version != "v5":
            raise ValueError("visual task planner prompt_version must be v5")
        if roi_quantum != 1024:
            raise ValueError("roi_quantum is frozen at 1024")
        if roi_coordinate_frame != _ROI_COORDINATE_FRAME:
            raise ValueError("unsupported ROI coordinate frame")
        if roi_materialization_policy != _ROI_MATERIALIZATION_POLICY:
            raise ValueError("unsupported ROI materialization policy")
        if not large_image_policy:
            raise ValueError("large_image_policy must not be empty")
        if vqa_assistance_scope is not None and (
            not isinstance(vqa_assistance_scope, str)
            or re.fullmatch(_SCOPE_PATTERN, vqa_assistance_scope) is None
        ):
            raise ValueError("vqa_assistance_scope is not a stable scope string")
        self._client = client
        self._system_prompt = system_prompt
        self._prompt_version = prompt_version
        self._catalog = catalog
        self._runtime_executable_by_task = {
            task: frozenset(
                executable_categories_by_task[task]
                if executable_categories_by_task is not None
                else catalog.executable_leaves_for_task(_vqa_capability_owner(task))
            )
            for task in _VISUAL_CAPABILITY_TASKS
        }
        self._max_side = max_side
        self._roi_quantum = roi_quantum
        self._roi_coordinate_frame = roi_coordinate_frame
        self._roi_materialization_policy = roi_materialization_policy
        self._large_image_policy = large_image_policy
        self._vqa_assistance_scope = vqa_assistance_scope

    @property
    def prompt_version(self) -> str:
        """Return the frozen planner prompt version. / 返回冻结的规划 prompt 版本。"""
        return self._prompt_version

    @property
    def planning_parameters(self) -> dict[str, object]:
        """Return JSON-safe parameters frozen into run identity.
        返回写入运行身份的 JSON 安全冻结参数。"""
        parameters: dict[str, object] = {
            "planning_mode": "visual-task-plan-v5",
            "task_prompt_version": self._prompt_version,
            "preview_max_side": self._max_side,
            "roi_coordinate_frame": self._roi_coordinate_frame,
            "roi_quantum": self._roi_quantum,
            "roi_materialization_policy": self._roi_materialization_policy,
            "large_image_policy": self._large_image_policy,
        }
        if self._vqa_assistance_scope is not None:
            parameters["vqa_assistance_scope"] = self._vqa_assistance_scope
        return parameters

    @property
    def prompt_snapshot_filename(self) -> str:
        """Stable basename for the capability-bound prompt snapshot.
        能力绑定 Prompt 快照使用稳定 basename。"""
        return "visual_task_plan_v5.runtime.md"

    @property
    def system_prompt(self) -> str:
        """Return the exact system body sent to the planner.
        返回实际发送给规划器的完整 system 正文。"""
        binding = {
            "allowed_tasks": sorted(_ALL_TASK_NAMES),
            "catalog_version": self._catalog.catalog_version,
            "canonical_leaf_categories": list(self._catalog.leaf_categories),
            "parent_expansions": {
                parent: list(leaves)
                for parent, leaves in self._catalog.parent_expansions.items()
            },
            "aliases": dict(self._catalog.aliases),
            "task_executable_categories": {
                task: [
                    leaf
                    for leaf in self._catalog.executable_leaves_for_task(
                        _vqa_capability_owner(task)
                    )
                    if leaf in self._runtime_executable_by_task[task]
                ]
                for task in _VISUAL_CAPABILITY_TASKS
            },
            **self.planning_parameters,
        }
        return (
            f"{self._system_prompt}\n\n"
            f"planner_binding={json.dumps(binding, sort_keys=True)}"
        )

    async def plan(
        self,
        view: SampleDraft | UnifiedSample,
        *,
        data_root: Path,
        artifact_dir: Path,
        budget: CallBudget | None = None,
    ) -> VisualTaskPlan:
        """Make exactly one planner call with ordered images then raw text.
        按图像顺序后接原始文本，恰好执行一次规划调用。"""
        try:
            identity = require_model_cache_identity(
                self._client, component="visual_task_planner"
            )
        except MissingModelCacheIdentityError as exc:
            raise VisualTaskPlanError("CLIENT_UNAVAILABLE") from exc

        data_urls, image_hashes = self._safe_previews(view, data_root)
        content: list[dict[str, object]] = [
            *[
                {"type": "image_url", "image_url": {"url": data_url}}
                for data_url in data_urls
            ],
            {"type": "text", "text": view.question},
        ]
        messages: list[dict[str, object]] = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {"role": "user", "content": content},
        ]
        image_digest = "|".join(image_hashes)
        request_hash = build_request_hash(
            model=identity.model,
            generation=identity.generation_payload(),
            prompt_version=self._prompt_version,
            messages=messages,
            image_sha256=image_digest,
            response_schema=VisualTaskPlan.model_json_schema(),
            client_version=identity.client_version,
            model_revision=identity.revision,
        )
        if budget is not None:
            try:
                budget.reserve_qwen()
            except Exception as exc:
                raise VisualTaskPlanError("BUDGET_EXHAUSTED") from exc
        try:
            plan = await self._client.complete_json(
                messages=messages,
                response_model=VisualTaskPlan,
                request_meta=RequestMeta(
                    request_id=f"{view.sample_id}:visual_task_plan",
                    request_hash=request_hash,
                    prompt_version=self._prompt_version,
                    sample_id=view.sample_id,
                    image_sha256=image_digest,
                    artifact_dir=artifact_dir / "visual_task_plan",
                ),
            )
        except ValidationError as exc:
            raise VisualTaskPlanError("SCHEMA_INVALID") from exc
        except Exception as exc:
            raise VisualTaskPlanError("CLIENT_ERROR") from exc
        try:
            plan = VisualTaskPlan.model_validate(plan)
        except ValidationError as exc:
            raise VisualTaskPlanError("SCHEMA_INVALID") from exc
        return self._post_validate(plan, view)

    async def plan_with_views(
        self,
        view: SampleDraft | UnifiedSample,
        *,
        data_root: Path,
        artifact_dir: Path,
        budget: CallBudget | None = None,
    ) -> tuple[VisualTaskPlan, tuple[MaterializedVisualView, ...]]:
        """Plan once, then materialize deterministic final-agent views.
        只规划一次，然后物化确定性的最终 Agent 图像视图。"""
        plan = await self.plan(
            view,
            data_root=data_root,
            artifact_dir=artifact_dir,
            budget=budget,
        )
        return plan, self.materialize_views(plan, view, data_root=data_root)

    def materialize_views(
        self,
        plan: VisualTaskPlan,
        view: SampleDraft | UnifiedSample,
        *,
        data_root: Path,
    ) -> tuple[MaterializedVisualView, ...]:
        """Materialize one full view per image and at most one quantized ROI.
        为每张图像物化整图，最多额外物化一个量化 ROI。"""
        root = data_root.resolve()
        normalized_sizes: list[tuple[str, tuple[int, int]]] = []
        for image_ref in view.images:
            candidate = (root / image_ref.path).resolve()
            if not candidate.is_relative_to(root):
                raise VisualTaskPlanError("PREVIEW_DECODE_FAILED")
            try:
                normalized_sizes.append((image_ref.image_id, normalized_image_size(candidate)))
            except (OSError, ValueError) as exc:
                raise VisualTaskPlanError("PREVIEW_DECODE_FAILED") from exc

        target_index = (
            plan.region_request.image_index
            if plan.region_request.explicit
            else None
        )
        if target_index is not None and target_index >= len(normalized_sizes):
            raise VisualTaskPlanError("IMAGE_INDEX_INVALID")
        requested_roi = plan.region_request.roi_xyxy
        materialized: list[MaterializedVisualView] = []
        for index, (image_id, source_size) in enumerate(normalized_sizes):
            width, height = source_size
            # A legal explicit ROI is materialized at every source size; direct
            # clipping, never a full-image fallback, handles small sources.
            # 任意源尺寸的合法显式 ROI 都要物化；小图由直接截断处理，绝不回退整图。
            is_quantized = (
                target_index == index
                and requested_roi is not None
            )
            if is_quantized:
                try:
                    geometry = materialize_quantized_roi(
                        source_size,
                        requested_roi,
                        roi_quantum=self._roi_quantum,
                    )
                except ValueError as exc:
                    raise VisualTaskPlanError("ROI_MATERIALIZATION_FAILED") from exc
                materialized.append(
                    MaterializedVisualView(
                        image_id=image_id,
                        view_mode="quantized_roi",
                        source_size=source_size,
                        crop_xyxy=geometry.crop_xyxy,
                        crop_size=geometry.crop_size,
                        requested_roi_xyxy_0_999=geometry.requested_roi_xyxy_0_999,
                        requested_pixel_xyxy=geometry.requested_pixel_xyxy,
                        roi_quantum=geometry.roi_quantum,
                        quantized_side=geometry.quantized_side,
                        ideal_square_xyxy=geometry.ideal_square_xyxy,
                        was_clipped=geometry.was_clipped,
                    )
                )
            else:
                materialized.append(
                    MaterializedVisualView(
                        image_id=image_id,
                        view_mode="full_image",
                        source_size=source_size,
                        crop_xyxy=(0, 0, width, height),
                        crop_size=source_size,
                    )
                )
        return tuple(materialized)

    @staticmethod
    def artifact_payload(
        plan: VisualTaskPlan,
        views: tuple[MaterializedVisualView, ...] = (),
    ) -> dict[str, object]:
        """Build the sanitized sample artifact without image bytes or paths.
        构建不含图像字节和路径的脱敏样本产物。"""
        payload = plan.model_dump(mode="json")
        payload["materialized_views"] = [view.model_dump(mode="json") for view in views]
        return payload

    def _safe_previews(
        self,
        view: SampleDraft | UnifiedSample,
        data_root: Path,
    ) -> tuple[list[str], list[str]]:
        """Read normalized shrink-only previews from the explicit root.
        从显式根目录读取只缩不放的规范化预览。"""
        root = data_root.resolve()
        data_urls: list[str] = []
        hashes: list[str] = []
        for image_ref in view.images:
            candidate = (root / image_ref.path).resolve()
            if not candidate.is_relative_to(root):
                raise VisualTaskPlanError("PREVIEW_DECODE_FAILED")
            try:
                data_url, digest = preview_from_path(
                    candidate,
                    max_side=self._max_side,
                )
            except (OSError, ValueError) as exc:
                raise VisualTaskPlanError("PREVIEW_DECODE_FAILED") from exc
            data_urls.append(data_url)
            hashes.append(digest)
        return data_urls, hashes

    def _post_validate(
        self,
        plan: VisualTaskPlan,
        view: SampleDraft | UnifiedSample,
    ) -> VisualTaskPlan:
        """Apply v5 leaf/category consistency and image-index policy.
        执行 v5 叶子类别一致性与图像索引策略校验。"""
        if plan.needs_visual_assistance:
            try:
                if plan.task == "grounding":
                    # Grounding keeps canonical categories that are outside
                    # the YOLO executable set so the evidence executor can
                    # route them through its direct-Qwen fallback. Other
                    # VQA capabilities remain strictly executable-only.
                    seen: set[str] = set()
                    for category in plan.object_categories:
                        if not self._catalog.is_leaf(category) or category in seen:
                            raise CatalogCategoryError(
                                "PLAN_CATEGORY_NOT_CANONICAL_LEAF",
                                categories=[category],
                            )
                        seen.add(category)
                else:
                    self._catalog.validate_plan_leaves(
                        plan.object_categories,
                        task=_vqa_capability_owner(plan.task),
                    )
            except CatalogCategoryError as exc:
                raise VisualTaskPlanError("SCHEMA_INVALID") from exc
            unavailable = [
                category
                for category in plan.object_categories
                if category not in self._runtime_executable_by_task[plan.task]
            ]
            if unavailable and plan.task != "grounding":
                raise VisualTaskPlanError("CAPABILITY_UNAVAILABLE")
            deduped = list(dict.fromkeys(plan.object_categories))
            if deduped != plan.object_categories:
                plan = plan.model_copy(update={"object_categories": deduped})
        if plan.task in COUNTING_TASKS:
            try:
                expected = self._catalog.executable_leaves_for_target(
                    plan.count_target or "",
                    task=plan.task,
                )
            except CatalogCategoryError as exc:
                raise VisualTaskPlanError("SCHEMA_INVALID") from exc
            if any(
                leaf not in self._runtime_executable_by_task[plan.task]
                for leaf in expected
            ):
                expected = ()
            if expected:
                if tuple(plan.object_categories) != expected:
                    raise VisualTaskPlanError("SCHEMA_INVALID")
                if not plan.needs_visual_assistance:
                    raise VisualTaskPlanError("SCHEMA_INVALID")
            elif plan.object_categories or plan.needs_visual_assistance:
                raise VisualTaskPlanError("SCHEMA_INVALID")
        if plan.region_request.explicit:
            image_index = plan.region_request.image_index
            if image_index is None or image_index >= len(view.images):
                raise VisualTaskPlanError("IMAGE_INDEX_INVALID")
        return plan
