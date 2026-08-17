"""Grounding evidence executor with the doc-16 materialized-view seam.

The v2 path is wired to ``GroundingAgent`` and uses the exact planner crop
without halo expansion.

Grounding 证据执行器带有 v2 物化视图 seam。v2 路径已接入
``GroundingAgent``，使用规划器精确裁片且不扩张 halo。

Flow / 流程:

    planned ROI + requested leaves
      -> each ROI one full YOLO inference
      -> filter requested labels (unrequested labels are dropped)
      -> one final Grounding Qwen call:
           YOLO-hit leaf: choose existing box_id only
           missing leaf: may emit ROI-local [0,1] xyxy
      -> deterministic validation and whole-image 0..999 conversion

Hard constraints / 硬约束:

- never imports agents/general_vqa/evidence, uses no SegFormer, and never
  looks at overlays; 绝不 import agents/general_vqa/evidence，不使用
  SegFormer，不看 overlay；
- reads the same catalog version / YOLO label mapping / detection protocol as
  VQA; 与 VQA 读取同一 catalog version/YOLO label mapping 和 detection
  protocol；
- the final Qwen sees clean ROI + candidate text only — never confidence and
  never box-drawn images; final Qwen 只看 clean ROI + candidate text，不看
  置信度或带框图；
- YOLO-hit leaves may only select an existing box_id; missing leaves may free
  box directly; unknown box_id, out-of-authority free boxes, and invalid
  coordinates are stably rejected; an explicit failure is raised when cleanup
  leaves no valid box; YOLO-hit 类别只能选择已有 box_id；缺失叶子可直接框选；
  未知 box_id、越权自由框、非法坐标稳定拒绝；清理后无合法框显式失败；
- ROI-local [0,1] coordinates are converted to the existing whole-image
  normalized_0_999_top_left Grounding contract only in the final deterministic
  postprocess; the coordinate frame is never guessed. ROI-local [0,1] 只有在
  最终确定性后处理后才转现有整图 normalized_0_999_top_left Grounding
  contract；绝不猜坐标系。

Detector parameters stay inject-only (GroundingEvidencePolicy): a None value
means "not calibrated" and disables the corresponding behaviour. No production
default is filled anywhere; the caller expresses the disabled state explicitly
with an all-None policy. 检测参数保持仅注入（GroundingEvidencePolicy）：None
表示“未校准”并关闭对应行为。任何位置不填写生产默认值；调用方用全 None 策略
显式表达禁用状态。
"""

from __future__ import annotations

import io
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agents.base import CallBudget
from agents.evidence_catalog import CatalogCategoryError, EvidenceCatalog
from agents.schema import (
    MaterializedVisualView,
    VisualTaskPlan,
)
from agents.visual_base import PromptBinding
from data.schema import UnifiedSample
from models.base import (
    MissingModelCacheIdentityError,
    ObjectDetectionClient,
    ObjectDetectionOutput,
    RequestMeta,
    VisionLanguageClient,
    build_request_hash,
    require_model_cache_identity,
)
from models.images import crop_image_box, image_to_data_url, image_sha256
# Existing whole-image Grounding output frame. / 现有整图 Grounding 输出制式。
_FINAL_FRAME = "normalized_0_999_top_left"
_MAX_999 = 999

# Mechanical non-binding ceiling passed to the detector runtime when the max
# detections policy is not calibrated (None = no policy cap). It is a transport
# bound, never a policy value; any real image has far fewer detections.
# max detections 策略未校准（None = 无策略上限）时传给检测运行时的机械非绑定
# 上限。它是传输边界而非策略值；任何真实图像的检测数都远小于该值。
_UNBOUNDED_MAX_DETECTIONS = 100_000

_ROI_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"


class GroundingEvidenceError(ValueError):
    """Stable error for grounding-evidence failures; the public message carries
    only the stable code, never raw model text, image bytes, or paths.
    Grounding 证据失败的稳定错误；公共消息只携带稳定 code，绝不携带原始模型
    文本、图像字节或路径。"""

    def __init__(self, code: str) -> None:
        super().__init__(f"GROUNDING_EVIDENCE_FAILED:{code}")
        self.code = code


# ── typed records / 类型化记录 ───────────────────────────────────────────


class GroundingRoiRecord(BaseModel):
    """One exact source-pixel geometry record for a materialized view.
    一条物化视图对应的精确源像素几何记录。"""

    model_config = ConfigDict(extra="forbid")

    roi_id: str = Field(min_length=1, pattern=_ROI_ID_PATTERN)
    image_id: str = Field(min_length=1)
    source_size: tuple[int, int]
    core_xyxy: tuple[int, int, int, int]
    expanded_xyxy: tuple[int, int, int, int]
    crop_size: tuple[int, int]


class GroundingCandidateRecord(BaseModel):
    """One YOLO candidate offered to the final Qwen: a stable box_id, the
    concrete leaf category, the source ROI, and the box in ROI-local [0,1]
    coordinates. The model-facing serialization is derived as integer 0..999
    xyxy JSON, while this record keeps the precise internal geometry. Confidence
    is never persisted here — it exists only in the private postprocess.
    提供给最终 Qwen 的一条 YOLO 候选：稳定 box_id、具体叶子类别、来源 ROI
    与 ROI-local [0,1] 坐标。模型侧序列化时派生为 0..999 整数 xyxy JSON，
    但此记录保留精确内部几何。置信度绝不在此持久化——只存在于私有后处理。"""

    model_config = ConfigDict(extra="forbid")

    box_id: str = Field(min_length=1, pattern=_ROI_ID_PATTERN)
    leaf_category: str = Field(min_length=1)
    roi_id: str = Field(min_length=1, pattern=_ROI_ID_PATTERN)
    roi_normalized_xyxy: tuple[float, float, float, float]


class GroundingFallbackBox(BaseModel):
    """Persisted internal free-box geometry in ROI-local [0,1] coordinates.
    The model-facing response uses GroundingModelFallbackBox and is converted
    into this stable artifact shape after validation. 持久化的内部自由框使用
    ROI-local [0,1] 坐标；模型侧响应由 GroundingModelFallbackBox 表示，并在
    校验后转换为这个稳定产物形态。"""

    model_config = ConfigDict(extra="forbid")

    leaf_category: str = Field(min_length=1)
    roi_id: str = Field(min_length=1, pattern=_ROI_ID_PATTERN)
    bbox: tuple[float, float, float, float]


class GroundingModelFallbackBox(BaseModel):
    """Model-facing free box in the Phase 2 integer 0..999 xyxy JSON frame.
    模型侧自由框使用 Phase 2 的 0..999 整数 xyxy JSON 坐标。"""

    model_config = ConfigDict(extra="forbid")

    leaf_category: str = Field(min_length=1)
    roi_id: str = Field(min_length=1, pattern=_ROI_ID_PATTERN)
    xyxy: tuple[int, int, int, int]


class GroundingQwenResponse(BaseModel):
    """Strict structured output of the one final Grounding Qwen call. The two
    lists are mutually exclusive authorities by leaf: YOLO-hit leaves may only
    appear via selected_box_ids, missing leaves only via fallback_boxes.
    最终 Grounding Qwen 一次调用的严格结构化输出。两个列表按叶子互斥授权：
    YOLO-hit 叶子只能经 selected_box_ids 出现，缺失叶子只能经
    fallback_boxes。"""

    model_config = ConfigDict(extra="forbid")

    selected_box_ids: list[str] = Field(default_factory=list)
    fallback_boxes: list[GroundingModelFallbackBox] = Field(default_factory=list)


class GroundingCallAudit(BaseModel):
    """Stable audit of one model call; error_code is a stable classification,
    never a raw message. 一次模型调用的稳定审计；error_code 是稳定分类，
    绝不携带原始消息。"""

    model_config = ConfigDict(extra="forbid")

    layer: Literal["yolo", "final_qwen"]
    roi_id: str | None = None
    input_size: tuple[int, int] | None = None
    logical_model_id: str = Field(min_length=1)
    weights_sha256: str | None = None
    status: Literal["succeeded", "failed"] = "succeeded"
    error_code: str | None = None


# The two downstream leaf states of 14C: "有 YOLO 候选" or "没有候选框、标签
# 不支持或 YOLO 不可用". The diagnostic prefixes stay auditable.
# 14C 的两种下游叶子状态；诊断前缀保持可审计。
GroundingLeafState = Literal["hit", "missing", "unsupported", "unavailable", "error"]


class GroundingEvidenceBundle(BaseModel):
    """JSON-safe persisted evidence of one grounding evidence pass. The
    cleaned Qwen output (selected_box_ids / fallback_boxes) and the stable
    drop counters make the authority enforcement auditable; confidence never
    appears here. 一次 Grounding 证据执行的 JSON 安全持久化证据。清理后的
    Qwen 输出（selected_box_ids / fallback_boxes）与稳定丢弃计数使权限强制
    可审计；置信度绝不出现于此。"""

    model_config = ConfigDict(extra="forbid")

    workflow: Literal["grounding_evidence"] = "grounding_evidence"
    catalog_version: str = Field(min_length=1)
    rois: list[GroundingRoiRecord] = Field(default_factory=list)
    candidates: list[GroundingCandidateRecord] = Field(default_factory=list)
    leaf_states: dict[str, GroundingLeafState] = Field(default_factory=dict)
    missing_leaves: list[str] = Field(default_factory=list)
    selected_box_ids: list[str] = Field(default_factory=list)
    fallback_boxes: list[GroundingFallbackBox] = Field(default_factory=list)
    dropped: dict[str, int] = Field(default_factory=dict)
    call_audit: list[GroundingCallAudit] = Field(default_factory=list)


class WholeImageBox(BaseModel):
    """One final deterministic box in the existing whole-image
    normalized_0_999_top_left contract. 一条现有整图 normalized_0_999_top_left
    契约下的最终确定性框。"""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    box: tuple[int, int, int, int]


class GroundingEvidenceResult(BaseModel):
    """Outcome of one grounding evidence pass: the persisted bundle plus the
    final whole-image boxes. 一次 Grounding 证据执行的结果：持久化 bundle 加
    最终整图框。"""

    model_config = ConfigDict(extra="forbid")

    bundle: GroundingEvidenceBundle
    whole_image_boxes: list[WholeImageBox] = Field(default_factory=list)


# ── inject-only detector policy / 仅注入检测策略 ─────────────────────────


@dataclass(frozen=True)
class GroundingEvidencePolicy:
    """Inject-only detector policy; every value defaults to None meaning "not
    calibrated" and disabling the corresponding behaviour. The executor never
    fills a production default; the caller expresses the disabled state with
    an explicit all-None policy. 仅注入检测策略；每个值默认 None 表示“未校准”
    并关闭对应行为。执行器绝不填写生产默认值；调用方用显式全 None 策略表达
    禁用状态。

    - confidence_threshold=None disables the YOLO phase entirely (capability
      off); confidence_threshold=None 整体关闭 YOLO 阶段（能力关闭）；
    - nms_iou_threshold=None disables runtime NMS (identity 1.0) and the
      deterministic dedup; nms_iou_threshold=None 关闭运行时 NMS（恒等 1.0）
      与确定性去重；
    - max_detections=None applies no policy cap (the runtime gets a non-binding
      mechanical ceiling). max_detections=None 不施加策略上限（运行时获得
      非绑定机械上限）。
    """

    confidence_threshold: float | None = None
    nms_iou_threshold: float | None = None
    max_detections: int | None = None

    def __post_init__(self) -> None:
        if self.confidence_threshold is not None:
            if not math.isfinite(self.confidence_threshold):
                raise ValueError("confidence_threshold must be finite")
            if not 0.0 <= self.confidence_threshold <= 1.0:
                raise ValueError("confidence_threshold must be within [0.0, 1.0]")
        if self.nms_iou_threshold is not None:
            if not math.isfinite(self.nms_iou_threshold):
                raise ValueError("nms_iou_threshold must be finite")
            if not 0.0 <= self.nms_iou_threshold <= 1.0:
                raise ValueError("nms_iou_threshold must be within [0.0, 1.0]")
        if self.max_detections is not None and self.max_detections < 1:
            raise ValueError("max_detections must be at least 1 when set")

    @property
    def yolo_enabled(self) -> bool:
        """The YOLO phase is active only when a threshold is calibrated.
        YOLO 阶段仅在阈值已校准时激活。"""
        return self.confidence_threshold is not None


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Pure-python intersection over union of two axis-aligned boxes.
    两个轴对齐框交并比的纯 Python 实现。"""
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _normalize_local(
    local_xyxy: tuple[float, float, float, float],
    crop_size: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    """Map crop-local pixel coordinates to ROI-local [0,1]; values outside the
    crop are clamped, and a degenerate or non-finite box returns None.
    将 crop 局部像素坐标映射为 ROI-local [0,1]；超出裁切范围的值被 clamp，
    退化或非有限框返回 None。"""
    width, height = crop_size
    if width <= 0 or height <= 0:
        return None
    x1, y1, x2, y2 = local_xyxy
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    box = (
        max(0.0, min(1.0, x1 / width)),
        max(0.0, min(1.0, y1 / height)),
        max(0.0, min(1.0, x2 / width)),
        max(0.0, min(1.0, y2 / height)),
    )
    if box[0] >= box[2] or box[1] >= box[3]:
        return None
    return box


def _normalized_box_to_999(
    box: tuple[float, float, float, float],
) -> list[int]:
    """Serialize ROI-local [0,1] xyxy as integer 0..999 JSON geometry.
    将 ROI-local [0,1] xyxy 序列化为 0..999 整数 JSON 几何。"""
    return [
        max(0, min(_MAX_999, round(value * _MAX_999))) for value in box
    ]


def _fallback_box_to_normalized(
    box: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    """Convert model-facing integer 0..999 xyxy back to ROI-local [0,1].
    将模型侧 0..999 整数 xyxy 转回 ROI-local [0,1]。"""
    return tuple(value / _MAX_999 for value in box)  # type: ignore[return-value]


def _valid_fallback_bbox(bbox: tuple[int, int, int, int]) -> bool:
    """A free box must be integer, within 0..999, and non-degenerate.
    自由框必须是整数、位于 0..999 且非退化。"""
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in bbox):
        return False
    if not all(0 <= value <= _MAX_999 for value in bbox):
        return False
    return bbox[0] < bbox[2] and bbox[1] < bbox[3]


def _audit_identity(
    outputs: list[ObjectDetectionOutput],
    client: object | None,
) -> tuple[str, str | None]:
    """Stable audit identity of one call: the logical model id and weights
    digest come from the call's own outputs when available, otherwise from the
    client's cache identity; never a physical path. 一次调用的稳定审计身份：
    优先取调用自身输出的逻辑模型 id 与权重摘要，否则取客户端缓存身份；绝不
    是物理路径。"""
    if outputs:
        return outputs[0].logical_model_id, outputs[0].weights_sha256
    identity = getattr(client, "cache_identity", None)
    model = getattr(identity, "model", None)
    generation = getattr(identity, "generation", None)
    digest: str | None = None
    if isinstance(generation, Mapping):
        value = generation.get("weights_sha256")
        if isinstance(value, str) and len(value) == 64:
            digest = value
    if isinstance(model, str) and model:
        return model, digest
    return "unknown", digest


@dataclass(frozen=True)
class _RawDetection:
    """In-memory YOLO hit before cross-ROI dedup and box_id assignment.
    Confidence exists only here. 跨 ROI 去重与 box_id 分配前的内存 YOLO 命中。
    置信度只存在于此处。"""

    confidence: float
    leaf_category: str
    roi_id: str
    local_normalized_xyxy: tuple[float, float, float, float]
    global_xyxy: tuple[float, float, float, float]


class GroundingEvidenceExecutor:
    """Execute the 14C grounding evidence flow with injected clients. The
    executor is deterministic: same plan, same images, same policy, same fake
    clients -> same result. The final Qwen is called exactly once whenever the
    seam runs, and all accumulation state is re-created per call so one
    executor serves many plans without cross-plan leakage.
    用注入客户端执行 14C Grounding 证据流程。执行器确定性：相同计划、相同图
    像、相同策略、相同 fake clients -> 相同结果。只要 seam 运行，最终 Qwen
    恰好调用一次；所有累积状态在每次调用时重新创建，一个执行器可服务多个计
    划而无跨计划泄漏。"""

    def __init__(
        self,
        *,
        catalog: EvidenceCatalog,
        qwen_client: VisionLanguageClient,
        prompt: PromptBinding,
        policy: GroundingEvidencePolicy,
        yolo_client: ObjectDetectionClient | None = None,
        yolo_device: str | None = None,
        yolo_image_size: int | None = None,
    ) -> None:
        """prompt carries the final-Grounding prompt binding (text + version);
        the final prompt itself is deferred (14C §13 item 6) and is therefore
        injected, never defaulted here. The YOLO image size is required when
        the YOLO phase is enabled and irrelevant when the capability is off.
        prompt 携带最终 Grounding prompt 绑定（text + version）；最终 prompt
        本身被延期（14C §13 第 6 项），因此注入而非在此默认。YOLO 阶段启用时
        必须提供 yolo_image_size；能力关闭时该值无关。"""
        if yolo_image_size is not None and yolo_image_size <= 0:
            raise ValueError("yolo_image_size must be positive when set")
        if policy.yolo_enabled and yolo_image_size is None:
            raise ValueError("yolo_image_size is required when the YOLO phase is enabled")
        self._catalog = catalog
        self._qwen_client = qwen_client
        self._prompt = prompt
        self._policy = policy
        self._yolo_client = yolo_client
        self._yolo_device = yolo_device
        self._yolo_image_size = yolo_image_size

    # ── main entry / 主入口 ─────────────────────────────────────────────

    async def run(
        self,
        plan: VisualTaskPlan,
        sample: UnifiedSample,
        images: Mapping[str, Image.Image],
        *,
        fallback_image_id: str,
        artifact_dir: Path,
        budget: CallBudget | None = None,
        materialized_views: tuple[MaterializedVisualView, ...],
    ) -> GroundingEvidenceResult:
        """Run the 14C flow over one plan and one sample. The plan must carry
        a validated evidence request; all accumulated state is re-created per
        call so one executor serves many plans without cross-plan leakage.
        对一个计划与一条样本运行 14C 流程。计划必须携带已校验 evidence
        request；所有累积状态在每次调用时重新创建，一个执行器可服务多个计划
        而无跨计划泄漏。"""
        if not sample.images:
            raise GroundingEvidenceError("SAMPLE_WITHOUT_IMAGES")
        try:
            if not plan.needs_visual_assistance:
                raise GroundingEvidenceError("PLAN_WITHOUT_VISUAL_ASSISTANCE")
            if not materialized_views:
                raise GroundingEvidenceError("MATERIALIZED_VIEWS_MISSING")
            leaves = self._catalog.expand_composites(plan.object_categories)
        except CatalogCategoryError as exc:
            raise GroundingEvidenceError("PLAN_INVALID") from exc
        sizes = {image_id: image.size for image_id, image in images.items()}
        try:
            records = self._materialized_regions(materialized_views, sizes)
        except ValueError as exc:
            raise GroundingEvidenceError("PLAN_INVALID") from exc

        detections, outcomes, audits, dropped = self._yolo_phase(
            images, records, leaves
        )
        candidates = self._assign_box_ids(detections)
        leaf_states, missing_leaves = self._aggregate_leaves(
            leaves, outcomes, candidates
        )

        response, audits = await self._final_qwen(
            sample,
            images,
            records,
            candidates,
            missing_leaves,
            artifact_dir=artifact_dir,
            budget=budget,
            audits=audits,
        )

        bundle = self._postprocess(
            response,
            records=records,
            candidates=candidates,
            leaves=leaves,
            leaf_states=leaf_states,
            missing_leaves=missing_leaves,
            audits=audits,
            dropped=dropped,
        )
        result = self._convert_to_whole_image(bundle, records)
        if not result.whole_image_boxes:
            raise GroundingEvidenceError("NO_VALID_BOXES")
        return result

    def _materialized_regions(
        self,
        views: tuple[MaterializedVisualView, ...],
        sizes: Mapping[str, tuple[int, int]],
    ) -> list[GroundingRoiRecord]:
        """Build exact fixed/full ROI records from v2 materialized views.
        从 v2 物化视图构建精确的固定/整图 ROI 记录。"""
        if not views:
            raise ValueError("materialized views must not be empty")
        records: list[GroundingRoiRecord] = []
        for index, view in enumerate(views):
            size = sizes.get(view.image_id)
            if size != view.source_size:
                raise ValueError("materialized view source size does not match image")
            roi_id = (
                "full"
                if len(views) == 1 and view.view_mode == "full_image"
                else f"{view.view_mode}-{index}"
            )
            records.append(
                GroundingRoiRecord(
                    roi_id=roi_id,
                    image_id=view.image_id,
                    source_size=view.source_size,
                    core_xyxy=view.crop_xyxy,
                    expanded_xyxy=view.crop_xyxy,
                    crop_size=view.crop_size,
                )
            )
        return records

    # ── helpers / 辅助 ──────────────────────────────────────────────────

    # ── YOLO phase / YOLO 阶段 ──────────────────────────────────────────

    def _yolo_phase(
        self,
        images: Mapping[str, Image.Image],
        records: list[GroundingRoiRecord],
        leaves: tuple[str, ...],
    ) -> tuple[
        list[_RawDetection],
        list[tuple[str, str, GroundingLeafState, str | None]],
        list[GroundingCallAudit],
        dict[str, int],
    ]:
        """One YOLO inference per ROI, filter every requested leaf from one
        output. Confidence is consumed internally only: thresholding, top-k
        retention, and the cross-ROI greedy dedup in whole-image coordinates.
        Per-ROI independence: an error in one ROI keeps the other ROIs'
        evidence. 每个 ROI 运行一次 YOLO，从一次输出过滤全部请求叶子。
        confidence 仅在内部消费：阈值、top-k 保留与 whole-image 坐标下的跨
        ROI 贪心去重。逐 ROI 独立：单 ROI 出错保留其他 ROI 的证据。"""
        raw: list[_RawDetection] = []
        outcomes: list[tuple[str, str, GroundingLeafState, str | None]] = []
        audits: list[GroundingCallAudit] = []
        dropped: dict[str, int] = {}
        yolo_active = (
            self._yolo_client is not None and self._policy.yolo_enabled
        )
        for record in records:
            crop = self._render_crop(images[record.image_id], record)
            outputs: list[ObjectDetectionOutput] | None = None
            failed_code: str | None = None
            if yolo_active:
                try:
                    outputs = self._yolo_client.detect(
                        crop,
                        confidence=self._policy.confidence_threshold,
                        iou=(
                            self._policy.nms_iou_threshold
                            if self._policy.nms_iou_threshold is not None
                            else 1.0
                        ),
                        image_size=self._yolo_image_size,
                        device=self._yolo_device,
                        max_detections=(
                            self._policy.max_detections
                            if self._policy.max_detections is not None
                            else _UNBOUNDED_MAX_DETECTIONS
                        ),
                    )
                except Exception as exc:
                    failed_code = type(exc).__name__
            if outputs is not None or failed_code is not None:
                logical_model_id, digest = _audit_identity(
                    outputs or [], self._yolo_client
                )
                audits.append(
                    GroundingCallAudit(
                        layer="yolo",
                        roi_id=record.roi_id,
                        input_size=crop.size,
                        logical_model_id=logical_model_id,
                        weights_sha256=digest,
                        status="failed" if failed_code is not None else "succeeded",
                        error_code=failed_code,
                    )
                )
            left, top = record.expanded_xyxy[0], record.expanded_xyxy[1]
            for leaf in leaves:
                if not self._catalog.capability_enabled(leaf, "yolo"):
                    outcomes.append((record.roi_id, leaf, "unsupported", None))
                    continue
                if not yolo_active:
                    outcomes.append((record.roi_id, leaf, "unavailable", None))
                    continue
                if failed_code is not None:
                    outcomes.append((record.roi_id, leaf, "error", failed_code))
                    continue
                assert outputs is not None
                leaf_labels = set(self._catalog.leaf_yolo_labels(leaf))
                leaf_outputs = [o for o in outputs if o.label in leaf_labels]
                if not leaf_outputs:
                    outcomes.append((record.roi_id, leaf, "missing", None))
                    continue
                retained = sorted(
                    leaf_outputs, key=lambda o: o.confidence, reverse=True
                )
                if self._policy.max_detections is not None:
                    retained = retained[: self._policy.max_detections]
                normalized: list[tuple[float, float, float, float]] = []
                for detection in retained:
                    box = _normalize_local(detection.xyxy, crop.size)
                    if box is None:
                        dropped["degenerate_detection"] = (
                            dropped.get("degenerate_detection", 0) + 1
                        )
                        continue
                    normalized.append(box)
                    raw.append(
                        _RawDetection(
                            confidence=detection.confidence,
                            leaf_category=leaf,
                            roi_id=record.roi_id,
                            local_normalized_xyxy=box,
                            global_xyxy=(
                                detection.xyxy[0] + left,
                                detection.xyxy[1] + top,
                                detection.xyxy[2] + left,
                                detection.xyxy[3] + top,
                            ),
                        )
                    )
                if normalized:
                    outcomes.append((record.roi_id, leaf, "hit", None))
                else:
                    outcomes.append((record.roi_id, leaf, "missing", None))
        deduped = self._dedup_cross_roi(raw)
        return deduped, outcomes, audits, dropped

    def _dedup_cross_roi(
        self,
        raw: list[_RawDetection],
    ) -> list[_RawDetection]:
        """Greedy cross-ROI dedup in whole-image coordinates: sort by internal
        confidence descending (stable sort, ties keep roi/leaf order), keep a
        detection only when its global box overlaps no kept box by IoU >= the
        policy threshold. The higher internal confidence wins; confidence never
        leaves this method. Disabled when the threshold is not calibrated.
        在 whole-image 坐标下贪心跨 ROI 去重：按内部 confidence 降序（稳定排
        序，同分保持 roi/leaf 顺序），仅当全局框与已保留框的 IoU 低于策略阈
        值时保留。内部置信度更高者胜出；confidence 绝不离开本方法。阈值未校
        准时关闭。"""
        if self._policy.nms_iou_threshold is None:
            return raw
        kept: list[_RawDetection] = []
        for item in sorted(raw, key=lambda d: d.confidence, reverse=True):
            if any(
                _iou(item.global_xyxy, other.global_xyxy)
                >= self._policy.nms_iou_threshold
                for other in kept
            ):
                continue
            kept.append(item)
        return kept

    def _assign_box_ids(
        self,
        deduped: list[_RawDetection],
    ) -> list[GroundingCandidateRecord]:
        """Assign stable box_ids (per-ROI 1-based counters) in the deterministic
        order: ROI order, then leaf order, then confidence descending (the raw
        insertion order already has that shape after a stable sort).
        按确定性顺序（ROI 顺序、叶子顺序、置信度降序——稳定排序后的原始插入
        顺序即此形状）分配稳定 box_id（逐 ROI 1 基计数器）。"""
        candidates: list[GroundingCandidateRecord] = []
        counter: dict[str, int] = {}
        for item in deduped:
            number = counter.get(item.roi_id, 0) + 1
            counter[item.roi_id] = number
            candidates.append(
                GroundingCandidateRecord(
                    box_id=f"{item.roi_id}-box-{number}",
                    leaf_category=item.leaf_category,
                    roi_id=item.roi_id,
                    roi_normalized_xyxy=item.local_normalized_xyxy,
                )
            )
        return candidates

    def _render_crop(
        self,
        image: Image.Image,
        record: GroundingRoiRecord,
    ) -> Image.Image:
        """Crop the exact source-pixel box from the v2 materialized view.
        按 v2 物化视图的精确源像素框裁切图像。"""

        crop = crop_image_box(image, record.expanded_xyxy)
        if crop.size != record.crop_size:
            raise GroundingEvidenceError("ROI_CROP_DRIFT")
        return crop

    # ── aggregation / 聚合 ──────────────────────────────────────────────

    def _aggregate_leaves(
        self,
        leaves: tuple[str, ...],
        outcomes: list[tuple[str, str, GroundingLeafState, str | None]],
        candidates: list[GroundingCandidateRecord],
    ) -> tuple[dict[str, GroundingLeafState], list[str]]:
        """Deterministic per-leaf aggregation across ROIs: a leaf is hit when
        it owns at least one retained candidate; otherwise the severity order
        error > unavailable > missing > unsupported over its per-ROI outcomes.
        A leaf that is not hit stays in missing_leaves and receives the Qwen
        free-box authority. 跨 ROI 确定性逐叶子聚合：叶子持有至少一条保留候选
        即 hit；否则按逐 ROI 结果的严重度顺序 error > unavailable > missing
        > unsupported。未 hit 的叶子留在 missing_leaves 并获得 Qwen 自由框
        权限。"""
        leaf_states: dict[str, GroundingLeafState] = {}
        missing_leaves: list[str] = []
        for leaf in leaves:
            if any(candidate.leaf_category == leaf for candidate in candidates):
                leaf_states[leaf] = "hit"
                continue
            states = [state for _, leaf_id, state, _ in outcomes if leaf_id == leaf]
            state: GroundingLeafState = "unsupported"
            for candidate_state in ("error", "unavailable", "missing"):
                if candidate_state in states:
                    state = candidate_state
                    break
            leaf_states[leaf] = state
            missing_leaves.append(leaf)
        return leaf_states, missing_leaves

    # ── final Qwen / 最终 Qwen ──────────────────────────────────────────

    async def _final_qwen(
        self,
        sample: UnifiedSample,
        images: Mapping[str, Image.Image],
        records: list[GroundingRoiRecord],
        candidates: list[GroundingCandidateRecord],
        missing_leaves: list[str],
        *,
        artifact_dir: Path,
        budget: CallBudget | None,
        audits: list[GroundingCallAudit],
    ) -> tuple[GroundingQwenResponse, list[GroundingCallAudit]]:
        """The single final Grounding Qwen call. The identity is required up
        front; the budget is consumed only when a model call is actually
        attempted. The user payload carries the clean ROI images plus candidate
        text — never confidence, never box-drawn images, never the ground
        truth. 唯一一次最终 Grounding Qwen 调用。身份必须前置验证；只在真正
        尝试模型调用时才消费 budget。用户载荷携带 clean ROI 图像加候选文本
        ——绝不携带置信度、带框图或 ground truth。"""
        try:
            identity = require_model_cache_identity(
                self._qwen_client, component="grounding_evidence"
            )
        except MissingModelCacheIdentityError as exc:
            raise GroundingEvidenceError("CLIENT_UNAVAILABLE") from exc

        data_urls: list[str] = []
        image_digests: list[str] = []
        preview_sizes: list[tuple[int, int]] = []
        for record in records:
            crop = self._render_crop(images[record.image_id], record)
            preview_sizes.append(crop.size)
            buffer = io.BytesIO()
            crop.save(buffer, format="PNG")
            data = buffer.getvalue()
            data_urls.append(image_to_data_url(data, "image/png"))
            image_digests.append(image_sha256(data))

        user_payload = {
            "question": sample.question,
            "catalog_version": self._catalog.catalog_version,
            "coordinate_frame": "roi_normalized_0_999_top_left",
            "box_format": "integer_xyxy_json",
            "rois": [
                {
                    "roi_id": record.roi_id,
                    "image_id": record.image_id,
                    "source_size": list(record.source_size),
                    "crop_size": list(record.crop_size),
                }
                for record in records
            ],
            "candidates": [
                {
                    "box_id": candidate.box_id,
                    "leaf_category": candidate.leaf_category,
                    "roi_id": candidate.roi_id,
                    "xyxy": _normalized_box_to_999(candidate.roi_normalized_xyxy),
                }
                for candidate in candidates
            ],
            "missing_leaves": missing_leaves,
        }
        content: list[dict[str, object]] = [
            *[{"type": "image_url", "image_url": {"url": url}} for url in data_urls],
            {"type": "text", "text": json.dumps(user_payload, ensure_ascii=False)},
        ]
        messages: list[dict[str, object]] = [
            {"role": "system", "content": self._prompt.text},
            {"role": "user", "content": content},
        ]
        image_digest = "|".join(image_digests)
        request_hash = build_request_hash(
            model=identity.model,
            generation=identity.generation_payload(),
            prompt_version=self._prompt.version,
            messages=messages,
            image_sha256=image_digest,
            response_schema=GroundingQwenResponse.model_json_schema(),
            client_version=identity.client_version,
            model_revision=identity.revision,
        )
        if budget is not None:
            try:
                budget.reserve_qwen()
            except Exception as exc:
                raise GroundingEvidenceError("BUDGET_EXHAUSTED") from exc
        failed_code: str | None = None
        try:
            response = await self._qwen_client.complete_json(
                messages=messages,
                response_model=GroundingQwenResponse,
                request_meta=RequestMeta(
                    request_id=f"{sample.sample_id}:grounding_final",
                    request_hash=request_hash,
                    prompt_version=self._prompt.version,
                    sample_id=sample.sample_id,
                    image_sha256=image_digest,
                    artifact_dir=artifact_dir / "grounding_evidence",
                ),
            )
        except ValidationError as exc:
            # A structurally invalid response fails per 14C §8.4; item-level
            # semantic problems are dropped in the postprocess instead.
            # 结构非法响应按 14C §8.4 失败；条目级语义问题由后处理丢弃。
            failed_code = "SCHEMA_INVALID"
            raise GroundingEvidenceError(failed_code) from exc
        except Exception as exc:
            failed_code = "CLIENT_ERROR"
            raise GroundingEvidenceError(failed_code) from exc
        finally:
            logical_model_id, digest = _audit_identity([], self._qwen_client)
            audits.append(
                GroundingCallAudit(
                    layer="final_qwen",
                    input_size=preview_sizes[0] if preview_sizes else None,
                    logical_model_id=logical_model_id,
                    weights_sha256=digest,
                    status="failed" if failed_code is not None else "succeeded",
                    error_code=failed_code,
                )
            )
        return response, audits

    # ── deterministic postprocess / 确定性后处理 ─────────────────────────

    def _postprocess(
        self,
        response: GroundingQwenResponse,
        *,
        records: list[GroundingRoiRecord],
        candidates: list[GroundingCandidateRecord],
        leaves: tuple[str, ...],
        leaf_states: dict[str, GroundingLeafState],
        missing_leaves: list[str],
        audits: list[GroundingCallAudit],
        dropped: dict[str, int],
    ) -> GroundingEvidenceBundle:
        """Deterministic authority enforcement and cleanup: unknown box_id,
        out-of-authority free boxes (hit leaf / unrequested leaf / unknown
        ROI), and invalid coordinates are dropped with stable counters. The
        cleaned selection and fallback boxes are the only persisted Qwen
        output. 确定性权限强制与清理：未知 box_id、越权自由框（hit 叶子/未请
        求叶子/未知 ROI）与非法坐标以稳定计数丢弃。清理后的选择与自由框是唯
        一持久化的 Qwen 输出。"""
        dropped = dict(dropped)
        candidates_by_id = {candidate.box_id: candidate for candidate in candidates}
        records_by_id = {record.roi_id: record for record in records}
        leaf_set = set(leaves)
        missing_set = set(missing_leaves)

        selected: list[str] = []
        seen: set[str] = set()
        for box_id in response.selected_box_ids:
            if box_id in seen:
                dropped["box_id_duplicate"] = dropped.get("box_id_duplicate", 0) + 1
                continue
            seen.add(box_id)
            if box_id not in candidates_by_id:
                dropped["unknown_box_id"] = dropped.get("unknown_box_id", 0) + 1
                continue
            selected.append(box_id)

        fallbacks: list[GroundingFallbackBox] = []
        for box in response.fallback_boxes:
            if box.leaf_category not in leaf_set:
                dropped["free_box_unrequested_leaf"] = (
                    dropped.get("free_box_unrequested_leaf", 0) + 1
                )
                continue
            if box.leaf_category not in missing_set:
                dropped["free_box_for_hit_leaf"] = (
                    dropped.get("free_box_for_hit_leaf", 0) + 1
                )
                continue
            if box.roi_id not in records_by_id:
                dropped["free_box_unknown_roi"] = (
                    dropped.get("free_box_unknown_roi", 0) + 1
                )
                continue
            if not _valid_fallback_bbox(box.xyxy):
                dropped["free_box_invalid_coordinates"] = (
                    dropped.get("free_box_invalid_coordinates", 0) + 1
                )
                continue
            fallbacks.append(
                GroundingFallbackBox(
                    leaf_category=box.leaf_category,
                    roi_id=box.roi_id,
                    bbox=_fallback_box_to_normalized(box.xyxy),
                )
            )

        return GroundingEvidenceBundle(
            catalog_version=self._catalog.catalog_version,
            rois=records,
            candidates=candidates,
            leaf_states=leaf_states,
            missing_leaves=missing_leaves,
            selected_box_ids=selected,
            fallback_boxes=fallbacks,
            dropped=dropped,
            call_audit=audits,
        )

    def _convert_to_whole_image(
        self,
        bundle: GroundingEvidenceBundle,
        records: list[GroundingRoiRecord],
    ) -> GroundingEvidenceResult:
        """Deterministic ROI-local [0,1] -> whole-image pixels ->
        normalized_0_999_top_left conversion, then the final same-leaf
        geometric dedup. Degenerate-after-rounding boxes are dropped honestly;
        a box is never fabricated. 确定性 ROI-local [0,1] -> 整图像素 ->
        normalized_0_999_top_left 转换，再做最终同叶子几何去重。取整后退化的
        框被如实丢弃；绝不伪造框。"""
        records_by_id = {record.roi_id: record for record in records}
        candidates_by_id = {candidate.box_id: candidate for candidate in bundle.candidates}
        merged: list[tuple[str, tuple[float, float, float, float], GroundingRoiRecord]] = []
        for box_id in bundle.selected_box_ids:
            candidate = candidates_by_id[box_id]
            record = records_by_id[candidate.roi_id]
            merged.append((candidate.leaf_category, candidate.roi_normalized_xyxy, record))
        for box in bundle.fallback_boxes:
            record = records_by_id[box.roi_id]
            merged.append(
                (
                    box.leaf_category,
                    box.bbox,
                    record,
                )
            )

        converted: list[tuple[str, tuple[int, int, int, int]]] = []
        for label, roi_normalized, record in merged:
            box = self._to_whole_image_999(roi_normalized, record)
            if box is None:
                continue
            converted.append((label, box))
        converted = self._final_dedup(converted)
        return GroundingEvidenceResult(
            bundle=bundle,
            whole_image_boxes=[
                WholeImageBox(label=label, box=box) for label, box in converted
            ],
        )

    def _to_whole_image_999(
        self,
        roi_normalized_xyxy: tuple[float, float, float, float],
        record: GroundingRoiRecord,
    ) -> tuple[int, int, int, int] | None:
        """ROI-local [0,1] -> whole-image pixels (offset by the expanded crop
        origin) -> normalized_0_999_top_left; returns None for a box that
        becomes degenerate after rounding. The x/y pixel to normalized
        conversion follows the documented frame semantics: the image edge maps
        to 999. ROI-local [0,1] -> 整图像素（以 expanded crop 原点为偏移）->
        normalized_0_999_top_left；取整后退化的框返回 None。像素到归一化的转
        换遵循文档化制式语义：图像边缘映射到 999。"""
        x1, y1, x2, y2 = roi_normalized_xyxy
        left, top = record.expanded_xyxy[0], record.expanded_xyxy[1]
        crop_w, crop_h = record.crop_size
        width, height = record.source_size
        if crop_w <= 0 or crop_h <= 0 or width <= 0 or height <= 0:
            return None
        px = (
            x1 * crop_w + left,
            y1 * crop_h + top,
            x2 * crop_w + left,
            y2 * crop_h + top,
        )
        box = tuple(
            max(0, min(_MAX_999, round(value / width * _MAX_999)))
            if index % 2 == 0
            else max(0, min(_MAX_999, round(value / height * _MAX_999)))
            for index, value in enumerate(px)
        )
        if box[0] >= box[2] or box[1] >= box[3]:
            return None
        return box  # type: ignore[return-value]

    def _final_dedup(
        self,
        converted: list[tuple[str, tuple[int, int, int, int]]],
    ) -> list[tuple[str, tuple[int, int, int, int]]]:
        """Final same-leaf geometric dedup over the merged selection and
        fallback boxes: identical instances of one leaf appear once, keeping
        the deterministic order (selected candidates first in box_id order,
        then fallback boxes in response order). Disabled when the threshold is
        not calibrated. 对合并的选择框与自由框做最终同叶子几何去重：同一叶子
        的同一实例只出现一次，保持确定性顺序（先按 box_id 顺序的已选候选，
        再按响应顺序的自由框）。阈值未校准时关闭。"""
        if self._policy.nms_iou_threshold is None:
            return converted
        kept: list[tuple[str, tuple[int, int, int, int]]] = []
        for label, box in converted:
            if any(
                label == kept_label
                and _iou(
                    tuple(value / _MAX_999 for value in box),
                    tuple(value / _MAX_999 for value in kept_box),
                )
                >= self._policy.nms_iou_threshold
                for kept_label, kept_box in kept
            ):
                continue
            kept.append((label, box))
        return kept
