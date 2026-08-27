"""VQA object-evidence executor consuming validated v5 planner leaves.

The evidence path is wired to ``GeneralVQAAgent`` and receives the exact
materialized views plus canonical executable leaves from the v5 planner.

VQA 对象证据执行器消费 v5 planner 已校验的 canonical leaves，并与 direct
inference 消费同一组精确物化视图。

Frozen state machine / 冻结状态机：

    expand requested composite categories to ordered leaf categories
    partition every ROI crop into deterministic 1024x1024 tiles (14.7)
    for each tile: run YOLO once, then filter all requested leaves
    for each still-missing leaf:
        if catalog has approved SegFormer capability:
            run SegFormer inference once per (binding, ROI), then filter
        if still missing:
            leave the leaf for the single final-Qwen visual fallback
    preserve all successful evidence from other leaves

States / 状态：hit / missing / unsupported / unavailable / error / not_run；
不存在 valid_empty（成功但筛选为空 = missing）。已 hit 的叶子绝不重跑或覆盖。

Bounded scheduling / 有界调度：一次 execution 生命周期内单个
``ThreadPoolExecutor(max_workers=max_tile_concurrency)``；全部 ROI tile
records/images 在调用模型前同步确定性生成；YOLO 与 SegFormer jobs 均按稳定
job index 提交，主线程按稳定 tile 顺序聚合，绝不按完成顺序。

VQA-only constraints / VQA 专属约束：

- YOLO 调用次数按 tile，SegFormer 按（ROI，binding），都不按类别增长；
  YOLO 模型输入永远是 1024×1024 tile，剩余部分 LANCZOS 拉伸；SegFormer
  模型输入是整张 ROI 右侧/底部黑色 padding 到 1024 倍数后整体 LANCZOS
  缩放的 1024×1024 方形（pad-multiple-1024-resize-square-v1）；
- 只保留目录请求标签，未请求输出全部丢弃；
- YOLO confidence 仅供阈值/去重/冲突裁决，绝不进入最终 bundle 或公共 trace；
- SegFormer 只保留 mask/存在性证据，不转 box、不生成 instance count；mask
  在各 ROI 内独立保留，绝不跨 ROI 像素融合或比较置信度；输出 class map 与
  catalog raw labels 不一致时严格失败；
- 跨 ROI 重复 YOLO 目标在 whole-image 坐标去重，内部置信度高者胜出；
- 最终 bundle 按 roi_id 和稳定 leaf order 组装，绝不按并发完成顺序；
- 不记录 raw exception、tensor、完整 raw response、Base64、secret 或物理模型路径。

Unfrozen parameters (inject-only; no production defaults are filled anywhere):
confidence threshold, NMS IoU threshold, max detections, ROI partial-failure
sample status. The executor keeps per-ROI/per-leaf outcomes and NEVER decides
the sample's final success status — that decision belongs to the runtime after
the user freezes it. 未冻结参数（仅注入；任何位置不填写生产默认值）：置信度
阈值、NMS IoU 阈值、最大检测数、ROI 部分失败时的 sample 状态。执行器保留
逐 ROI/逐类别结果，绝不决定 sample 最终成功状态——该决定在用户冻结后由
运行时负责。
"""

from __future__ import annotations

import array
import math
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from agents.evidence_catalog import EvidenceCatalog
from agents.general_vqa.evidence.geometry import (
    MODEL_INPUT_SIZE,
    local_to_global,
    model_xyxy_to_roi_xyxy,
    partition_roi,
)
from agents.general_vqa.evidence.rendering import (
    prepare_model_tile,
    prepare_segformer_roi,
    render_roi_crop,
    restore_segformer_class_id_mask,
    segformer_palette,
)
from agents.general_vqa.evidence.schema import (
    EvidenceLayer,
    EvidencePreprocessing,
    EvidenceState,
    EvidenceTileRecord,
    LayerStateRecord,
    ModelCallAudit,
    RoiEvidenceRecord,
    SegFormerEvidenceRecord,
    SegFormerPreprocessRecord,
    VqaEvidenceBundle,
    YoloDetectionRecord,
)
from agents.schema import MaterializedVisualView, VisualTaskPlan
from models.base import (
    ObjectDetectionClient,
    ObjectDetectionOutput,
    SemanticMaskClient,
    SemanticMaskOutput,
)


def _to_class_grid(class_id_map: Any) -> Image.Image:
    """Convert any two-dimensional integer class-id structure into an in-memory
    "I" grid without importing NumPy or torch: either a shape-bearing array or
    a list of equal-width rows. Non-integer values fail closed.
    在不导入 NumPy 或 torch 的情况下，把任意二维整数 class-id 结构转换为内存
    "I" 网格：支持带 shape 的数组或等宽行 list。非整数值严格失败。"""

    shape = getattr(class_id_map, "shape", None)
    if shape is not None:
        if len(shape) != 2:
            raise ValueError("class_id_map must be two-dimensional")
        height, width = int(shape[0]), int(shape[1])
        flattened = class_id_map.reshape(-1)
        values = (
            flattened.tolist() if hasattr(flattened, "tolist") else list(flattened)
        )
    elif isinstance(class_id_map, (list, tuple)) and class_id_map:
        height = len(class_id_map)
        width = len(class_id_map[0])
        if any(
            not isinstance(row, (list, tuple)) or len(row) != width
            for row in class_id_map
        ):
            raise ValueError("class_id_map rows must have equal width")
        values = [value for row in class_id_map for value in row]
    else:
        raise ValueError("class_id_map must be a two-dimensional integer grid")
    if width <= 0 or height <= 0:
        raise ValueError("class_id_map dimensions must be positive")
    for value in values:
        integer = int(value)
        if integer != value:
            raise ValueError("class_id_map contains a non-integer class ID")
    data = array.array("i", (int(value) for value in values))
    return Image.frombytes("I", (width, height), data.tobytes())


def _leaf_boolean_grid(grid: Image.Image, class_ids: frozenset[int]) -> Image.Image:
    """Boolean presence grid of one leaf over an integer class-id grid: pixels
    whose class id belongs to the leaf become 255. Unrequested classes stay 0
    (background) and never reach the segments/legend. PIL's point() with a
    callable is a no-op on I-mode images, so the grid is scanned via raw bytes
    instead. 一张整数 class-id grid 上某叶子的布尔存在网格：类别 id 属于该叶子
    的像素为 255。未请求类别保持 0（背景），绝不进入 segments/legend。PIL
    的 point() 在 I 模式下对 callable 是空操作，因此改为扫描原始字节。"""

    values = array.array("i", grid.tobytes())
    present = bytes(255 if class_id in class_ids else 0 for class_id in values)
    return Image.frombytes("L", grid.size, present)


def _pixel_xyxy_to_999(
    xyxy: tuple[float, float, float, float],
    size: tuple[int, int],
) -> list[int]:
    """Serialize crop-pixel xyxy in the Phase 2 0..999 integer JSON frame.
    将裁切图像素 xyxy 序列化为 Phase 2 的 0..999 整数 JSON 坐标。"""
    width, height = size
    values = (
        round(xyxy[0] / width * 999),
        round(xyxy[1] / height * 999),
        round(xyxy[2] / width * 999),
        round(xyxy[3] / height * 999),
    )
    return [
        max(0, min(999, int(values[0]))),
        max(0, min(999, int(values[1]))),
        max(0, min(999, int(values[2]))),
        max(0, min(999, int(values[3]))),
    ]


@dataclass(frozen=True)
class EvidencePolicy:
    """Inject-only policy for executor parameters whose values are not yet
    frozen by the user. Every value is required — no production default is
    filled anywhere. 执行器未冻结参数的注入策略。所有值必填——任何位置不
    填写生产默认值。"""

    confidence_threshold: float
    nms_iou_threshold: float
    max_detections: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.confidence_threshold):
            raise ValueError("confidence_threshold must be finite")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0.0, 1.0]")
        if not math.isfinite(self.nms_iou_threshold):
            raise ValueError("nms_iou_threshold must be finite")
        if not 0.0 <= self.nms_iou_threshold <= 1.0:
            raise ValueError("nms_iou_threshold must be within [0.0, 1.0]")
        if self.max_detections < 1:
            raise ValueError("max_detections must be at least 1")


@dataclass(frozen=True)
class RoiLeafOutcome:
    """One per-(ROI, leaf, layer) in-memory diagnostic. error_code is a stable
    classification (exception type name or a fixed code), never a raw message.
    一条逐（ROI，叶子，层）的内存诊断。error_code 是稳定分类（异常类型名或
    固定 code），绝不携带原始消息。"""

    roi_id: str
    leaf_category: str
    layer: EvidenceLayer
    state: EvidenceState
    error_code: str | None = None


@dataclass(frozen=True)
class EvidenceExecution:
    """Result of one executor pass: the JSON-safe bundle plus in-memory-only
    diagnostics. masks and outcomes never enter persisted artifacts; the
    bundle is the only thing the final Qwen call consumes. The executor
    deliberately exposes no sample-level status field — that decision is
    unfrozen and belongs to the runtime. 一次执行的结果：JSON 安全 bundle 加
    纯内存诊断。masks 与 outcomes 绝不进入持久化产物；bundle 是最终 Qwen
    调用唯一消费的东西。执行器刻意不暴露任何 sample 级状态字段——该决定
    未冻结，属于运行时。"""

    bundle: VqaEvidenceBundle
    layer_states: tuple[LayerStateRecord, ...]
    outcomes: tuple[RoiLeafOutcome, ...]
    masks: Mapping[tuple[str, str], Any]
    # Deterministic SegFormer mask palette (frozen 14.12.2), computed once per
    # executor from catalog segformer-capable leaves in catalog order. It is
    # in-memory-only: never persisted, but folded into the final-Qwen request
    # hash through the rendered mask image digests and evidence_identity.
    # 确定性 SegFormer mask 调色表（冻结 14.12.2），按 catalog 中启用 segformer
    # 的叶子顺序每执行器计算一次。仅存内存：绝不持久化，但通过渲染 mask 图像
    # 摘要与 evidence_identity 折叠进最终 Qwen 请求 hash。
    palette: Mapping[str, tuple[int, int, int]] = field(default_factory=dict)


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


def _segformer_audit_identity(
    output: SemanticMaskOutput | None,
    client: SemanticMaskClient,
) -> tuple[str, str | None]:
    """Stable audit identity of one segment call: the logical model id comes
    from the call's own diagnostics when available, otherwise from the client
    cache identity; the weights digest likewise. Never a physical path.
    一次 segment 调用的稳定审计身份：逻辑模型 id 优先取调用自身 diagnostics，
    否则取客户端缓存身份；权重摘要同理。绝不出现物理路径。"""
    if output is not None:
        diagnostics = getattr(output, "diagnostics", None)
        if isinstance(diagnostics, Mapping):
            model_id = diagnostics.get("logical_model_id")
            if isinstance(model_id, str) and model_id:
                return model_id, output.weights_sha256
    return _audit_identity([], client)


class ObjectEvidenceExecutor:
    """Execute the frozen VQA evidence state machine with injected clients.
    The executor is synchronous and deterministic: same plan, same images,
    same policy -> same bundle. 执行冻结 VQA 证据状态机，客户端注入。执行器
    同步且确定：相同计划、相同图像、相同策略 -> 相同 bundle。"""

    def __init__(
        self,
        *,
        catalog: EvidenceCatalog,
        policy: EvidencePolicy | None,
        yolo_client: ObjectDetectionClient | None,
        yolo_device: str | None,
        yolo_image_size: int | None,
        segmenter_clients: Mapping[str, SemanticMaskClient],
        preprocessing: EvidencePreprocessing,
    ) -> None:
        # The YOLO client and its device/image-size policy are all-or-none;
        # the detector policy is inject-only and required exactly when a
        # YOLO client is present. The frozen tile protocol fixes the model
        # reference size at 1024 square, so any other injected size fails
        # closed instead of silently letterboxing to the wrong resolution.
        #  YOLO client 与 device/image-size 策略全有或全无；detector 策略仅
        # 注入，且仅在存在 YOLO client 时必须提供。冻结 tile 协议固定模型
        # 参考尺寸为 1024 方形，因此注入任何其他尺寸都严格失败，而不是悄悄
        # letterbox 到错误分辨率。
        if (yolo_client is None) != (yolo_device is None):
            raise ValueError("yolo_device is required exactly when a yolo client is present")
        if (yolo_client is None) != (yolo_image_size is None):
            raise ValueError(
                "yolo_image_size is required exactly when a yolo client is present"
            )
        if yolo_client is not None:
            if yolo_image_size is None or yolo_image_size <= 0:
                raise ValueError("yolo_image_size must be positive")
            if yolo_image_size != MODEL_INPUT_SIZE:
                raise ValueError(
                    "the frozen evidence tile protocol requires yolo_image_size 1024"
                )
            if policy is None:
                raise ValueError("a yolo client requires the inject-only detector policy")
        if not isinstance(segmenter_clients, Mapping) or any(
            not isinstance(binding, str) or not binding for binding in segmenter_clients
        ):
            raise ValueError("segmenter_clients must map non-empty binding ids to clients")
        self._catalog = catalog
        self._policy = policy
        self._yolo_client = yolo_client
        self._yolo_device = yolo_device
        self._yolo_image_size = yolo_image_size
        self._segmenter_clients = segmenter_clients
        self._preprocessing = preprocessing
        # One deterministic palette per executor over the catalog's segformer
        # leaf order; rendering and the final Qwen content both draw from it.
        # 每执行器一份按 catalog segformer 叶子顺序的确定性调色表；渲染与
        # 最终 Qwen content 都取自它。
        self._palette = segformer_palette(
            [
                leaf
                for leaf in catalog.leaf_categories
                if catalog.capability_enabled(leaf, "segformer")
            ]
        )

    def execute(
        self,
        plan: VisualTaskPlan,
        images: Mapping[str, Image.Image],
        *,
        fallback_image_id: str,
        materialized_views: tuple[MaterializedVisualView, ...],
    ) -> EvidenceExecution:
        """Execute evidence against the already materialized views.
        使用已物化的视图执行证据流程。"""
        return self._execute_plan(
            plan,
            images,
            fallback_image_id=fallback_image_id,
            materialized_views=materialized_views,
        )

    def _execute_plan(
        self,
        plan: VisualTaskPlan,
        images: Mapping[str, Image.Image],
        *,
        fallback_image_id: str,
        materialized_views: tuple[MaterializedVisualView, ...],
    ) -> EvidenceExecution:
        """Run the frozen state machine over one plan. The plan must be the
        object_evidence_vqa family with a validated evidence request; the
        bundle is assembled by roi_id and stable leaf order. All accumulation
        state is re-created per call, so one executor serves many plans without
        any cross-plan leakage. A single bounded worker pool serves one
        execution lifecycle and is closed before the bundle is assembled.
        对一个计划运行冻结状态机。计划必须是 object_evidence_vqa 家族且携带
        已校验证据请求；bundle 按 roi_id 与稳定 leaf order 组装。所有累积状态
        在每次调用时重新创建，一个执行器可服务多个计划而无跨计划泄漏。一次
        execution 生命周期使用单个有界 worker pool，在组装 bundle 前关闭。"""
        self._audits: list[ModelCallAudit] = []
        self._outcomes: list[RoiLeafOutcome] = []
        self._masks: dict[tuple[str, str], Any] = {}
        self._segformer_preprocess: list[SegFormerPreprocessRecord] = []

        if not plan.needs_visual_assistance:
            raise ValueError("v2 evidence executor requires visual assistance")
        if not materialized_views:
            raise ValueError("v2 evidence executor requires materialized views")
        leaves = self._catalog.validate_plan_leaves(
            plan.object_categories,
            task="general_vqa",
        )
        records = self._materialized_regions(materialized_views, images)
        tile_plan = self._materialize_tiles(records, images)

        with ThreadPoolExecutor(
            max_workers=self._preprocessing.max_tile_concurrency
        ) as pool:
            hit_leaves, detections = self._yolo_phase(
                pool, tile_plan, records, leaves, images
            )
            segments = self._segformer_phase(
                pool, records, leaves, hit_leaves, images
            )
        layer_states, final_states, missing = self._aggregate(leaves)
        bundle = VqaEvidenceBundle(
            catalog_version=self._catalog.catalog_version,
            preprocessing_version=self._preprocessing.version,
            tiles=[tile_record for tile_record, _, _ in tile_plan],
            segformer_preprocess=self._segformer_preprocess,
            rois=[record for record in records],
            detections=detections,
            segments=segments,
            missing_leaves=missing,
            leaf_states={leaf: final_states[leaf] for leaf in leaves},
            call_audit=self._audits,
        )
        return EvidenceExecution(
            bundle=bundle,
            layer_states=tuple(layer_states),
            outcomes=tuple(self._outcomes),
            masks=dict(self._masks),
            palette=dict(self._palette),
        )

    def _materialized_regions(
        self,
        views: tuple[MaterializedVisualView, ...],
        images: Mapping[str, Image.Image],
    ) -> list[RoiEvidenceRecord]:
        """Convert frozen views into exact evidence geometry.
        将冻结的视图转换为精确证据几何。"""
        if not views:
            raise ValueError("materialized views must not be empty")
        records: list[RoiEvidenceRecord] = []
        for index, view in enumerate(views):
            image = images.get(view.image_id)
            if image is None or image.size != view.source_size:
                raise ValueError("materialized view source size does not match image")
            roi_id = (
                "full"
                if len(views) == 1 and view.view_mode == "full_image"
                else f"{view.view_mode}-{index}"
            )
            box = view.crop_xyxy
            records.append(
                RoiEvidenceRecord(
                    roi_id=roi_id,
                    image_id=view.image_id,
                    source_size=view.source_size,
                    core_xyxy=box,
                    expanded_xyxy=box,
                    crop_size=view.crop_size,
                )
            )
        return records

    def _materialize_tiles(
        self,
        records: list[RoiEvidenceRecord],
        images: Mapping[str, Image.Image],
    ) -> list[tuple[EvidenceTileRecord, Image.Image, RoiEvidenceRecord]]:
        """Synchronously materialize every ROI crop and its deterministic
        1024x1024 tiles before any model call; this stable plan order is the
        only merge order, never completion order. 在任何模型调用前同步物化每
        个 ROI 裁切及其确定性 1024x1024 tiles；这份稳定 plan 顺序是唯一合并
        顺序，绝不使用完成顺序。"""
        plan: list[tuple[EvidenceTileRecord, Image.Image, RoiEvidenceRecord]] = []
        for record in records:
            crop = render_roi_crop(images[record.image_id], record)
            for tile_record in partition_roi(
                record, tile_size=self._preprocessing.tile_size
            ):
                plan.append((tile_record, prepare_model_tile(crop, tile_record), record))
        return plan

    # ── worker jobs / worker 任务 ───────────────────────────────────────

    def _call_yolo_tile(
        self,
        tile_image: Image.Image,
        tile_record: EvidenceTileRecord | None = None,
    ) -> tuple[list[ObjectDetectionOutput] | None, str | None]:
        """One YOLO call on one prepared 1024x1024 tile, isolated in this
        worker: exceptions become a stable type-name code, and every returned
        detection must reference the strict 1024-square model input or the
        tile call fails closed. When M3_GPU_MONITOR=1 the CUDA memory hook
        records before/after/after_error events with the tile geometry.
        对一张已准备的 1024x1024 tile 执行一次 YOLO 调用，并在 worker 内隔离：
        异常转稳定类型名；所有返回检测必须引用严格 1024 方形模型输入，否则该
        tile 调用严格失败。M3_GPU_MONITOR=1 时 CUDA 显存 hook 记录
        before/after/after_error 事件并附带 tile 几何。"""
        try:
            from scripts.gpu_memory_monitor import log_cuda_memory_event
        except Exception:
            log_cuda_memory_event = None
        meta: dict[str, Any] = {}
        if tile_record is not None:
            meta = {
                "tile_id": tile_record.tile_id,
                "roi_id": tile_record.roi_id,
                "source_tile_xyxy": list(tile_record.source_tile_xyxy),
                "source_tile_size": list(tile_record.source_tile_size),
                "model_input_size": list(tile_record.model_input_size),
                "resize_applied": tile_record.resize_applied,
                "tile_image_size": list(tile_image.size),
            }
        if log_cuda_memory_event is not None:
            log_cuda_memory_event("yolo", "before", **meta)
        try:
            outputs = self._yolo_client.detect(
                tile_image,
                confidence=self._policy.confidence_threshold,
                iou=self._policy.nms_iou_threshold,
                image_size=self._yolo_image_size,
                device=self._yolo_device,
                max_detections=self._policy.max_detections,
            )
        except Exception as exc:
            if log_cuda_memory_event is not None:
                log_cuda_memory_event(
                    "yolo", "after_error", error=type(exc).__name__, **meta
                )
            return None, type(exc).__name__
        if log_cuda_memory_event is not None:
            log_cuda_memory_event("yolo", "after", **meta)
        if not outputs:
            return [], None
        for output in outputs:
            if (
                output.input_width != MODEL_INPUT_SIZE
                or output.input_height != MODEL_INPUT_SIZE
            ):
                return None, "unexpected_model_input_size"
        return outputs, None

    def _call_segformer_roi(
        self,
        client: SemanticMaskClient,
        model_input: Image.Image,
        roi_id: str | None = None,
        binding: str | None = None,
    ) -> tuple[SemanticMaskOutput | None, str | None]:
        """One segment call on one prepared 1024x1024 model input of the pad
        protocol, isolated in this worker: exceptions become a stable
        type-name code, and the returned map must stay aligned to the
        1024-square model input or the call fails closed. When
        M3_GPU_MONITOR=1 the CUDA memory hook records before/after/after_error
        events with the ROI/binding identity. 对 pad 协议下的一张已准备
        1024x1024 模型输入执行一次 segment 调用，并在 worker 内隔离：异常转
        稳定类型名；返回 map 必须保持与 1024 方形模型输入对齐，否则该调用
        严格失败。M3_GPU_MONITOR=1 时 CUDA 显存 hook 记录
        before/after/after_error 事件并附带 ROI/binding 身份。"""
        try:
            from scripts.gpu_memory_monitor import log_cuda_memory_event
        except Exception:
            log_cuda_memory_event = None
        meta: dict[str, Any] = {}
        if roi_id is not None:
            meta = {
                "roi_id": roi_id,
                "binding": binding,
                "model_input_size": list(model_input.size),
            }
        if log_cuda_memory_event is not None:
            log_cuda_memory_event("segformer", "before", **meta)
        try:
            output = client.segment(model_input)
        except Exception as exc:
            if log_cuda_memory_event is not None:
                log_cuda_memory_event(
                    "segformer", "after_error", error=type(exc).__name__, **meta
                )
            return None, type(exc).__name__
        if log_cuda_memory_event is not None:
            log_cuda_memory_event("segformer", "after", **meta)
        if output.original_size != (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
            return None, "unexpected_model_input_size"
        return output, None

    # ── helpers / 辅助 ─────────────────────────────────────────────────

    def _yolo_phase(
        self,
        pool: ThreadPoolExecutor,
        tile_plan: list[tuple[EvidenceTileRecord, Image.Image, RoiEvidenceRecord]],
        records: list[RoiEvidenceRecord],
        leaves: tuple[str, ...],
        images: Mapping[str, Image.Image],
    ) -> tuple[set[str], list[YoloDetectionRecord]]:
        """One detector call per prepared tile, dispatched into the shared
        bounded pool and merged in stable tile order. A leaf with no YOLO
        capability never triggers a call; requests with no YOLO-capable leaf
        at all make zero calls. Confidence stays internal: thresholding,
        per-(roi, leaf) top-k retention, and the cross-ROI dedup in
        whole-image coordinates. 对每个已准备 tile 各调用一次 detector，派发到
        共享有界 pool 并按稳定 tile 顺序合并。无 YOLO 能力的叶子绝不触发调用；
        请求中完全没有 YOLO 能力叶子时零调用。confidence 仅内部消费：阈值、
        逐（ROI，叶子）top-k 保留与 whole-image 坐标下的跨 ROI 去重。"""
        yolo_leaves = [
            leaf for leaf in leaves if self._catalog.capability_enabled(leaf, "yolo")
        ]
        for record in records:
            for leaf in leaves:
                if leaf not in yolo_leaves:
                    self._outcomes.append(
                        RoiLeafOutcome(record.roi_id, leaf, "yolo", "unsupported")
                    )
        if self._yolo_client is None:
            for record in records:
                for leaf in yolo_leaves:
                    self._outcomes.append(
                        RoiLeafOutcome(record.roi_id, leaf, "yolo", "unavailable")
                    )
            return set(), []
        if not yolo_leaves:
            return set(), []
        futures = [
            pool.submit(self._call_yolo_tile, tile_image, tile_record)
            for tile_record, tile_image, _ in tile_plan
        ]
        results = [future.result() for future in futures]
        candidates: dict[
            tuple[str, str], list[tuple[float, ObjectDetectionOutput, int]]
        ] = {}
        failed_tiles: dict[tuple[str, str], str] = {}
        for index, (tile_record, _, record) in enumerate(tile_plan):
            outputs, failed_code = results[index]
            logical_model_id, digest = _audit_identity(
                outputs or [], self._yolo_client
            )
            self._audits.append(
                ModelCallAudit(
                    layer="yolo",
                    roi_id=record.roi_id,
                    tile_id=tile_record.tile_id,
                    input_size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                    logical_model_id=logical_model_id,
                    weights_sha256=digest,
                    status="failed" if failed_code is not None else "succeeded",
                    error_code=failed_code,
                )
            )
            if failed_code is not None:
                for leaf in yolo_leaves:
                    failed_tiles.setdefault((record.roi_id, leaf), failed_code)
                continue
            for leaf in yolo_leaves:
                leaf_labels = set(self._catalog.leaf_yolo_labels(leaf))
                for output in outputs:
                    if output.label in leaf_labels:
                        candidates.setdefault((record.roi_id, leaf), []).append(
                            (output.confidence, output, index)
                        )
        detected: list[tuple[float, YoloDetectionRecord]] = []
        degenerate: set[tuple[str, str]] = set()
        hits: set[tuple[str, str]] = set()
        for (roi_id, leaf), items in candidates.items():
            retained = sorted(items, key=lambda item: item[0], reverse=True)[
                : self._policy.max_detections
            ]
            for confidence, output, index in retained:
                tile_record, _, record = tile_plan[index]
                try:
                    local_box = model_xyxy_to_roi_xyxy(output.xyxy, tile_record)
                except ValueError:
                    # Clipped-away degenerate boxes are dropped; a minimum box
                    # is never fabricated. 裁剪后退化的框直接丢弃；绝不伪造最小框。
                    degenerate.add((roi_id, leaf))
                    continue
                detected.append(
                    (
                        confidence,
                        YoloDetectionRecord(
                            leaf_category=leaf,
                            roi_id=roi_id,
                            local_xyxy=local_box,
                            local_roi_size=record.crop_size,
                            global_xyxy=local_to_global(local_box, record),
                            global_image_size=images[record.image_id].size,
                        ),
                    )
                )
                hits.add((roi_id, leaf))
        hit_leaves: set[str] = set()
        for record in records:
            for leaf in yolo_leaves:
                key = (record.roi_id, leaf)
                if key in hits:
                    hit_leaves.add(leaf)
                    self._outcomes.append(
                        RoiLeafOutcome(record.roi_id, leaf, "yolo", "hit")
                    )
                elif key in failed_tiles:
                    self._outcomes.append(
                        RoiLeafOutcome(
                            record.roi_id, leaf, "yolo", "error", failed_tiles[key]
                        )
                    )
                elif key in degenerate:
                    self._outcomes.append(
                        RoiLeafOutcome(
                            record.roi_id, leaf, "yolo", "missing", "degenerate_box"
                        )
                    )
                else:
                    self._outcomes.append(
                        RoiLeafOutcome(record.roi_id, leaf, "yolo", "missing")
                    )
        return hit_leaves, self._dedup_global(detected)

    def _dedup_global(
        self,
        detected: list[tuple[float, YoloDetectionRecord]],
    ) -> list[YoloDetectionRecord]:
        """Greedy cross-ROI dedup in whole-image coordinates: sort by internal
        confidence descending (stable sort, ties keep roi/leaf order), keep a
        detection only when its global box overlaps no kept box by IoU >= the
        policy threshold. The higher internal confidence wins; confidence never
        leaves this method. 在 whole-image 坐标下贪心跨 ROI 去重：按内部
        confidence 降序（稳定排序，同分保持 roi/leaf 顺序）排序，仅当全局框
        与已保留框的 IoU 低于策略阈值时保留。内部置信度更高者胜出；confidence
        绝不离开本方法。"""
        kept: list[YoloDetectionRecord] = []
        for confidence, record in sorted(
            detected, key=lambda item: item[0], reverse=True
        ):
            if any(
                _iou(record.global_xyxy, other.global_xyxy)
                >= self._policy.nms_iou_threshold
                for other in kept
            ):
                continue
            kept.append(record)
        return kept

    def _segformer_phase(
        self,
        pool: ThreadPoolExecutor,
        records: list[RoiEvidenceRecord],
        leaves: tuple[str, ...],
        hit_leaves: set[str],
        images: Mapping[str, Image.Image],
    ) -> list[SegFormerEvidenceRecord]:
        """Resolve each still-missing leaf through its verified segmenter
        binding: one call per (ROI, binding) into the shared bounded pool,
        merged in roi order -> binding order. Only requested leaves become
        evidence — unrequested classes stay background, no box/count is
        derived, no cross-checkpoint comparison happens, and a YOLO hit is
        never re-run or overwritten. The output class map must match the
        catalog raw labels or the leaf fails closed. The fresh SegFormer
        protocol is the pad-multiple-1024-resize-square one only: under the
        legacy v1 version every leaf that would need a fresh SegFormer call
        fails closed instead of silently running the new protocol under the
        old version label. 通过已验证 segmenter binding 解析每个仍缺失叶子：
        每个（ROI，binding）调用一次（共享有界 pool），按 roi order -> binding
        order 合并。只有请求叶子成为证据——未请求类别保持背景，不派生框/计数，
        不做跨 checkpoint 比较，YOLO 命中绝不重跑或覆盖。输出 class map 必须
        与 catalog raw labels 一致，否则叶子严格失败。新鲜 SegFormer 协议仅限
        pad-multiple-1024-resize-square：在旧 v1 版本下，任何需要新鲜
        SegFormer 调用的叶子都严格失败，绝不悄悄在旧版本标签下运行新协议。"""
        binding_leaves: dict[str, dict[str, list[str]]] = {}
        binding_order: list[str] = []
        for record in records:
            for leaf in leaves:
                if leaf in hit_leaves:
                    if self._catalog.capability_enabled(leaf, "segformer"):
                        self._outcomes.append(
                            RoiLeafOutcome(record.roi_id, leaf, "segformer", "not_run")
                        )
                    continue
                if not self._catalog.capability_enabled(leaf, "segformer"):
                    self._outcomes.append(
                        RoiLeafOutcome(record.roi_id, leaf, "segformer", "unsupported")
                    )
                    continue
                binding = self._catalog.leaf_segformer_binding(leaf)
                if binding is None or binding not in self._segmenter_clients:
                    self._outcomes.append(
                        RoiLeafOutcome(record.roi_id, leaf, "segformer", "unavailable")
                    )
                    continue
                if binding not in binding_order:
                    binding_order.append(binding)
                binding_leaves.setdefault(binding, {}).setdefault(
                    record.roi_id, []
                ).append(leaf)
        if not binding_order:
            return []
        if self._preprocessing.version == "greedy-1024-stretch-v1":
            # The legacy stretch protocol is read-only for historical
            # artifacts; a fresh SegFormer call under it would silently run
            # the new pad protocol under the old version label. Fail closed
            # for every leaf that would need such a call.
            # 旧 stretch 协议只用于历史 artifact 只读解释；在其下发起新鲜
            # SegFormer 调用等于在旧版本标签下悄悄运行新 pad 协议。对所有
            # 需要此类调用的叶子严格失败。
            for binding in binding_order:
                for roi_id, roi_leaves in binding_leaves[binding].items():
                    for leaf in roi_leaves:
                        self._outcomes.append(
                            RoiLeafOutcome(
                                roi_id,
                                leaf,
                                "segformer",
                                "error",
                                "legacy_segformer_protocol_unsupported",
                            )
                        )
            return []
        # One deterministic strict 1024x1024 model input per (ROI, binding)
        # group, materialized synchronously before any model call; the stable
        # plan order is the only merge order, never completion order. The
        # geometry record is per ROI, the call is per (ROI, binding).
        # 每个（ROI，binding）组一个确定性严格 1024×1024 模型输入，在任何模型
        # 调用前同步物化；这份稳定 plan 顺序是唯一合并顺序，绝不使用完成顺序。
        # 几何记录按 ROI，调用按（ROI，binding）。
        groups: list[
            tuple[
                RoiEvidenceRecord,
                str,
                list[str],
                SegFormerPreprocessRecord,
                Image.Image,
            ]
        ] = []
        for record in records:
            crop = render_roi_crop(images[record.image_id], record)
            for binding in binding_order:
                roi_leaves = binding_leaves[binding].get(record.roi_id)
                if not roi_leaves:
                    continue
                preprocess, model_input = prepare_segformer_roi(
                    crop, roi_id=record.roi_id, source_size=record.crop_size
                )
                groups.append((record, binding, roi_leaves, preprocess, model_input))
        futures = [
            pool.submit(
                self._call_segformer_roi,
                self._segmenter_clients[binding],
                model_input,
                record.roi_id,
                binding,
            )
            for record, binding, _, _, model_input in groups
        ]
        results = [future.result() for future in futures]
        segments: list[SegFormerEvidenceRecord] = []
        for (record, binding, roi_leaves, preprocess, _), (output, failed_code) in zip(
            groups, results
        ):
            self._segformer_preprocess.append(preprocess)
            logical_model_id, digest = _segformer_audit_identity(
                output, self._segmenter_clients[binding]
            )
            self._audits.append(
                ModelCallAudit(
                    layer="segformer",
                    roi_id=record.roi_id,
                    tile_id=None,
                    input_size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
                    logical_model_id=logical_model_id,
                    weights_sha256=digest,
                    status="failed" if failed_code is not None else "succeeded",
                    error_code=failed_code,
                )
            )
            if failed_code is not None or output is None:
                # The whole (ROI, binding) call failed: only this ROI's leaves
                # on this binding record the stable error; other ROIs,
                # bindings and the already-successful YOLO evidence stay
                # untouched, and no half mask or guessed background is used.
                # 整次（ROI，binding）调用失败：仅该 ROI 中依赖该 binding 的
                # 叶子记录稳定 error；其他 ROI、binding 与已成功的 YOLO evidence
                # 不受影响，绝不使用半张 mask 或猜测 background。
                for leaf in roi_leaves:
                    self._outcomes.append(
                        RoiLeafOutcome(
                            record.roi_id,
                            leaf,
                            "segformer",
                            "error",
                            failed_code or "unknown_call_failure",
                        )
                    )
                continue
            # Strict class-map verification: every requested leaf's raw labels
            # must exist in the model's authoritative label map.
            # 严格 class map 校验：每个请求叶子的 raw labels 必须存在于模型
            # 权威标签映射中。
            id_to_label = output.id_to_label
            label_map = id_to_label if isinstance(id_to_label, Mapping) else {}
            leaf_class_ids: dict[str, frozenset[int]] = {}
            mismatched: set[str] = set()
            for leaf in roi_leaves:
                labels = set(self._catalog.leaf_segformer_labels(leaf) or ())
                class_ids = frozenset(
                    class_id
                    for class_id, label in label_map.items()
                    if label in labels
                )
                if class_ids:
                    leaf_class_ids[leaf] = class_ids
                else:
                    mismatched.add(leaf)
            for leaf in mismatched:
                self._outcomes.append(
                    RoiLeafOutcome(
                        record.roi_id,
                        leaf,
                        "segformer",
                        "error",
                        "class_map_mismatch",
                    )
                )
            # Restore the whole class-id map once (NEAREST to padded size,
            # crop [0:W, 0:H]), then extract boolean masks for every leaf, so
            # class ids are never re-interpolated per leaf and the final mask
            # sizes are equal by construction.
            # 先整体恢复 class-id map 一次（NEAREST 到 padded 尺寸，裁切
            # [0:W, 0:H]），再为每个叶子提取 boolean mask，使 class id 绝不逐
            # 叶子重复插值，各叶子最终 mask 尺寸天然一致。
            try:
                grid = _to_class_grid(output.class_id_map)
                restored = restore_segformer_class_id_mask(grid, preprocess)
            except ValueError:
                for leaf in roi_leaves:
                    if leaf not in mismatched:
                        self._outcomes.append(
                            RoiLeafOutcome(
                                record.roi_id,
                                leaf,
                                "segformer",
                                "error",
                                "mask_geometry",
                            )
                        )
                continue
            if restored.size != record.crop_size:
                for leaf in roi_leaves:
                    if leaf not in mismatched:
                        self._outcomes.append(
                            RoiLeafOutcome(
                                record.roi_id,
                                leaf,
                                "segformer",
                                "error",
                                "mask_geometry",
                            )
                        )
                continue
            for leaf in roi_leaves:
                if leaf in mismatched:
                    continue
                boolean = _leaf_boolean_grid(restored, leaf_class_ids[leaf])
                if boolean.getextrema()[1] > 0:
                    segments.append(
                        SegFormerEvidenceRecord(
                            leaf_category=leaf, roi_id=record.roi_id
                        )
                    )
                    self._masks[(record.roi_id, leaf)] = boolean
                    self._outcomes.append(
                        RoiLeafOutcome(record.roi_id, leaf, "segformer", "hit")
                    )
                else:
                    self._outcomes.append(
                        RoiLeafOutcome(record.roi_id, leaf, "segformer", "missing")
                    )
        # One geometry record per ROI with at least one fresh SegFormer call,
        # in ROI order; the record is binding-independent.
        # 每个发生过新鲜 SegFormer 调用的 ROI 一条几何记录，按 ROI 顺序；该
        # 记录与 binding 无关。
        deduped: list[SegFormerPreprocessRecord] = []
        seen_roi: set[str] = set()
        for preprocess in self._segformer_preprocess:
            if preprocess.roi_id not in seen_roi:
                seen_roi.add(preprocess.roi_id)
                deduped.append(preprocess)
        self._segformer_preprocess = deduped
        return segments

    def _aggregate(
        self,
        leaves: tuple[str, ...],
    ) -> tuple[list[LayerStateRecord], dict[str, EvidenceState], list[str]]:
        """Deterministic per-leaf layer aggregation: for one (leaf, layer) the
        state is hit > error > unavailable > missing > unsupported. The final
        leaf state is the deepest layer's state; a yolo hit leaf gets a
        not_run segformer record only when it has an approved segformer
        capability. This decides leaf states only — the sample's final success
        status is never decided here. 逐（叶子，层）确定性聚合：状态优先级
        hit > error > unavailable > missing > unsupported。叶子最终状态取最深
        层状态；yolo 命中叶子仅在具备批准 segformer 能力时给出 not_run
        segformer 记录。这里只决定叶子状态——sample 最终成功状态绝不在此
        决定。"""
        layer_states: list[LayerStateRecord] = []
        final_states: dict[str, EvidenceState] = {}
        missing: list[str] = []
        for leaf in leaves:
            yolo_state = self._aggregate_layer(leaf, "yolo")
            layer_states.append(
                LayerStateRecord(leaf_category=leaf, layer="yolo", state=yolo_state)
            )
            if yolo_state == "hit":
                final_states[leaf] = "hit"
                if self._catalog.capability_enabled(leaf, "segformer"):
                    layer_states.append(
                        LayerStateRecord(
                            leaf_category=leaf, layer="segformer", state="not_run"
                        )
                    )
                continue
            seg_state = self._aggregate_layer(leaf, "segformer")
            if seg_state is None:
                seg_state = "unsupported"
            layer_states.append(
                LayerStateRecord(leaf_category=leaf, layer="segformer", state=seg_state)
            )
            final_states[leaf] = seg_state
            # Only a leaf whose deepest layer is hit is complete; every other
            # final state stays missing for the final-Qwen fallback.
            # 只有最深层层状态为 hit 的叶子才算完备；其余终态均保持缺失，进入
            # 最终 Qwen 回退。
            if seg_state != "hit":
                missing.append(leaf)
        return layer_states, final_states, missing

    def _aggregate_layer(
        self,
        leaf: str,
        layer: EvidenceLayer,
    ) -> EvidenceState | None:
        """Severity-ordered aggregation of one (leaf, layer) across ROIs; None
        when the layer never produced an outcome (no call, no capability).
        对一个（叶子，层）跨 ROI 做严重度排序聚合；层从未产生结果（无调用、
        无能力）时为 None。"""
        states = [
            outcome.state
            for outcome in self._outcomes
            if outcome.leaf_category == leaf and outcome.layer == layer
        ]
        if not states:
            return None
        for state in ("hit", "error", "unavailable", "missing"):
            if state in states:
                return state  # type: ignore[return-value]
        return "unsupported"
