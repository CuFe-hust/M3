"""Frozen VQA object-evidence executor (C5, 14A1) — fake clients only, not
wired to GeneralVQAAgent.run() or SampleRunner.

C5 冻结 VQA 对象证据执行器——仅 fake clients，不接 GeneralVQAAgent.run() 或
SampleRunner。

Frozen state machine / 冻结状态机：

    expand requested composite categories to ordered leaf categories
    for each ROI: run YOLO once, then filter all requested leaves
    for each still-missing leaf:
        if catalog has approved SegFormer capability:
            run SegFormer inference once per ROI, then filter
        if still missing:
            leave the leaf for the single final-Qwen visual fallback
    preserve all successful evidence from other leaves

States / 状态：hit / missing / unsupported / unavailable / error / not_run；
不存在 valid_empty（成功但筛选为空 = missing）。已 hit 的叶子绝不重跑或覆盖。

VQA-only constraints / VQA 专属约束：

- YOLO/SegFormer 调用次数按 ROI，不按类别增长；模型输入只有 ROI 裁切图；
- 只保留目录请求标签，未请求输出全部丢弃；
- YOLO confidence 仅供阈值/去重/冲突裁决，绝不进入最终 bundle 或公共 trace；
- SegFormer 只保留 mask/存在性证据，不转 box、不生成 instance count；mask
  在各 ROI 内独立保留，绝不跨 ROI 像素融合或比较置信度；
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

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from PIL import Image

from agents.evidence_catalog import EvidenceCatalog
from agents.general_vqa.evidence.geometry import local_to_global, resolve_roi_records
from agents.general_vqa.evidence.rendering import render_roi_crop
from agents.general_vqa.evidence.schema import (
    EvidenceLayer,
    EvidenceState,
    LayerStateRecord,
    ModelCallAudit,
    RoiEvidenceRecord,
    SegFormerEvidenceRecord,
    VqaEvidenceBundle,
    YoloDetectionRecord,
)
from agents.schema import FirstQwenVisualPlan, RoiRegion
from models.base import (
    DenseSemanticClient,
    ObjectDetectionClient,
    ObjectDetectionOutput,
)

# Stable error code for an approved SegFormer label absent from the client's
# class map; the approved mapping itself is wrong and must never be guessed.
# 已批准 SegFormer 标签不在客户端类别映射中的稳定错误码；已批准映射本身
# 错误，绝不猜测。
SEGFORMER_CLASS_MAP_MISMATCH = "SEGFORMER_CLASS_MAP_MISMATCH"


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


def _mask_presence(probabilities: Any, class_indices: list[int], threshold: float) -> Any:
    """Duck-typed boolean presence mask: pixels whose argmax class is ANY of
    the leaf's class indices (OR, not AND — a pixel has exactly one argmax
    class, so consecutive ANDs of mutually exclusive classes would always be
    empty) with max probability at least the threshold. No box, no count, no
    runtime import — operates on the client's arrays through their own
    methods only. 鸭子类型布尔存在掩膜：argmax 类别属于叶子类别索引中任意一
    个（OR 而非 AND——像素只有一个 argmax 类别，对互斥类别连续 AND 必然为空）
    且最大概率不低于阈值的像素。不转框、不计数、不导入运行时——只通过客户端
    数组自身的方法操作。"""
    max_prob = probabilities.max(axis=0)
    best = probabilities.argmax(axis=0)
    present = max_prob >= threshold
    if not class_indices:
        # No approved labels: the leaf cannot be present anywhere; keep the
        # presence honest instead of treating every pixel as the leaf.
        # 无已批准标签：叶子任何位置都不可能存在；保持存在性诚实，而不是把
        # 每个像素都当成该叶子。
        return present & (best != best)
    class_mask = best == class_indices[0]
    for index in class_indices[1:]:
        class_mask = class_mask | (best == index)
    return present & class_mask


class ObjectEvidenceExecutor:
    """Execute the frozen VQA evidence state machine with injected clients.
    The executor is synchronous and deterministic: same plan, same images,
    same policy -> same bundle. 执行冻结 VQA 证据状态机，客户端注入。执行器
    同步且确定：相同计划、相同图像、相同策略 -> 相同 bundle。"""

    def __init__(
        self,
        *,
        catalog: EvidenceCatalog,
        policy: EvidencePolicy,
        yolo_client: ObjectDetectionClient | None,
        yolo_device: str,
        yolo_image_size: int,
        segformer_client: DenseSemanticClient | None,
        segformer_tile_size: int | None = None,
        segformer_tile_overlap: int | None = None,
        segformer_feature_stage: int | None = None,
    ) -> None:
        if yolo_image_size <= 0:
            raise ValueError("yolo_image_size must be positive")
        # Tile policy is inject-only: no production default exists, so the
        # values are required exactly when a SegFormer client is present and
        # otherwise stay None. 切片策略仅注入：无生产默认值，因此仅当存在
        # SegFormer client 时才必须提供，否则保持 None。
        if segformer_tile_size is not None and segformer_tile_size <= 0:
            raise ValueError("segformer_tile_size must be positive")
        if segformer_tile_overlap is not None and (
            segformer_tile_size is None
            or not 0 <= segformer_tile_overlap < segformer_tile_size
        ):
            raise ValueError("segformer_tile_overlap must be within [0, tile_size)")
        if segformer_feature_stage is not None and segformer_feature_stage < 0:
            raise ValueError("segformer_feature_stage must be non-negative")
        if segformer_client is not None and (
            segformer_tile_size is None
            or segformer_tile_overlap is None
            or segformer_feature_stage is None
        ):
            raise ValueError(
                "SegFormer tile policy is required when a segformer client is present"
            )
        self._catalog = catalog
        self._policy = policy
        self._yolo_client = yolo_client
        self._yolo_device = yolo_device
        self._yolo_image_size = yolo_image_size
        self._segformer_client = segformer_client
        self._segformer_tile_size = segformer_tile_size
        self._segformer_tile_overlap = segformer_tile_overlap
        self._segformer_feature_stage = segformer_feature_stage

    def execute(
        self,
        plan: FirstQwenVisualPlan,
        images: Mapping[str, Image.Image],
        *,
        fallback_image_id: str,
    ) -> EvidenceExecution:
        """Run the frozen state machine over one plan. The plan must be the
        object_evidence_vqa family with a validated evidence request; the
        bundle is assembled by roi_id and stable leaf order. All accumulation
        state is re-created per call, so one executor serves many plans without
        any cross-plan leakage. 对一个计划运行冻结状态机。计划必须是
        object_evidence_vqa 家族且携带已校验证据请求；bundle 按 roi_id 与稳定
        leaf order 组装。所有累积状态在每次调用时重新创建，一个执行器可服务
        多个计划而无跨计划泄漏。"""
        if plan.execution_family != "object_evidence_vqa":
            raise ValueError("executor requires an object_evidence_vqa plan")
        if plan.evidence_request is None:
            raise ValueError("object_evidence_vqa plan must carry an evidence_request")
        self._audits: list[ModelCallAudit] = []
        self._outcomes: list[RoiLeafOutcome] = []
        self._masks: dict[tuple[str, str], Any] = {}

        sizes = {image_id: image.size for image_id, image in images.items()}
        leaves = self._catalog.expand_composites(
            plan.evidence_request.composite_categories
        )
        regions = self._regions(plan, fallback_image_id)
        records = resolve_roi_records(plan, sizes, fallback_image_id=fallback_image_id)

        hit_leaves, detections = self._yolo_phase(
            images, regions, records, leaves
        )
        segments = self._segformer_phase(
            images, regions, records, leaves, hit_leaves
        )
        layer_states, final_states, missing = self._aggregate(leaves)
        bundle = VqaEvidenceBundle(
            catalog_version=self._catalog.catalog_version,
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
        )

    # ── helpers / 辅助 ─────────────────────────────────────────────────

    def _regions(
        self,
        plan: FirstQwenVisualPlan,
        fallback_image_id: str,
    ) -> list[RoiRegion]:
        """Plan ROIs in order, or the unique full-image ROI for an empty plan.
        按顺序返回计划 ROI；空计划返回唯一整图 ROI。"""
        if plan.roi_plan.rois:
            return list(plan.roi_plan.rois)
        return [
            RoiRegion(
                roi_id="full",
                image_id=fallback_image_id,
                xyxy=(0.0, 0.0, 1.0, 1.0),
            )
        ]

    def _yolo_phase(
        self,
        images: Mapping[str, Image.Image],
        regions: list[RoiRegion],
        records: list[RoiEvidenceRecord],
        leaves: tuple[str, ...],
    ) -> tuple[set[str], list[YoloDetectionRecord]]:
        """YOLO once per ROI, filter every requested leaf from one output.
        Confidence is consumed internally only: thresholding, top-k retention,
        and the cross-ROI greedy dedup in whole-image coordinates.
        每个 ROI 运行一次 YOLO，从一次输出过滤全部请求叶子。confidence 仅在
        内部消费：阈值、top-k 保留与 whole-image 坐标下的跨 ROI 贪心去重。"""
        hit_leaves: set[str] = set()
        detected: list[tuple[float, YoloDetectionRecord]] = []
        for region, record in zip(regions, records):
            crop = render_roi_crop(images[record.image_id], region, record)
            outputs: list[ObjectDetectionOutput] | None = None
            failed_code: str | None = None
            if self._yolo_client is not None:
                try:
                    outputs = self._yolo_client.detect(
                        crop,
                        confidence=self._policy.confidence_threshold,
                        iou=self._policy.nms_iou_threshold,
                        image_size=self._yolo_image_size,
                        device=self._yolo_device,
                        max_detections=self._policy.max_detections,
                    )
                except Exception as exc:
                    failed_code = type(exc).__name__
            if outputs is not None or failed_code is not None:
                logical_model_id, digest = _audit_identity(
                    outputs or [], self._yolo_client
                )
                self._audits.append(
                    ModelCallAudit(
                        layer="yolo",
                        roi_id=region.roi_id,
                        input_size=crop.size,
                        logical_model_id=logical_model_id,
                        weights_sha256=digest,
                        status="failed" if failed_code is not None else "succeeded",
                        error_code=failed_code,
                    )
                )
            for leaf in leaves:
                if not self._catalog.capability_enabled(leaf, "yolo"):
                    self._outcomes.append(
                        RoiLeafOutcome(region.roi_id, leaf, "yolo", "unsupported")
                    )
                    continue
                if self._yolo_client is None:
                    self._outcomes.append(
                        RoiLeafOutcome(region.roi_id, leaf, "yolo", "unavailable")
                    )
                    continue
                if failed_code is not None:
                    self._outcomes.append(
                        RoiLeafOutcome(
                            region.roi_id, leaf, "yolo", "error", failed_code
                        )
                    )
                    continue
                assert outputs is not None
                leaf_labels = set(self._catalog.leaf_yolo_labels(leaf))
                leaf_outputs = [o for o in outputs if o.label in leaf_labels]
                if not leaf_outputs:
                    self._outcomes.append(
                        RoiLeafOutcome(region.roi_id, leaf, "yolo", "missing")
                    )
                    continue
                retained = sorted(
                    leaf_outputs, key=lambda o: o.confidence, reverse=True
                )[: self._policy.max_detections]
                for detection in retained:
                    detected.append(
                        (
                            detection.confidence,
                            YoloDetectionRecord(
                                leaf_category=leaf,
                                roi_id=region.roi_id,
                                local_xyxy=detection.xyxy,
                                local_roi_size=crop.size,
                                global_xyxy=local_to_global(detection.xyxy, record),
                                global_image_size=images[record.image_id].size,
                            ),
                        )
                    )
                hit_leaves.add(leaf)
                self._outcomes.append(
                    RoiLeafOutcome(region.roi_id, leaf, "yolo", "hit")
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
        images: Mapping[str, Image.Image],
        regions: list[RoiRegion],
        records: list[RoiEvidenceRecord],
        leaves: tuple[str, ...],
        hit_leaves: set[str],
    ) -> list[SegFormerEvidenceRecord]:
        """SegFormer once per ROI while any still-missing leaf has an approved
        capability; a leaf that hits in one ROI is not_run in later ROIs and
        never re-filtered. Masks stay per-ROI in memory; only (leaf, roi)
        presence records are persisted. 只要仍有缺失且具备批准能力的叶子，每个
        ROI 运行一次 SegFormer；在某 ROI 命中的叶子在后续 ROI 为 not_run 且
        绝不重筛。mask 只在内存中按 ROI 保留；持久化只有（叶子，ROI）存在性
        记录。"""
        segments: list[SegFormerEvidenceRecord] = []
        for region, record in zip(regions, records):
            crop = render_roi_crop(images[record.image_id], region, record)
            still_missing = [
                leaf
                for leaf in leaves
                if leaf not in hit_leaves
                and self._catalog.capability_enabled(leaf, "segformer")
            ]
            output = None
            failed_code: str | None = None
            if still_missing and self._segformer_client is not None:
                try:
                    output = self._segformer_client.infer(
                        crop,
                        tile_size=self._segformer_tile_size,
                        tile_overlap=self._segformer_tile_overlap,
                        feature_stage=self._segformer_feature_stage,
                    )
                except Exception as exc:
                    failed_code = type(exc).__name__
            if output is not None or failed_code is not None:
                logical_model_id, digest = _audit_identity([], self._segformer_client)
                if output is not None and output.weights_sha256:
                    digest = output.weights_sha256
                self._audits.append(
                    ModelCallAudit(
                        layer="segformer",
                        roi_id=region.roi_id,
                        input_size=crop.size,
                        logical_model_id=logical_model_id,
                        weights_sha256=digest,
                        status="failed" if failed_code is not None else "succeeded",
                        error_code=failed_code,
                    )
                )
            for leaf in leaves:
                if leaf in hit_leaves:
                    if self._catalog.capability_enabled(leaf, "segformer"):
                        self._outcomes.append(
                            RoiLeafOutcome(region.roi_id, leaf, "segformer", "not_run")
                        )
                    continue
                if not self._catalog.capability_enabled(leaf, "segformer"):
                    self._outcomes.append(
                        RoiLeafOutcome(region.roi_id, leaf, "segformer", "unsupported")
                    )
                    continue
                if self._segformer_client is None:
                    self._outcomes.append(
                        RoiLeafOutcome(region.roi_id, leaf, "segformer", "unavailable")
                    )
                    continue
                if failed_code is not None:
                    self._outcomes.append(
                        RoiLeafOutcome(
                            region.roi_id, leaf, "segformer", "error", failed_code
                        )
                    )
                    continue
                if output is None:
                    # No call happened for this ROI (every capable leaf was
                    # already hit at an upper layer); a still missing leaf
                    # without a call keeps its prior decision.
                    # 本 ROI 未发生调用（全部具备能力的叶子已在更上层命中）；仍
                    # 缺失且无调用的叶子保持先前决策。
                    continue
                labels = self._catalog.leaf_segformer_labels(leaf) or ()
                try:
                    indices = [output.class_names.index(label) for label in labels]
                except ValueError:
                    self._outcomes.append(
                        RoiLeafOutcome(
                            region.roi_id,
                            leaf,
                            "segformer",
                            "error",
                            SEGFORMER_CLASS_MAP_MISMATCH,
                        )
                    )
                    continue
                presence = _mask_presence(
                    output.probabilities, indices, self._policy.confidence_threshold
                )
                if bool(presence.any()):
                    segments.append(
                        SegFormerEvidenceRecord(
                            leaf_category=leaf, roi_id=region.roi_id
                        )
                    )
                    self._masks[(region.roi_id, leaf)] = presence
                    hit_leaves.add(leaf)
                    self._outcomes.append(
                        RoiLeafOutcome(region.roi_id, leaf, "segformer", "hit")
                    )
                else:
                    self._outcomes.append(
                        RoiLeafOutcome(region.roi_id, leaf, "segformer", "missing")
                    )
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
