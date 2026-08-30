#!/usr/bin/env python3
"""Fine-tune Qwen3.5-9B LoRA through the production GeneralVQAAgent input path.

通过生产 GeneralVQAAgent 输入路径微调 Qwen3.5-9B LoRA。每条样本先消费冻结的
VisualTaskPlan，运行真实 ROI/YOLO/SegFormer 流程并持久化 evidence artifacts；随后
监督稳定 AgentResult JSON 结构、answer 与经过 planner 筛选的 YOLO 证据。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
from tempfile import NamedTemporaryFile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.base import AgentContext  # noqa: E402
from agents.general_vqa.evidence.rendering import make_preview  # noqa: E402
from agents.schema import AgentResult, VisualTaskPlan  # noqa: E402
from application.bootstrap import assemble_runtime  # noqa: E402
from application.settings import load_settings  # noqa: E402
from data.schema import UnifiedSample  # noqa: E402
from models.base import ModelCacheIdentity  # noqa: E402
from scripts import finetune_qwen35_9b_visual_planner_lora as base_ft  # noqa: E402

LOGGER = logging.getLogger("finetune_qwen35_9b_general_vqa_agent_lora")
IGNORE_INDEX = -100
MANIFEST_NAME = "general_vqa_agent_training_manifest.json"
AUDIT_NAME = "parameter_audit.json"
PREPARATION_VERSION = "general-vqa-agent-request-preparation-v2"
MAX_SUPERVISED_EVIDENCE_ITEMS = 200

TRAIN_SOURCES = (
    ("VRSBench", "VRSBench/train_5000_image_grouped_v1.jsonl"),
    ("DOTA", "DOTA/train.jsonl"),
    ("HRSCD", "HRSCD/train.jsonl"),
    ("MiniFrance", "MiniFrance/train.jsonl"),
    ("LRS-VQA-Supplement", "LRS-VQA-Supplement/train.jsonl"),
)
VALIDATION_SOURCES = (
    ("VRSBench", "VRSBench/validation_500_image_grouped_v1.jsonl"),
    ("DOTA", "DOTA/validation.jsonl"),
    ("HRSCD", "HRSCD/validation.jsonl"),
    ("MiniFrance", "MiniFrance/validation.jsonl"),
    ("LRS-VQA-Supplement", "LRS-VQA-Supplement/validation.jsonl"),
)


class TrainingConfigurationError(RuntimeError):
    """Stable training/preparation contract failure. / 稳定训练/准备契约失败。"""


@dataclass(frozen=True)
class SourceRecord:
    dataset: str
    split: str
    source_file: Path
    line_number: int
    record: dict[str, Any]
    data_root: Path


class CaptureQwenClient:
    """Capture the exact production messages instead of invoking final Qwen.
    截获生产最终 messages，不调用最终 Qwen。
    """

    def __init__(self) -> None:
        self.cache_identity = ModelCacheIdentity(
            model="Qwen/Qwen3.5-9B:general-vqa-training-capture",
            generation={"mode": "request_capture"},
            client_version=PREPARATION_VERSION,
        )
        self.target: AgentResult | None = None
        self.messages: list[dict[str, Any]] | None = None

    async def complete_json(self, **kwargs: Any) -> AgentResult:
        if kwargs.get("response_model") is not AgentResult or self.target is None:
            raise TrainingConfigurationError("capture_client_received_unexpected_call")
        self.messages = kwargs.get("messages")
        if not isinstance(self.messages, list):
            raise TrainingConfigurationError("capture_client_messages_missing")
        return self.target


@lru_cache(maxsize=4096)
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    os.replace(temporary, path)


def _safe_relative(path: str) -> PurePosixPath:
    if not isinstance(path, str) or not path or "\\" in path:
        raise TrainingConfigurationError("unsafe_relative_path")
    posix = PurePosixPath(path)
    windows = PureWindowsPath(path)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise TrainingConfigurationError("unsafe_relative_path")
    if any(part in {"", ".", ".."} for part in posix.parts):
        raise TrainingConfigurationError("unsafe_relative_path")
    return posix


def _dataset_root(dataset: str, args: argparse.Namespace) -> Path:
    if dataset == "VRSBench":
        return Path(args.vrsbench_root).resolve(strict=True)
    if dataset == "LRS-VQA-Supplement":
        return Path(args.supplement_root).resolve(strict=True)
    return Path(args.large_image_root).resolve(strict=True)


def load_source_records(
    args: argparse.Namespace,
    *,
    split: str,
    max_samples: int | None,
) -> list[SourceRecord]:
    annotation_root = Path(args.annotation_root).resolve(strict=True)
    sources = TRAIN_SOURCES if split == "train" else VALIDATION_SOURCES
    loaded: list[SourceRecord] = []
    seen_ids: set[str] = set()
    for dataset, relative in sources:
        path = annotation_root / relative
        if not path.is_file():
            raise TrainingConfigurationError(f"required_annotation_missing:{relative}")
        data_root = _dataset_root(dataset, args)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise TrainingConfigurationError("invalid_annotation_jsonl") from error
                sample_id = record.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id or sample_id in seen_ids:
                    raise TrainingConfigurationError("sample_id_missing_or_duplicate")
                seen_ids.add(sample_id)
                sample = UnifiedSample.model_validate(record["input"]["agent_input"]["sample"])
                plan = VisualTaskPlan.model_validate(record["input"]["visual_task_plan"])
                target = AgentResult.model_validate(record["output"]["agent_result"])
                if sample.ground_truth is not None:
                    raise TrainingConfigurationError("ground_truth_leakage")
                if plan.task != sample.task or target.agent_name != "general_vqa_agent":
                    raise TrainingConfigurationError("record_task_or_agent_mismatch")
                if record.get("supervision", {}).get("loss_scope") != [
                    "output.agent_result.answer"
                ]:
                    raise TrainingConfigurationError("unsupported_loss_scope")
                for image in sample.images:
                    relative_image = _safe_relative(image.path.as_posix())
                    candidate = (data_root / Path(*relative_image.parts)).resolve()
                    if not candidate.is_relative_to(data_root) or not candidate.is_file():
                        raise TrainingConfigurationError(
                            f"image_missing_or_unsafe:{dataset}:{line_number}"
                        )
                if sample_id in set(getattr(args, "exclude_episode_id", ())):
                    continue
                loaded.append(SourceRecord(dataset, split, path, line_number, record, data_root))
                if max_samples is not None and len(loaded) >= max_samples:
                    return loaded
    return loaded


def _source_identity(item: SourceRecord, config_sha256: str) -> str:
    sample = item.record["input"]["agent_input"]["sample"]
    digest = hashlib.sha256()
    digest.update(PREPARATION_VERSION.encode())
    digest.update(config_sha256.encode())
    digest.update(json.dumps(item.record, ensure_ascii=False, sort_keys=True).encode())
    for image in sample["images"]:
        posix = _safe_relative(image["path"])
        digest.update(sha256_file(item.data_root / Path(*posix.parts)).encode())
    return digest.hexdigest()


def _decode_data_url(value: str) -> tuple[bytes, str]:
    match = re.fullmatch(r"data:([^;,]+);base64,(.+)", value, flags=re.DOTALL)
    if match is None:
        raise TrainingConfigurationError("captured_image_is_not_base64_data_url")
    try:
        return base64.b64decode(match.group(2), validate=True), match.group(1)
    except ValueError as error:
        raise TrainingConfigurationError("captured_image_base64_invalid") from error


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    os.replace(temporary, path)


def _sanitize_captured_messages(
    messages: list[dict[str, Any]], sample_dir: Path, output_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sanitized: list[dict[str, Any]] = []
    visuals: list[dict[str, Any]] = []
    image_index = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            sanitized.append({"role": message["role"], "content": content})
            continue
        if not isinstance(content, list):
            raise TrainingConfigurationError("captured_message_content_invalid")
        items: list[dict[str, Any]] = []
        for block in content:
            if block.get("type") == "text":
                items.append({"type": "text", "text": block["text"]})
                continue
            if block.get("type") != "image_url":
                raise TrainingConfigurationError("captured_content_type_invalid")
            data, mime = _decode_data_url(block["image_url"]["url"])
            extension = {"image/png": ".png", "image/jpeg": ".jpg", "image/tiff": ".tif"}.get(mime)
            if extension is None:
                raise TrainingConfigurationError("captured_image_mime_unsupported")
            digest = hashlib.sha256(data).hexdigest()
            image_path = sample_dir / "rendered" / f"{image_index:03d}-{digest[:16]}{extension}"
            _atomic_bytes(image_path, data)
            relative = image_path.relative_to(output_dir).as_posix()
            items.append({"type": "image", "image": relative})
            visuals.append({
                "content_image_index": image_index,
                "mime_type": mime,
                "sha256": digest,
                "artifact_path": relative,
            })
            image_index += 1
        sanitized.append({"role": message["role"], "content": items})
    return sanitized, visuals


def _supervised_target(
    source_target: AgentResult,
    plan: VisualTaskPlan,
    evidence_bundle: Mapping[str, Any] | None,
) -> AgentResult:
    """Build the target from planner-filtered YOLO detections only.
    仅使用经过 planner 筛选的 YOLO 检测构造监督目标。

    SegFormer hits deliberately remain input-only evidence because the runtime
    contract does not convert semantic masks into boxes. Stable executor order
    is retained and capped at the AgentResult schema limit.
    SegFormer 命中有意只作为输入证据，因为运行时契约不把语义掩膜转换成框。
    保留 executor 的稳定顺序，并按 AgentResult Schema 上限截断。
    """
    if not plan.needs_visual_assistance or not isinstance(evidence_bundle, Mapping):
        return source_target.model_copy(
            update={"boxes": [], "evidence_items": [], "geometry": {}}
        )
    rois = evidence_bundle.get("rois")
    detections = evidence_bundle.get("detections")
    leaf_states = evidence_bundle.get("leaf_states")
    if (
        not isinstance(rois, list)
        or not isinstance(detections, list)
        or not isinstance(leaf_states, Mapping)
    ):
        raise TrainingConfigurationError("evidence_bundle_shape_invalid")
    hit_leaves = {leaf for leaf, state in leaf_states.items() if state == "hit"}
    roi_images = {
        roi.get("roi_id"): roi.get("image_id")
        for roi in rois
        if isinstance(roi, Mapping)
        and isinstance(roi.get("roi_id"), str)
        and isinstance(roi.get("image_id"), str)
    }
    items: list[dict[str, Any]] = []
    for detection in detections:
        if not isinstance(detection, Mapping):
            raise TrainingConfigurationError("yolo_detection_shape_invalid")
        label = detection.get("leaf_category")
        # Composite planner categories are expanded into canonical leaves by
        # the executor; the validated hit-state map is therefore authoritative.
        # planner 复合类别由 executor 展开为 canonical leaves；因此以已校验的
        # hit 状态映射作为权威集合。
        if not isinstance(label, str) or label not in hit_leaves:
            raise TrainingConfigurationError("yolo_detection_not_requested_by_plan")
        global_xyxy = detection.get("global_xyxy")
        global_size = detection.get("global_image_size")
        roi_id = detection.get("roi_id")
        image_id = roi_images.get(roi_id)
        if (
            not isinstance(global_xyxy, (list, tuple))
            or len(global_xyxy) != 4
            or not isinstance(global_size, (list, tuple))
            or len(global_size) != 2
            or image_id is None
        ):
            raise TrainingConfigurationError("yolo_detection_geometry_invalid")
        box = _pixel_xyxy_to_999(global_xyxy, global_size)
        if box[0] >= box[2] or box[1] >= box[3]:
            continue
        items.append(
            {
                "label": label,
                "box": box,
                "image_id": image_id,
                "coordinate_frame": "normalized_0_999_top_left",
            }
        )
        if len(items) == MAX_SUPERVISED_EVIDENCE_ITEMS:
            break
    validated = AgentResult.model_validate(
        {
            "agent_name": source_target.agent_name,
            "answer": source_target.answer,
            "boxes": [item["box"] for item in items],
            "evidence_items": items,
            "geometry": {},
            "status": source_target.status,
        }
    )
    # AgentResult validation records input-repair diagnostics in geometry;
    # training targets keep that runtime-owned field intentionally empty.
    # AgentResult 校验会在 geometry 中记录输入修复诊断；训练目标有意将这个
    # 运行时字段保持为空。
    return validated.model_copy(update={"geometry": {}})


def _pixel_xyxy_to_999(xyxy: Sequence[Any], size: Sequence[Any]) -> list[int]:
    """Normalize whole-image pixel xyxy to strict integer 0..999 geometry.
    将整图像素 xyxy 归一化为严格的 0..999 整数几何。
    """
    try:
        width, height = float(size[0]), float(size[1])
        values = [float(value) for value in xyxy]
    except (TypeError, ValueError, IndexError) as error:
        raise TrainingConfigurationError("yolo_detection_geometry_invalid") from error
    if width <= 0 or height <= 0:
        raise TrainingConfigurationError("yolo_detection_geometry_invalid")
    if not all(math.isfinite(value) for value in [width, height, *values]):
        raise TrainingConfigurationError("yolo_detection_geometry_invalid")
    scales = (width, height, width, height)
    return [
        max(0, min(999, int(round(value / scale * 999))))
        for value, scale in zip(values, scales, strict=True)
    ]


async def prepare_records(
    args: argparse.Namespace,
    records: Sequence[SourceRecord],
    *,
    split: str,
    capture: CaptureQwenClient,
    components: Any,
    output_dir: Path,
    config_sha256: str,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    agent = components.agent_registry.get("general_vqa_agent")
    gpu_monitor = None
    if os.environ.get("M3_GPU_MONITOR") == "1":
        try:
            from scripts.gpu_memory_monitor import CudaMemoryMonitor
            gpu_monitor = CudaMemoryMonitor(
                output_dir=output_dir / "gpu_monitor",
                tag=f"prepare_{split}",
                interval=5.0,
                record_history=os.environ.get("M3_GPU_MONITOR_HISTORY") == "1",
            )
            gpu_monitor.start()
        except Exception:
            LOGGER.exception("failed to start CudaMemoryMonitor")
    for index, item in enumerate(records):
        sample_id = item.record["sample_id"]
        safe_id = hashlib.sha256(sample_id.encode()).hexdigest()[:24]
        sample_dir = output_dir / "prepared_evidence" / split / f"sample-{safe_id}"
        identity = _source_identity(item, config_sha256)
        cached_path = sample_dir / "compiled_record.json"
        status_path = sample_dir / "status.json"
        if cached_path.is_file() and status_path.is_file():
            cached_status = json.loads(status_path.read_text(encoding="utf-8"))
            cached_record = json.loads(cached_path.read_text(encoding="utf-8"))
            if (
                cached_status.get("state") == "succeeded"
                and cached_status.get("identity") == identity
                and _cached_record_respects_evidence_limit(cached_record)
            ):
                prepared.append(cached_record)
                if gpu_monitor is not None:
                    gpu_monitor.sample("cached_skip", index=index, sample_id=sample_id, dataset=item.dataset)
                continue
        sample_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(status_path, {
            "schema_version": 1, "state": "running", "sample_id": sample_id,
            "identity": identity, "preparation_version": PREPARATION_VERSION,
        })
        try:
            sample = UnifiedSample.model_validate(item.record["input"]["agent_input"]["sample"])
            plan = VisualTaskPlan.model_validate(item.record["input"]["visual_task_plan"])
            target = AgentResult.model_validate(item.record["output"]["agent_result"])
            views = components.visual_task_planner.materialize_views(
                plan, sample, data_root=item.data_root
            )
            capture.target = target
            capture.messages = None
            context = AgentContext(
                artifact_dir=sample_dir,
                qwen_client=components.qwen_clients["general_vqa_agent"],
                call_budget=components.call_budget_factory.create_for_sample(sample.task),
                data_root=item.data_root,
                visual_bindings=components.visual_bindings,
                visual_task_plan=plan,
                visual_views=views,
            )
            if gpu_monitor is not None:
                gpu_monitor.sample(
                    "before_agent_run", index=index, sample_id=sample_id, dataset=item.dataset
                )
            execution = await agent.run(sample, context)
            if gpu_monitor is not None:
                gpu_monitor.sample(
                    "after_agent_run", index=index, sample_id=sample_id, dataset=item.dataset
                )
            if capture.messages is None:
                raise TrainingConfigurationError("final_qwen_request_was_not_captured")
            messages, visuals = _sanitize_captured_messages(
                capture.messages, sample_dir, output_dir
            )
            evidence_bundle = execution.additional_results.get("vqa_evidence.json")
            target = _supervised_target(target, plan, evidence_bundle)
            target_text = json.dumps(
                target.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")
            )
            compiled = {
                "schema_version": 1,
                "format": "general-vqa-agent-prepared-chat-v1",
                "episode_id": sample_id,
                "dataset": item.dataset,
                "split": split,
                "answer": target.answer,
                "supervision": {
                    "scope": "agent_result_without_geometry_value",
                    "evidence_source": "planner_filtered_yolo",
                    "evidence_items": len(target.evidence_items),
                    "segformer_box_supervision": False,
                },
                "messages": [
                    *messages,
                    {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
                ],
                "preparation_identity": identity,
            }
            atomic_json(sample_dir / "visual_task_plan.json", {
                **plan.model_dump(mode="json"),
                "materialized_views": [view.model_dump(mode="json") for view in views],
            })
            atomic_json(sample_dir / "evidence_bundle.json", {
                "workflow": "object_evidence_vqa" if plan.needs_visual_assistance else "direct_vqa",
                "bundle": execution.additional_results.get("vqa_evidence.json"),
            })
            atomic_json(sample_dir / "prepared_request.json", {
                "schema_version": 1,
                "messages": messages,
                "visuals": visuals,
                "prompt_version": execution.trace.get("prompt_version"),
                "request_hash": execution.trace.get("request_hash"),
            })
            atomic_json(cached_path, compiled)
            atomic_json(status_path, {
                "schema_version": 1, "state": "succeeded", "sample_id": sample_id,
                "identity": identity, "preparation_version": PREPARATION_VERSION,
                "needs_visual_assistance": plan.needs_visual_assistance,
            })
            prepared.append(compiled)
        except Exception as error:
            atomic_json(status_path, {
                "schema_version": 1, "state": "failed", "sample_id": sample_id,
                "identity": identity, "preparation_version": PREPARATION_VERSION,
                "error_type": type(error).__name__,
            })
            raise TrainingConfigurationError(
                f"sample_preparation_failed:{split}:{index}:{type(error).__name__}"
            ) from error
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            LOGGER.info("prepared %s %d/%d", split, index + 1, len(records))
    if gpu_monitor is not None:
        gpu_monitor.sample("prepare_done", split=split, total=len(records))
        gpu_monitor.stop()
    return prepared


class PreparedDataset:
    def __init__(self, records: Sequence[dict[str, Any]], root: Path, processor: Any, max_length: int) -> None:
        self.records = list(records)
        self.root = root
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return encode_prepared_record(
            self.records[index], root=self.root, processor=self.processor,
            max_length=self.max_length,
        )


def _template_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in record["messages"]:
        content = message["content"]
        if isinstance(content, str):
            result.append({"role": message["role"], "content": content})
            continue
        blocks = []
        for block in content:
            if block["type"] == "image":
                blocks.append({"type": "image"})
            else:
                blocks.append({"type": "text", "text": block["text"]})
        result.append({"role": message["role"], "content": blocks})
    return result


def encode_prepared_record(
    record: dict[str, Any], *, root: Path, processor: Any, max_length: int
) -> dict[str, Any]:
    messages = _template_messages(record)
    full_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    prompt_text = processor.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    target_text = record["messages"][2]["content"][0]["text"]
    target_offset = full_text.rfind(target_text)
    geometry_marker = '"geometry":{}'
    geometry_offset = target_text.find(geometry_marker)
    if target_offset < 0 or geometry_offset < 0:
        raise TrainingConfigurationError("supervision_span_not_found")
    geometry_value_start = target_offset + geometry_offset + len('"geometry":')
    geometry_value_end = geometry_value_start + 2
    tokenizer = processor.tokenizer
    tokenized = tokenizer(
        full_text, add_special_tokens=False, return_offsets_mapping=True
    )
    plain_ids = tokenized.input_ids
    offsets = tokenized.offset_mapping
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    if plain_ids[: len(prompt_ids)] != prompt_ids:
        raise TrainingConfigurationError("chat_template_is_not_prefix_stable")
    images = _load_prepared_images(record, root=root)
    encoded = processor(text=[full_text], images=images, return_tensors="pt", padding=False)
    return _finish_encoded_record(
        encoded,
        record=record,
        max_length=max_length,
        plain_ids=plain_ids,
        offsets=offsets,
        target_offset=target_offset,
        target_text=target_text,
        geometry_value_start=geometry_value_start,
        geometry_value_end=geometry_value_end,
    )


def _load_prepared_images(
    record: dict[str, Any], *, root: Path
) -> list[Any]:
    """Load cached prepared visuals through the final-Qwen preview contract.

    Fresh records are already <=1080 at the Agent boundary. Reapplying the
    shrink-only transform keeps older prepared caches safe without changing
    source/ROI coordinate authority or upscaling current artifacts.
    通过 final-Qwen 预览契约加载已缓存的 prepared 图像。新记录已在 Agent
    边界缩至 <=1080；重复应用只缩不放变换可安全兼容旧缓存，且不改变
    源图/ROI 坐标权威，也不放大当前产物。
    """
    from PIL import Image

    images: list[Any] = []
    for message in record["messages"][:2]:
        if not isinstance(message["content"], list):
            continue
        for block in message["content"]:
            if block["type"] != "image":
                continue
            posix = _safe_relative(block["image"])
            path = (root / Path(*posix.parts)).resolve(strict=True)
            if not path.is_relative_to(root):
                raise TrainingConfigurationError("prepared_image_escaped_root")
            with Image.open(path) as image:
                images.append(make_preview(image.convert("RGB")))
    return images


def _finish_encoded_record(
    encoded: dict[str, Any], *, record: dict[str, Any], max_length: int,
    plain_ids: list[int], offsets: list[tuple[int, int]], target_offset: int,
    target_text: str, geometry_value_start: int, geometry_value_end: int,
) -> dict[str, Any]:
    torch = base_ft._torch()
    input_ids = encoded["input_ids"].squeeze(0)
    delta = int(input_ids.shape[0]) - len(plain_ids)
    if int(input_ids.shape[0]) > max_length:
        raise TrainingConfigurationError(
            f"sequence_too_long:{record['episode_id']}:{int(input_ids.shape[0])}"
        )
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    supervised = 0
    for plain_index, (start, end) in enumerate(offsets):
        in_target = end > target_offset and start < target_offset + len(target_text)
        in_geometry_value = end > geometry_value_start and start < geometry_value_end
        if in_target and not in_geometry_value:
            encoded_index = plain_index + delta
            if not 0 <= encoded_index < int(input_ids.shape[0]):
                raise TrainingConfigurationError("target_token_alignment_failed")
            labels[encoded_index] = input_ids[encoded_index]
            supervised += 1
    if supervised < 1:
        raise TrainingConfigurationError("target_has_no_supervised_tokens")
    feature: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": encoded["attention_mask"].squeeze(0),
        "labels": labels,
    }
    if "mm_token_type_ids" in encoded:
        feature["mm_token_type_ids"] = encoded["mm_token_type_ids"].squeeze(0)
    for key in ("pixel_values", "image_grid_thw"):
        if key in encoded:
            feature[key] = encoded[key]
    return feature


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="models/Qwen3.5-9B")
    parser.add_argument("--base-model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--config", default="configs/local.yaml")
    parser.add_argument("--annotation-root", default="data/2026-08-24_vqa-agent-io")
    parser.add_argument("--vrsbench-root", default="data/VRSBench-full")
    parser.add_argument("--large-image-root", default="data/phase2-train-visualplanning-refined-v4")
    parser.add_argument("--supplement-root", default="data/20260824-visual-planner-supplement")
    parser.add_argument("--output-dir", default="outputs/finetune/qwen35-9b-general-vqa-agent-lora")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument(
        "--exclude-episode-id", action="append", default=[],
        help="Exclude an explicitly identified corrupt episode; repeatable.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=6144)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint")
    return parser


def _summary(records: Sequence[SourceRecord]) -> dict[str, Any]:
    return {
        "records": len(records),
        "datasets": dict(sorted(Counter(item.dataset for item in records).items())),
        "tasks": dict(sorted(Counter(
            item.record["input"]["agent_input"]["sample"]["task"] for item in records
        ).items())),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # The explicitly selected offline training corpora contain trusted remote-
    # sensing rasters larger than Pillow's decompression-bomb threshold. Scope
    # the opt-out to this training process; production image loading keeps its
    # default protection.
    # 显式选择的离线训练语料包含超过 Pillow 解压炸弹阈值的可信遥感大图。仅在
    # 当前训练进程中解除限制；生产图片加载仍保留默认保护。
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve(strict=True)
    config_sha256 = sha256_file(config_path)
    train_source = load_source_records(
        args, split="train", max_samples=args.max_train_samples
    )
    validation_source = load_source_records(
        args, split="validation", max_samples=args.max_eval_samples
    )
    capture = CaptureQwenClient()
    settings = load_settings(config_path, environ={})
    components = assemble_runtime(settings, project_root=REPO_ROOT, qwen_client=capture)
    train_records = asyncio.run(prepare_records(
        args, train_source, split="train", capture=capture, components=components,
        output_dir=output_dir, config_sha256=config_sha256,
    ))
    validation_records = asyncio.run(prepare_records(
        args, validation_source, split="validation", capture=capture, components=components,
        output_dir=output_dir, config_sha256=config_sha256,
    ))
    prepared_dir = output_dir / "prepared"
    train_path = prepared_dir / "train.jsonl"
    validation_path = prepared_dir / "validation.jsonl"
    atomic_jsonl(train_path, train_records)
    atomic_jsonl(validation_path, validation_records)
    preparation_manifest = {
        "schema_version": 1,
        "status": "succeeded",
        "preparation_version": PREPARATION_VERSION,
        "config_sha256": config_sha256,
        "excluded_episode_ids": sorted(set(args.exclude_episode_id)),
        "train": {**_summary(train_source), "sha256": sha256_file(train_path)},
        "validation": {**_summary(validation_source), "sha256": sha256_file(validation_path)},
    }
    atomic_json(output_dir / "preparation_manifest.json", preparation_manifest)
    evidence_service = components.visual_bindings.vqa_evidence
    if evidence_service is not None:
        # Preparation and LoRA optimization are disjoint GPU phases. Release
        # only the isolated evidence workers before loading Qwen weights.
        # preparation 与 LoRA 优化是互斥 GPU 阶段；加载 Qwen 权重前只释放隔离的
        # evidence workers。
        evidence_service.close_gpu_workers()
    if args.prepare_only:
        print(json.dumps(preparation_manifest, ensure_ascii=False, indent=2))
        return 0

    transformers = base_ft._transformers()
    torch = base_ft._torch()
    config = transformers.AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    if config.model_type != "qwen3_5":
        raise TrainingConfigurationError(f"expected_qwen3_5_got:{config.model_type}")
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        args.model_path, config=config, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    )
    processor = transformers.AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    transformers.set_seed(args.seed)
    roots = base_ft.locate_model_roots(model)
    targets = base_ft.enumerate_lora_targets(roots)
    peft_model = base_ft.freeze_and_attach_lora(
        model, targets, rank=args.lora_rank, alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    audit = base_ft.audit_lora_model(peft_model, roots, targets)
    atomic_json(output_dir / AUDIT_NAME, audit)
    peft_model.enable_input_require_grads()
    peft_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    peft_model.config.use_cache = False
    train_dataset = PreparedDataset(train_records, output_dir, processor, args.max_seq_length)
    validation_dataset = PreparedDataset(
        validation_records, output_dir, processor, args.max_seq_length
    )
    # Fail before Trainer construction if structured-target alignment drifts.
    # 若结构化目标 token 对齐漂移，在构造 Trainer 前失败。
    train_probe = train_dataset[0]
    validation_probe = validation_dataset[0]
    probe = {
        "train_supervised_tokens": int((train_probe["labels"] != IGNORE_INDEX).sum()),
        "validation_supervised_tokens": int((validation_probe["labels"] != IGNORE_INDEX).sum()),
    }
    manifest = {
        "schema_version": 1,
        "status": "running",
        "base_model_id": args.base_model_id,
        "preparation": preparation_manifest,
        "supervision": {
            "scope": "output.agent_result_without_geometry_value",
            "ignore_index": IGNORE_INDEX,
            "structure_loss": True,
            "evidence_source": "planner_filtered_yolo",
            "segformer_box_supervision": False,
        },
        "lora": {"rank": args.lora_rank, "alpha": args.lora_alpha,
                 "dropout": args.lora_dropout, "targets": targets},
        "optimization": {
            "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
            "epochs": args.epochs, "batch_size": 1,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "warmup_ratio": args.warmup_ratio, "precision": "bfloat16",
            "max_seq_length": args.max_seq_length, "seed": args.seed,
        },
        "probe": probe,
    }
    atomic_json(output_dir / MANIFEST_NAME, manifest)
    training_args = transformers.TrainingArguments(
        output_dir=str(output_dir), per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio, lr_scheduler_type="cosine",
        max_grad_norm=args.max_grad_norm, bf16=True, fp16=False,
        gradient_checkpointing=True, logging_steps=args.logging_steps,
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=args.save_total_limit, dataloader_num_workers=0,
        remove_unused_columns=False, report_to="none", seed=args.seed,
        data_seed=args.seed, optim="adamw_torch",
    )
    trainer = transformers.Trainer(
        model=peft_model, args=training_args, train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=base_ft.VisualPlannerCollator(processor.tokenizer.pad_token_id),
        processing_class=processor,
    )
    resume = args.resume_from_checkpoint
    if resume == "auto":
        latest = base_ft._latest_checkpoint(output_dir)
        if latest is None:
            raise TrainingConfigurationError("no_checkpoint_available_for_auto_resume")
        resume = str(latest)
    result = trainer.train(resume_from_checkpoint=resume)
    final_eval = trainer.evaluate(metric_key_prefix="final_eval")
    final_adapter = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    processor.save_pretrained(final_adapter)
    manifest["status"] = "completed"
    manifest["result"] = {
        "global_step": int(trainer.state.global_step),
        "training_loss": float(result.training_loss),
        "final_eval": {key: value for key, value in final_eval.items()
                       if isinstance(value, (str, int, float, bool)) or value is None},
        "final_adapter": "final_adapter",
    }
    atomic_json(output_dir / MANIFEST_NAME, manifest)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
