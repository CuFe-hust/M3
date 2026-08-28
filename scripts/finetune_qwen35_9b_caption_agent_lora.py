#!/usr/bin/env python3
"""Fine-tune Qwen3.5-9B LoRA through the production GeneralVQAAgent input path.

通过生产 GeneralVQAAgent 输入路径微调 Qwen3.5-9B LoRA。每条样本先消费冻结的
VisualTaskPlan，运行真实 ROI/YOLO/SegFormer 流程并持久化 evidence artifacts；随后
仅对 AgentResult.answer token 计算 causal-LM loss。
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
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
from tempfile import NamedTemporaryFile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agents.base import AgentContext  # noqa: E402
from agents.schema import AgentResult, VisualTaskPlan  # noqa: E402
from application.bootstrap import assemble_runtime  # noqa: E402
from application.settings import load_settings  # noqa: E402
from data.schema import UnifiedSample  # noqa: E402
from models.base import ModelCacheIdentity  # noqa: E402
from scripts import finetune_qwen35_9b_visual_planner_lora as base_ft  # noqa: E402

LOGGER = logging.getLogger("finetune_qwen35_9b_caption_agent_lora")
IGNORE_INDEX = -100
MANIFEST_NAME = "caption_agent_training_manifest.json"
AUDIT_NAME = "parameter_audit.json"
PREPARATION_VERSION = "caption-agent-request-preparation-v1"

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
            model="Qwen/Qwen3.5-9B:caption-training-capture",
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
    if dataset == "XLRS-Bench":
        return Path(args.xlrs_root).resolve(strict=True)
    raise TrainingConfigurationError(f"unsupported_dataset_for_caption_training:{dataset}")


def load_source_records(
    args: argparse.Namespace,
    *,
    split: str,
    max_samples: int | None,
) -> list[SourceRecord]:
    annotation_root = Path(args.annotation_root).resolve(strict=True)
    if split == "train":
        sources = [
            ("VRSBench", "vrsbench_600_caption_v5/samples_vqa_agent_sft_v2_train.jsonl"),
            ("XLRS-Bench", "xlrs-train/samples_vqa_agent_sft_v2_train.jsonl"),
        ]
    else:
        sources = [
            ("VRSBench", "vrsbench_600_caption_v5/samples_vqa_agent_sft_v2_val.jsonl"),
            ("XLRS-Bench", "xlrs-train/samples_vqa_agent_sft_v2_val.jsonl"),
        ]
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
                plan_raw = record.get("input", {}).get("visual_task_plan")
                if isinstance(plan_raw, dict) and plan_raw.get("region_request") is None:
                    plan_raw["region_request"] = {}
                sample_id = record.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id or sample_id in seen_ids:
                    raise TrainingConfigurationError("sample_id_missing_or_duplicate")
                seen_ids.add(sample_id)
                sample = UnifiedSample.model_validate(record["input"]["agent_input"]["sample"])
                plan = VisualTaskPlan.model_validate(record["input"]["visual_task_plan"])
                target = AgentResult.model_validate(record["output"]["agent_result"])
                if sample.ground_truth is not None:
                    raise TrainingConfigurationError("ground_truth_leakage")
                if plan.task != sample.task or target.agent_name != "caption_agent":
                    raise TrainingConfigurationError("record_task_or_agent_mismatch")
                if record.get("supervision", {}).get("loss_scope") not in (
                    ["output.agent_result.answer"],
                    ["output.agent_result.answer", "output.agent_result.agent_name"],
                ):
                    raise TrainingConfigurationError("unsupported_loss_scope")
                for image in sample.images:
                    relative_image = _safe_relative(image.path.as_posix())
                    candidate = (data_root / Path(*relative_image.parts)).resolve()
                    if not candidate.is_relative_to(data_root) or not candidate.is_file():
                        raise TrainingConfigurationError(
                            f"image_missing_or_unsafe:{dataset}:{line_number}"
                        )
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
    agent = components.agent_registry.get("caption_agent")
    for index, item in enumerate(records):
        sample_id = item.record["sample_id"]
        safe_id = hashlib.sha256(sample_id.encode()).hexdigest()[:24]
        sample_dir = output_dir / "prepared_evidence" / split / f"sample-{safe_id}"
        identity = _source_identity(item, config_sha256)
        cached_path = sample_dir / "compiled_record.json"
        status_path = sample_dir / "status.json"
        if cached_path.is_file() and status_path.is_file():
            cached_status = json.loads(status_path.read_text(encoding="utf-8"))
            if cached_status.get("state") == "succeeded" and cached_status.get("identity") == identity:
                prepared.append(json.loads(cached_path.read_text(encoding="utf-8")))
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
                qwen_client=components.qwen_clients["caption_agent"],
                call_budget=components.call_budget_factory.create_for_sample(sample.task),
                data_root=item.data_root,
                visual_bindings=components.visual_bindings,
                visual_task_plan=plan,
                visual_views=views,
            )
            execution = await agent.run(sample, context)
            if capture.messages is None:
                raise TrainingConfigurationError("final_qwen_request_was_not_captured")
            messages, visuals = _sanitize_captured_messages(
                capture.messages, sample_dir, output_dir
            )
            target_text = json.dumps(
                target.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":")
            )
            compiled = {
                "schema_version": 1,
                "format": "caption-agent-prepared-chat-v1",
                "episode_id": sample_id,
                "dataset": item.dataset,
                "split": split,
                "answer": target.answer,
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
    torch = base_ft._torch()
    messages = _template_messages(record)
    full_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    prompt_text = processor.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    target_text = record["messages"][2]["content"][0]["text"]
    answer_literal = json.dumps(record["answer"], ensure_ascii=False)
    answer_key = '"answer":'
    key_offset = target_text.find(answer_key)
    answer_offset = target_text.find(answer_literal, key_offset + len(answer_key))
    target_offset = full_text.rfind(target_text)
    if min(key_offset, answer_offset, target_offset) < 0:
        raise TrainingConfigurationError("answer_span_not_found")
    answer_char_start = target_offset + answer_offset
    answer_char_end = answer_char_start + len(answer_literal)
    # Supervise the agent_name field too: locate the quoted literal of the
    # target agent_name ("caption_agent") right after the "agent_name" key so
    # the model learns the fixed contract value instead of only the answer
    # body. / 同时监督 agent_name 字段：在 "agent_name" 键后定位目标
    # agent_name 的带引号字面量（"caption_agent"），让模型学会固定契约值，
    # 而不仅是 answer 正文。
    agent_name_marker = '"agent_name":'
    agent_name_key_offset = target_text.find(agent_name_marker)
    if agent_name_key_offset < 0:
        raise TrainingConfigurationError("agent_name_span_not_found")
    agent_name_value_start = agent_name_key_offset + len(agent_name_marker)
    # The assistant target's agent_name is fixed at "caption_agent" by the
    # source-record validation; locate its quoted JSON literal directly.
    # assistant 目标的 agent_name 由源记录校验固定为 "caption_agent"；
    # 直接定位其带引号 JSON 字面量。
    agent_name_literal = '"caption_agent"'
    literal_offset = target_text.find(agent_name_literal, agent_name_value_start)
    if literal_offset < 0:
        raise TrainingConfigurationError("agent_name_literal_not_found")
    agent_name_char_start = target_offset + literal_offset
    agent_name_char_end = agent_name_char_start + len(agent_name_literal)
    tokenizer = processor.tokenizer
    tokenized = tokenizer(
        full_text, add_special_tokens=False, return_offsets_mapping=True
    )
    plain_ids = tokenized.input_ids
    offsets = tokenized.offset_mapping
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    if plain_ids[: len(prompt_ids)] != prompt_ids:
        raise TrainingConfigurationError("chat_template_is_not_prefix_stable")
    from PIL import Image

    images = []
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
                images.append(image.convert("RGB"))
    encoded = processor(text=[full_text], images=images, return_tensors="pt", padding=False)
    input_ids = encoded["input_ids"].squeeze(0)
    delta = int(input_ids.shape[0]) - len(plain_ids)
    if int(input_ids.shape[0]) > max_length:
        raise TrainingConfigurationError(
            f"sequence_too_long:{record['episode_id']}:{int(input_ids.shape[0])}"
        )
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    supervised = 0
    supervised_spans = (
        (agent_name_char_start, agent_name_char_end),
        (answer_char_start, answer_char_end),
    )
    for plain_index, (start, end) in enumerate(offsets):
        for span_start, span_end in supervised_spans:
            if end > span_start and start < span_end:
                encoded_index = plain_index + delta
                if not 0 <= encoded_index < int(input_ids.shape[0]):
                    raise TrainingConfigurationError("answer_token_alignment_failed")
                labels[encoded_index] = input_ids[encoded_index]
                supervised += 1
                break
    if supervised < 1:
        raise TrainingConfigurationError("answer_has_no_supervised_tokens")
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
    parser.add_argument("--annotation-root", default="data/20260824-caption-train-annotations")
    parser.add_argument("--vrsbench-root", default="data/vrsbench")
    parser.add_argument("--xlrs-root", default="data/20260824-caption-train-annotations/xlrs-train")
    parser.add_argument("--output-dir", default="outputs/finetune/qwen35-9b-caption-agent-lora")
    parser.add_argument(
        "--from-prepared",
        action="store_true",
        help="reuse prepared/train.jsonl+validation.jsonl instead of running prepare",
    )
    parser.add_argument(
        "--prepared-root",
        default=None,
        help="output dir that produced the prepared records (default: --output-dir)",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=6144)
    parser.add_argument("--max-pixels", type=int, default=1048576)
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
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve(strict=True)
    config_sha256 = sha256_file(config_path)
    if args.from_prepared:
        # Reuse an already-prepared dataset: skip source loading and the
        # prepare pass entirely. Image paths inside the prepared records are
        # relative to the prepared root (the output dir that produced them).
        # 复用已生成的 prepared 数据：完全跳过源加载与 prepare 阶段。
        # prepared 记录内的图片路径相对于产生它们的 prepared root。
        prepared_root = Path(args.prepared_root or args.output_dir).resolve()
        train_path = prepared_root / "prepared" / "train.jsonl"
        validation_path = prepared_root / "prepared" / "validation.jsonl"
        if not train_path.is_file() or not validation_path.is_file():
            raise TrainingConfigurationError(
                f"prepared_files_missing:{train_path}:{validation_path}"
            )
        train_records = [
            json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        validation_records = [
            json.loads(line) for line in validation_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        train_source: list[SourceRecord] = []
        validation_source: list[SourceRecord] = []
        preparation_manifest = {
            "schema_version": 1,
            "status": "reused",
            "preparation_version": PREPARATION_VERSION,
            "config_sha256": config_sha256,
            "from_prepared": True,
            "train": {"records": len(train_records), "sha256": sha256_file(train_path)},
            "validation": {"records": len(validation_records), "sha256": sha256_file(validation_path)},
        }
    else:
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
        prepared_root = output_dir
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
            "train": {**_summary(train_source), "sha256": sha256_file(train_path)},
            "validation": {**_summary(validation_source), "sha256": sha256_file(validation_path)},
        }
        atomic_json(output_dir / "preparation_manifest.json", preparation_manifest)
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
    processor = transformers.AutoProcessor.from_pretrained(
        args.model_path, local_files_only=True, max_pixels=args.max_pixels
    )
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
    train_dataset = PreparedDataset(train_records, prepared_root, processor, args.max_seq_length)
    validation_dataset = PreparedDataset(
        validation_records, prepared_root, processor, args.max_seq_length
    )
    # Fail before Trainer construction if answer-only token alignment drifts.
    # 若 answer-only token 对齐漂移，在构造 Trainer 前失败。
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
            "scope": ["output.agent_result.answer", "output.agent_result.agent_name"],
            "ignore_index": IGNORE_INDEX,
            "structure_loss": False,
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
