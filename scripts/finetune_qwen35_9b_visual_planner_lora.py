#!/usr/bin/env python3
"""LoRA SFT for Qwen3.5-9B on compiled visual-planner chats.
Qwen3.5-9B visual-planner compiled chat 的 LoRA SFT。

The script is intentionally separate from the runtime model/agent packages. It
consumes the already compiled ``training/{train,val}.jsonl`` files verbatim,
freezes the complete base model (including vision), attaches LoRA only to
explicitly enumerated language-model projections in Qwen3.5's hybrid decoder.

本脚本与运行时 model/agent 包保持隔离。它原样消费已经编译好的
``training/{train,val}.jsonl``，冻结包括视觉塔在内的完整 base model，并且只在
Qwen3.5 混合 decoder 中显式枚举出的语言模型 projection 上挂 LoRA，并只在
坐标与其他计划字段统一作为 assistant JSON token 接受语言模型 CE 监督。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Sequence


LOGGER = logging.getLogger("qwen35_visual_planner_lora")
IGNORE_INDEX = -100
MANIFEST_NAME = "qwen35_visual_planner_training_manifest.json"
AUDIT_NAME = "parameter_audit.json"
ALLOWED_TASKS = frozenset(
    {
        "caption",
        "change_caption",
        "change_qa",
        "counting",
        "fine_grained_counting",
        "general_vqa",
        "grounding",
        "multiple_choice_vqa",
        "scene_classification",
        "spatial_relation",
    }
)
FULL_ATTN_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
LINEAR_ATTN_PROJECTIONS = (
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
)
MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


class TrainingConfigurationError(RuntimeError):
    """Stable configuration/data-contract failure. / 稳定的配置或数据契约错误。"""


@dataclass(frozen=True)
class IndexedRecord:
    """Byte offset plus audited task identity. / 字节偏移与已审计任务身份。"""

    offset: int
    episode_id: str
    task: str
    roi_explicit: bool = False


@dataclass(frozen=True)
class ModelRoots:
    """Structurally located Qwen3.5 roots. / 通过结构定位的 Qwen3.5 根模块。"""

    vision: Any
    vision_name: str
    language: Any
    language_name: str


def _torch() -> Any:
    import torch  # noqa: PLC0415

    return torch


def _transformers() -> Any:
    import transformers  # noqa: PLC0415

    return transformers


def _peft() -> Any:
    import peft  # noqa: PLC0415

    return peft


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest. / 流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically publish JSON without a partial file. / 原子发布 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=False)
        handle.write("\n")
    os.replace(temporary, path)


def _target_from_record(record: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Return the exact assistant target text and its declared task.
    返回 assistant 精确目标文本及其中声明的 task。
    """
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise TrainingConfigurationError("record_messages_must_be_system_user_assistant")
    if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
        raise TrainingConfigurationError("record_roles_must_be_system_user_assistant")
    if not isinstance(messages[0].get("content"), str):
        raise TrainingConfigurationError("system_content_must_be_text")
    user_content = messages[1].get("content")
    if not isinstance(user_content, list):
        raise TrainingConfigurationError("user_content_must_be_multimodal_list")
    image_count = sum(
        1 for item in user_content if isinstance(item, dict) and item.get("type") == "image"
    )
    text_count = sum(
        1 for item in user_content if isinstance(item, dict) and item.get("type") == "text"
    )
    if image_count not in (1, 2) or text_count != 1:
        raise TrainingConfigurationError("user_content_requires_one_or_two_images_and_one_text")
    assistant_content = messages[2].get("content")
    if (
        not isinstance(assistant_content, list)
        or len(assistant_content) != 1
        or not isinstance(assistant_content[0], dict)
        or assistant_content[0].get("type") != "text"
        or not isinstance(assistant_content[0].get("text"), str)
    ):
        raise TrainingConfigurationError("assistant_content_must_be_one_text_item")
    target_text = assistant_content[0]["text"]
    try:
        target = json.loads(target_text)
    except json.JSONDecodeError as error:
        raise TrainingConfigurationError("assistant_target_must_be_json") from error
    task = target.get("task") if isinstance(target, dict) else None
    if task not in ALLOWED_TASKS:
        raise TrainingConfigurationError("assistant_target_has_invalid_task")
    if target.get("version") != "visual-task-plan-v5":
        raise TrainingConfigurationError("assistant_target_has_invalid_version")
    region = target.get("region_request")
    if not isinstance(region, dict) or not isinstance(region.get("explicit"), bool):
        raise TrainingConfigurationError("assistant_target_has_invalid_region_request")
    explicit = region["explicit"]
    image_index = region.get("image_index")
    box = region.get("roi_xyxy")
    if not explicit:
        if image_index is not None or box is not None:
            raise TrainingConfigurationError("implicit_region_must_have_null_geometry")
    else:
        if not isinstance(image_index, int) or not 0 <= image_index < image_count:
            raise TrainingConfigurationError("explicit_region_has_invalid_image_index")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(not isinstance(value, int) or isinstance(value, bool) for value in box)
        ):
            raise TrainingConfigurationError("explicit_region_has_invalid_box")
        x0, y0, x1, y1 = box
        if not (0 <= x0 < x1 <= 999 and 0 <= y0 < y1 <= 999):
            raise TrainingConfigurationError("explicit_region_box_is_out_of_range")
    return target_text, task, target


def resolve_image_path(dataset_root: Path, relative: str) -> Path:
    """Resolve one image while rejecting absolute and escaping paths.
    解析单张图像，同时拒绝绝对路径与目录逃逸。
    """
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise TrainingConfigurationError("unsafe_image_path")
    posix = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise TrainingConfigurationError("unsafe_image_path")
    root = dataset_root.resolve(strict=True)
    candidate = (root / Path(*posix.parts)).resolve(strict=True)
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise TrainingConfigurationError(f"image_not_a_safe_file:{relative}")
    return candidate


def image_paths_from_record(record: dict[str, Any], dataset_root: Path) -> list[Path]:
    """Resolve ordered user image blocks. / 按 user block 顺序解析图像。"""
    paths: list[Path] = []
    for item in record["messages"][1]["content"]:
        if item.get("type") == "image":
            paths.append(resolve_image_path(dataset_root, item.get("image")))
    return paths


class CompiledChatIndex:
    """Validated random-access index over one compiled JSONL split.
    编译后 JSONL split 的已校验随机访问索引。
    """

    def __init__(
        self,
        path: Path,
        dataset_root: Path,
        *,
        expected_split: str,
        max_samples: int | None = None,
    ) -> None:
        if not path.is_file():
            raise TrainingConfigurationError(f"dataset_split_missing:{path.name}")
        self.path = path
        self.dataset_root = dataset_root
        self.records: list[IndexedRecord] = []
        seen_ids: set[str] = set()
        with path.open("rb") as handle:
            while max_samples is None or len(self.records) < max_samples:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise TrainingConfigurationError("invalid_training_jsonl") from error
                episode_id = record.get("episode_id")
                if not isinstance(episode_id, str) or not episode_id or episode_id in seen_ids:
                    raise TrainingConfigurationError("episode_id_missing_or_duplicate")
                if record.get("format") != "visual-planner-compiled-chat-v1":
                    raise TrainingConfigurationError("unsupported_training_record_format")
                if record.get("schema_version") != 1 or record.get("split") != expected_split:
                    raise TrainingConfigurationError("training_record_schema_or_split_mismatch")
                _target_text, task, target = _target_from_record(record)
                image_paths_from_record(record, dataset_root)
                self.records.append(
                    IndexedRecord(
                        offset,
                        episode_id,
                        task,
                        bool(target["region_request"]["explicit"]),
                    )
                )
                seen_ids.add(episode_id)
        if not self.records:
            raise TrainingConfigurationError(f"empty_dataset_split:{path.name}")

    def __len__(self) -> int:
        return len(self.records)

    def read(self, index: int) -> dict[str, Any]:
        with self.path.open("rb") as handle:
            handle.seek(self.records[index].offset)
            return json.loads(handle.readline())

    def task_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(entry.task for entry in self.records).items()))

    def roi_explicit_count(self) -> int:
        return sum(1 for entry in self.records if entry.roi_explicit)


def _template_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Copy messages while removing filesystem values from image blocks.
    复制 messages，并从 image block 中移除文件系统值。
    """
    messages: list[dict[str, Any]] = []
    for message in record["messages"]:
        content = message["content"]
        if isinstance(content, str):
            copied: Any = content
        else:
            copied = []
            for item in content:
                if item.get("type") == "image":
                    copied.append({"type": "image"})
                elif item.get("type") == "text":
                    copied.append({"type": "text", "text": item["text"]})
                else:
                    raise TrainingConfigurationError("unsupported_message_content_type")
        messages.append({"role": message["role"], "content": copied})
    return messages


def encode_record(
    record: dict[str, Any],
    *,
    dataset_root: Path,
    processor: Any,
    max_seq_length: int,
) -> dict[str, Any]:
    """Encode one example and supervise only the exact assistant response.
    编码单条样本，并且只监督 assistant 的精确响应。

    Qwen3.5's template puts the empty ``<think>...</think>`` block in the
    generation prefix when thinking is disabled. Consequently, the supervised
    suffix is exactly target JSON + ``<|im_end|>`` (plus template newline).
    """
    torch = _torch()
    _target_text, task, _target = _target_from_record(record)
    messages = _template_messages(record)
    full_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prompt_text = processor.apply_chat_template(
        messages[:2],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    tokenizer = processor.tokenizer
    full_text_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    prompt_text_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    if full_text_ids[: len(prompt_text_ids)] != prompt_text_ids:
        raise TrainingConfigurationError("chat_template_is_not_prefix_stable")

    # Load images once; the processor expands every image placeholder in the
    # same prefix, so the expansion delta shifts all assistant positions by a
    # single deterministic amount. / 图像只加载一次；所有图像占位符都位于
    # assistant 之前，因此 expansion delta 可确定性对齐监督起点。
    from PIL import Image  # noqa: PLC0415

    images = []
    for path in image_paths_from_record(record, dataset_root):
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    encoded = processor(
        text=[full_text],
        images=images,
        return_tensors="pt",
        padding=False,
    )
    input_ids = encoded["input_ids"].squeeze(0)
    expanded_delta = int(input_ids.shape[0]) - len(full_text_ids)
    assistant_start = len(prompt_text_ids) + expanded_delta
    if assistant_start < 1 or assistant_start >= int(input_ids.shape[0]):
        raise TrainingConfigurationError("assistant_alignment_failed")
    if int(input_ids.shape[0]) > max_seq_length:
        raise TrainingConfigurationError(
            f"sequence_too_long:{record['episode_id']}:{int(input_ids.shape[0])}"
        )
    labels = input_ids.clone()
    labels[:assistant_start] = IGNORE_INDEX
    if not bool(torch.any(labels != IGNORE_INDEX)):
        raise TrainingConfigurationError("assistant_has_no_supervised_tokens")

    feature: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": encoded["attention_mask"].squeeze(0),
        "labels": labels,
        "task": task,
        "episode_id": record["episode_id"],
    }
    if "mm_token_type_ids" in encoded:
        feature["mm_token_type_ids"] = encoded["mm_token_type_ids"].squeeze(0)
    for key in ("pixel_values", "image_grid_thw"):
        if key in encoded:
            feature[key] = encoded[key]
    return feature


class VisualPlannerDataset:
    """Lazy processor-backed dataset. / 惰性 processor 数据集。"""

    def __init__(self, index: CompiledChatIndex, processor: Any, max_seq_length: int) -> None:
        self.index = index
        self.processor = processor
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, Any]:
        return encode_record(
            self.index.read(item),
            dataset_root=self.index.dataset_root,
            processor=self.processor,
            max_seq_length=self.max_seq_length,
        )


class VisualPlannerCollator:
    """Right-pad text tensors and concatenate per-image visual tensors.
    右填充文本 tensor，并拼接逐图视觉 tensor。
    """

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        torch = _torch()
        max_length = max(int(feature["input_ids"].shape[0]) for feature in features)

        def padded(key: str, value: int) -> Any:
            rows = []
            for feature in features:
                tensor = feature[key]
                rows.append(
                    torch.nn.functional.pad(
                        tensor, (0, max_length - int(tensor.shape[0])), value=value
                    )
                )
            return torch.stack(rows)

        batch = {
            "input_ids": padded("input_ids", self.pad_token_id),
            "attention_mask": padded("attention_mask", 0),
            "labels": padded("labels", IGNORE_INDEX),
        }
        if all("mm_token_type_ids" in feature for feature in features):
            batch["mm_token_type_ids"] = padded("mm_token_type_ids", 0)
        for key in ("pixel_values", "image_grid_thw"):
            values = [feature[key] for feature in features if key in feature]
            if values:
                if len(values) != len(features):
                    raise TrainingConfigurationError(f"inconsistent_batch_key:{key}")
                batch[key] = torch.cat(values, dim=0)
        return batch
def locate_model_roots(model: Any) -> ModelRoots:
    """Locate vision/language roots without fuzzy name matching.
    不使用模糊名称匹配，按结构定位视觉与语言根。
    """
    torch = _torch()
    vision_hits: list[tuple[str, Any]] = []
    language_hits: list[tuple[str, Any]] = []
    for name, module in model.named_modules():
        if isinstance(getattr(module, "blocks", None), torch.nn.ModuleList) and hasattr(
            module, "patch_embed"
        ):
            vision_hits.append((name, module))
        if isinstance(getattr(module, "layers", None), torch.nn.ModuleList) and isinstance(
            getattr(module, "embed_tokens", None), torch.nn.Module
        ):
            language_hits.append((name, module))
    if len(vision_hits) != 1 or len(language_hits) != 1:
        raise TrainingConfigurationError("qwen35_model_roots_are_ambiguous")
    vision_name, vision = vision_hits[0]
    language_name, language = language_hits[0]
    if not vision_name or not language_name or vision_name == language_name:
        raise TrainingConfigurationError("qwen35_model_roots_are_invalid")
    return ModelRoots(vision, vision_name, language, language_name)


def enumerate_lora_targets(roots: ModelRoots) -> list[str]:
    """Enumerate all supported hybrid-decoder projections by full path.
    以完整路径枚举混合 decoder 中全部受支持 projection。
    """
    torch = _torch()
    targets: list[str] = []
    for layer_index, layer in enumerate(roots.language.layers):
        full_attention = getattr(layer, "self_attn", None)
        linear_attention = getattr(layer, "linear_attn", None)
        if (full_attention is None) == (linear_attention is None):
            raise TrainingConfigurationError(f"decoder_layer_attention_kind_invalid:{layer_index}")
        attention = full_attention if full_attention is not None else linear_attention
        projections = (
            FULL_ATTN_PROJECTIONS if full_attention is not None else LINEAR_ATTN_PROJECTIONS
        )
        kind = "self_attn" if full_attention is not None else "linear_attn"
        for projection in projections:
            module = getattr(attention, projection, None)
            if not isinstance(module, torch.nn.Linear):
                raise TrainingConfigurationError(
                    f"decoder_projection_missing_or_not_linear:{layer_index}:{kind}:{projection}"
                )
            targets.append(
                f"{roots.language_name}.layers.{layer_index}.{kind}.{projection}"
            )
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise TrainingConfigurationError(f"decoder_layer_mlp_missing:{layer_index}")
        for projection in MLP_PROJECTIONS:
            module = getattr(mlp, projection, None)
            if not isinstance(module, torch.nn.Linear):
                raise TrainingConfigurationError(
                    f"decoder_mlp_projection_missing:{layer_index}:{projection}"
                )
            targets.append(
                f"{roots.language_name}.layers.{layer_index}.mlp.{projection}"
            )
    if len(targets) != len(set(targets)):
        raise TrainingConfigurationError("duplicate_lora_target")
    if any(target.startswith(roots.vision_name + ".") for target in targets):
        raise TrainingConfigurationError("lora_target_escaped_into_vision")
    return targets


def freeze_and_attach_lora(
    model: Any,
    targets: Sequence[str],
    *,
    rank: int,
    alpha: int,
    dropout: float,
) -> Any:
    """Freeze the base and attach LoRA to exact LLM paths. / 冻结 base 并精确挂 LoRA。"""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    peft = _peft()
    # A list of full paths is shortened by PEFT when adapter_config.json is
    # saved (for example to just ``q_proj``), which can match same-name vision
    # modules on reload. A regex string is persisted verbatim and therefore
    # keeps the adapter language-only after save/reload. / PEFT 保存时会把完整
    # 路径列表缩短成 ``q_proj`` 等叶子名，重载时可能误命中视觉同名模块；
    # regex 字符串会原样持久化，从而在保存/重载后仍保持 language-only。
    exact_target_pattern = "(?:" + "|".join(re.escape(target) for target in targets) + ")"
    config = peft.LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=peft.TaskType.CAUSAL_LM,
        target_modules=exact_target_pattern,
    )
    return peft.get_peft_model(model, config)


def initialize_lora_from_adapter(
    peft_model: Any,
    adapter_path: Path,
    *,
    expected_targets: Sequence[str],
    rank: int,
    alpha: int,
    dropout: float,
) -> dict[str, Any]:
    """Load only LoRA A/B from a prior adapter and reject every other legacy tensor.
    仅从旧 adapter 加载 LoRA A/B，并明确审计、丢弃其他旧张量。
    """
    config_path = adapter_path / "adapter_config.json"
    weights_path = adapter_path / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise TrainingConfigurationError("init_adapter_is_incomplete")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingConfigurationError("init_adapter_config_invalid") from error
    if (
        config.get("r") != rank
        or config.get("lora_alpha") != alpha
        or float(config.get("lora_dropout", -1.0)) != dropout
        or config.get("bias") != "none"
    ):
        raise TrainingConfigurationError("init_adapter_lora_config_mismatch")
    try:
        from safetensors.torch import load_file  # noqa: PLC0415

        state = load_file(str(weights_path), device="cpu")
    except Exception as error:
        raise TrainingConfigurationError("init_adapter_weights_invalid") from error
    lora_state = {
        key: value
        for key, value in state.items()
        if ".lora_A." in key or ".lora_B." in key
    }
    discarded = sorted(set(state) - set(lora_state))
    if any("visual_planner_roi_head" not in key for key in discarded):
        raise TrainingConfigurationError("init_adapter_has_unexpected_non_lora_tensor")
    expected_tensor_count = 2 * len(expected_targets)
    if len(lora_state) != expected_tensor_count:
        raise TrainingConfigurationError(
            f"init_adapter_lora_tensor_count_mismatch:{len(lora_state)}:{expected_tensor_count}"
        )
    result = _peft().set_peft_model_state_dict(
        peft_model,
        lora_state,
        adapter_name="default",
    )
    unexpected = sorted(getattr(result, "unexpected_keys", ()))
    if unexpected:
        raise TrainingConfigurationError(
            f"init_adapter_unexpected_keys:{len(unexpected)}"
        )
    return {
        "mode": "prior_lora_weights_only",
        "adapter_config_sha256": sha256_file(config_path),
        "adapter_weights_sha256": sha256_file(weights_path),
        "loaded_lora_tensors": len(lora_state),
        "discarded_legacy_roi_head_tensors": len(discarded),
        "legacy_roi_head_discarded": bool(discarded),
    }


def _logical_parameter_name(name: str) -> str:
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _lora_parent(name: str) -> str:
    segments = name.split(".")
    for index, segment in enumerate(segments):
        if segment.startswith("lora_"):
            return ".".join(segments[:index])
    return ""


def audit_lora_model(peft_model: Any, roots: ModelRoots, targets: Sequence[str]) -> dict[str, Any]:
    """Fail closed unless every trainable parameter is expected LLM LoRA.
    除非所有可训练参数均为预期 LLM LoRA，否则失败。
    """
    target_set = set(targets)
    hit_targets: set[str] = set()
    trainable: list[dict[str, Any]] = []
    total_parameters = 0
    trainable_parameters = 0
    for raw_name, parameter in peft_model.named_parameters():
        name = _logical_parameter_name(raw_name)
        count = int(parameter.numel())
        total_parameters += count
        if not parameter.requires_grad:
            continue
        parent = _lora_parent(name)
        if not parent or parent not in target_set:
            raise TrainingConfigurationError(f"unexpected_trainable_parameter:{name}")
        if parent.startswith(roots.vision_name + "."):
            raise TrainingConfigurationError(f"vision_lora_is_forbidden:{parent}")
        hit_targets.add(parent)
        trainable_parameters += count
        trainable.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "numel": count,
            }
        )
    missing = sorted(target_set - hit_targets)
    if missing:
        raise TrainingConfigurationError(f"lora_targets_not_attached:{len(missing)}")
    return {
        "vision_root": roots.vision_name,
        "language_root": roots.language_name,
        "target_count": len(targets),
        "targets": list(targets),
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "trainable": sorted(trainable, key=lambda item: item["name"]),
        "checks": {
            "all_targets_attached": True,
            "base_model_frozen": True,
            "vision_frozen": True,
            "only_llm_lora_trainable": True,
            "auxiliary_roi_head_absent": True,
        },
    }


def _latest_checkpoint(output_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    if output_dir.is_dir():
        for child in output_dir.iterdir():
            match = _CHECKPOINT_RE.match(child.name)
            adapter_weights = (
                child / "adapter_model.safetensors"
            ).is_file() or (child / "adapter_model.bin").is_file()
            complete = all(
                (child / filename).is_file()
                for filename in (
                    "adapter_config.json",
                    "trainer_state.json",
                    "optimizer.pt",
                    "scheduler.pt",
                )
            ) and adapter_weights
            if child.is_dir() and match and complete:
                candidates.append((int(match.group(1)), child))
    return max(candidates, default=(0, None))[1]


def _safe_model_identity(model_id: str) -> str:
    """Reject local paths as persisted logical model identities.
    拒绝将本地路径持久化为逻辑模型身份。
    """
    if (
        not model_id
        or PurePosixPath(model_id).is_absolute()
        or PureWindowsPath(model_id).is_absolute()
        or model_id.startswith((".", "~"))
    ):
        raise TrainingConfigurationError("base_model_id_must_be_path_independent")
    return model_id


def build_request_identity(
    args: argparse.Namespace,
    *,
    train_path: Path,
    validation_path: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Build the exact weight-affecting identity used for safe resume.
    构造用于安全 resume 的完整权重影响身份。
    """
    model_config = Path(args.model_path) / "config.json"
    if not model_config.is_file():
        raise TrainingConfigurationError("model_config_missing")
    initialization = {"mode": "new_lora"}
    if args.init_adapter_path:
        adapter = Path(args.init_adapter_path)
        initialization = {
            "mode": "prior_lora_weights_only",
            "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
            "adapter_weights_sha256": sha256_file(
                adapter / "adapter_model.safetensors"
            ),
        }
    return {
        "base_model_id": _safe_model_identity(args.base_model_id),
        "model_config_sha256": sha256_file(model_config),
        "data": {
            **summary,
            "train_sha256": sha256_file(train_path),
            "validation_sha256": sha256_file(validation_path),
        },
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "bias": "none",
        },
        "initialization": initialization,
        "optimization": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "warmup_ratio": args.warmup_ratio,
            "max_grad_norm": args.max_grad_norm,
            "max_seq_length": args.max_seq_length,
            "seed": args.seed,
            "torch_dtype": args.torch_dtype,
            "attn_implementation": args.attn_implementation,
            "gradient_checkpointing": args.gradient_checkpointing,
        },
    }


def validate_resume_identity(output_dir: Path, request_identity: dict[str, Any]) -> None:
    """Reject resume when persisted and requested identities differ.
    当持久化身份与新请求不同时拒绝 resume。
    """
    manifest_path = output_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise TrainingConfigurationError("resume_manifest_missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingConfigurationError("resume_manifest_invalid") from error
    if manifest.get("request_identity") != request_identity:
        raise TrainingConfigurationError("resume_request_identity_mismatch")
    if manifest.get("status") == "completed":
        raise TrainingConfigurationError("resume_run_already_completed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3.5-9B multimodal visual-planner LoRA SFT."
    )
    parser.add_argument("--model-path", default="models/Qwen3.5-9B")
    parser.add_argument("--base-model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--dataset-root", default="data/phase2-train-visualplanning-refined-v4"
    )
    parser.add_argument(
        "--output-dir", default="outputs/finetune/qwen35-9b-visual-planner-lora"
    )
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--torch-dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--max-seq-length", type=int, default=6144)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--preflight-samples", type=int, default=10)
    parser.add_argument("--inspect-only", action="store_true")

    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument(
        "--init-adapter-path",
        help="Completed prior adapter whose LoRA A/B initialize a new training run.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Checkpoint directory, or 'auto' for the latest checkpoint-N.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    _safe_model_identity(args.base_model_id)
    if args.max_seq_length < 256:
        raise TrainingConfigurationError("max_seq_length_too_small")
    if args.lora_rank < 1 or args.lora_alpha < 1:
        raise TrainingConfigurationError("lora_rank_and_alpha_must_be_positive")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise TrainingConfigurationError("lora_dropout_out_of_range")
    if args.learning_rate <= 0 or args.epochs <= 0:
        raise TrainingConfigurationError("learning_rate_and_epochs_must_be_positive")
    if args.gradient_accumulation_steps < 1:
        raise TrainingConfigurationError("gradient_accumulation_steps_must_be_positive")
    if args.preflight_samples < 0:
        raise TrainingConfigurationError("preflight_samples_must_be_nonnegative")
    if args.init_adapter_path and args.resume_from_checkpoint:
        raise TrainingConfigurationError("init_adapter_and_resume_are_mutually_exclusive")
    if args.init_adapter_path and not Path(args.init_adapter_path).is_dir():
        raise TrainingConfigurationError("init_adapter_not_found")


def _dataset_summary(train: CompiledChatIndex, validation: CompiledChatIndex) -> dict[str, Any]:
    return {
        "train": {
            "records": len(train),
            "task_counts": train.task_counts(),
            "roi_explicit_records": train.roi_explicit_count(),
        },
        "validation": {
            "records": len(validation),
            "task_counts": validation.task_counts(),
            "roi_explicit_records": validation.roi_explicit_count(),
        },
    }


def _resolve_dtype(name: str) -> Any:
    torch = _torch()
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _preflight(dataset: VisualPlannerDataset, count: int) -> dict[str, Any]:
    # Prefer the first occurrence of each task so the default ten-sample check
    # covers every task family, including the two-image change paths. Fill any
    # remaining slots in source order. / 优先选择每类任务的首条记录，使默认十条
    # 覆盖全部任务（包括双图变化路径）；剩余名额按源顺序补齐。
    selected: list[int] = []
    seen_tasks: set[str] = set()
    for index, entry in enumerate(dataset.index.records):
        if entry.task not in seen_tasks:
            selected.append(index)
            seen_tasks.add(entry.task)
            if len(selected) >= count:
                break
    if len(selected) < count:
        selected_set = set(selected)
        for index in range(len(dataset)):
            if index not in selected_set:
                selected.append(index)
                if len(selected) >= count:
                    break
    lengths: list[int] = []
    supervised: list[int] = []
    tasks: Counter[str] = Counter()
    for index in selected:
        feature = dataset[index]
        lengths.append(int(feature["input_ids"].shape[0]))
        supervised.append(int((feature["labels"] != IGNORE_INDEX).sum()))
        tasks[feature["task"]] += 1
    return {
        "checked": len(lengths),
        "task_counts": dict(sorted(tasks.items())),
        "sequence_length_min": min(lengths) if lengths else None,
        "sequence_length_max": max(lengths) if lengths else None,
        "supervised_tokens_min": min(supervised) if supervised else None,
        "supervised_tokens_max": max(supervised) if supervised else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    dataset_root = Path(args.dataset_root).resolve(strict=True)
    train_path = dataset_root / "training" / "train.jsonl"
    validation_path = dataset_root / "training" / "val.jsonl"
    train_index = CompiledChatIndex(
        train_path,
        dataset_root,
        expected_split="train",
        max_samples=args.max_train_samples,
    )
    validation_index = CompiledChatIndex(
        validation_path,
        dataset_root,
        expected_split="val",
        max_samples=args.max_eval_samples,
    )
    summary = _dataset_summary(train_index, validation_index)
    if args.inspect_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    request_identity = build_request_identity(
        args,
        train_path=train_path,
        validation_path=validation_path,
        summary=summary,
    )
    output_dir = Path(args.output_dir)
    resume: str | None = args.resume_from_checkpoint
    if resume == "auto":
        latest = _latest_checkpoint(output_dir)
        if latest is None:
            raise TrainingConfigurationError("no_checkpoint_available_for_auto_resume")
        resume = str(latest)
    elif resume is not None and not Path(resume).is_dir():
        raise TrainingConfigurationError("resume_checkpoint_not_found")
    if resume is not None:
        if Path(resume).resolve().parent != output_dir.resolve():
            raise TrainingConfigurationError("resume_checkpoint_must_belong_to_output_dir")
        validate_resume_identity(output_dir, request_identity)
    if output_dir.exists() and any(output_dir.iterdir()) and resume is None:
        raise TrainingConfigurationError("output_dir_is_nonempty_without_resume")
    output_dir.mkdir(parents=True, exist_ok=True)

    transformers = _transformers()
    torch = _torch()
    config = transformers.AutoConfig.from_pretrained(
        args.model_path, local_files_only=args.local_files_only
    )
    if config.model_type != "qwen3_5":
        raise TrainingConfigurationError(f"expected_qwen3_5_got:{config.model_type}")
    LOGGER.info("loading Qwen3.5 base model (local_files_only=%s)", args.local_files_only)
    model = transformers.AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        config=config,
        dtype=_resolve_dtype(args.torch_dtype),
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    )
    processor = transformers.AutoProcessor.from_pretrained(
        args.model_path, local_files_only=args.local_files_only
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    transformers.set_seed(args.seed)
    roots = locate_model_roots(model)
    targets = enumerate_lora_targets(roots)
    peft_model = freeze_and_attach_lora(
        model,
        targets,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    initialization_audit = {"mode": "new_lora"}
    if args.init_adapter_path:
        initialization_audit = initialize_lora_from_adapter(
            peft_model,
            Path(args.init_adapter_path),
            expected_targets=targets,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
    audit = audit_lora_model(peft_model, roots, targets)
    audit["initialization"] = initialization_audit
    atomic_write_json(output_dir / AUDIT_NAME, audit)
    LOGGER.info(
        "parameter audit passed: %d LoRA targets, %d / %d parameters trainable (%.4f%%)",
        audit["target_count"],
        audit["trainable_parameters"],
        audit["total_parameters"],
        100.0 * audit["trainable_fraction"],
    )

    if args.gradient_checkpointing:
        peft_model.enable_input_require_grads()
        peft_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        peft_model.config.use_cache = False

    train_dataset = VisualPlannerDataset(train_index, processor, args.max_seq_length)
    validation_dataset = VisualPlannerDataset(
        validation_index, processor, args.max_seq_length
    )
    preflight = _preflight(train_dataset, args.preflight_samples)
    LOGGER.info("data preflight: %s", json.dumps(preflight, ensure_ascii=False))

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "request_identity": request_identity,
        "initialization": initialization_audit,
        "base_model_id": args.base_model_id,
        "model_type": config.model_type,
        "architectures": list(config.architectures or []),
        "data": {
            **summary,
            "train_sha256": sha256_file(train_path),
            "validation_sha256": sha256_file(validation_path),
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "format": "visual-planner-compiled-chat-v1",
            "target_version": "visual-task-plan-v5",
        },
        "supervision": {
            "objective": "assistant_json_causal_lm_cross_entropy",
            "ignore_index": IGNORE_INDEX,
            "global_reduction": "mean_over_supervised_assistant_tokens",
            "roi_coordinates": "ordinary_assistant_json_tokens",
            "task_specific_heads": False,
            "auxiliary_roi_head": False,
        },
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "bias": "none",
            "target_count": len(targets),
            "targets": targets,
            "modules_to_save": [],
        },
        "optimization": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "warmup_ratio": args.warmup_ratio,
            "max_seq_length": args.max_seq_length,
            "seed": args.seed,
            "torch_dtype": args.torch_dtype,
            "attn_implementation": args.attn_implementation,
            "gradient_checkpointing": args.gradient_checkpointing,
        },
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": _peft().__version__,
        },
        "preflight": preflight,
    }
    atomic_write_json(output_dir / MANIFEST_NAME, manifest)

    use_bf16 = args.torch_dtype == "bfloat16"
    use_fp16 = args.torch_dtype == "float16"
    training_args = transformers.TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=args.max_grad_norm,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        optim="adamw_torch",
    )
    resume_start_step = 0
    if resume is not None:
        try:
            resume_state = json.loads(
                (Path(resume) / "trainer_state.json").read_text(encoding="utf-8")
            )
            resume_start_step = int(resume_state.get("global_step", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise TrainingConfigurationError("resume_trainer_state_invalid") from error
    trainer = transformers.Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=VisualPlannerCollator(processor.tokenizer.pad_token_id),
        processing_class=processor,
    )
    result = trainer.train(resume_from_checkpoint=resume)
    # Always persist one evaluation at the exact final step, even when the
    # periodic interval does not divide total steps. / 即使总 step 不能被周期
    # 整除，也始终在最终 step 持久化一次验证结果。
    final_evaluation = trainer.evaluate(metric_key_prefix="final_eval")
    final_adapter = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    processor.save_pretrained(final_adapter)
    new_steps = int(trainer.state.global_step) - resume_start_step
    result_metrics = {
        key: value
        for key, value in result.metrics.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    if new_steps == 0:
        result_metrics.pop("train_loss", None)
    manifest["status"] = "completed"
    manifest["result"] = {
        "global_step": int(trainer.state.global_step),
        "new_steps_this_invocation": new_steps,
        "training_loss": float(result.training_loss) if new_steps > 0 else None,
        "metrics": result_metrics,
        "final_evaluation": {
            key: value
            for key, value in final_evaluation.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        },
        "final_adapter": "final_adapter",
    }
    atomic_write_json(output_dir / MANIFEST_NAME, manifest)
    LOGGER.info("training completed; final adapter saved under final_adapter/")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    raise SystemExit(main())
