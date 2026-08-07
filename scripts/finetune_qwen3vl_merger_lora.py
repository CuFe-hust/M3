"""SFT LoRA training for the Qwen3-VL merger layers.
针对 Qwen3-VL merger 层的 LoRA SFT 训练脚本。

Qwen3-VL-8B has four ``Qwen3VLVisionPatchMerger`` modules: the final
``visual.merger`` plus three ``visual.deepstack_merger_list.*`` deepstack
mergers. This script attaches LoRA to the two ``nn.Linear`` layers inside each
of the four mergers (eight linear layers in total) and keeps every other
weight frozen, so the whole model stays on a pure LoRA schedule.
Qwen3-VL-8B 共有四个 ``Qwen3VLVisionPatchMerger`` 模块：最终的
``visual.merger`` 和三个 ``visual.deepstack_merger_list.*`` deepstack merger。
本脚本为四个 merger 内的两个 ``nn.Linear``（共八个线性层）挂 LoRA，
其余权重全部冻结，使整个模型保持纯 LoRA 训练。

The dataset format is the reference Qwen-VL-Series-Finetune conversation
format: a JSON/JSONL array whose items are
``{"id", "image", "conversations": [human, gpt]}``. Run
``scripts/prepare_vrsbench_sft.py`` first to produce it from VRSBench JSONL.
数据集格式与 Qwen-VL-Series-Finetune 参考仓库一致：JSON/JSONL 数组，元素为
``{"id", "image", "conversations": [human, gpt]}``。先运行
``scripts/prepare_vrsbench_sft.py`` 从 VRSBench JSONL 生成该格式。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)


IGNORE_INDEX = -100

# Chat-template tokens used by Qwen3-VL. / Qwen3-VL 使用的对话模板 token。
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"


@dataclass
class ModelArguments:
    """Model loading arguments. / 模型加载参数。"""

    model_id: str = field(
        default="Qwen/Qwen3-VL-8B-Instruct",
        metadata={"help": "Hugging Face id or local checkpoint directory."},
    )
    local_files_only: bool = field(
        default=False,
        metadata={"help": "Refuse network access while loading the checkpoint."},
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"choices": ("float32", "float16", "bfloat16", "auto")},
    )
    attn_implementation: str = field(
        default="sdpa",
        metadata={"choices": ("sdpa", "flash_attention_2", "eager")},
    )


@dataclass
class DataArguments:
    """Dataset and preprocessing arguments. / 数据集与预处理参数。"""

    train_file: str = field(metadata={"help": "Path to the SFT train JSON/JSONL."})
    eval_file: str | None = field(
        default=None, metadata={"help": "Optional SFT validation JSON/JSONL."}
    )
    image_folder: str | None = field(
        default=None,
        metadata={"help": "Root that relative image paths are joined against."},
    )
    image_min_pixels: int = field(
        default=256 * 32 * 32,
        metadata={"help": "Minimum image pixels; Qwen3-VL uses multiples of 32."},
    )
    image_max_pixels: int = field(
        default=1280 * 32 * 32,
        metadata={"help": "Maximum image pixels; Qwen3-VL uses multiples of 32."},
    )
    max_seq_length: int = field(
        default=4096,
        metadata={"help": "Truncate tokenized prompt+response to this length."},
    )
    max_train_samples: int | None = field(
        default=None, metadata={"help": "Cap the train dataset for smoke runs."}
    )
    max_eval_samples: int | None = field(
        default=None, metadata={"help": "Cap the eval dataset for smoke runs."}
    )


@dataclass
class LoRAArguments:
    """LoRA hyperparameters exposed on the command line.
    通过命令行暴露的 LoRA 超参数。
    """

    lora_rank: int = field(default=16, metadata={"help": "LoRA rank r."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_bias: str = field(
        default="none",
        metadata={"choices": ("none", "all", "lora_only"), "help": "LoRA bias mode."},
    )
    freeze_merger_base: bool = field(
        default=True,
        metadata={
            "help": "Pure LoRA keeps merger base weights frozen; set False to also "
            "fully train the merger base weights (reference-repo style)."
        },
    )


def check_runtime() -> None:
    """Verify that the installed Transformers can train the deepstack path.
    校验已安装 Transformers 可训练 deepstack 路径。
    """
    from packaging.version import Version

    import transformers

    if Version(transformers.__version__) < Version("5.6.0"):
        raise RuntimeError(
            "Qwen3-VL merger LoRA training requires transformers>=5.6.0; "
            f"found {transformers.__version__}. The reference environment uses "
            "transformers==5.14.1."
        )
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    except ImportError as error:
        raise RuntimeError("This Transformers build does not provide Qwen3-VL.") from error
    if not hasattr(Qwen3VLTextModel, "_deepstack_process"):
        raise RuntimeError(
            "This Transformers build lacks the Qwen3-VL deepstack fusion path; "
            "upgrade to transformers==5.14.1 or use the reference repo monkey patch."
        )


def resolve_dtype(torch_module: Any, name: str) -> Any:
    """Resolve the CLI dtype name to a torch dtype.
    将 CLI dtype 名称解析为 torch dtype。
    """
    if name == "auto":
        return "auto"
    mapping = {
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {name}")
    return mapping[name]


def load_records(path: str) -> list[dict[str, Any]]:
    """Load SFT records from a JSON array or JSONL file.
    从 JSON 数组或 JSONL 文件加载 SFT 记录。
    """
    source = Path(path)
    if not source.is_file():
        raise SystemExit(f"Training data file not found: {source}")
    if source.suffix == ".jsonl":
        records: list[dict[str, Any]] = []
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"Invalid JSON at {source}:{line_number}: {error}") from error
        return records
    try:
        loaded = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON file {source}: {error}") from error
    if not isinstance(loaded, list):
        raise RuntimeError(f"{source} must contain a JSON array of records.")
    return loaded


def validate_record(record: dict[str, Any]) -> None:
    """Validate one SFT record. / 校验一条 SFT 记录。"""
    if not record.get("id") or not record.get("image"):
        raise RuntimeError(f"SFT record missing id/image: {record!r}")
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        raise RuntimeError(f"SFT record must have two conversations: {record!r}")
    roles = [item.get("from") for item in conversations]
    if roles != ["human", "gpt"]:
        raise RuntimeError(f"SFT conversations must be human then gpt: {record!r}")
    if not conversations[0].get("value") or not conversations[1].get("value"):
        raise RuntimeError(f"SFT conversation text is empty: {record!r}")


def replace_image_tokens(text: str) -> str:
    """Replace the LLaVA-style <image> token with Qwen vision tokens.
    将 LLaVA 风格 <image> token 替换为 Qwen 视觉 token。
    """
    return text.replace(
        LLAVA_IMAGE_TOKEN,
        f"{VISION_START}{IMAGE_PAD}{VISION_END}",
    )


class VRSBenchSFTDataset(Dataset):
    """Tokenized SFT dataset for one image per record.
    每条记录一张图片的 SFT 数据集。
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
        processor: Any,
        image_folder: str | None,
        image_min_pixels: int,
        image_max_pixels: int,
        max_seq_length: int,
    ) -> None:
        super().__init__()
        self.records = records
        self.processor = processor
        self.image_folder = image_folder
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        validate_record(record)
        image_path = self._resolve_image_path(record["image"])
        image = Image.open(image_path).convert("RGB")
        conversations = record["conversations"]
        human_text = replace_image_tokens(conversations[0]["value"])
        gpt_text = conversations[1]["value"]
        user_input = f"{IM_START}user\n{human_text}{IM_END}\n{IM_START}assistant\n"
        gpt_response = f"{gpt_text}{IM_END}\n"

        inputs = self.processor(
            text=[user_input],
            images=[image],
            padding=False,
            return_tensors="pt",
            min_pixels=self.image_min_pixels,
            max_pixels=self.image_max_pixels,
        )
        prompt_ids = inputs["input_ids"][0]
        prompt_mm = inputs.get("mm_token_type_ids")
        if prompt_mm is None:
            prompt_mm = torch.zeros_like(prompt_ids, dtype=torch.long)
        else:
            prompt_mm = prompt_mm[0].to(dtype=torch.long)
        response_ids = self.processor.tokenizer(
            gpt_response, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]

        input_ids = torch.cat([prompt_ids, response_ids], dim=0)
        labels = torch.cat(
            [torch.full_like(prompt_ids, IGNORE_INDEX), response_ids], dim=0
        )
        mm_token_type_ids = torch.cat(
            [prompt_mm, torch.zeros_like(response_ids, dtype=torch.long)], dim=0
        )
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[: self.max_seq_length]
            labels = labels[: self.max_seq_length]
            mm_token_type_ids = mm_token_type_ids[: self.max_seq_length]

        data: dict[str, Any] = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
            "mm_token_type_ids": mm_token_type_ids,
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
        }
        return data

    def _resolve_image_path(self, image_field: str | list[str]) -> Path:
        if isinstance(image_field, list):
            if len(image_field) != 1:
                raise RuntimeError("Merger LoRA training expects exactly one image per record.")
            image_field = image_field[0]
        path = Path(image_field)
        if path.is_absolute():
            return path
        if not self.image_folder:
            raise RuntimeError(
                f"Relative image path {image_field!r} requires --image_folder."
            )
        return Path(self.image_folder) / path


class DataCollatorForSupervisedDataset:
    """Pad text tensors and stack vision tensors for the Trainer.
    为 Trainer 填充文本张量并拼接视觉张量。
    """

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids = _pad_sequence([f["input_ids"] for f in features], self.pad_token_id)
        labels = _pad_sequence([f["labels"] for f in features], IGNORE_INDEX)
        mm_token_type_ids = _pad_sequence(
            [f["mm_token_type_ids"] for f in features], 0
        )
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.pad_token_id),
            "mm_token_type_ids": mm_token_type_ids,
        }
        pixel_values = [f["pixel_values"] for f in features]
        image_grid_thw = [f["image_grid_thw"] for f in features]
        batch["pixel_values"] = torch.cat(pixel_values, dim=0)
        batch["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)
        return batch


def _pad_sequence(sequences: list[torch.Tensor], padding_value: int) -> torch.Tensor:
    """Right-pad a list of 1-D tensors to the same length.
    将一维张量列表右填充到相同长度。
    """
    max_length = max(len(sequence) for sequence in sequences)
    padded = torch.full(
        (len(sequences), max_length),
        padding_value,
        dtype=sequences[0].dtype,
    )
    for row, sequence in enumerate(sequences):
        padded[row, : len(sequence)] = sequence
    return padded


def find_merger_linear_names(model: nn.Module) -> list[str]:
    """Return every nn.Linear whose module path contains "merger".
    返回模块路径包含 "merger" 的所有 nn.Linear。
    """
    names = [
        name
        for name, module in model.named_modules()
        if "merger" in name and isinstance(module, nn.Linear)
    ]
    return names


def apply_merger_lora(
    model: nn.Module,
    rank: int,
    alpha: int,
    dropout: float,
    bias: str,
    freeze_merger_base: bool,
) -> tuple[nn.Module, list[str], int, int]:
    """Attach LoRA to the four merger modules and freeze everything else.
    为四个 merger 模块挂 LoRA 并冻结其余全部参数。
    """
    target_modules = find_merger_linear_names(model)
    if len(target_modules) < 8:
        raise RuntimeError(
            "Expected at least 8 merger linear layers (4 mergers x 2 linears) for "
            f"Qwen3-VL, found {len(target_modules)}: {target_modules}"
        )
    from peft import LoraConfig, get_peft_model

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    peft_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias=bias,
    )
    model = get_peft_model(model, peft_config)
    if not freeze_merger_base:
        for name, parameter in model.named_parameters():
            if "merger" in name and "lora_" not in name:
                parameter.requires_grad = True

    trainable_parameters = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    lora_trainable = [name for name in trainable_parameters if "lora_" in name]
    if not lora_trainable:
        raise RuntimeError("No LoRA parameter is trainable; check target module names.")
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return model, target_modules, trainable, total


def load_model(model_args: ModelArguments) -> nn.Module:
    """Load the untouched Qwen3-VL generation model.
    加载未修改的 Qwen3-VL 生成模型。
    """
    dtype = resolve_dtype(torch, model_args.torch_dtype)
    config = AutoConfig.from_pretrained(
        model_args.model_id,
        local_files_only=model_args.local_files_only,
    )
    if config.model_type != "qwen3_vl":
        raise RuntimeError(
            f"This script supports qwen3_vl only, got model_type={config.model_type!r}."
        )
    return AutoModelForImageTextToText.from_pretrained(
        model_args.model_id,
        config=config,
        torch_dtype=dtype,
        attn_implementation=model_args.attn_implementation,
        local_files_only=model_args.local_files_only,
    )


def _auto_resume_flag(output_dir: str, explicit: str | None) -> str | bool | None:
    """Return True to auto-resume the latest checkpoint when no explicit value is set.
    未显式指定时返回 True 以自动恢复最新 checkpoint。
    """
    if explicit is not None:
        return explicit
    if any(Path(output_dir).glob("checkpoint-*")):
        return True
    return None


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, LoRAArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()
    check_runtime()

    training_args.remove_unused_columns = False
    model = load_model(model_args)
    processor = AutoProcessor.from_pretrained(
        model_args.model_id,
        local_files_only=model_args.local_files_only,
    )
    model, target_modules, trainable, total = apply_merger_lora(
        model=model,
        rank=lora_args.lora_rank,
        alpha=lora_args.lora_alpha,
        dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        freeze_merger_base=lora_args.freeze_merger_base,
    )
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    train_records = load_records(data_args.train_file)
    if data_args.max_train_samples is not None:
        train_records = train_records[: data_args.max_train_samples]
    eval_records = (
        load_records(data_args.eval_file)
        if data_args.eval_file is not None
        else None
    )
    if eval_records is not None and data_args.max_eval_samples is not None:
        eval_records = eval_records[: data_args.max_eval_samples]

    train_dataset = VRSBenchSFTDataset(
        records=train_records,
        processor=processor,
        image_folder=data_args.image_folder,
        image_min_pixels=data_args.image_min_pixels,
        image_max_pixels=data_args.image_max_pixels,
        max_seq_length=data_args.max_seq_length,
    )
    eval_dataset = (
        VRSBenchSFTDataset(
            records=eval_records,
            processor=processor,
            image_folder=data_args.image_folder,
            image_min_pixels=data_args.image_min_pixels,
            image_max_pixels=data_args.image_max_pixels,
            max_seq_length=data_args.max_seq_length,
        )
        if eval_records is not None
        else None
    )
    data_collator = DataCollatorForSupervisedDataset(
        pad_token_id=processor.tokenizer.pad_token_id
    )

    if training_args.local_rank in (0, -1):
        print(
            f"Merger LoRA targets ({len(target_modules)}): "
            + ", ".join(target_modules[:4])
            + " ..."
        )
        print(
            f"Trainable parameters: {trainable:,} / {total:,} "
            f"({trainable / total * 100:.4f}%)"
        )
        print(f"Train records: {len(train_records)}; eval records: {len(eval_records) or 0}")

    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=processor,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    resume_from_checkpoint = _auto_resume_flag(
        training_args.output_dir, training_args.resume_from_checkpoint
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_state()

    if training_args.local_rank in (0, -1):
        Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
        model.config.save_pretrained(training_args.output_dir)
        model.save_pretrained(training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
