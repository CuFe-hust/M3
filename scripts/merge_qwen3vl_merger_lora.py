#!/usr/bin/env python3
"""Merge the Qwen3-VL-8B merger LoRA adapter into a full checkpoint.

将 Qwen3-VL-8B merger LoRA 适配器合并回完整权重。默认路径与远端训练输出
布局一致：base = models/qwen3_vl_8b/weights，adapter =
outputs/finetune/qwen3-vl-8b-merger-lora，输出 =
models/qwen3_vl_8b/merged。合并后的目录可直接被 models.qwen_transformers /
models.entry.create_model 加载（config + 权重 + processor + 辅助配置齐全）。
脚本只在执行时惰性导入 torch/peft/transformers，因此 --help 无需这些依赖。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

# Auxiliary base files that model/processor save_pretrained does not emit;
# copied from the base checkpoint only when missing in the output directory.
# 模型/processor 的 save_pretrained 不会产出的辅助文件；仅当输出目录缺失时
# 从 base 复制。
_AUXILIARY_FILES = (
    "generation_config.json",
    "chat_template.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "README.md",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the merge CLI. / 构建合并 CLI。"""

    parser = argparse.ArgumentParser(
        description=(
            "Merge the Qwen3-VL-8B merger LoRA adapter into a full "
            "checkpoint via PEFT."
        )
    )
    parser.add_argument(
        "--model-id",
        type=Path,
        default=Path("models/qwen3_vl_8b/weights"),
        help="Base Qwen3-VL checkpoint directory (default: local 8B weights).",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=Path("outputs/finetune/qwen3-vl-8b-merger-lora"),
        help="Directory produced by finetune_qwen3vl_merger_lora.py.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("models/qwen3_vl_8b/merged"),
        help="Directory for the merged checkpoint (must not already exist).",
    )
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=("float32", "float16", "bfloat16", "auto"),
        help="Dtype used while loading and merging; default bfloat16.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help=(
            "Torch device for merging; cpu is safe for an 8.77B model, "
            "cuda:0 is much faster."
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Refuse network access while loading the base checkpoint; "
            "default on for offline safety."
        ),
    )
    return parser


def _resolve_dtype(name: str) -> Any:
    """Resolve the CLI dtype name to a torch dtype.
    将 CLI dtype 名称解析为 torch dtype。"""

    import torch

    if name == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def _validate_adapter(adapter_path: Path) -> None:
    """Fail fast when the adapter directory is not a loadable PEFT LoRA.
    适配器目录不是可加载的 PEFT LoRA 时快速失败。"""

    if not adapter_path.is_dir():
        raise ValueError(f"adapter path is not a directory: {adapter_path}")
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"missing adapter_config.json under {adapter_path}")
    weights = (
        adapter_path / "adapter_model.safetensors",
        adapter_path / "adapter_model.bin",
    )
    if not any(path.is_file() for path in weights):
        raise ValueError(f"missing adapter_model weights under {adapter_path}")


def _copy_auxiliary_files(model_id: Path, output_path: Path) -> list[str]:
    """Copy small base-side config files that save_pretrained does not emit;
    never overwrite an existing output file. 复制 save_pretrained 不会产出的
    小配置；绝不覆盖输出目录已有文件。"""

    copied: list[str] = []
    for filename in _AUXILIARY_FILES:
        source = model_id / filename
        target = output_path / filename
        if source.is_file() and not target.exists():
            shutil.copy2(source, target)
            copied.append(filename)
    return copied


def main() -> int:
    args = build_parser().parse_args()
    try:
        _validate_adapter(args.adapter_path)
        output_path = Path(args.output_path)
        if output_path.exists():
            raise ValueError(
                f"output path already exists; choose a new path or remove it: "
                f"{output_path}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = _resolve_dtype(args.torch_dtype)
        base_model = AutoModelForImageTextToText.from_pretrained(
            args.model_id,
            torch_dtype=dtype,
            device_map=args.device,
            local_files_only=args.local_files_only,
        )
        peft_model = PeftModel.from_pretrained(base_model, args.adapter_path)
        merged_model = peft_model.merge_and_unload()
        merged_model.save_pretrained(output_path, safe_serialization=True)
        del merged_model, peft_model, base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        processor = AutoProcessor.from_pretrained(
            args.model_id,
            local_files_only=args.local_files_only,
        )
        processor.save_pretrained(output_path)
        copied = _copy_auxiliary_files(Path(args.model_id), output_path)
        print(
            f"Merged checkpoint saved to {output_path} "
            f"(auxiliary files copied: {', '.join(copied) or 'none'})"
        )
        return 0
    except KeyboardInterrupt:
        print("merge interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        # Public failure output never carries raw exception text or secrets.
        # 公共失败输出绝不携带原始异常文本或密钥。
        print(f"merge failed: {type(error).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
