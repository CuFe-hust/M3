"""Merge a Qwen3-VL merger LoRA adapter into a full checkpoint.
将 Qwen3-VL merger LoRA 适配器合并回完整权重。

The merged directory can be loaded directly by the existing Qwen3-VL wrappers
and served through ``main.py`` / ``models.entry.create_model``.
合并后的目录可直接被现有 Qwen3-VL 封装加载，并可通过
``main.py`` / ``models.entry.create_model`` 使用。
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    """Build the merge CLI. / 构建合并 CLI。"""
    parser = argparse.ArgumentParser(
        description="Merge a Qwen3-VL merger LoRA adapter into a full checkpoint."
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Base Qwen3-VL checkpoint (local directory or Hugging Face id).",
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="Directory produced by finetune_qwen3vl_merger_lora.py.",
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Directory for the merged checkpoint.",
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
        help="Torch device for merging; default cpu to save GPU memory.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Refuse network access while loading the base checkpoint.",
    )
    return parser


def resolve_dtype(name: str):
    """Resolve the CLI dtype name to a torch dtype.
    将 CLI dtype 名称解析为 torch dtype。
    """
    import torch

    if name == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def main() -> int:
    args = build_parser().parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    dtype = resolve_dtype(args.torch_dtype)
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map=args.device,
        local_files_only=args.local_files_only,
    )
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_path)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(output_path, safe_serialization=True)
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
    )
    processor.save_pretrained(output_path)
    print(f"Merged checkpoint saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
