#!/usr/bin/env python3
"""Backward-compatible ChangeAgent wrapper over the generic SFT core."""

from __future__ import annotations

import sys
from typing import Sequence

from scripts.finetune_multimodal_sft import main as _generic_main


def _normalize_args(args: list[str]) -> list[str]:
    mapping = {
        "--train_file": "--train-file",
        "--eval_file": "--validation-manifest",
        "--output_dir": "--output-dir",
        "--model_id": "--model-id",
        "--max_train_samples": "--max-train-samples",
        "--max_eval_samples": "--max-eval-samples",
        "--gradient_accumulation_steps": "--gradient-accumulation",
        "--num_train_epochs": "--epochs",
        "--resume_from_checkpoint": "--resume-from",
        "--model_name_or_path": "--model-path",
        "--lora_rank": "--lora-rank",
        "--lora_alpha": "--lora-alpha",
        "--lora_dropout": "--lora-dropout",
        "--lora_lr": "--lora-lr",
        "--merger_lr": "--connector-lr",
        "--connector_lr": "--connector-lr",
        "--weight_decay": "--weight-decay",
        "--warmup_ratio": "--warmup-ratio",
        "--max_grad_norm": "--max-grad-norm",
        "--smoke_gradients": "--smoke-gradients",
        "--preflight_only": "--preflight-only",
        "--logging_steps": "--logging-steps",
    }
    normalized: list[str] = []
    for value in args:
        normalized.append(mapping.get(value, value))
    normalized.extend(["--model-adapter", "auto", "--data-profile", "change_agent"])
    return normalized


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--data_profile" in args or "--data-profile" in args:
        raise ValueError("finetune_change_qwen_sft fixes --data_profile=change_agent")
    if "--repeat_group_key" not in args and "--repeat-group-key" not in args:
        args = ["--repeat_group_key", "task", *args]
    return _generic_main(_normalize_args(args))


if __name__ == "__main__":
    main()
