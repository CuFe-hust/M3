#!/usr/bin/env python3
"""Legacy Phase2 CLI compatibility wrapper.

The implementation moved to ``training.multimodal_sft``. The historical
module is retained at ``scripts._legacy_finetune_qwen3vl_phase2`` so old unit
tests, checkpoint readers and helper imports remain available while the public
entry point resolves through the generic trainer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import _legacy_finetune_qwen3vl_phase2 as _legacy
from scripts.finetune_multimodal_sft import main as _generic_main

for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_legacy, _name))


def __getattr__(name: str) -> Any:
    return getattr(_legacy, name)


def _translate_legacy_args(argv: Sequence[str]) -> list[str]:
    """Map the stable subset of legacy flags to the generic CLI."""

    args = list(argv)
    if "--deepspeed" in args or "--fsdp" in args:
        raise _legacy.ConfigurationError(
            "DeepSpeed/FSDP are not supported by the generic composite checkpoint path"
        )
    translated: list[str] = []
    mapping = {
        "--model-id": "--model-id", "--model_id": "--model-id",
        "--model-path": "--model-path", "--model_name_or_path": "--model-path",
        "--train-file": "--train-file", "--train_file": "--train-file",
        "--output-dir": "--output-dir", "--output_dir": "--output-dir",
        "--max-train-samples": "--max-train-samples", "--max_train_samples": "--max-train-samples",
        "--max-eval-samples": "--max-eval-samples", "--max_eval_samples": "--max-eval-samples",
        "--gradient-accumulation-steps": "--gradient-accumulation", "--gradient_accumulation_steps": "--gradient-accumulation",
        "--num-train-epochs": "--epochs", "--num_train_epochs": "--epochs",
        "--max-steps": "--max-steps", "--max_steps": "--max-steps",
        "--seed": "--seed",
        "--resume-from-checkpoint": "--resume-from", "--resume_from_checkpoint": "--resume-from",
        "--image-root": "--image-root",
        "--eval-file": "--validation-manifest", "--eval_file": "--validation-manifest",
        "--lora-rank": "--lora-rank", "--lora_rank": "--lora-rank",
        "--lora-alpha": "--lora-alpha", "--lora_alpha": "--lora-alpha",
        "--lora-dropout": "--lora-dropout", "--lora_dropout": "--lora-dropout",
        "--lora-lr": "--lora-lr", "--lora_lr": "--lora-lr",
        "--merger-lr": "--connector-lr", "--merger_lr": "--connector-lr",
        "--connector-lr": "--connector-lr", "--connector_lr": "--connector-lr",
        "--weight-decay": "--weight-decay", "--weight_decay": "--weight-decay",
        "--warmup-ratio": "--warmup-ratio", "--warmup_ratio": "--warmup-ratio",
        "--max-grad-norm": "--max-grad-norm", "--max_grad_norm": "--max-grad-norm",
        "--repeat-group-key": "--repeat-group-key", "--repeat_group_key": "--repeat-group-key",
        "--repeat-weights": "--repeat-weights", "--repeat_weights": "--repeat-weights",
        "--smoke-gradients": "--smoke-gradients", "--smoke_gradients": "--smoke-gradients",
        "--preflight-only": "--preflight-only", "--preflight_only": "--preflight-only",
        "--logging-steps": "--logging-steps", "--logging_steps": "--logging-steps",
        "--torch-dtype": "--dtype", "--torch_dtype": "--dtype", "--dtype": "--dtype",
        "--local-files-only": "--local-files-only", "--local_files_only": "--local-files-only",
    }
    index = 0
    while index < len(args):
        value = args[index]
        replacement = mapping.get(value)
        if replacement is None:
            index += 1
            continue
        translated.append(replacement)
        if index + 1 < len(args) and not args[index + 1].startswith("--"):
            translated.append(args[index + 1])
            index += 2
        else:
            index += 1
    if "--model-id" not in translated:
        translated.extend(["--model-id", "Qwen/Qwen3-VL-8B-Instruct"])
    if "--train-file" not in translated:
        raise _legacy.ConfigurationError("--train-file is required")
    translated.extend(["--model-adapter", "auto", "--data-profile", "phase2"])
    return translated


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    return _generic_main(_translate_legacy_args(args))


if __name__ == "__main__":
    raise SystemExit(main())
