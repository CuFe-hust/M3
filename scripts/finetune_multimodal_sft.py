#!/usr/bin/env python3
"""Generic multimodal SFT entry point.

The CLI selects a task/data profile and a model adapter independently.  Use
``--plan-only`` to run the no-weight adapter probe and print the contract
boundary before any model is loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.multimodal_sft.checkpoint import CheckpointContractError
from training.multimodal_sft.data import profile_for
from training.multimodal_sft.parameter_plan import TuningPolicy
from training.multimodal_sft.registry import UnsupportedModelAdapter, default_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model-agnostic multimodal SFT trainer")
    parser.add_argument("--model-id", "--model-path", "--model_id", "--model_path", dest="model_id", required=True)
    parser.add_argument("--model-adapter", default="auto")
    parser.add_argument("--data-profile", default="phase2", choices=("phase2", "change_agent"))
    parser.add_argument("--train-file", "--train-manifest", "--train_file", dest="train_file")
    parser.add_argument("--validation-manifest", "--eval-file", "--eval_file", dest="validation_manifest")
    parser.add_argument("--output-dir", default="outputs/multimodal_sft")
    parser.add_argument("--tuning-policy", default="lora_plus_projector", choices=("lora_only", "projector_only", "lora_plus_projector", "full_language_lora"))
    parser.add_argument("--lora-rank", "--lora_rank", type=int, default=64)
    parser.add_argument("--lora-alpha", "--lora_alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", "--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora-lr", "--lora_lr", type=float, default=1e-4)
    parser.add_argument("--connector-lr", "--connector_lr", "--merger-lr", "--merger_lr", dest="connector_lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", "--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", "--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--batch-size", "--per-device-train-batch-size", "--per_device_train_batch_size", dest="batch_size", type=int, default=1, help="Phase 1D deferred: only micro-batch size 1 is supported")
    parser.add_argument("--gradient-accumulation", "--gradient-accumulation-steps", "--gradient_accumulation_steps", dest="gradient_accumulation", type=int, default=1)
    parser.add_argument("--epochs", "--num-train-epochs", "--num_train_epochs", dest="epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resume-from", "--resume-from-checkpoint", "--resume_from_checkpoint", dest="resume_from", type=str)
    parser.add_argument("--preflight-only", "--preflight_only", action="store_true")
    parser.add_argument("--smoke-gradients", "--smoke_gradients", action="store_true")
    parser.add_argument("--max-train-samples", "--max_train_samples", type=int)
    parser.add_argument("--max-eval-samples", "--max_eval_samples", type=int)
    parser.add_argument("--repeat-group-key", "--repeat_group_key", dest="repeat_group_key")
    parser.add_argument("--repeat-weights", "--repeat_weights", action="append", default=[])
    parser.add_argument("--logging-steps", "--logging_steps", type=int, default=10)
    parser.add_argument("--save-steps", "--save_steps", type=int, default=0)
    parser.add_argument("--save-total-limit", "--save_total_limit", type=int)
    parser.add_argument("--deepspeed")
    parser.add_argument("--fsdp")
    parser.add_argument("--fsdp-config", "--fsdp_config")
    parser.add_argument("--image-root", action="append", default=[])
    parser.add_argument("--dtype", "--torch-dtype", "--torch_dtype", dest="dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", "--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan-only", action="store_true", help="Probe adapter only; never load model weights or train")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.batch_size != 1:
            raise ValueError("GENERIC_BATCHING_NOT_YET_AVAILABLE: micro-batch size must be 1 before Phase 1D")
        if args.deepspeed or args.fsdp or args.fsdp_config:
            raise ValueError("distributed_backend=unsupported_in_generic_phase1: DeepSpeed/FSDP are not implemented")
        profile = profile_for(args.data_profile)
        registry = default_registry()
        adapter, probe = registry.resolve(
            args.model_id,
            model_adapter=args.model_adapter,
            local_files_only=args.local_files_only,
        )
        selected_policy = replace(
            TuningPolicy.from_name(args.tuning_policy),
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        result = {
            "adapter": adapter.name,
            "probe": probe.as_dict(),
            "data_profile": profile.name,
            "tuning_policy": selected_policy.as_dict(),
            "plan_only": bool(args.plan_only),
        }
        if args.plan_only:
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if not args.train_file:
            raise CheckpointContractError("--train-file is required unless --plan-only is used")
        model, processor, loaded_probe = adapter.load(
            args.model_id,
            dtype=args.dtype,
            device=args.device,
            local_files_only=args.local_files_only,
        )
        from training.multimodal_sft.trainer_core import GenericTrainerCore, TrainingConfig

        episodes = profile.read(args.train_file)
        eval_episodes = profile.read(args.validation_manifest) if args.validation_manifest else None
        trainer = GenericTrainerCore(adapter=adapter, data_profile=profile)
        repeat_weights: dict[str, int] = {}
        for item in args.repeat_weights:
            if "=" not in item:
                raise CheckpointContractError("--repeat-weights must use group=count")
            group, raw_count = item.split("=", 1)
            repeat_weights[group] = int(raw_count)
        training_result = trainer.fit(
            model=model,
            processor=processor,
            episodes=episodes,
            config=TrainingConfig(
                output_dir=args.output_dir,
                epochs=args.epochs,
                lora_lr=args.lora_lr,
                connector_lr=args.connector_lr,
                weight_decay=args.weight_decay,
                warmup_ratio=args.warmup_ratio,
                max_grad_norm=args.max_grad_norm,
                max_steps=args.max_steps,
                gradient_accumulation_steps=args.gradient_accumulation,
                seed=args.seed,
                preflight_only=args.preflight_only,
                smoke_gradients=args.smoke_gradients,
                max_train_samples=args.max_train_samples,
                max_eval_samples=args.max_eval_samples,
                logging_steps=args.logging_steps,
                batch_size=args.batch_size,
                repeat_group_key=args.repeat_group_key,
                repeat_weights=repeat_weights,
                save_steps=args.save_steps,
                save_total_limit=args.save_total_limit,
                data_contract={"image_roots": list(args.image_root), "batch_size": args.batch_size},
            ),
            policy=selected_policy,
            probe=loaded_probe,
            model_identity=loaded_probe.identity.as_dict(),
            eval_episodes=eval_episodes,
        )
        result["steps"] = training_result.steps
        result["manifest"] = str(training_result.manifest_path) if training_result.manifest_path else None
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (UnsupportedModelAdapter, CheckpointContractError, ValueError) as exc:
        details = getattr(exc, "details", None)
        suffix = f" details={details!r}" if details else ""
        print(f"multimodal SFT rejected: {type(exc).__name__}: {exc}{suffix}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
