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
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-adapter", default="auto")
    parser.add_argument("--data-profile", default="phase2", choices=("phase2", "change_agent"))
    parser.add_argument("--train-file")
    parser.add_argument("--output-dir", default="outputs/multimodal_sft")
    parser.add_argument("--tuning-policy", default="lora_plus_projector", choices=("lora_only", "projector_only", "lora_plus_projector", "full_language_lora"))
    parser.add_argument("--dtype", default="auto", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan-only", action="store_true", help="Probe adapter only; never load model weights or train")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        profile = profile_for(args.data_profile)
        registry = default_registry()
        adapter, probe = registry.resolve(
            args.model_id,
            model_adapter=args.model_adapter,
            local_files_only=args.local_files_only,
        )
        result = {
            "adapter": adapter.name,
            "probe": probe.as_dict(),
            "data_profile": profile.name,
            "tuning_policy": TuningPolicy.from_name(args.tuning_policy).as_dict(),
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
        trainer = GenericTrainerCore(adapter=adapter, data_profile=profile)
        training_result = trainer.fit(
            model=model,
            processor=processor,
            episodes=episodes,
            config=TrainingConfig(output_dir=args.output_dir),
            policy=TuningPolicy.from_name(args.tuning_policy),
            probe=loaded_probe,
            model_identity=loaded_probe.identity.as_dict(),
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
