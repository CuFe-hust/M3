#!/usr/bin/env python3
"""Evaluate saved Grounding Agent trainable states on a validation JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.multimodal_sft.adapters import _hf
from training.multimodal_sft.data import profile_for
from training.multimodal_sft.image_roots import ImageRootRegistry
from training.multimodal_sft.parameter_plan import ParameterPlan
from training.multimodal_sft.registry import default_registry
from training.multimodal_sft.trainer_core import GenericTrainerCore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--image-root", action="append", default=[])
    parser.add_argument("--output")
    parser.add_argument("--max-seq-length", type=int, default=32768)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    return parser


def evaluate_checkpoint(args: argparse.Namespace, checkpoint: Path, rows: list[dict]) -> dict:
    manifest = json.loads((checkpoint / "training_manifest.json").read_text(encoding="utf-8"))
    plan_payload = json.loads((checkpoint / "parameter_plan.json").read_text(encoding="utf-8"))
    plan = ParameterPlan(**plan_payload)
    adapter, _probe = default_registry().resolve(args.model_id, local_files_only=True)
    model, processor, _loaded_probe = adapter.load(
        args.model_id,
        dtype=args.dtype,
        device="cpu",
        local_files_only=True,
    )
    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PEFT is required for checkpoint evaluation") from exc
    peft_model = PeftModel.from_pretrained(model, str(checkpoint / "adapter"), is_trainable=False)
    _hf.restore_full_train_state_for_export(
        model=peft_model,
        checkpoint_dir=checkpoint,
        parameter_plan=plan,
    )
    validate_state = getattr(adapter, "validate_checkpoint_state", None)
    if callable(validate_state):
        validate_state(checkpoint, plan)
    peft_model.to(args.device)
    peft_model.eval()
    profile = profile_for("grounding")
    trainer = GenericTrainerCore(adapter=adapter, data_profile=profile)
    roots = ImageRootRegistry.from_specs(args.image_root)
    loss = trainer.evaluate(
        model=peft_model,
        processor=processor,
        episodes=rows,
        image_roots=roots,
        epoch=0,
        seed=1234,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
    )
    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_type": manifest.get("checkpoint_type"),
        "examples": len(rows),
        "loss": loss,
        "adapter": manifest.get("adapter_name"),
        "tuning_policy": manifest.get("tuning_policy"),
    }
    del peft_model, model, processor
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass
    return result


def main() -> int:
    args = build_parser().parse_args()
    profile = profile_for("grounding")
    rows = profile.read(args.validation_file)
    if args.max_eval_samples is not None:
        rows = rows[: args.max_eval_samples]
    if not rows:
        raise SystemExit("validation file contains no rows")
    results = [evaluate_checkpoint(args, Path(item), rows) for item in args.checkpoint]
    payload = {"data_profile": "grounding", "results": results}
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
