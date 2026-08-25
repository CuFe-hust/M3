#!/usr/bin/env python3
"""Audit exact untruncated token lengths for a formal ChangeAgent SFT corpus."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.multimodal_sft.data import profile_for
from training.multimodal_sft.image_roots import ImageRootRegistry
from training.multimodal_sft.registry import default_registry
from training.multimodal_sft.token_audit import audit_change_agent_tokens


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit ChangeAgent SFT token lengths without loading model weights")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-adapter", default="qwen3_5")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--image-root", action="append", default=[])
    parser.add_argument("--prompt-ref")
    parser.add_argument("--prompt-file")
    parser.add_argument("--threshold", action="append", type=int, default=[])
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = profile_for(
        "change_agent",
        data_manifest=args.data_manifest,
        prompt_ref=args.prompt_ref,
        prompt_file=args.prompt_file,
    )
    image_roots = ImageRootRegistry.from_specs(args.image_root)
    adapter, probe = default_registry().resolve(
        args.model_id,
        model_adapter=args.model_adapter,
        local_files_only=args.local_files_only,
    )
    if adapter.name not in {"qwen3_5", "qwen3_vl"} or not probe.passed:
        raise SystemExit("token audit requires a passing Qwen multimodal adapter probe")
    processor = adapter.load_processor(args.model_id, local_files_only=args.local_files_only)
    train_rows = profile.read(args.train_file)
    validation_rows = profile.read(args.validation_manifest)
    result = audit_change_agent_tokens(
        profile=profile,
        processor=processor,
        image_roots=image_roots,
        split_episodes={"train": train_rows, "validation": validation_rows},
        thresholds=tuple(args.threshold or (4096, 8192)),
        progress_every=args.progress_every,
        progress=lambda message: print(message, file=sys.stderr, flush=True),
    )
    result.update({
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "adapter": adapter.name,
        "adapter_probe": probe.as_dict(),
        "processor_identity": adapter.processor_identity(processor),
        "data_identity": profile.identity_contract(image_roots),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["recommended_max_seq_length"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
