#!/usr/bin/env python3
"""Generic multimodal SFT checkpoint exporter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.multimodal_sft.exporter import ExportContractError, GenericExporter
from training.multimodal_sft.registry import UnsupportedModelAdapter, default_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a multimodal SFT checkpoint through its adapter")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-adapter", default="auto")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-forward", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        adapter, _probe = default_registry().resolve(
            args.model_id,
            model_adapter=args.model_adapter,
            local_files_only=args.local_files_only,
        )
        result = GenericExporter(adapter).export(
            model_id=args.model_id,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            local_files_only=args.local_files_only,
            verify_forward=args.verify_forward,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        return 0
    except (UnsupportedModelAdapter, ExportContractError, ValueError) as exc:
        details = getattr(exc, "details", None)
        suffix = f" details={details!r}" if details else ""
        print(f"multimodal SFT export rejected: {type(exc).__name__}: {exc}{suffix}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
