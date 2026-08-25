#!/usr/bin/env python3
"""Build a deterministic rendered ChangeAgent fixture for generic export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.multimodal_sft.change_export_fixture import (
    ChangeExportFixtureError,
    build_change_export_fixture,
)
from training.multimodal_sft.image_roots import ImageRootRegistry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a rendered Change export fixture.")
    parser.add_argument("--validation", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", action="append", default=[])
    parser.add_argument("--prompt-ref")
    parser.add_argument("--prompt-file")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        fixture = build_change_export_fixture(
            validation=args.validation,
            manifest=args.manifest,
            image_roots=ImageRootRegistry.from_specs(args.image_root),
            prompt_ref=args.prompt_ref,
            prompt_file=args.prompt_file,
            output=args.output,
        )
    except ChangeExportFixtureError as exc:
        print(f"change export fixture rejected: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": args.output, "metadata": fixture["metadata"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
