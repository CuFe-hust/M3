#!/usr/bin/env python3
"""Build the formal offline ChangeAgent multi-source corpus. / 构建正式离线多源语料。"""

from __future__ import annotations

import argparse
import json
import sys

from training.multimodal_sft.change_corpus import ChangeCorpusBuildError, build_corpus, inspect_source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the formal offline ChangeAgent multi-source corpus.")
    parser.add_argument("--source-spec")
    parser.add_argument("--output-dir")
    parser.add_argument("--prompt-ref")
    parser.add_argument("--inspect-source")
    args = parser.parse_args(argv)
    try:
        if args.inspect_source:
            print(json.dumps(inspect_source(args.inspect_source), ensure_ascii=False, indent=2))
            return 0
        if not args.source_spec or not args.output_dir or not args.prompt_ref:
            parser.error("--source-spec, --output-dir and --prompt-ref are required unless --inspect-source is used")
        manifest = build_corpus(args.source_spec, args.output_dir, args.prompt_ref)
        print(json.dumps({"output_dir": args.output_dir, "counts": manifest["counts"]}, ensure_ascii=False, indent=2))
        return 0
    except ChangeCorpusBuildError as exc:
        print(f"error: {exc.code}: {exc.detail}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
