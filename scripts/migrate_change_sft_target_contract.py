#!/usr/bin/env python3
"""Create a safe v2 reference mirror; never overwrite old corpus. / 创建安全 v2 参考镜像。"""

from __future__ import annotations

import argparse
import json
import sys

from training.multimodal_sft.change_target_migration import ChangeTargetMigrationError, migrate_reference_corpus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate a legacy Change SFT corpus into a non-authoritative v2 reference.")
    parser.add_argument("--old-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(migrate_reference_corpus(old_dir=args.old_dir, output_dir=args.output_dir), ensure_ascii=False, indent=2))
        return 0
    except ChangeTargetMigrationError as exc:
        print(f"error: {exc.code}: {exc.detail}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
