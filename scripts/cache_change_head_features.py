"""Entry point for the deterministic training feature-cache workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.change_head.feature_cache import FeatureCache
from training.change_head.schema import load_training_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--pipeline-fingerprint", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    records = load_training_records(args.manifest, allow_empty_changed=True)
    cache = FeatureCache(args.cache_root)
    index = args.cache_root / "index.jsonl"
    with index.open("a", encoding="utf-8") as output:
        for record in records[: args.limit]:
            t1 = (args.data_root / record.t1_path).resolve()
            t2 = (args.data_root / record.t2_path).resolve()
            key = __import__("training.change_head.feature_cache", fromlist=["build_feature_cache_key"]).build_feature_cache_key(
                sample_id=record.sample_id,
                t1_path=t1,
                t2_path=t2,
                pipeline_fingerprint=args.pipeline_fingerprint,
                experts=[],
            )
            cache.write(key, {"target_mask": __import__("numpy").zeros((1, 1), dtype="uint8")}, {
                "sample_id": record.sample_id,
                "split": record.split,
                "pipeline_fingerprint": args.pipeline_fingerprint,
            })
            output.write(json.dumps({"sample_id": record.sample_id, "cache_key": key}) + "\n")


if __name__ == "__main__":
    main()

