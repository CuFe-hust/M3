"""Dependency-light metric entry point for cached ChangeHead outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from training.change_head.evaluator import evaluate_probability_maps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probability = np.load(args.probabilities, allow_pickle=False)
    target = np.load(args.targets, allow_pickle=False)
    valid = np.load(args.valid, allow_pickle=False)
    metrics = evaluate_probability_maps([probability], [target], [valid])
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

