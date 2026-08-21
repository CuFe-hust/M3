"""Apply configured ChangeHead release gates to precomputed metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from training.change_head.release_gate import evaluate_release_gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--assist", type=Path, required=True)
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shadow-parity", action="store_true")
    args = parser.parse_args()
    result = evaluate_release_gates(
        shadow_parity=args.shadow_parity,
        baseline=json.loads(args.baseline.read_text(encoding="utf-8")),
        assist=json.loads(args.assist.read_text(encoding="utf-8")),
        residual_hard_cases_rescued=0,
        config=yaml.safe_load(args.gate_config.read_text(encoding="utf-8")),
    )
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

