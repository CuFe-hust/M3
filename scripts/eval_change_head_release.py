"""Apply configured ChangeHead release gates to precomputed metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from training.change_head.release_gate import evaluate_release_gates


def _numeric_deltas(baseline: dict[str, object], candidate: dict[str, object]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key in sorted(set(baseline) | set(candidate)):
        before = baseline.get(key)
        after = candidate.get(key)
        if isinstance(before, (int, float)) and isinstance(after, (int, float)):
            deltas[key] = float(after) - float(before)
    return deltas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--assist", type=Path, required=True)
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shadow-parity", action="store_true")
    parser.add_argument("--hard-case-comparison", type=Path, default=None)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    assist = json.loads(args.assist.read_text(encoding="utf-8"))
    comparison = (
        json.loads(args.hard_case_comparison.read_text(encoding="utf-8"))
        if args.hard_case_comparison is not None
        else {}
    )
    if comparison:
        if not isinstance(comparison, list):
            raise SystemExit("--hard-case-comparison must contain a JSON list")
        rescued = sum(
            bool(item.get("baseline_fail")) and bool(item.get("candidate_pass"))
            for item in comparison
        )
        regressed = sum(
            bool(item.get("baseline_pass")) and bool(item.get("candidate_fail"))
            for item in comparison
        )
        assist["hard_case_total"] = len(comparison)
        assist["hard_case_rescued"] = rescued
        assist["hard_case_regressed"] = regressed
    gate_config = yaml.safe_load(args.gate_config.read_text(encoding="utf-8"))
    result = evaluate_release_gates(
        shadow_parity=args.shadow_parity,
        baseline=baseline,
        assist=assist,
        residual_hard_cases_rescued=int(assist.get("hard_case_rescued", 0)),
        residual_hard_cases_regressed=int(assist.get("hard_case_regressed", 0)),
        config=gate_config,
    )
    result["baseline"] = baseline
    result["candidate"] = assist
    result["deltas"] = _numeric_deltas(baseline, assist)
    result["config"] = gate_config
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
