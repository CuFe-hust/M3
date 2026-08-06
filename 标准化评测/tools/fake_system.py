"""Deterministic external-system stand-in used only by evaluation fixtures."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--behavior",
        choices=("ok", "missing", "duplicate", "malformed", "error", "nonzero", "timeout"),
        default="ok",
    )
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    args = parser.parse_args(argv)

    if args.behavior == "timeout":
        time.sleep(max(args.sleep_seconds, 0.0))
        return 0

    rows = _read_requests(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.behavior == "missing":
        args.output.write_text("", encoding="utf-8")
    elif args.behavior == "malformed":
        args.output.write_text("not-json\n", encoding="utf-8")
    else:
        predictions = list(_predictions(rows, args.behavior))
        if args.behavior == "duplicate":
            predictions = [prediction for prediction in predictions for _ in range(2)]
        _write_rows(args.output, predictions)

    return 17 if args.behavior == "nonzero" else 0


def _read_requests(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _predictions(rows: Iterable[dict[str, Any]], behavior: str) -> Iterable[dict[str, Any]]:
    for request in rows:
        sample_id = request.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("fixture request requires a nonempty sample_id")
        if behavior == "error":
            yield {
                "sample_id": sample_id,
                "status": "error",
                "error_code": "inference_error",
                "error": "fixture error",
                "latency_ms": 1.0,
            }
            continue
        prediction = request.get("fixture_prediction")
        if isinstance(prediction, str) and prediction:
            yield {
                "sample_id": sample_id,
                "status": "ok",
                "prediction": prediction,
                "raw_output": "fixture_prediction",
                "latency_ms": 1.0,
            }
            continue
        if isinstance(prediction, dict) and isinstance(prediction.get("boxes"), list) and prediction["boxes"]:
            yield {
                "sample_id": sample_id,
                "status": "ok",
                "boxes": prediction["boxes"],
                "raw_output": "fixture_prediction",
                "latency_ms": 1.0,
            }
            continue
        if not isinstance(prediction, (str, dict)):
            raise ValueError("fake_system only accepts explicitly marked fixture_prediction values")
        raise ValueError("fake_system fixture_prediction must contain text or boxes")


def _write_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"fake_system: {error}", file=sys.stderr)
        raise SystemExit(2) from error
