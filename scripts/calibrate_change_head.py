"""Fit a conservative validation-only ChangeHead calibration file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    best_temperature = 1.0
    best_loss = float("inf")
    for temperature in np.linspace(0.25, 4.0, 76):
        probability = 1.0 / (1.0 + np.exp(-logits / temperature))
        loss = -np.mean(
            targets * np.log(np.clip(probability, 1e-6, 1.0))
            + (1.0 - targets) * np.log(np.clip(1.0 - probability, 1e-6, 1.0))
        )
        if loss < best_loss:
            best_loss = float(loss)
            best_temperature = float(temperature)
    return best_temperature


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logits", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logits = np.load(args.logits)
    targets = np.load(args.targets).astype(np.float32)
    temperature = fit_temperature(logits.astype(np.float32), targets)
    probability = 1.0 / (1.0 + np.exp(-logits / temperature))
    threshold = 0.95
    reliability = float(max(0.5, min(1.0, 1.0 - abs(float(np.mean(probability - targets))))))
    args.output.write_text(json.dumps({
        "schema_version": 1,
        "temperature": temperature,
        "rescue_probability_threshold": threshold,
        "rescue_min_component_area_ratio": 0.0005,
        "validation_reliability": reliability,
        "optional_expert_missing_reliability_factor": 0.90,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

