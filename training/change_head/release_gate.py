"""Machine-readable release gates for ChangeHead promotion."""

from __future__ import annotations

from typing import Any


def evaluate_release_gates(
    *,
    shadow_parity: bool,
    baseline: dict[str, Any],
    assist: dict[str, Any],
    residual_hard_cases_rescued: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    critical = config.get("critical_no_change", {})
    normal = config.get("normal_changed", {})
    residual = config.get("residual_hard_cases", {})
    gates = {
        "shadow_parity": bool(shadow_parity),
        "critical_no_change": (
            int(assist.get("critical_new_fp_samples", 0)) <= int(critical.get("max_new_false_positive_samples", 0))
            and int(assist.get("critical_new_fp_components", 0)) <= int(critical.get("max_new_false_positive_components", 0))
        ),
        "normal_changed": (
            float(baseline.get("normal_proposal_recall", 0.0))
            - float(assist.get("normal_proposal_recall", 0.0))
            <= float(normal.get("max_proposal_recall_drop", 0.0))
            and float(baseline.get("normal_proposal_f1", 0.0))
            - float(assist.get("normal_proposal_f1", 0.0))
            <= float(normal.get("max_proposal_f1_drop", 0.0))
        ),
        "residual_hard_cases": (
            not residual.get("require_net_improvement", True)
            or residual_hard_cases_rescued >= int(residual.get("min_additional_rescued_samples", 1))
        ),
    }
    gates["broad_validation"] = (
        float(baseline.get("proposal_f1", 0.0))
        - float(assist.get("proposal_f1", 0.0))
        <= float(config.get("broad_validation", {}).get("max_proposal_f1_drop", 0.01))
    )
    return {"passed": all(gates.values()), "gates": gates}

