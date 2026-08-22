"""Machine-readable release gates for ChangeHead promotion."""

from __future__ import annotations

from typing import Any, Mapping


def _drop(baseline: Mapping[str, Any], candidate: Mapping[str, Any], key: str) -> float:
    return float(baseline.get(key, 0.0)) - float(candidate.get(key, 0.0))


def _hard_case_counts(
    *,
    assist: Mapping[str, Any],
    residual_hard_cases_rescued: int,
    residual_hard_cases_regressed: int,
) -> tuple[int, int]:
    rescued = int(assist.get("hard_case_rescued", residual_hard_cases_rescued))
    regressed = int(assist.get("hard_case_regressed", residual_hard_cases_regressed))
    return max(0, rescued), max(0, regressed)


def evaluate_release_gates(
    *,
    shadow_parity: bool,
    baseline: dict[str, Any],
    assist: dict[str, Any],
    residual_hard_cases_rescued: int = 0,
    residual_hard_cases_regressed: int = 0,
    config: dict[str, Any],
) -> dict[str, Any]:
    critical = config.get("critical_no_change", {})
    normal = config.get("normal_changed", {})
    residual = config.get("residual_hard_cases", {})
    broad = config.get("broad_validation", {})
    building = config.get("building_edge", {})
    baseline_edge = baseline.get("building_edge_proposal_recall")
    assist_edge = assist.get("building_edge_proposal_recall")
    edge_available = baseline_edge is not None and assist_edge is not None
    edge_allowed_missing = bool(building.get("allow_missing_subset", False))
    edge_pass = (
        float(baseline_edge) - float(assist_edge)
        <= float(building.get("max_proposal_recall_drop", 0.0))
        if edge_available
        else edge_allowed_missing
    )
    rescued, regressed = _hard_case_counts(
        assist=assist,
        residual_hard_cases_rescued=residual_hard_cases_rescued,
        residual_hard_cases_regressed=residual_hard_cases_regressed,
    )
    net = rescued - regressed
    nochange_fp_increase = (
        float(assist.get("scene_nochange_fp_rate", 0.0))
        - float(baseline.get("scene_nochange_fp_rate", 0.0))
    )
    gates = {
        "shadow_parity": bool(shadow_parity),
        "critical_no_change": (
            int(assist.get("critical_new_fp_samples", 0))
            <= int(critical.get("max_new_false_positive_samples", 0))
            and int(assist.get("critical_new_fp_components", 0))
            <= int(critical.get("max_new_false_positive_components", 0))
        ),
        "normal_changed": (
            _drop(baseline, assist, "normal_proposal_recall")
            <= float(normal.get("max_proposal_recall_drop", 0.0))
            and _drop(baseline, assist, "normal_proposal_f1")
            <= float(normal.get("max_proposal_f1_drop", 0.0))
        ),
        "residual_hard_cases": (
            not residual.get("require_net_improvement", True)
            or net >= int(residual.get("min_net_improvement", residual.get("min_additional_rescued_samples", 1)))
        ),
        "building_edge": edge_pass,
        "broad_proposal_f1": _drop(baseline, assist, "proposal_f1")
        <= float(broad.get("max_proposal_f1_drop", 0.01)),
        "broad_pixel_f1": _drop(baseline, assist, "pixel_f1")
        <= float(broad.get("max_pixel_f1_drop", 0.01)),
        "nochange_scene_rate": nochange_fp_increase
        <= float(critical.get("max_scene_fp_rate_increase", 0.0)),
    }
    gates["broad_validation"] = bool(
        gates["broad_proposal_f1"] and gates["broad_pixel_f1"]
    )
    details = {
        "building_edge": {
            "status": "available" if edge_available else "skipped",
            "pass": edge_pass,
            "baseline_recall": baseline_edge,
            "candidate_recall": assist_edge,
            "drop": None if not edge_available else float(baseline_edge) - float(assist_edge),
        },
        "hard_cases": {
            "total": int(assist.get("hard_case_total", 0)),
            "rescued": rescued,
            "regressed": regressed,
            "net_improvement": net,
        },
        "nochange": {
            "baseline_rate": float(baseline.get("scene_nochange_fp_rate", 0.0)),
            "candidate_rate": float(assist.get("scene_nochange_fp_rate", 0.0)),
            "increase": nochange_fp_increase,
        },
    }
    passed = all(bool(value) for value in gates.values())
    return {
        "passed": passed,
        "gates": gates,
        "gate_details": details,
        "release": {
            "eligible_for_shadow": bool(gates["shadow_parity"]),
            "eligible_for_assist": passed,
        },
    }
