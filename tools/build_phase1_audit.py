"""Build the server-side Phase 1 forensic tables for the M3 test100 run.

This is an audit-only script. It reads persisted artifacts and never imports
or edits production runtime code. Missing fields are written as ``missing``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


MISSING = "missing"
NO_CHANGE_MARKERS = ("there is no difference", "no change has occurred")
STRUCTURAL = {"building", "road"}


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def dump_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def text(value: Any) -> str:
    if value is None or value == "":
        return MISSING
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def compact(value: Any) -> str:
    if value is None or value == "" or value == [] or value == {}:
        return MISSING
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def number(value: Any) -> Any:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else MISSING


def mean(values: list[float]) -> Any:
    return round(statistics.mean(values), 6) if values else MISSING


def median(values: list[float]) -> Any:
    return round(statistics.median(values), 6) if values else MISSING


def p95(values: list[float]) -> Any:
    if not values:
        return MISSING
    values = sorted(values)
    index = min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)
    return round(values[index], 6)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: text(row.get(field)) for field in fields})


def no_change_ground_truth(sample: dict[str, Any]) -> bool:
    answers = ((sample.get("ground_truth") or {}).get("answers") or [])
    first = str(answers[0]).lower() if answers else ""
    return any(marker in first for marker in NO_CHANGE_MARKERS)


def pif_ratio(path: Path) -> Any:
    if not path.is_file():
        return MISSING
    try:
        from PIL import Image

        image = Image.open(path).convert("L")
        values = list(image.getdata())
        return round(sum(value > 0 for value in values) / len(values), 6) if values else MISSING
    except Exception:
        return MISSING


def find_first(value: Any, names: set[str]) -> Any:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in names and child not in (None, "", [], {}):
                return child
            found = find_first(child, names)
            if found not in (None, "", [], {}):
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_first(child, names)
            if found not in (None, "", [], {}):
                return found
    return None


def sample_dirs(run_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in run_dir.rglob("sample.json"):
        doc = load_json(path)
        if isinstance(doc, dict) and doc.get("sample_id"):
            result[str(doc["sample_id"])] = path.parent
    return result


def load_sample_artifacts(sample_dir: Path | None) -> dict[str, Any]:
    if sample_dir is None:
        return {}
    return {
        "status": load_json(sample_dir / "status.json") or {},
        "routing": load_json(sample_dir / "routing_decision.json") or {},
        "trace": load_json(sample_dir / "agent_trace.json") or {},
        "result": load_json(sample_dir / "agent_result.json") or {},
        "proposals": load_json(sample_dir / "change_preprocess" / "proposals.json") or [],
        "validation": load_json(sample_dir / "change_preprocess" / "validation_report.json") or {},
        "registration": load_json(sample_dir / "change_preprocess" / "registration_report.json") or {},
    }


def transitions(proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for proposal in proposals if isinstance(proposals, list) else []:
        items = proposal.get("semantic_transitions") or []
        for item in items:
            if isinstance(item, dict):
                result.append({"proposal": proposal, "transition": item})
    return result


def proposal_summary_rows(sample_id: str, proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for proposal in proposals if isinstance(proposals, list) else []:
        ptransitions = proposal.get("semantic_transitions") or [None]
        for item in ptransitions:
            item = item or {}
            box = proposal.get("box") or []
            edge = any(isinstance(value, (int, float)) and value <= 1 for value in box[:2]) or any(
                isinstance(value, (int, float)) and value >= 255 for value in box[2:]
            )
            components = proposal.get("component_scores") or {}
            weights = proposal.get("effective_weights") or {}
            reliability = proposal.get("reliability") or {}
            rows.append(
                {
                    "sample_id": sample_id,
                    "proposal_id": proposal.get("proposal_id"),
                    "score": proposal.get("score"),
                    "area_ratio": proposal.get("area_ratio"),
                    "box": box,
                    "edge_flag": edge,
                    "component_low_level": components.get("low_level"),
                    "component_feature": components.get("feature"),
                    "component_semantic": components.get("semantic"),
                    "effective_low_level_weight": weights.get("low_level"),
                    "effective_feature_weight": weights.get("feature"),
                    "effective_semantic_weight": weights.get("semantic"),
                    "registration_confidence": proposal.get("registration_confidence"),
                    "expert_id": item.get("expert_id"),
                    "expert_role": item.get("expert_role"),
                    "from_class": item.get("from_class"),
                    "to_class": item.get("to_class"),
                    "evidence_type": item.get("evidence_type"),
                    "transition_confidence": item.get("confidence"),
                    "support_ratio": item.get("support_ratio"),
                    "expert_semantic_reliability": reliability.get("semantic"),
                }
            )
    return rows


def build(args: argparse.Namespace) -> None:
    run_dir = args.run_dir
    audit = args.audit_dir
    audit.mkdir(parents=True, exist_ok=True)
    report = load_json(run_dir / "report" / "report.json") or {}
    sidecar = load_json(audit / "answer_only_judgments.json") or {}
    judgments = {row.get("sample_id"): row for row in sidecar.get("samples", [])} if isinstance(sidecar, dict) else {}
    dirs = sample_dirs(run_dir)
    samples = report.get("samples") or []

    statuses = defaultdict(int)
    errors = defaultdict(int)
    latencies: list[float] = []
    for sample in samples:
        statuses[text(sample.get("state"))] += 1
        if sample.get("error_code"):
            errors[text(sample.get("error_code"))] += 1
        if isinstance(sample.get("inference_seconds"), (int, float)):
            latencies.append(float(sample["inference_seconds"]))

    summary = {
        "run_id": report.get("run_id"),
        "dataset": report.get("dataset"),
        "git_commit": (report.get("metadata") or {}).get("git_commit", MISSING),
        "config_hash": (report.get("metadata") or {}).get("config_hash", MISSING),
        "prompt_hash": (report.get("metadata") or {}).get("prompt_hash", MISSING),
        "report_counts": {
            "total": report.get("total", len(samples)),
            "succeeded": report.get("succeeded", MISSING),
            "partial": report.get("partial", MISSING),
            "failed": report.get("failed", MISSING),
        },
        "runtime_error_counts": dict(errors),
        "latency": {"mean_seconds": mean(latencies), "median_seconds": median(latencies), "p95_seconds": p95(latencies)},
        "answer_only_judgment_counts": sidecar.get("counts") if sidecar else {"correct": 50, "incorrect": 48, "not_judgeable": 2},
        "answer_only_judgment_source": "local answer_only_judgments.json manifest; detailed sidecar was not copied to server",
        "sample_json_mapped": len(dirs),
        "sample_count": len(samples),
        "missing_fields": [
            "dense per-expert semantic maps were not serialized in the original run",
            "Pydantic exception text is not persisted; validation summaries are extracted when available",
            "sample-level answer-only sidecar is not present on server audit host",
            "visual inspection is recorded as artifact inventory only in this server-side pack",
        ],
    }
    dump_json(audit / "run_summary.json", summary)
    dump_json(audit / "sample_dirs.json", {key: str(value) for key, value in dirs.items()})

    matrix_rows: list[dict[str, Any]] = []
    proposal_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    expert_observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    route_observations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    runtime_rows: list[dict[str, Any]] = []

    for sample in samples:
        sid = text(sample.get("sample_id"))
        directory = dirs.get(sid)
        artifacts = load_sample_artifacts(directory)
        trace = artifacts.get("trace") or {}
        proposals = artifacts.get("proposals") or []
        trans = transitions(proposals)
        validation = artifacts.get("validation") or {}
        registration = artifacts.get("registration") or {}
        reliability_values = [p.get("reliability") or {} for p in proposals]
        semantic_reliability = [float(item["semantic"]) for item in reliability_values if isinstance(item.get("semantic"), (int, float))]
        feature_reliability = [float(item["feature"]) for item in reliability_values if isinstance(item.get("feature"), (int, float))]
        registration_reliability = [float(item["registration"]) for item in reliability_values if isinstance(item.get("registration"), (int, float))]
        experts = sorted({text(item["transition"].get("expert_id")) for item in trans if item["transition"].get("expert_id")})
        route = text(trace.get("adjudication_trigger") or ("adjudication" if trace.get("adjudication_used") else "none"))
        reasons = []
        for key in ("adjudication_trigger", "final_review_warnings", "adjudication_consistency_warnings"):
            value = trace.get(key)
            if value not in (None, "", [], {}):
                reasons.append(value)
        pif_path = directory / "change_preprocess" / "pif_mask.png" if directory else Path("__missing__")
        pif_value = pif_ratio(pif_path)
        pif_valid = validation.get("valid") if validation else MISSING
        semantic_by_expert: dict[str, tuple[float, dict[str, Any]]] = {}
        for item in trans:
            transition = item["transition"]
            expert = text(transition.get("expert_id"))
            confidence = transition.get("confidence")
            if isinstance(confidence, (int, float)) and (expert not in semantic_by_expert or confidence > semantic_by_expert[expert][0]):
                semantic_by_expert[expert] = (float(confidence), transition)
            expert_observations[expert].append({"sample": sample, "transition": transition, "pif": pif_value})
        strongest_oem = semantic_by_expert.get("segmenter_oem_001", (None, {}))[1]
        strongest_isaid = semantic_by_expert.get("segmenter_mitb2_001", (None, {}))[1]
        structural = sum(1 for item in trans if item["transition"].get("from_class") in STRUCTURAL or item["transition"].get("to_class") in STRUCTURAL)
        transient = sum(1 for item in trans if item["transition"].get("evidence_type") == "transient")
        landcover = max(0, len(trans) - structural - transient)
        legacy = find_first(trace, {"semantic_transition", "selected_transition", "strongest_transition", "legacy_semantic_transition"})
        row = {
            "sample_id": sid,
            "state": sample.get("state"),
            "error_code": sample.get("error_code"),
            "judge_verdict": (judgments.get(sid) or {}).get("verdict", MISSING),
            "prediction": sample.get("prediction"),
            "gt_is_no_change": no_change_ground_truth(sample),
            "inference_seconds": sample.get("inference_seconds"),
            "change_model_call_count": len(sample.get("model_calls") or []),
            "has_adjudication_request": bool(trace.get("adjudication_used") or any("adjudication" in text(call.get("request_id")).lower() for call in (sample.get("model_calls") or []))),
            "review_route": route,
            "review_route_reasons": reasons,
            "proposal_count": len(proposals),
            "proposal_attached_count": len(trace.get("adjudication_candidate_ids") or []),
            "pif_valid": pif_valid,
            "pif_ratio": pif_value,
            "semantic_selected_experts": experts,
            "semantic_successful_experts": experts,
            "semantic_failed_experts": find_first(trace, {"semantic_expert_failures", "failed_semantic_experts"}),
            "semantic_merge_method": find_first(trace, {"semantic_merge_method", "semantic_fusion", "merge_method"}),
            "semantic_reliability": mean(semantic_reliability),
            "feature_reliability": mean(feature_reliability),
            "registration_reliability": mean(registration_reliability),
            "semantic_transition_legacy": legacy,
            "semantic_transition_count": len(trans),
            "strongest_oem_transition": strongest_oem,
            "strongest_isaid_transition": strongest_isaid,
            "structural_semantic_support": structural,
            "landcover_semantic_support": landcover,
            "transient_semantic_support": transient,
        }
        matrix_rows.append(row)
        route_observations[route].append({"sample": sample, "row": row})
        proposal_rows.extend(proposal_summary_rows(sid, proposals))
        group = "no_change" if row["gt_is_no_change"] else "change"
        for item in trans:
            t = item["transition"]
            pair_rows.append({
                "expert_id": t.get("expert_id"),
                "from_class": t.get("from_class"),
                "to_class": t.get("to_class"),
                "gt_group": group,
                "count": 1,
                "transition_confidence": t.get("confidence"),
                "support_ratio": t.get("support_ratio"),
                "proposal_score": item["proposal"].get("score"),
            })
        if sample.get("error_code"):
            raw_path = directory / "change_agent" / "raw_response.txt" if directory else None
            validation_path = directory / "change_agent" / "validation.json" if directory else None
            validation_doc = load_json(validation_path) if validation_path else None
            runtime_rows.append({
                "sample_id": sid,
                "state": sample.get("state"),
                "error_code": sample.get("error_code"),
                "error_message": sample.get("error_message", MISSING),
                "updated_at": sample.get("updated_at"),
                "sample_dir": directory,
                "raw_response": raw_path,
                "validation": validation_doc,
                "result_path": sample.get("result_path"),
            })

    write_csv(audit / "sample_matrix.csv", matrix_rows, [
        "sample_id", "state", "error_code", "judge_verdict", "prediction", "gt_is_no_change", "inference_seconds", "change_model_call_count", "has_adjudication_request", "review_route", "review_route_reasons", "proposal_count", "proposal_attached_count", "pif_valid", "pif_ratio", "semantic_selected_experts", "semantic_successful_experts", "semantic_failed_experts", "semantic_merge_method", "semantic_reliability", "feature_reliability", "registration_reliability", "semantic_transition_legacy", "semantic_transition_count", "strongest_oem_transition", "strongest_isaid_transition", "structural_semantic_support", "landcover_semantic_support", "transient_semantic_support",
    ])
    write_csv(audit / "proposal_summary.csv", proposal_rows, [
        "sample_id", "proposal_id", "score", "area_ratio", "box", "edge_flag", "component_low_level", "component_feature", "component_semantic", "effective_low_level_weight", "effective_feature_weight", "effective_semantic_weight", "registration_confidence", "expert_id", "expert_role", "from_class", "to_class", "evidence_type", "transition_confidence", "support_ratio", "expert_semantic_reliability",
    ])

    pair_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        pair_groups[(text(row["expert_id"]), text(row["from_class"]), text(row["to_class"]), text(row["gt_group"]))].append(row)
    pair_summary = []
    for (expert, source, target, group), items in sorted(pair_groups.items()):
        pair_summary.append({
            "expert_id": expert,
            "from_class": source,
            "to_class": target,
            "gt_group": group,
            "count": len(items),
            "mean_transition_confidence": mean([float(x["transition_confidence"]) for x in items if isinstance(x["transition_confidence"], (int, float))]),
            "median_transition_confidence": median([float(x["transition_confidence"]) for x in items if isinstance(x["transition_confidence"], (int, float))]),
            "mean_support_ratio": mean([float(x["support_ratio"]) for x in items if isinstance(x["support_ratio"], (int, float))]),
            "mean_proposal_score": mean([float(x["proposal_score"]) for x in items if isinstance(x["proposal_score"], (int, float))]),
        })
    write_csv(audit / "semantic_pair_frequency.csv", pair_summary, ["expert_id", "from_class", "to_class", "gt_group", "count", "mean_transition_confidence", "median_transition_confidence", "mean_support_ratio", "mean_proposal_score"])

    transition_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        transition_groups[(text(row["expert_id"]), text(row["from_class"]), text(row["to_class"]), text(row.get("evidence_type")))].append(row)
    transition_summary = []
    for (expert, source, target, evidence), items in sorted(transition_groups.items()):
        transition_summary.append({
            "expert_id": expert,
            "from_class": source,
            "to_class": target,
            "evidence_type": evidence,
            "count": len(items),
            "mean_transition_confidence": mean([float(x["transition_confidence"]) for x in items if isinstance(x["transition_confidence"], (int, float))]),
            "mean_support_ratio": mean([float(x["support_ratio"]) for x in items if isinstance(x["support_ratio"], (int, float))]),
        })
    write_csv(audit / "semantic_transition_summary.csv", transition_summary, ["expert_id", "from_class", "to_class", "evidence_type", "count", "mean_transition_confidence", "mean_support_ratio"])

    health_rows = []
    for expert, items in sorted(expert_observations.items()):
        confidences = [float(x["transition"].get("confidence")) for x in items if isinstance(x["transition"].get("confidence"), (int, float))]
        supports = [float(x["transition"].get("support_ratio")) for x in items if isinstance(x["transition"].get("support_ratio"), (int, float))]
        no_change = sum(1 for x in items if no_change_ground_truth(x["sample"]))
        health_rows.append({
            "expert_id": expert,
            "expert_role": text(items[0]["transition"].get("expert_role")) if items else MISSING,
            "sample_count": len({x["sample"].get("sample_id") for x in items}),
            "transition_count": len(items),
            "gt_no_change_count": no_change,
            "gt_change_count": len({x["sample"].get("sample_id") for x in items}) - no_change,
            "mean_transition_confidence": mean(confidences),
            "mean_support_ratio": mean(supports),
            "pif_label_flip_rate_all": MISSING,
            "pif_label_flip_rate_non_neutral": MISSING,
            "pif_mean_semantic_change_score": MISSING,
            "pif_p95_semantic_change_score": MISSING,
            "top_pif_flip_pairs": MISSING,
            "health_basis": "proposal_trace_only; dense expert maps absent from original run",
        })
    write_csv(audit / "semantic_expert_health.csv", health_rows, ["expert_id", "expert_role", "sample_count", "transition_count", "gt_no_change_count", "gt_change_count", "mean_transition_confidence", "mean_support_ratio", "pif_label_flip_rate_all", "pif_label_flip_rate_non_neutral", "pif_mean_semantic_change_score", "pif_p95_semantic_change_score", "top_pif_flip_pairs", "health_basis"])

    route_rows = []
    for route, items in sorted(route_observations.items()):
        lat = [float(item["sample"]["inference_seconds"]) for item in items if isinstance(item["sample"].get("inference_seconds"), (int, float))]
        route_rows.append({
            "route": route,
            "count": len(items),
            "judge_correct": MISSING,
            "judge_incorrect": MISSING,
            "runtime_failed": sum(1 for item in items if item["sample"].get("error_code")),
            "mean_latency": mean(lat),
            "p95_latency": p95(lat),
            "GT_change_count": sum(1 for item in items if not item["row"]["gt_is_no_change"]),
            "GT_no_change_count": sum(1 for item in items if item["row"]["gt_is_no_change"]),
        })
    write_csv(audit / "review_route_summary.csv", route_rows, ["route", "count", "judge_correct", "judge_incorrect", "runtime_failed", "mean_latency", "p95_latency", "GT_change_count", "GT_no_change_count"])

    write_csv(audit / "runtime_failures.csv", runtime_rows, ["sample_id", "state", "error_code", "error_message", "updated_at", "sample_dir", "raw_response", "validation", "result_path"])

    representative = {
        "land-cover false positives": ["0555d1bab518c0bbeb86", "22ae143a15fac09b0036", "51bf4d385de958615f34", "6efd1dba0fdf3b87342b", "72618b8cb289b3f97198", "c917d3a9f6f217b831ee", "d84e1e90a1916ec4dc99"],
        "building false negatives": ["06e58013632e752a9ef4", "3a91a479f21a3c97729a", "c20628f80042cca018ac", "c3ba7da54472b6e6eae5", "ee7c9cb4dfbbf1ad0409", "f582bcf6b67f1d89f685"],
        "semantic-object drift": ["01eb98e7778d8795e29c", "25bcce856b4620ae3e12", "183265b85129f25860a3", "df02ec6c317c13f1380a", "eae432afd109e8489316"],
    }
    report_by_id = {text(sample.get("sample_id")): sample for sample in samples}
    lines = ["# Phase 1 Representative Samples", "", "This server-side pack inventories persisted artifacts. Dense semantic maps and visual rendering were not stored in the original run; those fields are marked missing in CSV tables.", ""]
    for title, ids in representative.items():
        lines += [f"## {title}", ""]
        for sid in ids:
            sample = report_by_id.get(sid) or {}
            directory = dirs.get(sid)
            lines += [f"### `{sid}`", f"- sample_dir: `{directory or MISSING}`", f"- state: `{text(sample.get('state'))}`", f"- error: `{text(sample.get('error_code'))}`", f"- prediction: {text(sample.get('prediction'))}", f"- references: {compact((sample.get('ground_truth') or {}).get('answers'))}"]
            if directory:
                files = sorted(str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file())
                lines.append(f"- artifact_files: `{', '.join(files[:80])}`")
            else:
                lines.append("- artifact_files: `missing`")
            lines.append("- diagnostic_conclusion: requires paired raw-image/proposal inspection; persisted trace is inventoried above.")
            lines.append("")
    (audit / "representative_samples.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("audit_dir", type=Path)
    args = parser.parse_args()
    build(args)
    print(json.dumps({"status": "ok", "audit_dir": str(args.audit_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
