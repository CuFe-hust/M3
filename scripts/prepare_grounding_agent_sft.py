#!/usr/bin/env python3
"""Convert grounding targets plus simulated planner/detector outputs to SFT JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.schema import AgentResult
from training.multimodal_sft.profiles.grounding import GROUNDING_TARGET_SCHEMA


class PreparationError(ValueError):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PreparationError(f"file does not exist: {path}")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PreparationError(f"invalid JSON at {path}:{index + 1}") from exc
        if not isinstance(row, dict):
            raise PreparationError(f"row is not an object at {path}:{index + 1}")
        rows.append(row)
    return rows


def index_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        episode_id = str(row.get("episode_id") or "")
        if not episode_id or episode_id in result:
            raise PreparationError(f"missing or duplicate episode_id in {path}")
        result[episode_id] = row
    return result


def planner_output(row: Mapping[str, Any], episode_id: str) -> dict[str, Any]:
    direct = row.get("planner_output")
    if isinstance(direct, Mapping):
        return dict(direct)
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise PreparationError(f"planner output missing for {episode_id}")
    for message in reversed(messages):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, list):
            text = "".join(str(item.get("text", "")) for item in content if isinstance(item, Mapping))
        else:
            text = str(content or "")
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PreparationError(f"planner assistant output is not JSON for {episode_id}") from exc
        if not isinstance(value, dict):
            raise PreparationError(f"planner output is not an object for {episode_id}")
        return value
    raise PreparationError(f"planner assistant message missing for {episode_id}")


def evidence_items(row: Mapping[str, Any], episode_id: str) -> list[dict[str, Any]]:
    value = row.get("evidence_items", [])
    if not isinstance(value, list):
        raise PreparationError(f"evidence_items is not a list for {episode_id}")
    cleaned: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise PreparationError(f"invalid evidence item for {episode_id}")
        cleaned.append({"label": item.get("label"), "box": item.get("box")})
    try:
        validated = AgentResult(agent_name="grounding_agent", answer="evidence", evidence_items=cleaned, status="completed")
    except Exception as exc:  # noqa: BLE001 - fail closed at conversion boundary
        raise PreparationError(f"invalid evidence item for {episode_id}") from exc
    return [{"label": item.label, "box": item.box} for item in validated.evidence_items]


def convert(args: argparse.Namespace) -> int:
    planner = index_rows(Path(args.planner_jsonl) if args.planner_jsonl else None)
    evidence = index_rows(Path(args.evidence_jsonl) if args.evidence_jsonl else None)
    output: list[dict[str, Any]] = []
    for row in read_jsonl(Path(args.input_jsonl)):
        episode_id = str(row.get("episode_id") or "")
        if not episode_id:
            raise PreparationError("input row is missing episode_id")
        planner_row = planner.get(episode_id, row)
        if args.require_planner and episode_id not in planner:
            raise PreparationError(f"planner row missing for {episode_id}")
        evidence_row = evidence.get(episode_id, {})
        if args.require_evidence and episode_id not in evidence:
            raise PreparationError(f"evidence row missing for {episode_id}")
        answer = row.get("answer_text", row.get("answer"))
        if not isinstance(answer, str) or not answer.strip():
            raise PreparationError(f"dataset answer is missing for {episode_id}")
        items = evidence_items(evidence_row, episode_id)
        target_result = {
            "agent_name": "grounding_agent",
            "answer": answer,
            "evidence_items": items,
            "status": "completed",
        }
        AgentResult.model_validate(target_result)
        relative_image = str(row.get("image") or "")
        if args.image_prefix:
            relative_image = f"{args.image_prefix.rstrip('/')}/{Path(relative_image).name}"
        output.append({
            "episode_id": episode_id,
            "split": str(row.get("split", args.split)),
            "task_profile": "grounding",
            "image_source": args.image_source,
            "image": relative_image,
            "question": str(row.get("question") or row.get("prompt") or ""),
            "planner_output": planner_output(planner_row, episode_id),
            "evidence_items": items,
            "target": {"response_schema": GROUNDING_TARGET_SCHEMA, "result": target_result},
            "metadata": {
                "source_dataset": row.get("dataset"),
                "source_split": row.get("split"),
                "source_coordinate_frame": row.get("coordinate_frame"),
                "source_box_999": row.get("box_999"),
                "source_answer_text": answer,
                "construction": {
                    "planner": bool(planner_row),
                    "detector_evidence": bool(evidence_row),
                    "target_contract": GROUNDING_TARGET_SCHEMA,
                },
            },
        })
    destination = Path(args.output_jsonl)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in output),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(destination), "rows": len(output), "target_schema": GROUNDING_TARGET_SCHEMA}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--image-source", required=True)
    parser.add_argument("--planner-jsonl")
    parser.add_argument("--evidence-jsonl")
    parser.add_argument("--require-planner", action="store_true")
    parser.add_argument("--require-evidence", action="store_true")
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-prefix", help="Rewrite image paths to prefix/basename, e.g. Images_train for VRSBench")
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(convert(build_parser().parse_args()))
    except PreparationError as exc:
        print(f"grounding preparation rejected: {exc}", file=sys.stderr)
        raise SystemExit(2)
