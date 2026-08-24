#!/usr/bin/env python3
"""Prepare canonical, ordered ChangeAgent natural-language SFT episodes.

The adapters are intentionally read-only and offline.  They do not download
datasets, synthesize masks/proposals, or start training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from agents.change.schema import CANONICAL_NO_CHANGE, ChangeInitialResult


CHANGE_SFT_SCHEMA_VERSION = 1


REJECTION_CODES = {
    "missing_t1", "missing_t2", "unsafe_image_path", "unknown_task", "missing_question",
    "missing_answer", "invalid_role_order", "invalid_target_schema", "excluded_parent_sample",
    "context_dependent_multiturn", "duplicate_episode_id", "split_leakage",
}


def _validate_output_episode(episode: dict[str, Any]) -> None:
    """Keep preparation import-safe: validate the pure JSON target locally."""
    if episode.get("task") not in {"change_caption", "change_qa"}:
        raise ValueError("unknown_task")
    if episode["task"] == "change_qa" and not str(episode.get("question") or "").strip():
        raise ValueError("missing_question")
    if tuple(image.get("role") for image in episode.get("images", [])[:2]) != ("raw_full_t1", "raw_full_t2"):
        raise ValueError("invalid_role_order")
    if episode.get("target", {}).get("response_schema") != "ChangeInitialResult":
        raise ValueError("invalid_target_schema")
    ChangeInitialResult.model_validate(episode["target"]["result"])


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip() or "\\" in value or "\x00" in value:
        return None
    if value.startswith("/") or value.startswith("//") or len(value) >= 2 and value[1] == ":":
        return None
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return None
    return value


def _read_records(source: Path) -> list[dict[str, Any]]:
    raw = source.read_text(encoding="utf-8")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(parsed, dict):
        for key in ("images", "records", "data", "samples", "annotations"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
    if not isinstance(parsed, list):
        raise ValueError("source must be a JSON array/object containing records, or JSONL")
    return [item for item in parsed if isinstance(item, dict)]


def _value(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _pair_paths(record: dict[str, Any], split: str) -> tuple[str | None, str | None, bool, bool]:
    t1 = _value(record, "t1", "image_t1", "before", "image_a", "A")
    t2 = _value(record, "t2", "image_t2", "after", "image_b", "B")
    images = record.get("images")
    if isinstance(images, list) and len(images) >= 2:
        t1 = t1 or (images[0].get("path") if isinstance(images[0], dict) else images[0])
        t2 = t2 or (images[1].get("path") if isinstance(images[1], dict) else images[1])
    filename = _value(record, "filename", "file_name", "image")
    if isinstance(filename, str):
        t1 = t1 or f"images/{split}/A/{filename}"
        t2 = t2 or f"images/{split}/B/{filename}"
    safe_t1, safe_t2 = _safe_relative_path(t1), _safe_relative_path(t2)
    return safe_t1, safe_t2, t1 is not None and safe_t1 is None, t2 is not None and safe_t2 is None


def _answer_result(answer: str) -> dict[str, Any]:
    result = {
        "agent_name": "change_agent", "answer": answer, "boxes": [], "evidence": [],
        "evidence_items": [], "geometry": {}, "status": "completed",
    }
    return ChangeInitialResult.model_validate(result).model_dump(mode="json")


def _episode(
    *, episode_id: str, parent_id: str, dataset: str, split: str, task: str,
    question: str, t1: str, t2: str, answer: str, image_source: str, provenance: dict[str, Any],
) -> dict[str, Any]:
    images = [
        {"image_source": image_source, "path": t1, "role": "raw_full_t1"},
        {"image_source": image_source, "path": t2, "role": "raw_full_t2"},
    ]
    payload = {
        "decision_stage": "initial", "question": question, "task": task,
        "coordinate_frame": "normalized_0_999_top_left", "input_mode": "raw_only",
        "temporal_roles": ["t1", "t2"],
        "image_manifest": [{"index": "0", "role": "raw_full_t1"}, {"index": "1", "role": "raw_full_t2"}],
    }
    return {
        "schema_version": CHANGE_SFT_SCHEMA_VERSION, "episode_id": episode_id,
        "parent_sample_id": parent_id, "dataset": dataset, "split": split, "task": task,
        "input_contract": "semantic_pair_v1", "question": question, "images": images,
        "request_payload": payload,
        "target": {"response_schema": "ChangeInitialResult", "result": _answer_result(answer)},
        "augmentation_policy": {"temporal_geometry": "locked_identity", "photometric": "disabled"},
        "provenance": provenance,
    }


def _group_validation(parent_id: str, seed: str, ratio: int = 10) -> bool:
    return int(hashlib.sha256(f"{seed}|{parent_id}".encode("utf-8")).hexdigest()[:8], 16) % ratio == 0


def _levir_candidates(records: Iterable[dict[str, Any]]) -> Iterable[tuple[dict[str, Any] | None, str | None]]:
    for index, record in enumerate(records):
        source_split = str(_value(record, "split", "filepath") or "train").lower()
        if source_split in {"val", "validation"}:
            split = "validation"
        elif source_split == "train":
            split = "train"
        else:
            continue  # test is never SFT.
        parent = str(_value(record, "id", "image_id", "filename", "file_name") or f"levir/{source_split}/{index}")
        t1, t2, t1_unsafe, t2_unsafe = _pair_paths(record, source_split)
        if t1_unsafe or t2_unsafe:
            yield None, "unsafe_image_path"
            continue
        if not t1:
            yield None, "missing_t1"
            continue
        if not t2:
            yield None, "missing_t2"
            continue
        captions = _value(record, "captions", "sentences", "answers")
        if isinstance(captions, str):
            captions = [captions]
        if not isinstance(captions, list):
            captions = [_value(record, "caption", "answer")]
        for caption_index, caption in enumerate(captions):
            if isinstance(caption, dict):
                caption = _value(caption, "raw", "caption", "text", "answer")
            if not isinstance(caption, str) or not caption.strip():
                yield None, "missing_answer"
                continue
            answer = caption.strip()
            if answer.upper() in {"NO_CHANGE", "NO CHANGE"}:
                answer = CANONICAL_NO_CHANGE
            yield _episode(
                episode_id=f"levir/{source_split}/{parent}/caption/{caption_index}", parent_id=parent,
                dataset="LEVIR-CC", split=split, task="change_caption", question="", t1=t1, t2=t2,
                answer=answer, image_source="levir",
                provenance={"source_dataset": "LEVIR-CC", "source_record_id": parent, "answer_origin": "human"},
            ), None


def _changechat_candidates(records: Iterable[dict[str, Any]], seed: str) -> Iterable[tuple[dict[str, Any] | None, str | None]]:
    for index, record in enumerate(records):
        parent = str(_value(record, "id", "sample_id", "image_id") or f"changechat/{index}")
        turns = _value(record, "turns", "conversations", "messages")
        if isinstance(turns, list) and len(turns) > 2:
            yield None, "context_dependent_multiturn"
            continue
        task = _value(record, "task", "task_type") or "change_caption"
        if task not in {"change_caption", "change_qa"}:
            yield None, "unknown_task"
            continue
        question = _value(record, "question", "instruction", "query") or ""
        answer = _value(record, "answer", "response", "output")
        if isinstance(turns, list) and len(turns) == 2:
            question = question or str(_value(turns[0], "value", "content", "text") or "")
            answer = answer or _value(turns[1], "value", "content", "text")
        if task == "change_qa" and (not isinstance(question, str) or not question.strip()):
            yield None, "missing_question"
            continue
        if not isinstance(answer, str) or not answer.strip():
            yield None, "missing_answer"
            continue
        t1, t2, t1_unsafe, t2_unsafe = _pair_paths(record, str(record.get("split") or "train"))
        if t1_unsafe or t2_unsafe:
            yield None, "unsafe_image_path"
            continue
        if not t1:
            yield None, "missing_t1"
            continue
        if not t2:
            yield None, "missing_t2"
            continue
        split_raw = str(record.get("split") or "").lower()
        split = "validation" if split_raw in {"val", "validation"} else "train"
        if not split_raw:
            split = "validation" if _group_validation(parent, seed) else "train"
        yield _episode(
            episode_id=f"changechat/{split}/{parent}/{task}/0", parent_id=parent,
            dataset="ChangeChat", split=split, task=task, question=str(question), t1=t1, t2=t2,
            answer=answer.strip(), image_source="changechat",
            provenance={"source_dataset": "ChangeChat", "source_record_id": parent, "answer_origin": "human"},
        ), None


def _prompt(args: argparse.Namespace) -> tuple[str, str]:
    if bool(args.prompt_ref) == bool(args.prompt_file):
        raise ValueError("provide exactly one of --prompt-ref or --prompt-file")
    if args.prompt_file:
        path = Path(args.prompt_file)
        return str(path), path.read_text(encoding="utf-8")
    candidate = Path(args.prompt_ref)
    if candidate.is_file():
        return str(candidate), candidate.read_text(encoding="utf-8")
    repo_prompt = Path(__file__).resolve().parents[1] / "prompts" / f"{args.prompt_ref}.md"
    if repo_prompt.is_file():
        return str(args.prompt_ref), repo_prompt.read_text(encoding="utf-8")
    raise ValueError("--prompt-ref does not resolve to a readable repository prompt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare ordered ChangeAgent Qwen SFT JSONL (offline only).")
    parser.add_argument("--source-type", required=True, choices=("levir_caption", "changechat"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--excluded-sample-ids", default="", help="Comma-separated parent ids to exclude.")
    parser.add_argument("--prompt-ref")
    parser.add_argument("--prompt-file")
    parser.add_argument("--seed", default="change-qwen-sft-v1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not args.source.is_file():
            raise ValueError(f"source not found: {args.source}")
        prompt_ref, prompt_text = _prompt(args)
        records = _read_records(args.source)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    excluded = {item.strip() for item in args.excluded_sample_ids.split(",") if item.strip()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = {"train": [], "validation": []}
    rejections: list[dict[str, str]] = []
    seen: set[str] = set()
    candidates = _levir_candidates(records) if args.source_type == "levir_caption" else _changechat_candidates(records, args.seed)
    for candidate, code in candidates:
        if candidate is None:
            rejections.append({"reason": code or "invalid_target_schema"})
            continue
        parent = candidate["parent_sample_id"]
        if parent in excluded:
            rejections.append({"source_record_id": parent, "reason": "excluded_parent_sample"})
            continue
        if candidate["episode_id"] in seen:
            rejections.append({"source_record_id": parent, "reason": "duplicate_episode_id"})
            continue
        try:
            _validate_output_episode(candidate)
        except Exception as error:
            rejections.append({"source_record_id": parent, "reason": getattr(error, "code", "invalid_target_schema")})
            continue
        seen.add(candidate["episode_id"])
        episodes[candidate["split"]].append(candidate)
    train_parents = {ep["parent_sample_id"] for ep in episodes["train"]}
    val_parents = {ep["parent_sample_id"] for ep in episodes["validation"]}
    if train_parents & val_parents:
        print("error: split_leakage", file=sys.stderr)
        return 1
    def write(name: str, values: list[dict]) -> str:
        payload = "".join(_safe_json(item) + "\n" for item in values)
        (args.output_dir / name).write_text(payload, encoding="utf-8", newline="\n")
        return _sha256_bytes(payload.encode("utf-8"))
    train_sha = write("train.jsonl", episodes["train"])
    val_sha = write("validation.jsonl", episodes["validation"])
    rejected_payload = "".join(_safe_json(item) + "\n" for item in rejections)
    (args.output_dir / "rejected.jsonl").write_text(rejected_payload, encoding="utf-8", newline="\n")
    manifest = {
        "schema_version": CHANGE_SFT_SCHEMA_VERSION, "tool": "scripts/prepare_change_qwen_sft.py",
        "source_type": args.source_type, "seed": args.seed,
        "source": {"basename": args.source.name, "sha256": _sha256_file(args.source)},
        "change_prompt": {"ref": prompt_ref, "sha256": _sha256_bytes(prompt_text.encode("utf-8"))},
        "outputs": {"train.jsonl_sha256": train_sha, "validation.jsonl_sha256": val_sha,
                    "rejected.jsonl_sha256": _sha256_bytes(rejected_payload.encode("utf-8"))},
        "counts": {split: {"total": len(values), "by_task": dict(Counter(ep["task"] for ep in values))} for split, values in episodes.items()},
        "rejected": {"total": len(rejections), "by_reason": dict(Counter(item["reason"] for item in rejections))},
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"train.jsonl: {len(episodes['train'])} episodes")
    print(f"validation.jsonl: {len(episodes['validation'])} episodes")
    return 0


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


if __name__ == "__main__":
    raise SystemExit(main())
