#!/usr/bin/env python3
"""Compile raw VRSBench train annotations into GeneralVQAAgent SFT chats.

将 VRSBench 原始训练标注整理为 GeneralVQAAgent 的 SFT 对话，同时保留
VisualTaskPlanner 的原始图文输入视图。源标注只读，输出使用稳定排序和原子替换。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "data" / "VRSBench-full" / "Annotations_train"
OUTPUT = REPO_ROOT / "data" / "20260824_VQA_agent_finetuning"

sys.path.insert(0, str(REPO_ROOT))
from agents.schema import AgentResult, VisualEvidence  # noqa: E402
from data.adapters.vrsbench.task_normalizer import normalize_task  # noqa: E402

SYSTEM_PROMPT = (
    "Answer the question concisely from the image. Preserve up to four representative "
    "relevant localized objects as labeled evidence_items; copy all evidence-item boxes "
    "into boxes in the same order. Coordinates are integer whole-image 0..999 raster "
    "coordinates in JSON with the origin at the top-left, positive x to the right, and "
    "positive y downward. A box is one flat array [x1,y1,x2,y2], never a pair of corner "
    "arrays. Use an empty evidence list only when the answer genuinely has no localizable "
    "visual support. Do not include hidden reasoning.\n\nReturn valid JSON only. Set "
    "agent_name to 'general_vqa_agent'; put the concise final answer in answer, retain "
    "relevant labeled boxes or points in evidence_items, copy evidence boxes into boxes, "
    "use concise factual evidence strings, and set status to 'completed'. When boxes are "
    "returned, use integer 0..999 whole-image xyxy coordinates in JSON, with each box as "
    "a flat [x1,y1,x2,y2] array."
)


def _stable_paths() -> list[Path]:
    """Return a reproducible source order. / 返回可复现的源文件顺序。"""
    return sorted(
        SOURCE.glob("*.json"),
        key=lambda path: (hashlib.sha256(path.name.encode()).hexdigest(), path.name),
    )


def _box_999(value: Any) -> list[int] | None:
    """Convert a valid normalized source box without repairing it. / 转换合法源框且不修复。"""
    if not isinstance(value, list) or len(value) != 4:
        return None
    if not all(isinstance(item, (int, float)) and 0 <= item <= 1 for item in value):
        return None
    x1, y1, x2, y2 = value
    if x1 >= x2 or y1 >= y2:
        return None
    return [round(item * 999) for item in value]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def _evidence(question: str, objects: list[dict[str, Any]]) -> tuple[list[str], list[VisualEvidence]]:
    """Choose source-backed gaze targets, falling back to an explicit scene focus.
    选择有源标注依据的注视目标；无可定位对象时显式回退为全景中心注视。
    """
    question_tokens = _tokens(question)
    candidates: list[tuple[int, int, dict[str, Any], list[int]]] = []
    for index, obj in enumerate(objects):
        box = _box_999(obj.get("obj_coord"))
        if box is None:
            continue
        searchable = " ".join(
            str(obj.get(key, ""))
            for key in ("obj_cls", "referring_sentence", "obj_position", "obj_rel_position")
        )
        overlap = len(question_tokens & _tokens(searchable))
        candidates.append((overlap, -index, obj, box))
    matched = [item for item in candidates if item[0] > 0]
    selected = sorted(matched or candidates, reverse=True, key=lambda item: (item[0], item[1]))[:4]
    if not selected:
        item = VisualEvidence(
            label="scene_focus", point=[500, 500], confidence=1.0,
            coordinate_frame="normalized_0_999_top_left",
        )
        return ["The whole-image scene provides the visual support for the answer."], [item]
    evidence_items = [
        VisualEvidence(
            label=str(obj.get("obj_cls") or "object"), box=box, confidence=1.0,
            coordinate_frame="normalized_0_999_top_left",
        )
        for _, _, obj, box in selected
    ]
    evidence = [
        f"A source-annotated {item.label} is visible at the indicated location."
        for item in evidence_items
    ]
    return evidence, evidence_items


def _record(path: Path, data: dict[str, Any], qa: dict[str, Any]) -> dict[str, Any] | None:
    question = str(qa.get("question", "")).strip()
    answer = str(qa.get("answer", "")).strip()
    normalization = normalize_task(question, str(qa.get("type", "")))
    task = normalization.normalized_task
    if task == "counting" or task not in {"general_vqa", "spatial_relation"}:
        return None
    image_name = str(data.get("image") or f"{path.stem}.png")
    image = f"../VRSBench-full/Images_train/{image_name}"
    sample_id = f"vrsbench/train/{path.stem}/qa/{qa.get('ques_id')}"
    evidence, evidence_items = _evidence(question, list(data.get("objects", [])))
    result = AgentResult(
        agent_name="general_vqa_agent", answer=answer, evidence=evidence,
        evidence_items=evidence_items, status="completed",
    )
    payload = {
        "question": question,
        "task": task,
        "coordinate_frame": "normalized_0_999_top_left",
        "box_format": "integer_xyxy_json",
        "answer_constraints": normalization.answer_constraints,
        "semantic_subtype": normalization.semantic_subtype,
    }
    return {
        "schema_version": 1,
        "sample_id": sample_id,
        "dataset": "VRSBench",
        "split": "train",
        "image": image,
        "source": {"annotation_file": path.name, "ques_id": qa.get("ques_id"), "question_type": qa.get("type")},
        "planner_messages": [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}
        ],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": json.dumps(payload, ensure_ascii=False)}]},
            {"role": "assistant", "content": json.dumps(result.model_dump(mode="json"), ensure_ascii=False)},
        ],
    }


def _atomic_json(path: Path, value: Any) -> None:
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    counts: Counter[str] = Counter()
    output_path = OUTPUT / "train.jsonl"
    with NamedTemporaryFile("w", encoding="utf-8", dir=OUTPUT, delete=False) as handle:
        for path in _stable_paths():
            data = json.loads(path.read_text(encoding="utf-8"))
            counts["annotation_files"] += 1
            for qa in data.get("qa_pairs", []):
                counts["qa_pairs_total"] += 1
                record = _record(path, data, qa)
                if record is None:
                    counts["excluded_counting"] += 1
                    continue
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["records"] += 1
                counts[f"task:{json.loads(record['messages'][1]['content'][1]['text'])['task']}"] += 1
                assistant = json.loads(record["messages"][2]["content"])
                if assistant["evidence_items"][0]["point"] is not None:
                    counts["scene_focus_fallback"] += 1
                else:
                    counts["source_box_gaze"] += 1
        temporary = Path(handle.name)
    os.replace(temporary, output_path)
    manifest = {
        "schema_version": 1,
        "created_date": "2026-08-24",
        "purpose": "VQA agent fine-tuning",
        "source": "../VRSBench-full/Annotations_train",
        "output": "train.jsonl",
        "counts": dict(sorted(counts.items())),
        "notes": [
            "Counting questions are excluded because CountingAgent owns the counting task.",
            "planner_messages mirrors planner inference input: ordered image blocks followed by raw question text.",
            "messages mirrors GeneralVQAAgent SFT input/output; assistant content is serialized AgentResult JSON.",
            "confidence=1.0 denotes deterministic source-annotation provenance, not a model confidence estimate.",
            "scene_focus is an explicit whole-image gaze fallback, not an object detection box.",
        ],
    }
    _atomic_json(OUTPUT / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
