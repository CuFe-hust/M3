#!/usr/bin/env python3
"""Compile the reviewed LRS-VQA supplement into VisualTaskPlan v5 SFT data.

The compiler is deterministic and offline. Source answers are retained only in
audit provenance and never enter model-visible system/user/assistant messages.
将已审核的 LRS-VQA 补充集确定性编译为 VisualTaskPlan v5 SFT 数据；源答案仅保留
在审计来源中，绝不进入模型可见的 system/user/assistant 消息。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import uuid
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from agents.evidence_catalog import EvidenceCatalog
from agents.general_vqa.evidence.rendering import preview_from_path
from agents.schema import VisualTaskPlan
from scripts.refine_visual_planner_dataset import _question_evidence_categories


FORMAT = "visual-planner-compiled-chat-v1"
SOURCE_GROUP = "LRS-VQA-Supplement"
SPLIT_SEED = "m3-lrs-vqa-supplement-image-split-v1"
VAL_PERCENT = 10
PROTOCOL_RELATIVE = PurePosixPath("protocols/protocol-e2553b41e4b2a5af.json")


def _json(value: Any, *, sort_keys: bool = True) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=sort_keys, separators=(",", ":"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(content)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: Any) -> dict[str, Any]:
    content = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    _atomic_write(path, content)
    return {"bytes": len(content), "sha256": _sha256_bytes(content)}


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    content = "".join(_json(row) + "\n" for row in rows).encode()
    _atomic_write(path, content)
    return {"examples": len(rows), "bytes": len(content), "sha256": _sha256_bytes(content)}


def _safe_source(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or any(part in {"", ".", ".."} for part in posix.parts):
        raise ValueError("UNSAFE_IMAGE_PATH")
    candidate = (root / Path(*posix.parts)).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("UNSAFE_IMAGE_PATH")
    return candidate


def _split(image_digest: str) -> str:
    rank = int(_sha256_bytes(f"{SPLIT_SEED}|{image_digest}".encode())[:8], 16) % 100
    return "val" if rank < VAL_PERCENT else "train"


def _task(question: str, category: str) -> str:
    normalized = " ".join(question.casefold().split())
    if re.search(r"\b(how many|total number of|number of|amount of)\b", normalized):
        return "counting"
    if category == "rural or urban":
        return "scene_classification"
    # Keep relation routing conservative: the answer must itself express a
    # spatial relationship, not merely require inspecting a nearby object.
    # 保守识别空间关系：答案本身必须表达空间关系，而不只是观察邻近对象。
    if re.search(r"\b(where|what spatial relation|relative position|which direction)\b", normalized):
        return "spatial_relation"
    return "general_vqa"


def _count_target(question: str) -> str | None:
    normalized = " ".join(question.casefold().split())
    if "total number of planes" in normalized:
        return "planes"
    if normalized.startswith("how many fields"):
        return "fields"
    return None


def _reason_code(task: str, category: str) -> str:
    if task == "counting":
        return "quantity_question"
    if task == "scene_classification":
        return "scene_classification_request"
    return {
        "object category": "category_question",
        "object color": "color_question",
        "object shape": "shape_question",
        "object status": "status_question",
        "object background": "background_question",
    }.get(category, "general_question")


def _roi(record: Mapping[str, Any]) -> list[int]:
    width, height = record["image_size"]
    x0, y0, x1, y1 = record["hbox"]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("INVALID_ATTENTION_BOX")
    values = [
        round(x0 * 999 / width),
        round(y0 * 999 / height),
        round(x1 * 999 / width),
        round(y1 * 999 / height),
    ]
    values[2] = max(values[2], values[0] + 1)
    values[3] = max(values[3], values[1] + 1)
    return [min(999, value) for value in values]


def _ordered_target(plan: VisualTaskPlan) -> dict[str, Any]:
    region = plan.region_request
    return {
        "version": plan.version,
        "task": plan.task,
        "needs_visual_assistance": plan.needs_visual_assistance,
        "object_categories": list(plan.object_categories),
        "count_target": plan.count_target,
        "region_request": {
            "explicit": region.explicit,
            "image_index": region.image_index,
            "roi_xyxy": list(region.roi_xyxy) if region.roi_xyxy is not None else None,
        },
        "reason_codes": list(plan.reason_codes),
    }


def _preview(source: Path, output: Path, *, split: str, max_side: int) -> tuple[str, str]:
    # The reviewed FAIR images legitimately exceed Pillow's generic bomb
    # threshold. Disable it only around this trusted local source decode.
    # 已审核的 FAIR 大图会超过 Pillow 通用炸弹阈值；仅对可信本地源解码临时关闭。
    previous_limit = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = None
        data_url, digest = preview_from_path(source, max_side=max_side)
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit
    prefix, separator, encoded = data_url.partition(",")
    if not separator or prefix != "data:image/png;base64":
        raise ValueError("INVALID_PREVIEW")
    content = base64.b64decode(encoded, validate=True)
    if _sha256_bytes(content) != digest:
        raise ValueError("PREVIEW_DIGEST_MISMATCH")
    relative = f"training_images/{split}/{digest}.png"
    target = output / relative
    if target.exists() and _sha256_file(target) != digest:
        raise ValueError("PREVIEW_COLLISION")
    if not target.exists():
        _atomic_write(target, content)
    return relative, digest


def compile_dataset(source_root: Path, protocol_source: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_source.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "protocol-e2553b41e4b2a5af":
        raise ValueError("UNEXPECTED_PROTOCOL")
    catalog = EvidenceCatalog.from_file(Path(__file__).resolve().parents[1] / "agents/evidence_catalog.json")
    global_categories = tuple(protocol["annotation_evidence_policy"]["global_executable_categories"])
    preview_max_side = protocol["planner_binding"]["preview_max_side"]
    source_rows = [json.loads(line) for line in (source_root / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    compact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    training: dict[str, list[dict[str, Any]]] = defaultdict(list)
    decisions: list[dict[str, Any]] = []
    image_cache: dict[str, tuple[str, str, str]] = {}
    task_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()

    protocol_target = source_root / Path(*PROTOCOL_RELATIVE.parts)
    _atomic_write(protocol_target, protocol_source.read_bytes())
    for record in source_rows:
        question = record["text"]
        category = record["category"]
        source_image = _safe_source(source_root, record["image"])
        source_digest = _sha256_file(source_image)
        split = _split(source_digest)
        cached = image_cache.get(record["image"])
        if cached is None:
            suffix = source_image.suffix.casefold()
            image_relative = f"images/{split}/{source_digest}{suffix}"
            image_target = source_root / image_relative
            image_target.parent.mkdir(parents=True, exist_ok=True)
            if not image_target.exists():
                try:
                    os.link(source_image, image_target)
                except OSError:
                    shutil.copy2(source_image, image_target)
            preview_relative, preview_digest = _preview(
                source_image,
                source_root,
                split=split,
                max_side=preview_max_side,
            )
            cached = (image_relative, preview_relative, preview_digest)
            image_cache[record["image"]] = cached
        image_relative, preview_relative, preview_digest = cached
        task = _task(question, category)
        count_target = _count_target(question) if task == "counting" else None
        categories = _question_evidence_categories(
            question,
            task=task,
            catalog=catalog,
            global_executable=global_categories,
        )
        if task == "counting":
            executable = set(protocol["planner_binding"]["task_executable_categories"]["counting"])
            expanded = catalog.expand_target(count_target or "")
            if not expanded or any(item not in executable for item in expanded):
                categories = ()
            else:
                categories = tuple(item for item in catalog.leaf_categories if item in expanded)
        reason_codes = [_reason_code(task, category), "explicit_region"]
        plan = VisualTaskPlan.model_validate(
            {
                "version": "visual-task-plan-v5",
                "task": task,
                "needs_visual_assistance": bool(categories),
                "object_categories": list(categories),
                "count_target": count_target,
                "region_request": {"explicit": True, "image_index": 0, "roi_xyxy": _roi(record)},
                "reason_codes": reason_codes,
            }
        )
        target = _ordered_target(plan)
        target_text = _json(target, sort_keys=False)
        episode_id = f"lrs-vqa-supplement-{record['question_id']}"
        user_content = [
            {"type": "image", "image": image_relative},
            {"type": "text", "text": question},
        ]
        compact_record = {
            "dataset": "lrs_vqa_supplement",
            "episode_id": episode_id,
            "image": image_relative,
            "messages": [
                {"role": "system", "content_ref": PROTOCOL_RELATIVE.as_posix()},
                {"role": "user", "content": user_content},
            ],
            "protocol_id": protocol["protocol_id"],
            "protocol_version": "visual-task-plan-v5",
            "provenance": {
                "source_record_id": record["question_id"],
                "source_image": record["image"],
                "source_category": category,
                "source_image_sha256": source_digest,
                "preview_sha256": preview_digest,
            },
            "response_model": "VisualTaskPlan",
            "schema_version": 1,
            "source_group": SOURCE_GROUP,
            "split": split,
            "target": target,
            "target_text": target_text,
        }
        compact[split].append(compact_record)
        training[split].append(
            {
                "episode_id": episode_id,
                "format": FORMAT,
                "image": preview_relative,
                "messages": [
                    {"role": "system", "content": protocol["system_prompt"]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": preview_relative},
                            {"type": "text", "text": question},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
                ],
                "schema_version": 1,
                "source_group": SOURCE_GROUP,
                "split": split,
            }
        )
        decisions.append(
            {
                "episode_id": episode_id,
                "source_category": category,
                "task": task,
                "needs_visual_assistance": bool(categories),
                "object_categories": list(categories),
                "count_target": count_target,
                "roi_xyxy": _roi(record),
            }
        )
        task_counts[task] += 1
        category_counts.update(categories)

    files: dict[str, Any] = {}
    for split in ("train", "val"):
        compact[split].sort(key=lambda row: row["episode_id"])
        training[split].sort(key=lambda row: row["episode_id"])
        files[f"datasets/{split}.jsonl"] = _write_jsonl(source_root / f"datasets/{split}.jsonl", compact[split])
        files[f"training/{split}.jsonl"] = _write_jsonl(source_root / f"training/{split}.jsonl", training[split])
    files["audit/label_decisions.jsonl"] = _write_jsonl(source_root / "audit/label_decisions.jsonl", decisions)
    manifest = {
        "dataset": "LRS-VQA visual-planner supplement",
        "format": FORMAT,
        "protocol_id": protocol["protocol_id"],
        "source": "samples.jsonl",
        "source_sha256": _sha256_file(source_root / "samples.jsonl"),
        "examples": len(source_rows),
        "unique_images": len(image_cache),
        "split_policy": {"group_by": "source_image_sha256", "seed": SPLIT_SEED, "val_percent": VAL_PERCENT},
        "task_distribution": dict(sorted(task_counts.items())),
        "object_category_distribution": dict(sorted(category_counts.items())),
        "ground_truth_model_visible": False,
        "target_field_order": ["version", "task", "needs_visual_assistance", "object_categories", "count_target", "region_request", "reason_codes"],
        "files": files,
    }
    _write_json(source_root / "compiled_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/20260824-visual-planner-supplement"))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("data/phase2-train-visualplanning-refined-v4/protocols/protocol-e2553b41e4b2a5af.json"),
    )
    args = parser.parse_args(argv)
    manifest = compile_dataset(args.source.resolve(), args.protocol.resolve())
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
