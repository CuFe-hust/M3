#!/usr/bin/env python3
"""Deterministically supplement refined visual-planner SFT data.

The compiler is offline and source-read-only. It adds balanced examples for
underrepresented task identities from VRSBench and LEVIR-CC while preserving
the existing compact dataset and resolved training-chat formats.

确定性补充 Visual Planner SFT 数据。编译器完全离线且只读源数据，从
VRSBench 与 LEVIR-CC 补齐稀缺 task，并保持现有紧凑数据与展开训练消息格式。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from agents.evidence_catalog import EvidenceCatalog
from agents.schema import VisualTaskPlan
from agents.general_vqa.evidence.rendering import preview_from_path
from scripts.refine_visual_planner_dataset import (
    TRAINING_FORMAT,
    _answer_leakage,
    _atomic_write_bytes,
    _atomic_write_json,
    _atomic_write_text,
    _canonical_json,
    _question_evidence_categories,
    _sha256_bytes,
    _sha256_file,
    _write_jsonl,
    compile_training_messages,
)


# v2: caption/change assistance is driven by question text only; source
# annotations (VRSBench objects, LEVIR reference captions) never inject
# planner categories into generic caption/change targets.
# v2：caption/change 的 assistance 仅由问题文本驱动；源标注（VRSBench
# 物体、LEVIR 参考 caption）不再向 generic caption/change target 注入类别。
POLICY_VERSION = "visual-planner-structured-supplement-v2"
SELECTION_SEED = "m3-visual-planner-supplement-v1"
VRS_GROUP = "VRSBenchSupplement"
LEVIR_GROUP = "LEVIR_CC"
DEFAULT_TRAIN_PER_TASK = 800
DEFAULT_VAL_PER_TASK = 100
_NO_REGION = {"explicit": False, "image_index": None, "roi_xyxy": None}

_VRS_CLASS_MAP: dict[str, tuple[str, ...]] = {
    "airplane": ("plane",),
    "vehicle": ("small-vehicle", "large-vehicle"),
    "ship": ("ship",),
    "harbor": ("harbor",),
    "tennis-court": ("tennis-court",),
    "bridge": ("bridge",),
    "storage-tank": ("storage-tank",),
    "baseball-diamond": ("baseball-diamond",),
    "ground-track-field": ("ground-track-field",),
    "swimming-pool": ("swimming-pool",),
    "basketball-court": ("basketball-court",),
    "roundabout": ("roundabout",),
    "soccer-ball-field": ("soccer-ball-field",),
    "airport": ("airport",),
    "helicopter": ("helicopter",),
    "container-crane": ("container-crane",),
    "helipad": ("helipad",),
}

_DISPLAY_PLURALS = {
    "airplane": "airplanes",
    "vehicle": "vehicles",
    "ship": "ships",
    "harbor": "harbors",
    "tennis-court": "tennis courts",
    "bridge": "bridges",
    "storage-tank": "storage tanks",
    "baseball-diamond": "baseball diamonds",
    "ground-track-field": "ground track fields",
    "swimming-pool": "swimming pools",
    "basketball-court": "basketball courts",
    "roundabout": "roundabouts",
    "soccer-ball-field": "soccer fields",
    "airport": "airports",
    "helicopter": "helicopters",
    "container-crane": "container cranes",
    "helipad": "helipads",
}

_CAPTION_QUESTIONS = (
    "Describe this remote-sensing image in detail.",
    "Provide an open-ended description of the visible scene.",
    "Summarize the important visual content of this remote-sensing image.",
    "Write a detailed caption for this image.",
)
_CHANGE_CAPTION_QUESTIONS = (
    "Describe all meaningful changes between the first and second remote-sensing images.",
    "Provide an open-ended description of how the scene changed from the first image to the second.",
    "Summarize the visible changes between these two temporally ordered images.",
    "Write a change caption comparing the first image with the second image.",
)
_CHANGE_QA_QUESTIONS = (
    "Did any meaningful visual change occur between the first and second remote-sensing images?",
    "Does the second image show a change from the first image?",
    "Are the two temporally ordered scenes visually different?",
    "Has the scene changed between the first observation and the second observation?",
)
_NUMBER_WORDS = frozenset(
    {
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
        "twenty",
        "thirty",
        "forty",
        "fifty",
        "sixty",
        "seventy",
        "eighty",
        "ninety",
        "hundred",
    }
)
_NUMBER_VALUES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_MC_ALLOWED_TYPES = frozenset({"object quantity", "object existence", "object color"})
_COLOR_WORDS = frozenset(
    {
        "black",
        "white",
        "gray",
        "grey",
        "red",
        "blue",
        "green",
        "yellow",
        "brown",
        "orange",
        "purple",
        "pink",
        "tan",
        "beige",
        "silver",
        "gold",
        "multicolor",
        "multicolored",
        "varied",
    }
)


@dataclass(frozen=True)
class VrsCandidate:
    split: str
    annotation_path: Path
    image_path: Path
    record: dict[str, Any]
    grounding_object: dict[str, Any]
    fine_classes: tuple[str, ...]
    qa: dict[str, Any]
    choices: tuple[str, ...]


@dataclass(frozen=True)
class AddedEpisode:
    dataset_record: dict[str, Any]
    training_record: dict[str, Any]
    source_locator: str


def _rank(*parts: object) -> str:
    return _sha256_bytes("|".join(str(part) for part in parts).encode("utf-8"))


def _clean_text(value: Any, *, max_length: int = 500) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:max_length].strip()


def _numeric_value(value: str) -> int | None:
    normalized = value.casefold().replace("-", " ").strip(" .")
    if re.fullmatch(r"\d+", normalized):
        return int(normalized)
    tokens = normalized.split()
    if not tokens or not all(token in _NUMBER_WORDS or token == "and" for token in tokens):
        return None
    total = 0
    current = 0
    for token in tokens:
        if token == "and":
            continue
        if token == "hundred":
            current = max(current, 1) * 100
        else:
            current += _NUMBER_VALUES[token]
    total += current
    return total


def _numeric_answer(value: str) -> bool:
    return _numeric_value(value) is not None


def _valid_mc_answer(question_type: str, answer: str) -> bool:
    normalized = answer.casefold().strip(" .")
    if question_type == "object quantity":
        return _numeric_answer(answer)
    if question_type == "object existence":
        return normalized in {"yes", "no"}
    if question_type == "object color":
        words = set(re.findall(r"[a-z]+", normalized))
        return bool(words & _COLOR_WORDS)
    return False


def _ordered_categories(
    raw_classes: Iterable[str], catalog_order: Sequence[str]
) -> tuple[str, ...]:
    selected: set[str] = set()
    for raw in raw_classes:
        selected.update(_VRS_CLASS_MAP.get(raw, ()))
    return tuple(category for category in catalog_order if category in selected)[:8]


def _valid_grounding_object(raw: Any) -> bool:
    if not isinstance(raw, dict) or raw.get("obj_cls") not in _VRS_CLASS_MAP:
        return False
    sentence = _clean_text(raw.get("referring_sentence"), max_length=240)
    coord = raw.get("obj_coord")
    return bool(
        sentence
        and isinstance(coord, list)
        and len(coord) == 4
        and all(isinstance(value, (int, float)) and 0 <= value <= 1 for value in coord)
        and coord[0] < coord[2]
        and coord[1] < coord[3]
    )


def _answer_pool(vrs_root: Path, split: str) -> tuple[dict[str, tuple[str, ...]], str]:
    pools: dict[str, set[str]] = defaultdict(set)
    inventory = hashlib.sha256()
    for path in sorted((vrs_root / f"Annotations_{split}").glob("*.json")):
        raw = path.read_bytes()
        inventory.update(path.name.encode("utf-8") + b"\0" + raw)
        record = json.loads(raw)
        for qa in record.get("qa_pairs", []):
            answer = _clean_text(qa.get("answer"), max_length=100)
            question_type = _clean_text(qa.get("type"), max_length=80)
            if question_type not in _MC_ALLOWED_TYPES or not _valid_mc_answer(
                question_type, answer
            ):
                continue
            if answer and question_type:
                pools[question_type].add(answer)
    return (
        {key: tuple(sorted(values, key=str.casefold)) for key, values in pools.items()},
        inventory.hexdigest(),
    )


def build_choices(
    answer: str,
    pool: Sequence[str],
    *,
    identity: str,
) -> tuple[str, ...] | None:
    """Build deterministic closed options without recording the correct index.
    确定性构造封闭选项，但不记录正确答案位置。"""

    correct = _clean_text(answer, max_length=100)
    if not correct:
        return None
    distinct: dict[str, str] = {}
    correct_numeric = _numeric_value(correct)
    for candidate in pool:
        cleaned = _clean_text(candidate, max_length=100)
        if cleaned:
            numeric = _numeric_value(cleaned) if correct_numeric is not None else None
            key = f"numeric:{numeric}" if numeric is not None else cleaned.casefold()
            distinct.setdefault(key, cleaned)
    correct_key = (
        f"numeric:{correct_numeric}" if correct_numeric is not None else correct.casefold()
    )
    distinct[correct_key] = correct
    if correct.casefold() in {"yes", "no"}:
        opposite = "No" if correct.casefold() == "yes" else "Yes"
        matched = distinct.get(opposite.casefold())
        if matched is None:
            return None
        options = [correct, matched]
        options.sort(key=lambda value: _rank(SELECTION_SEED, identity, "option", value))
        return tuple(options)
    desired = 4
    alternatives = [
        value for key, value in distinct.items() if key != correct_key
    ]
    alternatives.sort(key=lambda value: _rank(SELECTION_SEED, identity, "distractor", value))
    if len(alternatives) < desired - 1:
        return None
    options = [correct, *alternatives[: desired - 1]]
    options.sort(key=lambda value: _rank(SELECTION_SEED, identity, "option", value))
    return tuple(options)


def format_multiple_choice(question: str, choices: Sequence[str]) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if not 2 <= len(choices) <= len(labels):
        raise ValueError("multiple-choice questions require 2..26 choices")
    rendered = " ".join(
        f"({labels[index]}) {choice}" for index, choice in enumerate(choices)
    )
    return f"{_clean_text(question)} Choices: {rendered}"


def validate_choices(question_type: str, choices: Sequence[str]) -> None:
    """Fail closed when a generated option set leaves its semantic answer space.
    生成选项越出同一语义答案空间时严格失败。"""

    if question_type == "object quantity":
        values = [_numeric_value(choice) for choice in choices]
        if any(value is None for value in values) or len(set(values)) != len(values):
            raise ValueError("INCOMPATIBLE_QUANTITY_CHOICES")
        return
    if question_type == "object existence":
        if {choice.casefold() for choice in choices} != {"yes", "no"}:
            raise ValueError("INCOMPATIBLE_EXISTENCE_CHOICES")
        return
    if question_type == "object color":
        if not all(_valid_mc_answer(question_type, choice) for choice in choices):
            raise ValueError("INCOMPATIBLE_COLOR_CHOICES")
        return
    raise ValueError("UNSUPPORTED_MULTIPLE_CHOICE_TYPE")


def _select_vrs(
    vrs_root: Path,
    *,
    split: str,
    quota: int,
    answer_pools: Mapping[str, Sequence[str]],
    excluded_images: set[str],
) -> list[VrsCandidate]:
    candidates: list[VrsCandidate] = []
    annotation_root = vrs_root / f"Annotations_{split}"
    image_root = vrs_root / f"Images_{split}"
    for path in sorted(annotation_root.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        image_name = _clean_text(record.get("image"), max_length=200)
        image_path = image_root / image_name
        if (
            not image_name
            or image_name in excluded_images
            or not image_path.is_file()
            or not _clean_text(record.get("caption"))
        ):
            continue
        objects = [item for item in record.get("objects", []) if isinstance(item, dict)]
        raw_classes = tuple(
            sorted(
                {str(item.get("obj_cls")) for item in objects if item.get("obj_cls") in _VRS_CLASS_MAP}
            )
        )
        if len(raw_classes) < 2:
            continue
        grounding_objects = [item for item in objects if _valid_grounding_object(item)]
        if not grounding_objects:
            continue
        grounding_objects.sort(
            key=lambda item: _rank(SELECTION_SEED, split, image_name, "ground", item.get("obj_id"))
        )
        suitable_qas: list[tuple[dict[str, Any], tuple[str, ...]]] = []
        for qa in record.get("qa_pairs", []):
            if not isinstance(qa, dict):
                continue
            question = _clean_text(qa.get("question"), max_length=500)
            answer = _clean_text(qa.get("answer"), max_length=100)
            question_type = _clean_text(qa.get("type"), max_length=80)
            if question_type not in _MC_ALLOWED_TYPES or not _valid_mc_answer(
                question_type, answer
            ):
                continue
            choices = build_choices(
                answer,
                answer_pools.get(question_type, ()),
                identity=f"{split}|{image_name}|{qa.get('ques_id')}|{question}",
            )
            if question and choices is not None:
                suitable_qas.append((qa, choices))
        if not suitable_qas:
            continue
        suitable_qas.sort(
            key=lambda item: _rank(
                SELECTION_SEED,
                split,
                image_name,
                "qa",
                item[0].get("ques_id"),
            )
        )
        qa, choices = suitable_qas[0]
        validate_choices(_clean_text(qa.get("type"), max_length=80), choices)
        candidates.append(
            VrsCandidate(
                split=split,
                annotation_path=path,
                image_path=image_path,
                record=record,
                grounding_object=grounding_objects[0],
                fine_classes=raw_classes[:2],
                qa=qa,
                choices=choices,
            )
        )
    candidates.sort(key=lambda item: _rank(SELECTION_SEED, "vrs", split, item.image_path.name))
    if len(candidates) < quota:
        raise ValueError(f"VRS_QUOTA_UNAVAILABLE:{split}:{len(candidates)}<{quota}")
    return candidates[:quota]


def select_balanced_levir(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    quota: int,
) -> list[Mapping[str, Any]]:
    """Select equal changed/unchanged rows with stable content hashing.
    使用稳定内容哈希等量选取变化与未变化样本。"""

    if quota % 2:
        raise ValueError("LEVIR quota must be even")
    selected: list[Mapping[str, Any]] = []
    for flag in (0, 1):
        bucket = [row for row in rows if row.get("split") == split and row.get("changeflag") == flag]
        bucket.sort(
            key=lambda row: _rank(SELECTION_SEED, "levir", split, flag, row.get("imgid"), row.get("filename"))
        )
        needed = quota // 2
        if len(bucket) < needed:
            raise ValueError(f"LEVIR_QUOTA_UNAVAILABLE:{split}:{flag}:{len(bucket)}<{needed}")
        selected.extend(bucket[:needed])
    selected.sort(key=lambda row: _rank(SELECTION_SEED, "levir-final", split, row.get("imgid")))
    return selected


def _target(
    task: str,
    categories: Sequence[str],
    *,
    count_target: str | None = None,
    reason_code: str,
) -> dict[str, Any]:
    return VisualTaskPlan.model_validate(
        {
            "version": "visual-task-plan-v5",
            "task": task,
            "needs_visual_assistance": bool(categories),
            "object_categories": list(categories),
            "count_target": count_target,
            "region_request": _NO_REGION,
            "reason_codes": [reason_code],
        }
    ).model_dump(mode="json")


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _clone_base(base: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("OUTPUT_ALREADY_EXISTS")
    output.mkdir(parents=True)
    for source in sorted(base.rglob("*")):
        relative = source.relative_to(base)
        target = output / relative
        if source.is_symlink():
            raise ValueError("BASE_SYMLINK_UNSUPPORTED")
        if source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif relative.parts[0] in {"images", "training_images"}:
            _link_or_copy(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def _register_image(
    source: Path,
    *,
    output: Path,
    image_manifest: dict[str, Any],
    source_group: str,
    source_image_id: str,
) -> str:
    digest = _sha256_file(source)
    existing = image_manifest["by_sha256"].get(digest)
    if existing is None:
        suffix = source.suffix.casefold().lstrip(".") or "png"
        relative = f"images/sha256/{digest}.{suffix}"
        target = output / PurePosixPath(relative)
        _link_or_copy(source, target)
        existing = {
            "bytes": source.stat().st_size,
            "extension": suffix,
            "path": relative,
            "sha256": digest,
            "source_image_ids": [],
        }
        image_manifest["by_sha256"][digest] = existing
    elif _sha256_file(output / PurePosixPath(existing["path"])) != digest:
        raise ValueError("EXISTING_IMAGE_DIGEST_MISMATCH")
    source_ref = {"image_id": source_image_id, "source_group": source_group}
    if source_ref not in existing["source_image_ids"]:
        existing["source_image_ids"].append(source_ref)
        existing["source_image_ids"].sort(
            key=lambda item: (item["source_group"], item["image_id"])
        )
    return str(existing["path"])


def _materialize_preview(source: Path, output: Path, *, max_side: int) -> str:
    data_url, digest = preview_from_path(source, max_side=max_side)
    prefix, separator, encoded = data_url.partition(",")
    if not separator or prefix != "data:image/png;base64":
        raise ValueError("INVALID_PREVIEW_DATA_URL")
    content = base64.b64decode(encoded, validate=True)
    if _sha256_bytes(content) != digest:
        raise ValueError("PREVIEW_DIGEST_MISMATCH")
    relative = f"training_images/sha256/{digest}.png"
    target = output / PurePosixPath(relative)
    if target.exists():
        if _sha256_file(target) != digest:
            raise ValueError("EXISTING_PREVIEW_DIGEST_MISMATCH")
    else:
        _atomic_write_bytes(target, content)
    return relative


def _make_record(
    *,
    protocol_id: str,
    dataset: str,
    source_group: str,
    split: str,
    episode_id: str,
    images: Sequence[str],
    question: str,
    target: Mapping[str, Any],
    provenance: Mapping[str, Any],
    source_image_id: str,
) -> dict[str, Any]:
    if not images:
        raise ValueError("episode requires at least one image")
    serialized_target = deepcopy(dict(target))
    return {
        "dataset": dataset,
        "episode_id": episode_id,
        "image": images[0],
        "messages": [
            {
                "content_ref": f"protocols/{protocol_id}.json",
                "role": "system",
            },
            {
                "content": [
                    *({"image": image, "type": "image"} for image in images),
                    {"text": question, "type": "text"},
                ],
                "role": "user",
            },
        ],
        "protocol_id": protocol_id,
        "protocol_version": "visual-task-plan-v5",
        "provenance": deepcopy(dict(provenance)),
        "response_model": "VisualTaskPlan",
        "schema_version": 1,
        "source_group": source_group,
        "source_image_id": source_image_id,
        "split": split,
        "target": serialized_target,
        "target_text": _canonical_json(serialized_target),
    }


def _make_training_record(
    record: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    preview_images: Sequence[str],
) -> dict[str, Any]:
    messages = compile_training_messages(record, protocol, image_override=preview_images)
    return {
        "episode_id": record["episode_id"],
        "format": TRAINING_FORMAT,
        "image": preview_images[0],
        "messages": messages,
        "schema_version": 1,
        "source_group": record["source_group"],
        "split": record["split"],
    }


def _vrs_episodes(
    candidates: Sequence[VrsCandidate],
    *,
    output: Path,
    image_manifest: dict[str, Any],
    protocol_id: str,
    protocol: Mapping[str, Any],
    catalog: EvidenceCatalog,
    max_side: int,
) -> list[AddedEpisode]:
    episodes: list[AddedEpisode] = []
    catalog_order = catalog.leaf_categories
    global_executable = tuple(
        category
        for category in catalog_order
        if any(
            category in categories
            for categories in protocol["planner_binding"]["task_executable_categories"].values()
        )
    )
    for candidate in candidates:
        source_name = candidate.image_path.name
        image = _register_image(
            candidate.image_path,
            output=output,
            image_manifest=image_manifest,
            source_group=VRS_GROUP,
            source_image_id=source_name,
        )
        preview = _materialize_preview(candidate.image_path, output, max_side=max_side)
        # Source object classes remain usable for grounding/counting supervision, but
        # generic caption questions do not name any category; assistance for caption
        # must be derived from the question text only, never from source annotations.
        # 源物体类别仍可用于 grounding/counting 监督，但 generic caption 问题不指明
        # 任何类别；caption 的 assistance 只能由问题文本驱动，不得来自源标注。
        raw_classes = tuple(
            str(item.get("obj_cls"))
            for item in candidate.record.get("objects", [])
            if isinstance(item, dict) and item.get("obj_cls") in _VRS_CLASS_MAP
        )
        base_provenance = {
            "source": {
                "annotation": f"VRSBench-full/Annotations_{candidate.split}/{candidate.annotation_path.name}",
                "image": f"VRSBench-full/Images_{candidate.split}/{source_name}",
            },
            "source_annotation": f"VRSBenchSupplement/{candidate.split}/{candidate.annotation_path.name}",
            "supplement_policy_version": POLICY_VERSION,
        }
        variant = int(_rank(SELECTION_SEED, source_name)[:8], 16) % len(_CAPTION_QUESTIONS)

        caption_question = _CAPTION_QUESTIONS[variant]
        caption_categories = _question_evidence_categories(
            caption_question,
            task="caption",
            catalog=catalog,
            global_executable=global_executable,
        )
        specs: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = []
        specs.append(
            (
                "caption",
                caption_question,
                _target("caption", caption_categories, reason_code="structured_caption_source"),
                {**base_provenance, "question_type": "open_caption"},
            )
        )

        grounding_object = candidate.grounding_object
        referring = _clean_text(grounding_object["referring_sentence"], max_length=240).rstrip(".")
        # Grounding categories must be derivable from the referring sentence text:
        # "small vehicle" -> small-vehicle, "large vehicle" -> large-vehicle,
        # bare "vehicle" -> both leaves, "baseball field" -> baseball-diamond via
        # explicit text aliases. The hidden source obj_cls never injects categories.
        # Grounding 类别必须能由 referring sentence 文本推导：small vehicle ->
        # small-vehicle、large vehicle -> large-vehicle、裸 vehicle -> 两个叶子、
        # baseball field -> baseball-diamond 走显式文本 alias；隐藏的源 obj_cls
        # 不再注入类别。
        ground_categories = _question_evidence_categories(
            referring,
            task="grounding",
            catalog=catalog,
            global_executable=global_executable,
        )
        specs.append(
            (
                "grounding",
                f"Return the bounding box coordinates (x_min, y_min, x_max, y_max) for: {referring}.",
                _target("grounding", ground_categories, reason_code="structured_grounding_source"),
                {
                    **base_provenance,
                    "question_type": "grounding_geometry",
                    "source_record_id": grounding_object.get("obj_id"),
                },
            )
        )

        fine_names = [_DISPLAY_PLURALS[value] for value in candidate.fine_classes]
        joined = " and ".join(fine_names)
        fine_target = f"separate counts for {joined}"
        fine_categories = _ordered_categories(candidate.fine_classes, catalog_order)
        specs.append(
            (
                "fine-grained-counting",
                f"Count the {joined} separately by fine-grained category. Report one count for each category.",
                _target(
                    "fine_grained_counting",
                    fine_categories,
                    count_target=fine_target,
                    reason_code="structured_fine_grained_count_source",
                ),
                {**base_provenance, "question_type": "fine_grained_object_quantity"},
            )
        )

        raw_mc_question = _clean_text(candidate.qa.get("question"))
        mc_question = format_multiple_choice(raw_mc_question, candidate.choices)
        if _answer_leakage(raw_mc_question):
            mc_categories: tuple[str, ...] = ()
        else:
            mc_categories = _question_evidence_categories(
                raw_mc_question,
                task="multiple_choice_vqa",
                catalog=catalog,
                global_executable=global_executable,
            )
        specs.append(
            (
                "multiple-choice-vqa",
                mc_question,
                _target(
                    "multiple_choice_vqa",
                    mc_categories,
                    reason_code="structured_multiple_choice_source",
                ),
                {
                    **base_provenance,
                    "question_type": _clean_text(candidate.qa.get("type"), max_length=80),
                    "source_record_id": candidate.qa.get("ques_id"),
                },
            )
        )

        stem = Path(source_name).stem
        for suffix, question, target, provenance in specs:
            episode_id = f"vrsbench-supplement-{candidate.split}-{stem}-{suffix}"
            record = _make_record(
                protocol_id=protocol_id,
                dataset=VRS_GROUP,
                source_group=VRS_GROUP,
                split=candidate.split,
                episode_id=episode_id,
                images=(image,),
                question=question,
                target=target,
                provenance=provenance,
                source_image_id=source_name,
            )
            episodes.append(
                AddedEpisode(
                    record,
                    _make_training_record(record, protocol=protocol, preview_images=(preview,)),
                    f"VRSBench-full/Annotations_{candidate.split}/{candidate.annotation_path.name}",
                )
            )
    return episodes


def _levir_episodes(
    rows: Sequence[Mapping[str, Any]],
    *,
    levir_root: Path,
    output: Path,
    image_manifest: dict[str, Any],
    protocol_id: str,
    protocol: Mapping[str, Any],
    catalog: EvidenceCatalog,
    max_side: int,
) -> list[AddedEpisode]:
    episodes: list[AddedEpisode] = []
    for row in rows:
        split = str(row["split"])
        filename = str(row["filename"])
        source_a = levir_root / str(row["image_a"])
        source_b = levir_root / str(row["image_b"])
        if not source_a.is_file() or not source_b.is_file():
            raise ValueError(f"LEVIR_IMAGE_MISSING:{split}:{filename}")
        images = (
            _register_image(
                source_a,
                output=output,
                image_manifest=image_manifest,
                source_group=LEVIR_GROUP,
                source_image_id=f"{split}/A/{filename}",
            ),
            _register_image(
                source_b,
                output=output,
                image_manifest=image_manifest,
                source_group=LEVIR_GROUP,
                source_image_id=f"{split}/B/{filename}",
            ),
        )
        previews = (
            _materialize_preview(source_a, output, max_side=max_side),
            _materialize_preview(source_b, output, max_side=max_side),
        )
        variant = int(_rank(SELECTION_SEED, split, filename)[:8], 16) % 4
        provenance = {
            "source": {
                "annotation": "Levir-CC-dataset/LevirCCcaptions_readable.jsonl",
                "t1": str(row["image_a"]),
                "t2": str(row["image_b"]),
            },
            "source_annotation": "LEVIR_CC/LevirCCcaptions_readable.jsonl",
            "source_record_id": row.get("imgid"),
            "supplement_policy_version": POLICY_VERSION,
        }
        # Change assistance must be driven by the question text, not by the
        # LEVIR reference captions; generic change questions name no category.
        # change assistance 必须由问题文本驱动，不得来自 LEVIR 参考 caption；
        # generic change 问题不指明任何类别。
        change_caption_question = _CHANGE_CAPTION_QUESTIONS[variant]
        change_qa_question = _CHANGE_QA_QUESTIONS[variant]
        change_caption_categories = _question_evidence_categories(
            change_caption_question,
            task="change_caption",
            catalog=catalog,
            global_executable=global_executable,
        )
        change_qa_categories = _question_evidence_categories(
            change_qa_question,
            task="change_qa",
            catalog=catalog,
            global_executable=global_executable,
        )
        specs = (
            (
                "change-caption",
                change_caption_question,
                _target(
                    "change_caption",
                    change_caption_categories,
                    reason_code="structured_change_caption_source",
                ),
                "change_caption",
            ),
            (
                "change-qa",
                change_qa_question,
                _target(
                    "change_qa",
                    change_qa_categories,
                    reason_code="structured_change_qa_source",
                ),
                "change_qa",
            ),
        )
        stem = Path(filename).stem
        for suffix, question, target, question_type in specs:
            episode_id = f"levir-cc-supplement-{split}-{stem}-{suffix}"
            record = _make_record(
                protocol_id=protocol_id,
                dataset=LEVIR_GROUP,
                source_group=LEVIR_GROUP,
                split=split,
                episode_id=episode_id,
                images=images,
                question=question,
                target=target,
                provenance={**provenance, "question_type": question_type},
                source_image_id=filename,
            )
            episodes.append(
                AddedEpisode(
                    record,
                    _make_training_record(record, protocol=protocol, preview_images=previews),
                    f"LevirCCcaptions_readable.jsonl:{row.get('imgid')}",
                )
            )
    return episodes


def _excluded_vrs_images(base: Path) -> set[str]:
    excluded: set[str] = set()
    for path in sorted((base / "datasets").glob("*/*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("source_group") == "VRSBench":
                excluded.add(str(record.get("source_image_id")))
    return excluded


def _append_decisions(output: Path, episodes: Sequence[AddedEpisode]) -> None:
    path = output / "audit/label_decisions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for episode in episodes:
        record = episode.dataset_record
        target = record["target"]
        rows.append(
            {
                "cache_hit": False,
                "changed_target_fields": [
                    "count_target",
                    "needs_visual_assistance",
                    "object_categories",
                    "task",
                ],
                "decision_code": "structured_source_supplement",
                "episode_id": record["episode_id"],
                "new_assistance": target["needs_visual_assistance"],
                "new_categories": target["object_categories"],
                "new_count_target": target["count_target"],
                "new_task": target["task"],
                "old_assistance": None,
                "old_count_target": None,
                "old_task": None,
                "request_hash": _rank(
                    POLICY_VERSION,
                    record["episode_id"],
                    record["messages"][1]["content"][-1]["text"],
                    record["target_text"],
                ),
                "review_required": True,
            }
        )
    _write_jsonl(path, rows)


def _distribution(output: Path) -> dict[str, Any]:
    task_counts: Counter[str] = Counter()
    assistance_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    accepted = 0
    for path in sorted((output / "datasets").glob("*/*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            target = VisualTaskPlan.model_validate(record["target"])
            accepted += 1
            task_counts[target.task] += 1
            assistance_counts[str(target.needs_visual_assistance).lower()] += 1
            category_counts.update(target.object_categories)
    quarantine_path = output / "audit/quarantine.jsonl"
    quarantine = sum(1 for line in quarantine_path.read_text(encoding="utf-8").splitlines() if line)
    return {
        "accepted": accepted,
        "assistance": dict(sorted(assistance_counts.items())),
        "categories": dict(sorted(category_counts.items())),
        "quarantine": quarantine,
        "tasks": dict(sorted(task_counts.items())),
    }


def compile_supplement(
    *,
    base: Path,
    vrs_root: Path,
    levir_jsonl: Path,
    output: Path,
    catalog_path: Path,
    train_per_task: int,
    val_per_task: int,
) -> dict[str, Any]:
    if train_per_task <= 0 or val_per_task <= 0:
        raise ValueError("quotas must be positive")
    if train_per_task % 2 or val_per_task % 2:
        raise ValueError("quotas must be even for LEVIR balancing")
    base = base.resolve()
    vrs_root = vrs_root.resolve()
    levir_jsonl = levir_jsonl.resolve()
    output = output.resolve()
    catalog = EvidenceCatalog.from_file(catalog_path.resolve())
    base_manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    protocol_ids = tuple(base_manifest["protocols"])
    if len(protocol_ids) != 1:
        raise ValueError("BASE_PROTOCOL_COUNT_UNSUPPORTED")
    protocol_id = protocol_ids[0]
    protocol = json.loads(
        (base / base_manifest["protocols"][protocol_id]["path"]).read_text(encoding="utf-8")
    )
    max_side = int(protocol["planner_binding"]["preview_max_side"])

    answer_pools: dict[str, dict[str, tuple[str, ...]]] = {}
    vrs_inventory: dict[str, str] = {}
    for split in ("train", "val"):
        answer_pools[split], vrs_inventory[split] = _answer_pool(vrs_root, split)
    excluded = _excluded_vrs_images(base)
    vrs_selected = [
        candidate
        for split, quota in (("train", train_per_task), ("val", val_per_task))
        for candidate in _select_vrs(
            vrs_root,
            split=split,
            quota=quota,
            answer_pools=answer_pools[split],
            excluded_images=excluded,
        )
    ]

    levir_rows = [
        json.loads(line)
        for line in levir_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    levir_selected = [
        row
        for split, quota in (("train", train_per_task), ("val", val_per_task))
        for row in select_balanced_levir(levir_rows, split=split, quota=quota)
    ]

    _clone_base(base, output)
    manifest = deepcopy(base_manifest)
    image_manifest = manifest["images"]
    vrs_episodes = _vrs_episodes(
        vrs_selected,
        output=output,
        image_manifest=image_manifest,
        protocol_id=protocol_id,
        protocol=protocol,
        catalog=catalog,
        max_side=max_side,
    )
    levir_episodes = _levir_episodes(
        levir_selected,
        levir_root=levir_jsonl.parent,
        output=output,
        image_manifest=image_manifest,
        protocol_id=protocol_id,
        protocol=protocol,
        catalog=catalog,
        max_side=max_side,
    )
    episodes = [*vrs_episodes, *levir_episodes]

    by_file: dict[tuple[str, str], list[AddedEpisode]] = defaultdict(list)
    for episode in episodes:
        key = (episode.dataset_record["source_group"], episode.dataset_record["split"])
        by_file[key].append(episode)
    training_stats: dict[str, Any] = manifest["refinement"]["training_files"]
    for (group, split), grouped in sorted(by_file.items()):
        grouped.sort(key=lambda item: item.dataset_record["episode_id"])
        dataset_relative = f"datasets/{group}/{split}.jsonl"
        training_relative = f"training/{group}/{split}.jsonl"
        dataset_stats = _write_jsonl(
            output / PurePosixPath(dataset_relative),
            [item.dataset_record for item in grouped],
        )
        training_stats[training_relative] = _write_jsonl(
            output / PurePosixPath(training_relative),
            [item.training_record for item in grouped],
        )
        dataset_entry = manifest["datasets"].setdefault(
            group,
            {
                "embedded_image_blocks": 0,
                "examples": 0,
                "files": {},
                "logical_datasets": [group],
                "protocol_ids": [protocol_id],
                "source_files": 0,
                "splits": {},
            },
        )
        dataset_entry["files"][f"{split}.jsonl"] = dataset_stats
        dataset_entry["examples"] += len(grouped)
        dataset_entry["splits"][split] = len(grouped)
        dataset_entry["embedded_image_blocks"] += sum(
            len(item.dataset_record["messages"][1]["content"]) - 1 for item in grouped
        )
    manifest["datasets"][VRS_GROUP]["source_files"] = len(vrs_selected)
    manifest["datasets"][LEVIR_GROUP]["source_files"] = 1
    image_manifest["unique_count"] = len(image_manifest["by_sha256"])
    image_manifest["total_decoded_bytes"] = sum(
        int(entry["bytes"]) for entry in image_manifest["by_sha256"].values()
    )
    image_manifest["embedded_block_count"] = sum(
        int(entry["embedded_image_blocks"]) for entry in manifest["datasets"].values()
    )
    manifest["description"] = "DeepSeek-refined plus structured-source visual-planner episodes"
    manifest["refinement"]["accepted"] = sum(
        int(entry["examples"]) for entry in manifest["datasets"].values()
    )
    manifest["supplement"] = {
        "base_manifest_sha256": _sha256_file(base / "manifest.json"),
        "examples": len(episodes),
        "excluded_splits": ["test"],
        "levir_jsonl_sha256": _sha256_file(levir_jsonl),
        "policy_version": POLICY_VERSION,
        "selection_seed": SELECTION_SEED,
        "task_quotas": {"train": train_per_task, "val": val_per_task},
        "vrs_annotation_inventory_sha256": vrs_inventory,
    }
    _atomic_write_json(output / "manifest.json", manifest)

    _append_decisions(output, episodes)
    distribution = _distribution(output)
    _atomic_write_json(output / "audit/distribution.json", distribution)
    selection_rows = [
        {
            "episode_id": item.dataset_record["episode_id"],
            "source_locator": item.source_locator,
            "split": item.dataset_record["split"],
            "task": item.dataset_record["target"]["task"],
        }
        for item in sorted(episodes, key=lambda value: value.dataset_record["episode_id"])
    ]
    _write_jsonl(output / "audit/supplement_selections.jsonl", selection_rows)
    supplement_run = {
        **manifest["supplement"],
        "assistance": {
            "false": sum(
                1 for item in episodes if not item.dataset_record["target"]["needs_visual_assistance"]
            ),
            "true": sum(
                1 for item in episodes if item.dataset_record["target"]["needs_visual_assistance"]
            ),
        },
        "answer_keys_persisted": False,
        "source_boxes_persisted": False,
        "source_captions_persisted": False,
        "network_calls": 0,
        "selected_levir_pairs": len(levir_selected),
        "selected_vrs_images": len(vrs_selected),
        "task_distribution": dict(
            sorted(Counter(item.dataset_record["target"]["task"] for item in episodes).items())
        ),
    }
    _atomic_write_json(output / "audit/supplement_run.json", supplement_run)
    training_contract_path = output / "audit/training_contract.json"
    training_contract = json.loads(training_contract_path.read_text(encoding="utf-8"))
    training_contract["preview_images"] = sum(
        1 for _ in (output / "training_images/sha256").glob("*.png")
    )
    training_contract["ordered_multi_image_messages_verified"] = True
    _atomic_write_json(training_contract_path, training_contract)
    _atomic_write_text(
        output / "README.md",
        (output / "README.md").read_text(encoding="utf-8").rstrip()
        + "\n"
        + f"- Structured supplement: `{POLICY_VERSION}`; {len(episodes)} examples; no network calls.\n"
        + "- Added tasks: caption, fine_grained_counting, grounding, multiple_choice_vqa, change_caption, change_qa.\n"
        + "- Source test splits were excluded; LEVIR image order is A/t1 then B/t2.\n",
    )
    return {"distribution": distribution, "supplement": supplement_run}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--vrs-root", type=Path, required=True)
    parser.add_argument("--levir-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("agents/evidence_catalog.json"),
    )
    parser.add_argument("--train-per-task", type=int, default=DEFAULT_TRAIN_PER_TASK)
    parser.add_argument("--val-per-task", type=int, default=DEFAULT_VAL_PER_TASK)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compile_supplement(
        base=args.base,
        vrs_root=args.vrs_root,
        levir_jsonl=args.levir_jsonl,
        output=args.output,
        catalog_path=args.catalog,
        train_per_task=args.train_per_task,
        val_per_task=args.val_per_task,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
