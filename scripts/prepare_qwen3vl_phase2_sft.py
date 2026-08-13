#!/usr/bin/env python3
"""Prepare canonical Phase 2 SFT episodes from VRSBench phase2-train + GeoChat.

Read-only conversion of the raw annotations under data/phase2-train into
canonical training episodes (train.jsonl / validation.jsonl) plus
manifest.json and rejected.jsonl. Episodes are decoupled from
Transformers / PEFT / Qwen chat templates; a later script renders prompts.

只读解析 data/phase2-train 下的 VRSBench 与 GeoChat 原始标注，导出与
Transformers / PEFT / Qwen chat template 解耦的 canonical training
episodes。本脚本不加载模型、不调用网络、不做图像增强，也不修改原始标注。

Key contracts / 关键契约:
- VRSBench grounding: image + referring sentence -> target boxes (box_999).
- VRSBench VQA: every QA gets a box_assisted view when the image has at
  least one valid annotation box; input boxes are presented as "Available
  annotated regions" by the prompt renderer, never as complete
  question-level evidence (no fuzzy question-object binding).
- 40% self-attention augmentation: extra unboxed copies for a stable
  sha256(seed + parent_episode_id) sorted, per-source-task stratified 40%
  of box_assisted parents (train split only). Selection never replaces the
  boxed view and naturally-unboxed QAs never enter the denominator.
- GeoChat: [refer] -> target boxes; [identify] -> input boxes + text
  answer; plain conversations keep turn order. All coordinates are
  converted 0..100 -> 0..999 via round(c*999/100) and validated before and
  after conversion; GeoChat class ids are preserved in provenance only.
  Records that violate the coordinate protocol or conversation structure
  are written to rejected.jsonl with stable error codes.
- Outputs never contain machine absolute paths; every field is JSON-safe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Counter, Iterator

EPISODE_SCHEMA_VERSION = 1
DEFAULT_SEED = "phase2-sft-v1"
DEFAULT_SELF_ATTENTION_RATIO = 0.40
_CHUNK_SIZE = 1 << 20

# GeoChat box protocol: {<x1><y1><x2><y2>|<class_id>}, integer coords 0..100.
# 坐标 0..100 整数，class id 为私有 id 仅保留在 provenance。
_GC_BOX_TOKEN = re.compile(r"\{<([^>]+)><([^>]+)><([^>]+)><([^>]+)>\|([^>]*)>\}")
_GC_BOX_LIKE = re.compile(r"\{<[^{}]*\}")
_GC_INT_FIELD = re.compile(r"[+-]?\d+")
_GC_SPACE_RUN = re.compile(r"[ \t]+")
_GC_PHRASE = re.compile(r"<p>(.*?)</p>", re.DOTALL)
_GC_IMAGE_TOKEN = "<image>"

# Spatial language that breaks under rotation/flip/augmentation.
# 几何增强（旋转/翻转）后语义可能失效的空间语言；命中即 orientation_locked。
_SPATIAL_TERMS = re.compile(
    r"\b(top|bottom|left|right|north|south|east|west|middle|center|centre|"
    r"upper|lower|corner|above|below|near|beside|front|back|behind|"
    r"left[- ]?most|right[- ]?most|top[- ]?most|bottom[- ]?most)\b",
    re.IGNORECASE,
)

# VRSBench official QA types that are inherently spatial.
# VRSBench 官方 question 类型中天然带空间语义的类型。
SPATIAL_VRSBENCH_TYPES = frozenset({"object position", "object direction"})

# Stable rejection codes; one record is rejected as a whole with one code.
# 稳定拒绝码；一条记录以一个拒绝码整体拒绝。
REJECTION_CODES = frozenset({
    "missing_image_field",
    "invalid_conversation_type",
    "invalid_role_order",
    "missing_turn_text",
    "image_token_count_mismatch",
    "unparseable_box",
    "non_finite_box",
    "out_of_range_box",
    "degenerate_box",
    # VRSBench block that cannot be parsed as JSON (curated source is
    # expected to have zero; recorded instead of silently dropped).
    # VRSBench 块无法解析为 JSON 时的拒绝码（预期源数据为零）。
    "parse_error",
})

# VRSBench phase2-train file names (see data/phase2-train/VRSBench/manifest.json).
# VRSBench phase2-train 输入文件名；test_raw 只是未提取的官方评测清单，不参与训练。
_VRSBENCH_TRAIN_FILE = "VRSBench_train.jsonl"
_VRSBENCH_VAL_FILE = "VRSBench_val.jsonl"
_GEOCHAT_FILE = "GeoChat_Instruct.json"

# Validated GeoChat rejections carry stable codes; parse_error is the
# documented extension for unparseable VRSBench blocks.
# 允许的稳定拒绝码集合（parse_error 为 VRSBench 解析失败的文档化扩展）。
assert REJECTION_CODES >= {
    "missing_image_field", "invalid_conversation_type", "invalid_role_order",
    "missing_turn_text", "image_token_count_mismatch", "unparseable_box",
    "non_finite_box", "out_of_range_box", "degenerate_box",
}


class BoxProtocolError(ValueError):
    """GeoChat box token violates the coordinate protocol.
    GeoChat 框 token 违反坐标协议。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


# ---------------------------------------------------------------------------
# Stream / IO helpers
# 流式读取与原子写出工具
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file. / 返回文件的十六进制 sha256。"""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    """Write text through a temp file then atomically replace.
    通过临时文件写入后原子替换，避免中断留下半个 JSON。"""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def iter_json_array(path: Path, hasher: Any | None = None) -> Iterator[Any]:
    """Stream a top-level JSON array without materializing all records.

    Reads the file in chunks and decodes one element at a time so the whole
    (up to ~260 MB) GeoChat file is never held in memory twice. When hasher
    is given, every chunk read is fed into it (input checksum).
    """
    decoder = json.JSONDecoder()
    buf = ""
    with open(path, "r", encoding="utf-8") as fh:
        # Skip whitespace and consume the leading '['.
        # 跳过空白并消费开头的 '['。
        while True:
            chunk = fh.read(_CHUNK_SIZE)
            if not chunk:
                raise ValueError(f"not a JSON array: {path.name}")
            if hasher is not None:
                hasher.update(chunk.encode("utf-8"))
            buf += chunk
            stripped = buf.lstrip()
            if stripped.startswith("["):
                buf = stripped[1:]
                break
            if len(stripped) > 4:
                raise ValueError(f"not a JSON array: {path.name}")
        while True:
            stripped = buf.lstrip()
            if not stripped:
                chunk = fh.read(_CHUNK_SIZE)
                if not chunk:
                    raise ValueError(f"truncated JSON array: {path.name}")
                if hasher is not None:
                    hasher.update(chunk.encode("utf-8"))
                buf = chunk
                continue
            if stripped.startswith("]"):
                return
            if stripped.startswith(","):
                buf = stripped[1:]
                continue
            try:
                obj, end = decoder.raw_decode(stripped)
            except json.JSONDecodeError:
                chunk = fh.read(_CHUNK_SIZE)
                if not chunk:
                    raise ValueError(f"truncated JSON array: {path.name}")
                if hasher is not None:
                    hasher.update(chunk.encode("utf-8"))
                buf += chunk
                continue
            yield obj
            buf = stripped[end:]


def iter_vrsbench_records(path: Path) -> Iterator[tuple[int, Any | None]]:
    """Yield (index, record) per image; record is None on a parse error.

    Supports both the pretty-printed blank-line separated JSON blocks and
    standard single-line JSONL. The format is detected from the first line.
    """
    with open(path, "r", encoding="utf-8") as fh:
        first = fh.readline()
        if not first.strip():
            return
        try:
            json.loads(first)
            single_line = True
        except json.JSONDecodeError:
            single_line = False
        if single_line:
            index = 1
            yield index, json.loads(first)
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                index += 1
                try:
                    yield index, json.loads(line)
                except json.JSONDecodeError:
                    yield index, None
        else:
            buf = [first]
            index = 0
            for line in fh:
                if line.strip() == "":
                    if buf:
                        index += 1
                        text = "".join(buf)
                        buf = []
                        try:
                            yield index, json.loads(text)
                        except json.JSONDecodeError:
                            yield index, None
                else:
                    buf.append(line)
            if buf:
                index += 1
                text = "".join(buf)
                try:
                    yield index, json.loads(text)
                except json.JSONDecodeError:
                    yield index, None


def _write_json_line(fh, obj: dict) -> None:
    """Serialize one JSON-safe dict as a single line (no NaN/Infinity).
    序列化一条 JSON-safe 字典为单行；allow_nan=False 强制拒绝 NaN/Infinity。"""
    fh.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")


def _write_rejected(fh, dataset: str, source_record_id: str, reason: str) -> None:
    """Write one stable rejection entry (no raw exception text).
    写入一条稳定拒绝记录（不含原始异常全文）。"""
    _write_json_line(fh, {
        "dataset": dataset,
        "source_record_id": source_record_id,
        "reason": reason,
    })


# ---------------------------------------------------------------------------
# Episode registration (single run per process)
# Episode 登记（单进程单次运行）
# ---------------------------------------------------------------------------

_SEEN_EPISODE_IDS: set[str] = set()


def yield_episode(episode: dict, writer: Callable[[dict], None]) -> None:
    """Register an episode (unique id) and write its JSON line.
    登记 episode 的唯一 id 并写出其 JSON 行。"""
    episode_id = episode["episode_id"]
    if episode_id in _SEEN_EPISODE_IDS:
        raise ValueError(f"duplicate episode_id: {episode_id}")
    _SEEN_EPISODE_IDS.add(episode_id)
    writer(episode)


# ---------------------------------------------------------------------------
# GeoChat parsing
# GeoChat 解析
# ---------------------------------------------------------------------------


def parse_geochat_boxes(text: str) -> list[dict]:
    """Parse and validate every GeoChat box token in text.

    Returns box entries (xyxy_999 + provenance class id). Raises
    BoxProtocolError with a stable code on the first invalid token.
    A box-like fragment that does not match the strict protocol is also an
    error, so malformed coordinates can never silently leak into episodes.
    """
    tokens = list(_GC_BOX_TOKEN.finditer(text))
    if not tokens:
        for frag in _GC_BOX_LIKE.finditer(text):
            # Box-looking fragment that failed the strict protocol.
            # 形似框但不符合严格协议的片段。
            raise BoxProtocolError(
                "unparseable_box",
                f"malformed box token: {frag.group(0)[:60]!r}",
            )
        return []
    boxes: list[dict] = []
    for m in tokens:
        fields = m.groups()[:4]
        # The class group may include the leading '<' of the class tag
        # (e.g. '<11>'), so strip it to keep only the id value.
        # class 分组可能包含 class 标签的开头 '<'（如 '<11>'），剥离后只留 id。
        class_id = m.group(5).lstrip("<")
        ints: list[int] = []
        for raw in fields:
            if not _GC_INT_FIELD.fullmatch(raw):
                if re.search(r"\d", raw):
                    # Numeric-looking but not an integer (e.g. "1.5", "nan").
                    # 形似数字但不是整数（如 "1.5"）。
                    raise BoxProtocolError(
                        "non_finite_box", f"non-integer coordinate: {raw!r}"
                    )
                raise BoxProtocolError(
                    "unparseable_box", f"non-numeric coordinate: {raw!r}"
                )
            ints.append(int(raw))
        x1, y1, x2, y2 = ints
        if any(c < 0 or c > 100 for c in ints):
            raise BoxProtocolError(
                "out_of_range_box", f"coordinate outside 0..100: {ints!r}"
            )
        if x1 >= x2 or y1 >= y2:
            raise BoxProtocolError(
                "degenerate_box", f"degenerate box before conversion: {ints!r}"
            )
        xyxy_999 = [round(c * 999 / 100) for c in ints]
        if not (xyxy_999[0] < xyxy_999[2] and xyxy_999[1] < xyxy_999[3]):
            raise BoxProtocolError(
                "degenerate_box",
                f"degenerate box after conversion: {xyxy_999!r}",
            )
        boxes.append({
            "xyxy_999": xyxy_999,
            "label": "",
            "description": "",
            "source_class_id": class_id,
        })
    return boxes


def _strip_box_tokens(text: str, tokens: list[Any]) -> str:
    """Remove box tokens and collapse surrounding spaces/tabs.
    移除框 token 并规整其周边空格（保留换行与其余原文）。"""
    out = text
    for m in tokens:
        out = out.replace(m.group(0), " ")
    return _GC_SPACE_RUN.sub(" ", out).strip()


def _preceding_phrase(text: str, token_start: int) -> str:
    """Return the <p>...</p> phrase immediately preceding a box token.
    返回紧邻框 token 之前的 <p>...</p> 短语（仅允许空白间隔）。"""
    head = text[:token_start]
    for m in reversed(list(_GC_PHRASE.finditer(head))):
        tail = head[m.end():]
        if tail.strip() == "":
            return m.group(1).strip()
        break
    return ""


def _augmentation_policy(texts: list[str]) -> dict:
    """Conservative geometry policy: spatial language -> orientation_locked.
    保守几何策略：含空间语言即 orientation_locked，否则 geometry_safe。"""
    if any(_SPATIAL_TERMS.search(t) for t in texts if t):
        return {"geometry": "orientation_locked", "reason": "spatial_language"}
    return {"geometry": "geometry_safe", "reason": "no_spatial_language"}


def _geochat_episode_id(index: int, rec_id: str) -> str:
    """GeoChat ids are not unique in the source, so the record ordinal
    prefixes the id to guarantee a globally unique, machine-stable id.
    GeoChat 源 id 不唯一，因此以记录序号为前缀保证全局唯一且跨机器稳定。"""
    return f"geochat/train/{index}/{rec_id}"


def build_geochat_episode(index: int, rec: Any) -> tuple[dict | None, str | None]:
    """Convert one GeoChat record into one episode or a rejection code.

    Returns (episode, None) on success or (None, rejection_code).
    """
    if not isinstance(rec, dict):
        return None, "invalid_conversation_type"
    image = rec.get("image")
    if not isinstance(image, str) or not image.strip():
        return None, "missing_image_field"
    parts = image.split("/")
    if (
        image.startswith(("/", "\\"))
        or "\\" in image
        or ".." in parts
        or (parts and ":" in parts[0])
    ):
        # Unsafe image path (absolute, backslash, .. escape, drive letter)
        # is treated as a missing/unsafe image field.
        # 不安全图片路径（绝对路径/反斜杠/.. 逃逸/盘符）按缺失字段拒绝。
        return None, "missing_image_field"
    convs = rec.get("conversations")
    if not isinstance(convs, list):
        return None, "invalid_conversation_type"
    if len(convs) < 2:
        return None, "invalid_role_order"
    for c in convs:
        if not isinstance(c, dict) or c.get("from") not in ("human", "gpt"):
            return None, "invalid_conversation_type"
    if convs[0].get("from") != "human":
        return None, "invalid_role_order"
    for i, c in enumerate(convs):
        if (i % 2 == 0 and c.get("from") != "human") or (
            i % 2 == 1 and c.get("from") != "gpt"
        ):
            return None, "invalid_role_order"
    if len(convs) % 2 != 0:
        # Trailing human turn without an assistant answer.
        # 末尾多余一条无人回答的 human 回合。
        return None, "invalid_role_order"
    texts = [c.get("value") for c in convs]
    if any(not isinstance(t, str) or not t.strip() for t in texts):
        return None, "missing_turn_text"
    if sum(t.count(_GC_IMAGE_TOKEN) for t in texts) != 1:
        return None, "image_token_count_mismatch"

    human_texts = [texts[i] for i in range(0, len(texts), 2)]
    joined_human = "\n".join(human_texts)
    has_refer = "[refer]" in joined_human
    has_identify = "[identify]" in joined_human
    if has_refer and has_identify:
        return None, "invalid_conversation_type"
    if has_refer:
        task_kind = "geochat_refer"
        source_task = "refer"
    elif has_identify:
        task_kind = "geochat_identify"
        source_task = "identify"
    else:
        task_kind = "geochat_conversation"
        source_task = "conversation"

    turns: list[dict] = []
    policy_texts: list[str] = []
    for i in range(0, len(convs), 2):
        human_value = texts[i]
        gpt_value = texts[i + 1]
        # User-side boxes (identify): structured into input_boxes.
        # user 侧框（identify）结构化到 input_boxes。
        try:
            user_tokens = list(_GC_BOX_TOKEN.finditer(human_value))
            input_boxes = parse_geochat_boxes(human_value)
        except BoxProtocolError as exc:
            return None, exc.code
        user_text = _strip_box_tokens(human_value, user_tokens)
        if task_kind == "geochat_refer":
            # Assistant answer is pure box protocol; kept read-only here and
            # re-rendered by the later prompt file from structured boxes.
            # refer 的 assistant 回答是纯框协议，此处保持只读原样，
            # 由后续渲染文件基于结构化框重新渲染。
            try:
                target_boxes = parse_geochat_boxes(gpt_value)
            except BoxProtocolError as exc:
                return None, exc.code
            if not target_boxes:
                return None, "unparseable_box"
            assistant_text = gpt_value.strip()
        else:
            try:
                gpt_tokens = list(_GC_BOX_TOKEN.finditer(gpt_value))
                parsed = parse_geochat_boxes(gpt_value)
            except BoxProtocolError as exc:
                return None, exc.code
            if task_kind == "geochat_identify":
                if not input_boxes:
                    return None, "unparseable_box"
                target_boxes: list[dict] = []
            else:
                # Inline boxes in plain answers are target regions; label is
                # the immediately preceding <p> phrase when present.
                # 普通回答中的内联框作为 target 区域，label 取紧邻的 <p> 短语。
                target_boxes = []
                for tm, entry in zip(gpt_tokens, parsed):
                    phrase = _preceding_phrase(gpt_value, tm.start())
                    if phrase:
                        entry["label"] = phrase
                        entry["description"] = phrase
                    target_boxes.append(entry)
            assistant_text = _strip_box_tokens(gpt_value, gpt_tokens)
        turns.append({
            "user_text": user_text,
            "assistant_text": assistant_text,
            "input_boxes": input_boxes,
            "target_boxes": target_boxes,
        })
        policy_texts.extend([user_text, assistant_text])

    rec_id = rec.get("id") or f"record_{index}"
    episode_id = _geochat_episode_id(index, rec_id)
    episode = {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "episode_id": episode_id,
        "parent_episode_id": episode_id,
        "dataset": "GeoChat",
        "split": "train",
        "image_source": "geochat",
        "image": image,
        "task_kind": task_kind,
        "source_task": source_task,
        "turns": turns,
        "augmentation_policy": _augmentation_policy(policy_texts),
        "provenance": {
            "source_record_id": f"geochat/{index}/{rec_id}",
            "kind": task_kind,
            "view": "raw",
            "n_turns": len(turns),
            "multi_turn": len(turns) > 1,
        },
    }
    return episode, None


# ---------------------------------------------------------------------------
# VRSBench conversion
# VRSBench 转换
# ---------------------------------------------------------------------------


def _valid_vrsbench_box(obj: dict) -> list[int] | None:
    """Return box_999 as validated ints or None when unusable.
    返回经校验的 box_999 整数列表，不可用时返回 None（只审计不修复）。"""
    if obj.get("box_valid") is not True:
        return None
    raw = obj.get("box_999")
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        box = [int(v) for v in raw]
    except (TypeError, ValueError):
        return None
    if any(v < 0 or v > 999 for v in box):
        return None
    if not (box[0] < box[2] and box[1] < box[3]):
        return None
    return box


def _vrsbench_box_entry(obj: dict, box_999: list[int]) -> dict:
    """Build the canonical box entry for one VRSBench object.
    为 VRSBench object 构建 canonical 框条目。"""
    return {
        "xyxy_999": box_999,
        "label": obj.get("obj_cls") or "",
        "description": obj.get("referring_sentence") or "",
        "source_object_id": obj.get("obj_id"),
    }


def _vrsbench_qa_policy(qa_type: str, question: str, answer: str) -> dict:
    """Spatial QA types are always orientation_locked; otherwise text check.
    空间类型 QA 一律 orientation_locked；其余按文本检查。"""
    if qa_type in SPATIAL_VRSBENCH_TYPES:
        return {"geometry": "orientation_locked", "reason": "source_type_spatial"}
    return _augmentation_policy([question, answer])


def process_vrsbench_record(
    rec: dict,
    out_split: str,
    stats: dict,
    sa_candidates: list[dict],
    writer: Callable[[dict], None],
) -> None:
    """Convert one VRSBench per-image record into episodes.

    Emits: one vrsbench_grounding episode per valid object; one
    vqa_box_assisted episode per QA when the image has at least one valid
    box (with self-attention candidates collected for train); otherwise one
    vqa_naturally_unboxed episode per QA. Invalid boxes are audited only.
    """
    rec_id = rec.get("id") or ""
    split_key = "train" if out_split == "train" else "validation"
    vs = stats["vrsbench"][split_key]
    vs["records"] += 1
    image = rec.get("image") or ""
    objects = rec.get("objects") or []
    qa_pairs = rec.get("qa_pairs") or []
    qa_occ: Counter[Any] = Counter()

    valid_boxes: list[tuple[dict, list[int]]] = []
    for obj in objects:
        vs["objects"] += 1
        box = _valid_vrsbench_box(obj)
        if box is None:
            vs["invalid_objects"] += 1
            continue
        valid_boxes.append((obj, box))

    for obj, box in valid_boxes:
        vs["valid_objects"] += 1
        episode_id = f"{rec_id}/obj/{obj.get('obj_id')}"
        episode = {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "episode_id": episode_id,
            "parent_episode_id": episode_id,
            "dataset": "VRSBench",
            "split": out_split,
            "image_source": "vrsbench",
            "image": image,
            "task_kind": "vrsbench_grounding",
            "source_task": "grounding",
            "turns": [{
                "user_text": obj.get("referring_sentence") or "",
                "assistant_text": "",
                "input_boxes": [],
                "target_boxes": [_vrsbench_box_entry(obj, box)],
            }],
            "augmentation_policy": _augmentation_policy(
                [obj.get("referring_sentence") or ""]
            ),
            "provenance": {
                "source_record_id": rec_id,
                "object_id": obj.get("obj_id"),
                "view": "grounding",
                "source_class": obj.get("obj_cls") or "",
            },
        }
        vs["by_task_kind"]["vrsbench_grounding"] += 1
        yield_episode(episode, writer)

    for qa in qa_pairs:
        vs["qa_pairs"] += 1
        question = qa.get("question") or ""
        answer = qa.get("answer") or ""
        source_task = qa.get("task") or ""
        qid = qa.get("ques_id")
        # A few source records repeat a ques_id inside one image; the
        # occurrence ordinal keeps episode ids globally unique while the
        # first occurrence keeps the documented id shape (qa/{ques_id}).
        # 少数源记录在同一图内重复 ques_id；出现序次后缀保证全局唯一，
        # 首次出现保持文档化的 id 形态（qa/{ques_id}）。
        qa_occ[qid] += 1
        parent_id = (
            f"{rec_id}/qa/{qid}"
            if qa_occ[qid] == 1
            else f"{rec_id}/qa/{qid}.{qa_occ[qid]}"
        )
        provenance = {
            "source_record_id": rec_id,
            "question_id": qa.get("ques_id"),
            "source_type": qa.get("type") or "",
        }
        policy = _vrsbench_qa_policy(qa.get("type") or "", question, answer)
        if valid_boxes:
            episode_id = f"{parent_id}/box_assisted"
            episode = {
                "schema_version": EPISODE_SCHEMA_VERSION,
                "episode_id": episode_id,
                "parent_episode_id": parent_id,
                "dataset": "VRSBench",
                "split": out_split,
                "image_source": "vrsbench",
                "image": image,
                "task_kind": "vqa_box_assisted",
                "source_task": source_task,
                "turns": [{
                    "user_text": question,
                    "assistant_text": answer,
                    "input_boxes": [
                        _vrsbench_box_entry(obj, box) for obj, box in valid_boxes
                    ],
                    "target_boxes": [],
                }],
                "augmentation_policy": policy,
                "provenance": {**provenance, "view": "box_assisted"},
            }
            vs["by_task_kind"]["vqa_box_assisted"] += 1
            yield_episode(episode, writer)
            if out_split == "train":
                # Self-attention candidate: unboxed extra copy, selected later
                # by a stable per-stratum 40% rule. Coordinates never enter
                # the user prompt; only a summary stays in provenance audit.
                # self-attention 候选：无框副本，稍后按稳定分层 40% 规则选择；
                # 坐标不进入 user prompt，provenance audit 只保留摘要。
                sa_episode = {
                    "schema_version": EPISODE_SCHEMA_VERSION,
                    "episode_id": f"{parent_id}/self_attention",
                    "parent_episode_id": parent_id,
                    "dataset": "VRSBench",
                    "split": "train",
                    "image_source": "vrsbench",
                    "image": image,
                    "task_kind": "vqa_self_attention",
                    "source_task": source_task,
                    "turns": [{
                        "user_text": question,
                        "assistant_text": answer,
                        "input_boxes": [],
                        "target_boxes": [],
                    }],
                    "augmentation_policy": policy,
                    "provenance": {
                        **provenance,
                        "view": "self_attention",
                        "audit": {
                            "box_summary": {
                                "count": len(valid_boxes),
                                "labels": [
                                    obj.get("obj_cls") or "" for obj, _ in valid_boxes
                                ],
                            }
                        },
                    },
                }
                sa_candidates.append({
                    "source_task": source_task,
                    "parent_episode_id": parent_id,
                    "episode": sa_episode,
                })
        else:
            episode_id = f"{parent_id}/naturally_unboxed"
            episode = {
                "schema_version": EPISODE_SCHEMA_VERSION,
                "episode_id": episode_id,
                "parent_episode_id": parent_id,
                "dataset": "VRSBench",
                "split": out_split,
                "image_source": "vrsbench",
                "image": image,
                "task_kind": "vqa_naturally_unboxed",
                "source_task": source_task,
                "turns": [{
                    "user_text": question,
                    "assistant_text": answer,
                    "input_boxes": [],
                    "target_boxes": [],
                }],
                "augmentation_policy": policy,
                "provenance": {**provenance, "view": "naturally_unboxed"},
            }
            vs["by_task_kind"]["vqa_naturally_unboxed"] += 1
            yield_episode(episode, writer)


def select_self_attention(
    candidates: list[dict], ratio: float, seed: str
) -> tuple[list[dict], dict[str, dict[str, int | float]]]:
    """Deterministic per-source-task stratified selection of 40% copies.

    Within each source_task stratum, parents are sorted by
    sha256(seed + parent_episode_id) and the first round(ratio * N) are
    selected. No Python hash() and no runtime randomness.
    每层按 sha256(seed + parent_episode_id) 排序并取前 round(ratio*N) 条。
    """
    strata: dict[str, list[dict]] = {}
    for cand in candidates:
        strata.setdefault(cand["source_task"], []).append(cand)
    episodes: list[dict] = []
    per_stratum: dict[str, dict[str, int | float]] = {}
    for task in sorted(strata):
        group = strata[task]
        group.sort(
            key=lambda c: hashlib.sha256(
                (seed + c["parent_episode_id"]).encode("utf-8")
            ).hexdigest()
        )
        n = len(group)
        k = round(ratio * n)
        per_stratum[task] = {
            "parents": n,
            "selected": k,
            "actual_ratio": k / n if n else 0.0,
        }
        for cand in group[:k]:
            episodes.append(cand["episode"])
    return episodes, per_stratum


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser. / 构建 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare canonical Phase 2 SFT episodes (train.jsonl / "
            "validation.jsonl / manifest.json / rejected.jsonl) from "
            "VRSBench phase2-train and GeoChat_Instruct.json. Read-only, "
            "offline, deterministic. 从 VRSBench phase2-train 与 GeoChat 生成 "
            "canonical 训练 episodes；只读、离线、确定性。"
        )
    )
    parser.add_argument("--vrsbench-dir", required=True, type=Path,
                        help="Directory containing VRSBench_train.jsonl and VRSBench_val.jsonl.")
    parser.add_argument("--geochat-file", required=True, type=Path,
                        help="Path to GeoChat_Instruct.json (JSON array).")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory for train.jsonl / validation.jsonl / manifest.json / rejected.jsonl (must not be a source data dir).")
    parser.add_argument("--self-attention-ratio", type=float,
                        default=DEFAULT_SELF_ATTENTION_RATIO,
                        help=f"Extra unboxed copy ratio for train box-assisted VQA parents (default: {DEFAULT_SELF_ATTENTION_RATIO}).")
    parser.add_argument("--seed", default=DEFAULT_SEED,
                        help=f"Stable seed for the stratified self-attention selection (default: {DEFAULT_SEED!r}).")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. / 入口。"""
    # Reset run-state so main() is safe to call more than once per process
    # (unit tests run several pipelines in one process).
    # 重置运行状态，保证同一进程内多次调用 main() 安全（测试会多次运行）。
    _SEEN_EPISODE_IDS.clear()
    args = build_parser().parse_args(argv)
    ratio = args.self_attention_ratio
    if not (0.0 < ratio <= 1.0):
        print(f"error: --self-attention-ratio must be in (0, 1], got {ratio}", file=sys.stderr)
        return 1
    vrsbench_dir = Path(args.vrsbench_dir).resolve()
    geochat_file = Path(args.geochat_file).resolve()
    out_dir = Path(args.output_dir)
    out_dir_resolved = out_dir.resolve()
    if not vrsbench_dir.is_dir():
        print(f"error: vrsbench dir not found: {vrsbench_dir}", file=sys.stderr)
        return 1
    if not geochat_file.is_file():
        print(f"error: geochat file not found: {geochat_file}", file=sys.stderr)
        return 1
    if out_dir_resolved == vrsbench_dir or out_dir_resolved == geochat_file.parent:
        print(
            "error: --output-dir must not be a source data directory "
            "(refusing to write outputs next to raw annotations)",
            file=sys.stderr,
        )
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)

    vrsbench_train_path = vrsbench_dir / _VRSBENCH_TRAIN_FILE
    vrsbench_val_path = vrsbench_dir / _VRSBENCH_VAL_FILE
    for p in (vrsbench_train_path, vrsbench_val_path):
        if not p.is_file():
            print(f"error: missing input file: {p.name}", file=sys.stderr)
            return 1

    train_tmp = out_dir / f".train.jsonl.{os.getpid()}.tmp"
    val_tmp = out_dir / f".validation.jsonl.{os.getpid()}.tmp"
    rej_tmp = out_dir / f".rejected.jsonl.{os.getpid()}.tmp"

    stats: dict[str, Any] = {
        "vrsbench": {
            "train": {
                "records": 0, "objects": 0, "valid_objects": 0,
                "invalid_objects": 0, "qa_pairs": 0,
                "by_task_kind": Counter(),
            },
            "validation": {
                "records": 0, "objects": 0, "valid_objects": 0,
                "invalid_objects": 0, "qa_pairs": 0,
                "by_task_kind": Counter(),
            },
        },
        "geochat": {
            "records": 0,
            "by_kind": Counter(),
            "multi_turn_by_kind": Counter(),
            "by_task_kind": Counter(),
        },
        "rejected": {"VRSBench": Counter(), "GeoChat": Counter()},
        "sa": {"candidates": 0},
    }

    sa_candidates: list[dict] = []
    sa_strata: dict[str, dict[str, int | float]] = {}
    geochat_sha = ""
    try:
        with open(train_tmp, "w", encoding="utf-8", newline="\n") as train_fh, \
             open(val_tmp, "w", encoding="utf-8", newline="\n") as val_fh, \
             open(rej_tmp, "w", encoding="utf-8", newline="\n") as rej_fh:
            train_writer = lambda ep: _write_json_line(train_fh, ep)  # noqa: E731
            val_writer = lambda ep: _write_json_line(val_fh, ep)  # noqa: E731
            rej_writer = lambda ds, sid, reason: _write_rejected(rej_fh, ds, sid, reason)  # noqa: E731

            # --- VRSBench train -------------------------------------------
            for index, rec in iter_vrsbench_records(vrsbench_train_path):
                if rec is None:
                    stats["rejected"]["VRSBench"]["parse_error"] += 1
                    rej_writer(
                        "VRSBench", f"vrsbench/train/block_{index}", "parse_error",
                    )
                    continue
                if not isinstance(rec, dict) or not rec.get("image"):
                    stats["rejected"]["VRSBench"]["missing_image_field"] += 1
                    rec_id = (
                        rec.get("id") if isinstance(rec, dict) else None
                    ) or f"vrsbench/train/block_{index}"
                    rej_writer("VRSBench", rec_id, "missing_image_field")
                    continue
                process_vrsbench_record(rec, "train", stats, sa_candidates, train_writer)
            stats["sa"]["candidates"] = len(sa_candidates)

            # --- self-attention extra copies (train only) -----------------
            sa_episodes, sa_strata = select_self_attention(
                sa_candidates, ratio, args.seed
            )
            for episode in sa_episodes:
                stats["vrsbench"]["train"]["by_task_kind"]["vqa_self_attention"] += 1
                yield_episode(episode, train_writer)

            # --- GeoChat (train only; no official split) ------------------
            geochat_hasher = hashlib.sha256()
            for index, rec in enumerate(
                iter_json_array(geochat_file, geochat_hasher), start=1
            ):
                stats["geochat"]["records"] += 1
                episode, code = build_geochat_episode(index, rec)
                if code is not None:
                    stats["rejected"]["GeoChat"][code] += 1
                    rec_id = (
                        rec.get("id") if isinstance(rec, dict) else None
                    ) or f"record_{index}"
                    rej_writer("GeoChat", f"geochat/{index}/{rec_id}", code)
                    continue
                gs = stats["geochat"]
                gs["by_kind"][episode["task_kind"]] += 1
                if episode["provenance"].get("multi_turn"):
                    gs["multi_turn_by_kind"][episode["task_kind"]] += 1
                gs["by_task_kind"][episode["task_kind"]] += 1
                yield_episode(episode, train_writer)
            geochat_sha = geochat_hasher.hexdigest()

            # --- VRSBench validation --------------------------------------
            for index, rec in iter_vrsbench_records(vrsbench_val_path):
                if rec is None:
                    stats["rejected"]["VRSBench"]["parse_error"] += 1
                    rej_writer(
                        "VRSBench", f"vrsbench/val/block_{index}", "parse_error",
                    )
                    continue
                if not isinstance(rec, dict) or not rec.get("image"):
                    stats["rejected"]["VRSBench"]["missing_image_field"] += 1
                    rec_id = (
                        rec.get("id") if isinstance(rec, dict) else None
                    ) or f"vrsbench/val/block_{index}"
                    rej_writer("VRSBench", rec_id, "missing_image_field")
                    continue
                process_vrsbench_record(rec, "validation", stats, sa_candidates, val_writer)
    except (ValueError, OSError) as exc:
        for p in (train_tmp, val_tmp, rej_tmp):
            p.unlink(missing_ok=True)
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # --- finalize outputs (atomic replace) --------------------------------
    vrsbench_train_sha = sha256_file(vrsbench_train_path)
    vrsbench_val_sha = sha256_file(vrsbench_val_path)
    for tmp, name in (
        (train_tmp, "train.jsonl"),
        (val_tmp, "validation.jsonl"),
        (rej_tmp, "rejected.jsonl"),
    ):
        os.replace(tmp, out_dir / name)
    train_sha = sha256_file(out_dir / "train.jsonl")
    val_sha = sha256_file(out_dir / "validation.jsonl")
    rej_sha = sha256_file(out_dir / "rejected.jsonl")

    # --- manifest ----------------------------------------------------------
    geochat_total = stats["geochat"]["records"]
    geochat_rejected = sum(stats["rejected"]["GeoChat"].values())
    geochat_episodes = sum(stats["geochat"]["by_task_kind"].values())
    vrsb_train = stats["vrsbench"]["train"]
    vrsb_val = stats["vrsbench"]["validation"]
    vrsbench_rejected = sum(stats["rejected"]["VRSBench"].values())
    sa_total_selected = sum(s["selected"] for s in sa_strata.values())
    combined_train: Counter[str] = Counter()
    combined_train.update(vrsb_train["by_task_kind"])
    combined_train.update(stats["geochat"]["by_task_kind"])

    manifest: dict[str, Any] = {
        "schema_version": EPISODE_SCHEMA_VERSION,
        "tool": "scripts/prepare_qwen3vl_phase2_sft.py",
        "generation": {
            "seed": args.seed,
            "self_attention_ratio": ratio,
            "coordinate_rule": {
                "input": (
                    "VRSBench box_999 (0..999 integer xyxy); "
                    "GeoChat {<x1><y1><x2><y2>|<class_id>} integer coords 0..100"
                ),
                "output": "0..999 integer xyxy",
                "geochat_conversion": "round(c * 999 / 100)",
                "validation": "range 0..100 and x1<x2 / y1<y2 checked before and after conversion",
                "version": 1,
            },
            "augmentation_policy": {
                "geometry_values": ["orientation_locked", "geometry_safe"],
                "orientation_locked_when": (
                    "VRSBench QA type in {object position, object direction} "
                    "or spatial terms appear in any turn text "
                    "(top/bottom/left/right/north/south/east/west/middle/center/"
                    "upper/lower/corner/above/below/near/beside/front/back/behind/"
                    "leftmost/rightmost/topmost/bottommost); unsure -> locked"
                ),
                "note": (
                    "imaging-degradation augmentations (brightness, blur, noise, "
                    "low contrast, JPEG, vignette) are coordinate-preserving and "
                    "remain allowed for orientation_locked episodes"
                ),
            },
            "episode_order": (
                "train.jsonl: VRSBench train episodes, then selected "
                "self-attention extra copies, then GeoChat episodes; "
                "validation.jsonl: VRSBench validation episodes only"
            ),
        },
        "inputs": {
            "vrsbench_train": {
                "file": _VRSBENCH_TRAIN_FILE,
                "sha256": vrsbench_train_sha,
            },
            "vrsbench_validation": {
                "file": _VRSBENCH_VAL_FILE,
                "sha256": vrsbench_val_sha,
            },
            "geochat": {"file": _GEOCHAT_FILE, "sha256": geochat_sha},
        },
        "outputs": {
            "train.jsonl_sha256": train_sha,
            "validation.jsonl_sha256": val_sha,
            "rejected.jsonl_sha256": rej_sha,
            "manifest.json_sha256": "",  # self checksum filled below
        },
        "counts": {
            "train": {
                "by_task_kind": dict(combined_train),
                "total": sum(combined_train.values()),
            },
            "validation": {
                "by_task_kind": dict(vrsb_val["by_task_kind"]),
                "total": sum(vrsb_val["by_task_kind"].values()),
            },
        },
        "dataset_counts": {
            "VRSBench": {
                "train": {
                    "records": vrsb_train["records"],
                    "objects": vrsb_train["objects"],
                    "valid_objects": vrsb_train["valid_objects"],
                    "invalid_objects": vrsb_train["invalid_objects"],
                    "qa_pairs": vrsb_train["qa_pairs"],
                },
                "validation": {
                    "records": vrsb_val["records"],
                    "objects": vrsb_val["objects"],
                    "valid_objects": vrsb_val["valid_objects"],
                    "invalid_objects": vrsb_val["invalid_objects"],
                    "qa_pairs": vrsb_val["qa_pairs"],
                },
            },
            "GeoChat": {
                "records": geochat_total,
                "refer": {
                    "episodes": stats["geochat"]["by_kind"]["geochat_refer"],
                    "multi_turn": stats["geochat"]["multi_turn_by_kind"]["geochat_refer"],
                },
                "identify": {
                    "episodes": stats["geochat"]["by_kind"]["geochat_identify"],
                    "multi_turn": stats["geochat"]["multi_turn_by_kind"]["geochat_identify"],
                },
                "conversation": {
                    "episodes": stats["geochat"]["by_kind"]["geochat_conversation"],
                    "multi_turn": stats["geochat"]["multi_turn_by_kind"]["geochat_conversation"],
                },
            },
        },
        "self_attention": {
            "ratio": ratio,
            "split": "train",
            "strata": sa_strata,
            "total_parents": stats["sa"]["candidates"],
            "total_selected": sa_total_selected,
        },
        "vrsbench_invalid_objects": {
            "train": vrsb_train["invalid_objects"],
            "validation": vrsb_val["invalid_objects"],
        },
        "geochat_rejections": {
            "by_reason": dict(stats["rejected"]["GeoChat"]),
            "total": geochat_rejected,
        },
        "vrsbench_rejections": {
            "by_reason": dict(stats["rejected"]["VRSBench"]),
            "total": vrsbench_rejected,
        },
        "closure": {
            "geochat_records == geochat_episodes + geochat_rejected": (
                geochat_total == geochat_episodes + geochat_rejected
            ),
            "vrsbench_train_objects == grounding_episodes + invalid_objects": (
                vrsb_train["objects"]
                == vrsb_train["by_task_kind"]["vrsbench_grounding"]
                + vrsb_train["invalid_objects"]
            ),
            "vrsbench_train_qa_pairs == box_assisted + naturally_unboxed": (
                vrsb_train["qa_pairs"]
                == vrsb_train["by_task_kind"]["vqa_box_assisted"]
                + vrsb_train["by_task_kind"]["vqa_naturally_unboxed"]
            ),
            "vrsbench_validation_qa_pairs == box_assisted + naturally_unboxed": (
                vrsb_val["qa_pairs"]
                == vrsb_val["by_task_kind"]["vqa_box_assisted"]
                + vrsb_val["by_task_kind"]["vqa_naturally_unboxed"]
            ),
            "self_attention_total <= box_assisted_train_total": (
                sa_total_selected <= vrsb_train["by_task_kind"]["vqa_box_assisted"]
            ),
            "all_episode_ids_unique": True,
        },
    }

    # Self checksum: computed over this document with the field left empty.
    # 自校验和：按本字段为空字符串时的文档字节计算。
    manifest_bytes = json.dumps(
        manifest, indent=2, ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest["outputs"]["manifest.json_sha256"] = manifest_sha
    _atomic_write(
        out_dir / "manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
    )

    print(f"train.jsonl:        {manifest['counts']['train']['total']} episodes")
    print(f"validation.jsonl:   {manifest['counts']['validation']['total']} episodes")
    print(f"rejected:           {geochat_rejected + vrsbench_rejected} "
          f"(GeoChat {geochat_rejected}, VRSBench {vrsbench_rejected})")
    print(f"self-attention:     {sa_total_selected} of "
          f"{stats['sa']['candidates']} box-assisted parents")
    print(f"manifest.json:      {out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
