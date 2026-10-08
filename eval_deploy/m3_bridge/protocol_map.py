"""Pure Protocol-2.0 <-> M3 mapping helpers (no M3 imports).

纯映射层：Protocol 2.0 请求 → M3 task/题干/规范化元数据，以及
M3 Agent 输出 → 协议答案。此模块不导入 M3，便于离线单测。

Coordinate convention: M3 uses integer 0..999 whole-image xyxy boxes
(agents.schema.VisualEvidence); the Protocol needs original-image pixel
boxes. Conversion mirrors models/images.py (x*W//999, (x*W+998)//999).
坐标系：M3 内部是 0..999 整图 xyxy；协议要求原图像素框；换算与
M3 models/images.py 的整数取整互逆。
"""

from __future__ import annotations

import os
import re
from difflib import SequenceMatcher
from typing import Any

from .errors import AdapterError

CHANGE_TASKS = frozenset({"change_caption", "change_qa"})

BBOX_PICK = os.environ.get("BRIDGE_BBOX_PICK", "largest_area")

# Prompt-level, training-free debiasing guidance appended per constraint type.
# Observed 2026-10-06: change-region answers anchored on "bottom" (14/23, 3
# correct), single-choice answers over-picked option E (7/7 wrong), and bbox
# answers covered only part of area-type targets or the wrong instance.
# 免训练提示词纠偏：变化题 bottom 锚定、单选 E 偏置、区域型目标框不全。
_CHANGE_GUIDANCE = (
    "Before answering, compare the two images systematically region by region "
    "(top / center / bottom, and left / right within the strongest candidate "
    "area). Choose the one region where the most significant semantic change "
    "actually occurred. Do not default to any particular region; the answer "
    "differs from question to question. Even if the change is subtle between "
    "seasons or illumination, you must still select the single best region."
)

_CHOICE_GUIDANCE = (
    "Evaluate every option against the image independently and choose the "
    "best-supported one. Do not favor or avoid any particular letter or "
    "position."
)

_BBOX_GUIDANCE = (
    "If the target is an area-type object (for example a parking lot, a grass "
    "area, a harbor, a road or a zone), the box must cover the FULL extent of "
    "that area, not only a part of it. If the description says 'the largest', "
    "'the only', or gives a spatial relation, first compare the candidate "
    "instances and return exactly the one described."
)

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9]*\s*\n?|\n?\s*```$")
_WRAP_CHARS = "\"'“”‘’"


def task_for(request: dict[str, Any]) -> str:
    """Map one protocol request to a single M3 task (constraint = task声明).

    把一个协议请求映射为唯一的 M3 task（约束即任务声明）。
    """
    constraint = request["response_constraint"]
    kind = constraint["type"]
    n_images = len(request.get("images", []))
    if kind == "bbox":
        return "grounding"
    if kind == "single_choice":
        return "multiple_choice_vqa"
    if kind == "integer":
        # 计数类走 general_vqa 直答：counting 任务需要 planner 的
        # count_target，且证据链最重（120s 风险）。
        return "general_vqa"
    if kind in ("enum", "short_text") and n_images == 2:
        return "change_qa"
    return "general_vqa"


def compose_question(request: dict[str, Any]) -> str:
    """Append the response-constraint instruction to the exam question.

    M3 不把 response_constraint 传给模型，因此在题干后追加约束指令。
    """
    question = str(request["question"]).strip()
    constraint = request["response_constraint"]
    kind = constraint["type"]
    if kind == "enum":
        values = ", ".join(str(v) for v in constraint["values"])
        prefix = ""
        if len(request.get("images") or []) == 2:
            prefix = _CHANGE_GUIDANCE + "\n"
        suffix = prefix + (
            "Choose exactly one of the following values and reply with that "
            f"value verbatim, with nothing else: {values}"
        )
    elif kind == "single_choice":
        pairs = choice_pairs(request)
        lines = "\n".join(f"{chr(65 + i)}. {text}" for i, (_, text) in enumerate(pairs))
        suffix = (
            "Choose exactly one option. Reply with ONLY the option letter "
            "(for example: B).\n" + lines + "\n" + _CHOICE_GUIDANCE
        )
    elif kind == "integer":
        suffix = "Reply with a single non-negative integer (digits only)."
    elif kind == "short_text":
        suffix = "Reply with a short single-line answer phrase, without formatting."
    elif kind == "bbox":
        suffix = (
            "Localize the target and return EXACTLY ONE bounding box as the "
            "first entry of boxes, using integer whole-image 0..999 xyxy "
            "coordinates. Always return your best single box even if you are "
            "not fully certain. " + _BBOX_GUIDANCE
        )
    else:  # pragma: no cover - validate_request upstream rejects unknown types
        suffix = ""
    return f"{question}\n\n{suffix}".strip()


# ── choices ────────────────────────────────────────────────────────────────


def choice_pairs(request: dict[str, Any]) -> list[tuple[str, str]]:
    """Ordered (site_letter, text) pairs for a single_choice request.

    单选题的（网站选项标识, 选项文本）有序列表。
    """
    pairs: list[tuple[str, str]] = []
    choices = request.get("choices")
    if isinstance(choices, dict):
        for key, value in choices.items():
            letter = str(key).strip().upper()
            if letter:
                pairs.append((letter, str(value).strip()))
    if not pairs:
        values = request["response_constraint"].get("values") or []
        for index, value in enumerate(values):
            pairs.append((chr(ord("A") + index), str(value).strip()))
    return pairs


def m3_choices(pairs: list[tuple[str, str]]) -> list[str]:
    """Choices passed to M3 with positional A..Z labels (independent of the
    site's letter scheme); the bridge maps the result back by position.

    传给 M3 的选项文本使用位置字母 A..Z（与网站标识解耦），桥接再按位置
    映射回网站标识。
    """
    return [f"{chr(65 + index)}. {text}" for index, (_, text) in enumerate(pairs)]


# ── answer normalization ───────────────────────────────────────────────────


def normalize_answer(request: dict[str, Any], payload: Any) -> str | list[int]:
    """Normalize one Agent payload into a protocol-legal answer.

    把一条 Agent 输出规范化为协议合法答案。
    """
    kind = request["response_constraint"]["type"]
    if kind == "bbox":
        return _bbox_answer(request, payload)
    answer = str(getattr(payload, "answer", "") or "")
    if kind == "enum":
        return _enum_answer(request, answer)
    if kind == "single_choice":
        return _single_choice_answer(request, answer)
    if kind == "integer":
        return _integer_answer(answer)
    return _short_text_answer(answer)


def _strip_wrappers(text: str) -> str:
    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    cleaned = cleaned.strip(_WRAP_CHARS).strip()
    cleaned = cleaned.rstrip(".!?。！？;；,，、")
    return cleaned.strip()


def _canon_value(text: str) -> str:
    """Casefold and normalize separators/synonyms for value matching.

    统一大小写、分隔符与常见方位同义词，用于枚举/区域值匹配。
    """
    normalized = text.strip().casefold().replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.replace("upper", "top").replace("lower", "bottom")


def _enum_answer(request: dict[str, Any], answer: str) -> str:
    values = [str(v) for v in request["response_constraint"]["values"]]
    cleaned = _strip_wrappers(answer)
    if cleaned in values:
        return cleaned
    lowered = {v.casefold(): v for v in values}
    key = cleaned.casefold()
    if key in lowered:
        return lowered[key]
    first_line = cleaned.splitlines()[0].strip() if cleaned else ""
    if first_line.casefold() in lowered:
        return lowered[first_line.casefold()]
    # Canonical matching: normalize separators and common spatial synonyms so
    # "top right" / "upper left" map onto canonical labels ("top_right").
    # 规范匹配：统一分隔符与常见方位同义词（"top right"/"upper left" → 规范标签）。
    canon_values = {_canon_value(v): v for v in values}
    canon_key = _canon_value(cleaned)
    if canon_key in canon_values:
        return canon_values[canon_key]
    # Containment: pick the longest value that appears in the answer text.
    # 包含匹配：取出现在答案文本中的最长允许值（如 "Yes, there is..."）。
    hits = [v for v in values if _canon_value(v) in canon_key]
    if hits:
        return max(hits, key=len)
    if not cleaned:
        raise AdapterError(502, "unanswerable")
    # Fuzzy rescue for otherwise unmatched phrases.
    # 其余未匹配表述的模糊兜底。
    best_value, best_score = "", 0.0
    for value in values:
        score = SequenceMatcher(None, canon_key, _canon_value(value)).ratio()
        if score > best_score:
            best_value, best_score = value, score
    if best_score >= 0.6:
        return best_value
    for value in values:
        if canon_key and canon_key in _canon_value(value):
            return value
    # Unmatched but format-legal: return the cleaned text (site scores it).
    # 未匹配但格式合法：返回清洗后的文本，由网站计分。
    return cleaned[:4096]


def _single_choice_answer(request: dict[str, Any], answer: str) -> str:
    pairs = choice_pairs(request)
    if not pairs:
        raise AdapterError(502, "unanswerable")
    text = answer.strip()
    if not text:
        raise AdapterError(502, "unanswerable")
    # 1) Pure label form: "B", "(B)", "B.", "B)".
    m = re.match(r"^\s*[\(\[]?([A-Za-z])[\)\].、:：\-]?\s*$", text)
    if m:
        index = ord(m.group(1).upper()) - ord("A")
        if 0 <= index < len(pairs):
            return pairs[index][0]
    # 2) Prefixed label + text: "B. water" / "B) water".
    m = re.match(r"^\s*[\(\[]?([A-Za-z])[\)\].、:：\-]\s*(.+)$", text)
    if m:
        index = ord(m.group(1).upper()) - ord("A")
        if 0 <= index < len(pairs):
            return pairs[index][0]
    # 3) Exact / contained text match (normalized).
    key = _strip_wrappers(text).casefold()
    for letter, choice_text in pairs:
        if choice_text and choice_text.casefold() == key:
            return letter
    for letter, choice_text in pairs:
        candidate = choice_text.casefold()
        if candidate and (candidate in key or key in candidate):
            return letter
    # 4) Fuzzy best match with a conservative floor.
    best_letter, best_score = None, 0.0
    for letter, choice_text in pairs:
        score = SequenceMatcher(None, key, choice_text.casefold()).ratio()
        if score > best_score:
            best_letter, best_score = letter, score
    if best_letter is not None and best_score >= 0.6:
        return best_letter
    raise AdapterError(502, "unanswerable")


def _integer_answer(answer: str) -> str:
    compact = answer.replace(",", " ").replace("，", " ")
    match = re.search(r"\d+", compact)
    if match is None:
        negative = re.search(r"-\d+", compact)
        if negative is None:
            raise AdapterError(502, "unanswerable")
        return "0"
    return str(int(match.group(0)))


def _short_text_answer(answer: str) -> str:
    cleaned = _FENCE_RE.sub("", answer.strip()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        raise AdapterError(502, "unanswerable")
    return cleaned[:4096]


def _bbox_answer(request: dict[str, Any], payload: Any) -> list[int]:
    width = request.get("image_width")
    height = request.get("image_height")
    if not (isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0):
        raise AdapterError(502, "no_localization")
    boxes: list[list[int]] = []
    raw_boxes = getattr(payload, "boxes", None) or []
    for box in raw_boxes:
        if (
            isinstance(box, (list, tuple))
            and len(box) == 4
            and all(isinstance(v, (int, float)) for v in box)
        ):
            boxes.append([int(round(float(v))) for v in box])
    if not boxes:
        for item in getattr(payload, "evidence_items", None) or []:
            box = getattr(item, "box", None)
            if isinstance(box, (list, tuple)) and len(box) == 4:
                boxes.append([int(v) for v in box])
    if not boxes:
        numbers = re.findall(r"-?\d+(?:\.\d+)?", str(getattr(payload, "answer", "") or ""))
        if len(numbers) >= 4:
            boxes.append([int(round(float(v))) for v in numbers[:4]])
    if not boxes:
        raise AdapterError(502, "no_localization")
    chosen = _pick_box(boxes)
    x1, y1, x2, y2 = chosen
    if any(v > 999 for v in chosen):
        # Looks like the model already answered in image pixels.
        # 模型可能直接给了像素坐标（且超出 0..999 范围）时按像素处理。
        pixel = [x1, y1, x2, y2]
    else:
        pixel = [
            (x1 * width) // 999,
            (y1 * height) // 999,
            (x2 * width + 998) // 999,
            (y2 * height + 998) // 999,
        ]
    pixel[0] = max(0, min(width, pixel[0]))
    pixel[1] = max(0, min(height, pixel[1]))
    pixel[2] = max(0, min(width, pixel[2]))
    pixel[3] = max(0, min(height, pixel[3]))
    if pixel[0] >= pixel[2]:
        pixel[2] = min(width, pixel[0] + 1)
        if pixel[0] >= pixel[2]:
            pixel[0] = max(0, pixel[2] - 1)
    if pixel[1] >= pixel[3]:
        pixel[3] = min(height, pixel[1] + 1)
        if pixel[1] >= pixel[3]:
            pixel[1] = max(0, pixel[3] - 1)
    if pixel[0] >= pixel[2] or pixel[1] >= pixel[3]:
        raise AdapterError(502, "no_localization")
    return [int(v) for v in pixel]


def _pick_box(boxes: list[list[int]]) -> list[int]:
    if BBOX_PICK == "first":
        return boxes[0]
    return max(boxes, key=lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1]))
