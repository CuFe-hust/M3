"""Clean Google Earth mentions from VRSBench train captions with regexes.
用正则清洗 VRSBench train caption 中的 Google Earth 提及。

Reads VRSBench_train_caption.jsonl and writes VRSBench_train_caption_cleaned.jsonl.
Only the caption field is rewritten; every other field is preserved as-is.
The val split is intentionally left untouched.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Google Earth matches "Google Earth", "GoogleEarth", and lowercase variants.
# Google Earth 匹配 “Google Earth”“GoogleEarth” 及小写变体。
_GE = r"Google\s*Earth"

# Verbs that introduce a Google Earth source phrase.
# 引入 Google Earth 来源短语的动词。
_SOURCE_VERBS = (
    r"(?:sourced|captured|taken|obtained|provided|retrieved|"
    r"downloaded|generated|gathered|presented)"
)

# Optional "with high resolution" tail that commonly follows the source phrase.
# 常见于来源短语之后的“with high resolution”可选结尾。
_HIGH_RES = r"(?:\s+(?:and\s+)?(?:with\s+)?(?:a\s+)?high(?:[- ])?resolution)?"

# "as seen/viewed/shown/presented/provided on Google Earth" source phrase.
# “as seen/viewed/shown/presented/provided on Google Earth”来源短语。
_AS_SOURCE = r"as\s+(?:seen|viewed|shown|presented|provided)"

# Hedge adverbs that can precede a Google Earth source phrase.
# 可能位于 Google Earth 来源短语之前的模糊副词。
_HEDGE = (
    r"(?:presumably|apparently|reportedly|possibly|likely|probably|"
    r"clearly|obviously|evidently)"
)

# Subject noun phrases used when dropping leftover source-only sentences.
# 删除仅剩来源信息的残句时使用的主语名词短语。
_SUBJECT = (
    r"(?:the|this|that|an?)\s+"
    r"(?:image|photo|photograph|snapshot|"
    r"aerial\s+(?:view|image|photo)|satellite\s+(?:image|view)|"
    r"remote\s*sensing\s+image|high(?:[- ])?resolution\s+image|"
    r"color\s+image|overhead\s+(?:view|image))"
)


def _cleanup_text(text: str) -> str:
    """Collapse whitespace and fix punctuation exposed by deletion.
    压缩空白并修复删除后暴露的标点问题。
    """
    s = re.sub(r"\s{2,}", " ", text)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([,;])\s*([,;])", r"\1", s)
    s = re.sub(r"([.!?])\s*([,;])", r"\1", s)
    s = re.sub(r"[,;]\s*\.", ".", s)
    s = re.sub(r"\.\s*\.", ".", s)
    s = s.strip()
    s = re.sub(r"^,\s*", "", s)
    if s and s[0] == ".":
        s = s[1:].strip()
    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s


def _drop_fragment_sentences(text: str) -> str:
    """Drop sentences that only mention the image source after cleaning.
    删除清洗后仅剩图片来源信息、无法独立成句的句子。
    """
    fragment_patterns = (
        # "The image." / "The satellite image." leftovers after phrase removal.
        # 短语删除后残留的 “The image.” / “The satellite image.”。
        re.compile(_SUBJECT + r"\s*[.!?]?\s*$", re.I),
        # "The image is." / "The image appears to be."
        # “The image is.” / “The image appears to be.”。
        re.compile(
            _SUBJECT
            + r"\s+(?:is|was|appears?\s+to\s+be|seems?\s+to\s+be)\s*[.!?]?\s*$",
            re.I,
        ),
        # "The source of the image."
        # “The source of the image.”。
        re.compile(r"the\s+source\s+of\s+the\s+image\s*[.!?]?\s*$", re.I),
        # "as seen." / "as viewed." leftovers.
        # “as seen.” / “as viewed.” 残留。
        re.compile(_AS_SOURCE + r"\s*[.!?]?\s*$", re.I),
        # "visible." leftover after "visible on Google Earth" removal.
        # 删除 “visible on Google Earth” 后残留的 “visible.”。
        re.compile(r"visible\s*[.!?]?\s*$", re.I),
    )
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [s for s in sentences if s and not any(p.match(s) for p in fragment_patterns)]
    if not kept:
        # Never return an empty caption; keep the cleaned text unchanged.
        # 绝不返回空 caption；保留清洗后的原文。
        return text
    return " ".join(kept)


def _check_fragment_sentences(text: str) -> bool:
    """Return True when a cleaned caption still contains a source-only fragment.
    当清洗后的 caption 仍包含仅剩来源信息的残句时返回 True。
    """
    fragment_patterns = (
        re.compile(_SUBJECT + r"\s*[.!?]?\s*$", re.I),
        re.compile(
            _SUBJECT
            + r"\s+(?:is|was|appears?\s+to\s+be|seems?\s+to\s+be)\s*[.!?]?\s*$",
            re.I,
        ),
        re.compile(r"the\s+source\s+of\s+the\s+image\s*[.!?]?\s*$", re.I),
        re.compile(_AS_SOURCE + r"\s*[.!?]?\s*$", re.I),
        re.compile(r"visible\s*[.!?]?\s*$", re.I),
    )
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if sentence and any(p.match(sentence) for p in fragment_patterns):
            return True
    return False


def clean_caption(text: str) -> str:
    """Remove Google Earth source mentions while keeping the sentence fluent.
    删除 Google Earth 来源提及，同时保持句子流畅。
    """
    # Captions without Google Earth mentions stay byte-identical.
    # 不含 Google Earth 提及的 caption 保持逐字节不变。
    if not re.search(_GE, text, re.I):
        return text
    s = text
    # 1) Comma-wrapped source phrase: ", sourced from Google Earth,"
    #    逗号包裹的来源短语：", sourced from Google Earth,"
    s = re.sub(
        r",\s*(?:which\s+(?:is|was)\s+)?"
        + _SOURCE_VERBS
        + r"\s+(?:from|by|via|through)\s+"
        + _GE
        + _HIGH_RES
        + r"\s*,?",
        "",
        s,
        flags=re.I,
    )
    # 2) "as seen/viewed ... on Google Earth" phrase. This must run before the
    #    plain verb phrase rule, otherwise "as presented by Google Earth"
    #    leaves a dangling "as,".
    #     “as seen/viewed ... on Google Earth”短语。
    #     必须位于普通动词短语规则之前，否则 "as presented by Google Earth"
    #     会残留悬空的 "as,"。
    s = re.sub(
        r"\s*,?\s+" + _AS_SOURCE + r"\s+(?:on|in|from|via|through|by)\s+"
        + _GE
        + r"\s*,?",
        " ",
        s,
        flags=re.I,
    )
    # 3) Plain source phrase, optionally with "is/was": "is sourced from Google Earth"
    #    普通来源短语，可选带 "is/was"："is sourced from Google Earth"
    s = re.sub(
        r"(?:\b(?:is|was|has been)\s+)?"
        r"(?:which\s+(?:is|was)\s+)?"
        + _SOURCE_VERBS
        + r"\s+(?:from|by|via|through)\s+"
        + _GE
        + _HIGH_RES
        + r"(?:\s+and\s+)?",
        " ",
        s,
        flags=re.I,
    )
    # 3b) Hedged source phrase: ", presumably from Google Earth,"
    #     带模糊副词的来源短语：", presumably from Google Earth,"
    s = re.sub(
        r"\s*,?\s+" + _HEDGE + r"\s+(?:from|by|on|via|through)\s+"
        + _GE
        + r"\s*,?",
        " ",
        s,
        flags=re.I,
    )
    # 3c) "originates/comes from Google Earth (and ...)"
    #     “originates/comes from Google Earth (and ...)”
    s = re.sub(
        r"\s+(?:originates?|comes?)\s+from\s+"
        + _GE
        + r"(?:\s+and\s+)?",
        " ",
        s,
        flags=re.I,
    )
    # 3d) "visible/viewed on/from Google Earth"
    #     “visible/viewed on/from Google Earth”
    s = re.sub(
        r"\s+(?:visible|viewed)\s+(?:on|in|from|via|through)\s+"
        + _GE
        + r"\b",
        " ",
        s,
        flags=re.I,
    )
    # 4) Copular construction: "is from Google Earth and ..."
    #    系动词结构："is from Google Earth and ..."
    s = re.sub(
        r"\s+is\s+from\s+" + _GE + r"(?:\s+and\s+)?", " ", s, flags=re.I
    )
    # 5) Copular noun phrase: "The source of the image is Google Earth."
    #    系动词名词短语："The source of the image is Google Earth."
    s = re.sub(
        r"\s+is\s+" + _GE + r"(?:\s+and\s+)?", " ", s, flags=re.I
    )
    # 6) Simple prepositional phrase: "image from Google Earth"
    #    简单介词短语："image from Google Earth"
    s = re.sub(
        r"\s+(?:from|by|on|via|through)\s+" + _GE + r"\b", "", s, flags=re.I
    )
    # 7) Attributive use: "Google Earth satellite image" -> "satellite image"
    #    定语用法："Google Earth satellite image" -> "satellite image"
    s = re.sub(
        _GE
        + r"\s+"
        + r"((?:(?:satellite|aerial|high(?:[- ])?resolution|overhead|"
        + r"remote(?:[ -]?sensing)?)\s+)*"
        + r"(?:image|imagery|photo|photograph|snapshot|view))",
        r"\1",
        s,
        flags=re.I,
    )
    # 8) Fallback: remove any remaining "Google Earth" token.
    #    兜底：删除任何剩余的 "Google Earth" 词。
    s = re.sub(_GE, "", s, flags=re.I)
    s = _cleanup_text(s)
    # 8) Drop source-only fragment sentences left behind by the rules above.
    #    删除上述规则处理后仅剩来源信息的残句。
    s = _drop_fragment_sentences(s)
    return _cleanup_text(s)


def build_parser() -> argparse.ArgumentParser:
    """Build the caption cleaning CLI. / 构建 caption 清洗 CLI。"""
    parser = argparse.ArgumentParser(
        description="Clean Google Earth mentions from VRSBench train captions."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing VRSBench_train_caption.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to --root.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_root = args.output_dir if args.output_dir is not None else args.root
    in_path = args.root / "VRSBench_train_caption.jsonl"
    out_path = out_root / "VRSBench_train_caption_cleaned.jsonl"
    if not in_path.is_file():
        raise SystemExit(f"Input file not found: {in_path}")
    out_root.mkdir(parents=True, exist_ok=True)
    stats = {"total": 0, "changed": 0, "unchanged": 0, "empty_after_clean": 0}
    fragment_examples: list[str] = []
    with in_path.open(encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            original = record["caption"]
            cleaned = clean_caption(original)
            stats["total"] += 1
            if cleaned != original:
                stats["changed"] += 1
            else:
                stats["unchanged"] += 1
            if not cleaned:
                stats["empty_after_clean"] += 1
            if _check_fragment_sentences(cleaned):
                if len(fragment_examples) < 5:
                    fragment_examples.append(cleaned)
            record["caption"] = cleaned
            fout.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    # Guard against incomplete patterns leaving Google Earth behind.
    # 防止未覆盖的模式残留 Google Earth。
    leftover = sum(
        1
        for line in out_path.read_text(encoding="utf-8").splitlines()
        if re.search(_GE, json.loads(line)["caption"], re.I)
    )
    if leftover:
        raise SystemExit(f"Leftover Google Earth mentions: {leftover}")
    stats["leftover_google_earth"] = leftover
    if fragment_examples:
        raise SystemExit(
            "Source-only fragment sentences remain: "
            + str(len(fragment_examples))
            + "+ examples: "
            + " || ".join(fragment_examples)
        )
    stats["fragment_sentences_after_clean"] = 0
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
