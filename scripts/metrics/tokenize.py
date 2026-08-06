"""Shared tokenization for text metrics.

English: whitespace tokens, lowercased.
Chinese: jieba word segmentation when available, else character-level.
"""

from __future__ import annotations

import re

try:  # jieba is an optional dependency; fall back to character-level for Chinese
    import jieba  # type: ignore

    _HAS_JIEBA = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_JIEBA = False

_CJK_RE = re.compile(r"[一-鿿㐀-䶿]")


def _is_cjk(text: str) -> bool:
    return _CJK_RE.search(text) is not None


def tokenize(text: str) -> list[str]:
    """Return deterministic lowercased tokens for caption/VQA text."""
    text = (text or "").strip().lower()
    if not text:
        return []
    if _is_cjk(text):
        if _HAS_JIEBA:
            import jieba

            return [token for token in jieba.cut(text) if token.strip()]
        return list(text.replace(" ", ""))
    return text.split()
