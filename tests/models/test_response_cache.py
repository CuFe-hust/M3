"""Contract tests for the model response cache.

模型响应缓存测试：命中/未命中、原子写入、十六进制 key 校验、
缓存不保存图片 Base64 或凭据。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from models.cache import (
    CacheEntry,
    CacheWriteError,
    CorruptCacheEntryError,
    JsonResponseCache,
    ModelCacheError,
)


def test_cache_miss_returns_none(tmp_path: Path) -> None:
    cache = JsonResponseCache(tmp_path)
    assert cache.load("a" * 64) is None


def test_cache_save_and_hit_roundtrip(tmp_path: Path) -> None:
    cache = JsonResponseCache(tmp_path)
    digest = "b" * 64
    entry = CacheEntry(raw_response='{"ok": true}', parsed={"ok": True})
    cache.save(digest, entry)
    loaded = cache.load(digest)
    assert loaded is not None
    assert loaded.raw_response == '{"ok": true}'
    assert loaded.parsed == {"ok": True}
    assert (tmp_path / f"{digest}.json").is_file()


def test_cache_save_is_atomic_no_tmp_leftovers(tmp_path: Path) -> None:
    cache = JsonResponseCache(tmp_path)
    digest = "c" * 64
    cache.save(digest, CacheEntry(raw_response="x", parsed={"a": 1}))
    assert list(tmp_path.glob("*.tmp")) == []


def test_cache_rejects_non_hex_keys(tmp_path: Path) -> None:
    cache = JsonResponseCache(tmp_path)
    for bad in ("", "z" * 64, "not-hex", "g" * 64):
        with pytest.raises(ValueError, match="hexadecimal"):
            cache._path(bad)
    # Uppercase hex, short hex, and letter-only hex are accepted; the file
    # name keeps the original spelling. 大写、短十六进制与纯字母十六进制均可
    # 接受；文件名保留原始拼写。
    assert cache._path("A" * 64).name == ("A" * 64) + ".json"
    assert cache._path("abc").name == "abc.json"


def test_cache_never_stores_base64_or_credentials(tmp_path: Path) -> None:
    cache = JsonResponseCache(tmp_path)
    digest = "d" * 64
    cache.save(digest, CacheEntry(raw_response="{}", parsed={}))
    content = (tmp_path / f"{digest}.json").read_text(encoding="utf-8")
    assert "base64" not in content
    assert "sk-" not in content
    assert "api_key" not in content


def test_cache_entry_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        CacheEntry.model_validate({"raw_response": "x", "parsed": {}, "secret": "sk-1"})


def test_cache_corrupt_entry_raises_specific_error(tmp_path: Path) -> None:
    """Corrupt JSON raises the precise CorruptCacheEntryError, never a broad
    exception. 损坏 JSON 必须抛出精确的 CorruptCacheEntryError。"""
    digest = "e" * 64
    (tmp_path / f"{digest}.json").write_text("{not json", encoding="utf-8")
    cache = JsonResponseCache(tmp_path)
    with pytest.raises(CorruptCacheEntryError, match="invalid"):
        cache.load(digest)


def test_cache_schema_invalid_entry_raises_specific_error(tmp_path: Path) -> None:
    """Valid JSON failing the entry schema raises CorruptCacheEntryError.
    合法 JSON 但不符合条目 Schema 时抛出 CorruptCacheEntryError。"""
    digest = "f" * 64
    (tmp_path / f"{digest}.json").write_text('{"parsed": {}}', encoding="utf-8")
    cache = JsonResponseCache(tmp_path)
    with pytest.raises(CorruptCacheEntryError, match="invalid"):
        cache.load(digest)


def test_cache_non_utf8_entry_raises_specific_error(tmp_path: Path) -> None:
    """Non-UTF-8 cache files raise CorruptCacheEntryError, not OSError.
    非 UTF-8 缓存文件抛出 CorruptCacheEntryError，而非 OSError。"""
    digest = "a1b2c3d4" * 8
    (tmp_path / f"{digest}.json").write_bytes(b"\xff\xfe\x00\x80")
    cache = JsonResponseCache(tmp_path)
    with pytest.raises(CorruptCacheEntryError, match="unreadable"):
        cache.load(digest)


def test_cache_errors_share_model_cache_base() -> None:
    """All cache failures share the ModelCacheError base type.
    所有缓存失败共享 ModelCacheError 基类。"""
    assert issubclass(CorruptCacheEntryError, ModelCacheError)
    assert issubclass(CacheWriteError, ModelCacheError)


# ── 写失败 / write failures (C) ────────────────────────────────────────────


def test_cache_save_mkdir_failure_raises_cache_write_error(tmp_path: Path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("file", encoding="utf-8")
    cache = JsonResponseCache(blocked)
    with pytest.raises(CacheWriteError, match="mkdir"):
        cache.save("a" * 64, CacheEntry(raw_response="x", parsed={}))
    assert list(tmp_path.glob("*.tmp")) == []


def test_cache_save_write_failure_raises_cache_write_error(
    tmp_path: Path, monkeypatch
) -> None:
    cache = JsonResponseCache(tmp_path)

    def _broken_write(self, *args, **kwargs):
        raise OSError("write boom")

    monkeypatch.setattr(Path, "write_text", _broken_write)
    with pytest.raises(CacheWriteError, match="write_text"):
        cache.save("b" * 64, CacheEntry(raw_response="x", parsed={}))
    assert list(tmp_path.glob("*.tmp")) == []


def test_cache_save_replace_failure_raises_cache_write_error(tmp_path: Path) -> None:
    digest = "c" * 64
    (tmp_path / f"{digest}.json").mkdir()
    cache = JsonResponseCache(tmp_path)
    with pytest.raises(CacheWriteError, match="replace"):
        cache.save(digest, CacheEntry(raw_response="x", parsed={}))
    # The temporary file must be cleaned up even when replace fails.
    # 即使替换失败也必须清理临时文件。
    assert list(tmp_path.glob("*.tmp")) == []


def test_cache_save_cleanup_failure_raises_cache_write_error(
    tmp_path: Path, monkeypatch
) -> None:
    cache = JsonResponseCache(tmp_path)

    def _broken_unlink(self):
        raise OSError("cleanup boom")

    # The temporary file "exists" when cleanup runs, and unlink fails.
    # 清理阶段临时文件"存在"，且 unlink 失败。
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(Path, "unlink", _broken_unlink)
    with pytest.raises(CacheWriteError, match="cleanup"):
        cache.save("d" * 64, CacheEntry(raw_response="x", parsed={}))


def test_cleanup_error_does_not_mask_primary_write_error(
    tmp_path: Path, monkeypatch
) -> None:
    """When both the write and the cleanup fail, the write error wins.
    写入与清理同时失败时，原始写错误优先。"""
    cache = JsonResponseCache(tmp_path)

    def _broken_write(self, *args, **kwargs):
        raise OSError("write boom")

    def _broken_unlink(self):
        raise OSError("cleanup boom")

    monkeypatch.setattr(Path, "write_text", _broken_write)
    monkeypatch.setattr(Path, "exists", lambda self: True)
    monkeypatch.setattr(Path, "unlink", _broken_unlink)
    with pytest.raises(CacheWriteError, match="write_text") as info:
        cache.save("e" * 64, CacheEntry(raw_response="x", parsed={}))
    assert "cleanup" not in str(info.value)


# ── CacheEntry 内容安全 / entry content safety (D) ─────────────────────────


def test_cache_entry_rejects_base64_raw_response() -> None:
    with pytest.raises(ValidationError, match="blocked value"):
        CacheEntry(raw_response="data:image/png;base64,AAAA", parsed={})


def test_cache_entry_rejects_sensitive_parsed_key() -> None:
    with pytest.raises(ValidationError, match="blocked key"):
        CacheEntry(raw_response="{}", parsed={"api_key": "sk-x"})
    with pytest.raises(ValidationError, match="blocked key"):
        CacheEntry(raw_response="{}", parsed={"nested": {"access_token": "x"}})


def test_cache_entry_rejects_bearer_and_case_space_values() -> None:
    with pytest.raises(ValidationError, match="blocked value"):
        CacheEntry(raw_response="{}", parsed={"answer": "Bearer abc"})
    with pytest.raises(ValidationError, match="blocked value"):
        CacheEntry(raw_response="{}", parsed={"answer": "  SK-SECRET"})
    with pytest.raises(ValidationError, match="blocked value"):
        CacheEntry(raw_response="{}", parsed={"answer": "Data:Image/png;base64,AAAA"})


def test_cache_entry_allows_plain_text_with_token_word() -> None:
    """Ordinary words like "token" inside normal text must not be rejected.
    普通文本中的 token 等单词不得误报。"""
    entry = CacheEntry(
        raw_response='{"answer": "the token count is 3"}',
        parsed={"answer": "the token count is 3", "count": 3},
    )
    assert entry.parsed["answer"] == "the token count is 3"
