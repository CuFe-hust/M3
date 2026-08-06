"""Contract tests for the model response cache.

模型响应缓存测试：命中/未命中、原子写入、十六进制 key 校验、
缓存不保存图片 Base64 或凭据。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from models.cache import CacheEntry, JsonResponseCache


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


def test_cache_corrupt_entry_raises(tmp_path: Path) -> None:
    digest = "e" * 64
    (tmp_path / f"{digest}.json").write_text("{not json", encoding="utf-8")
    cache = JsonResponseCache(tmp_path)
    with pytest.raises(Exception):
        cache.load(digest)
