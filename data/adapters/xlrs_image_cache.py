"""Deterministic image materialization for XLRS Hugging Face releases.

XLRS Hugging Face 发布的确定性图片物化：
- path 字符串：优先直接使用（不复制）；
- {path, bytes}：path 可用时优先，否则按内容 SHA-256 物化到外部 cache；
- PIL Image：有有效 filename 时优先，否则 PNG 编码后按内容哈希物化；
- cache 文件名由内容哈希确定（<sha256>.png），原子写入、同哈希复用；
- 绝不写入 dataset root；不使用随机临时文件名。
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any


class ImageMaterializationError(ValueError):
    """Raised when an image feature cannot be safely resolved or materialized.
    图片特征无法安全解析或物化时抛出。"""


def materialize_image(
    value: Any,
    *,
    release_root: Path,
    cache_root: Path,
    index: int,
) -> tuple[Path, dict[str, Any]]:
    """Resolve one HF image feature to a local file and a JSON-safe descriptor.
    将一条 HF 图片特征解析为本地文件与 JSON 安全描述符。"""
    if not isinstance(cache_root, Path):
        raise ImageMaterializationError(
            f"cache_root must be a Path, got {type(cache_root).__name__}"
        )
    if isinstance(value, str):
        candidate = release_root / value
        if not candidate.is_file():
            raise ImageMaterializationError(
                f"XLRS row {index} references missing image path: {value}"
            )
        return candidate, {"image_present": True, "image_source_type": "path"}
    if isinstance(value, dict):
        path_value = value.get("path")
        if isinstance(path_value, str) and path_value:
            candidate = release_root / path_value
            if candidate.is_file():
                return candidate, {"image_present": True, "image_source_type": "path"}
        bytes_value = value.get("bytes")
        if isinstance(bytes_value, (bytes, bytearray)):
            target = _materialize_bytes(bytes(bytes_value), cache_root)
            return target, {"image_present": True, "image_source_type": "bytes"}
        raise ImageMaterializationError(
            f"XLRS row {index} image dict has neither usable path nor bytes"
        )
    if _is_pil_image(value):
        filename = getattr(value, "filename", None)
        if isinstance(filename, str) and filename:
            candidate = release_root / filename
            if candidate.is_file():
                return candidate, {"image_present": True, "image_source_type": "path"}
        target = _materialize_pil(value, cache_root)
        return target, {"image_present": True, "image_source_type": "pil"}
    raise ImageMaterializationError(
        f"XLRS row {index} has an unsupported image value of type {type(value).__name__}"
    )


def cache_existing_path(
    path: Path,
    *,
    cache_root: Path,
    index: int,
) -> tuple[Path, dict[str, Any]]:
    """Copy a path-backed image into the external cache by content hash.
    Used when one row mixes path and bytes/PIL images so every image of the
    row shares the cache root. 按内容哈希把 path 图片复制到外部 cache；
    用于一行混合 path 与 bytes/PIL 图片时统一解析根。"""
    if not isinstance(cache_root, Path):
        raise ImageMaterializationError(
            f"cache_root must be a Path, got {type(cache_root).__name__}"
        )
    if not path.is_file():
        raise ImageMaterializationError(
            f"XLRS row {index} references missing image path: {path}"
        )
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    suffix = path.suffix.lower() or ".png"
    target = cache_root / f"{digest}{suffix}"
    if not target.is_file():
        cache_root.mkdir(parents=True, exist_ok=True)
        _atomic_write(target, data)
    return target, {"image_present": True, "image_source_type": "path_cached"}


def _materialize_bytes(data: bytes, cache_root: Path) -> Path:
    digest = hashlib.sha256(data).hexdigest()
    target = cache_root / f"{digest}.png"
    if target.is_file():
        return target
    cache_root.mkdir(parents=True, exist_ok=True)
    _atomic_write(target, data)
    return target


def _materialize_pil(image: Any, cache_root: Path) -> Path:
    buffer = _encode_png(image)
    digest = hashlib.sha256(buffer).hexdigest()
    target = cache_root / f"{digest}.png"
    if target.is_file():
        return target
    cache_root.mkdir(parents=True, exist_ok=True)
    _atomic_write(target, buffer)
    return target


def _encode_png(image: Any) -> bytes:
    try:
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception as error:  # noqa: BLE001 - any PIL failure is a materialization failure
        raise ImageMaterializationError(
            f"cannot encode PIL image to PNG: {type(error).__name__}: {error}"
        ) from error


def _atomic_write(target: Path, data: bytes) -> None:
    directory = target.parent
    directory.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=directory, prefix=".xlrs-", delete=False
        ) as handle:
            handle.write(data)
            temporary = Path(handle.name)
        temporary.replace(target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _is_pil_image(value: Any) -> bool:
    module = type(value).__module__ or ""
    return module.startswith("PIL.") or module == "PIL"
