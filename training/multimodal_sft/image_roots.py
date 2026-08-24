"""Explicit, symlink-safe image source roots for model-neutral profiles."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


class ImageRootError(ValueError):
    """An image source/path cannot be safely resolved."""

    def __init__(self, code: str, path: str = "", source: str = "") -> None:
        self.code = str(code)
        self.path = str(path)
        self.source = str(source)
        details = ":" + source if source else ""
        suffix = ":" + path if path else ""
        super().__init__(f"{self.code}{details}{suffix}")


class ImageRootRegistry:
    """Map a declared source name to one absolute root and resolve safely."""

    def __init__(self, roots: Mapping[str, str | Path]) -> None:
        normalized: dict[str, Path] = {}
        for source, root in roots.items():
            name = str(source).strip()
            if not name or "=" in name:
                raise ImageRootError("INVALID_IMAGE_SOURCE", source=name)
            path = Path(root)
            if not path.is_absolute():
                raise ImageRootError("IMAGE_ROOT_NOT_ABSOLUTE", source=name)
            resolved = path.resolve()
            if not resolved.is_dir():
                raise ImageRootError("IMAGE_ROOT_MISSING", source=name, path=str(resolved))
            normalized[name] = resolved
        self._roots = normalized

    @classmethod
    def from_specs(cls, specs: list[str] | tuple[str, ...]) -> "ImageRootRegistry":
        roots: dict[str, str] = {}
        for raw in specs:
            if "=" not in raw:
                raise ImageRootError("INVALID_IMAGE_ROOT_SPEC", path=str(raw))
            source, root = raw.split("=", 1)
            if not source or not root or source in roots:
                raise ImageRootError("INVALID_IMAGE_ROOT_SPEC", path=str(raw))
            roots[source] = root
        return cls(roots)

    @property
    def roots(self) -> Mapping[str, str]:
        return {source: str(root) for source, root in self._roots.items()}

    def contract(self) -> dict[str, Any]:
        payload = {source: str(root).replace("\\", "/") for source, root in sorted(self._roots.items())}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return {"sources": payload, "sha256": hashlib.sha256(canonical).hexdigest()}

    def resolve(self, image_source: str, relative_path: str) -> Path:
        source = str(image_source)
        if source not in self._roots:
            raise ImageRootError("UNKNOWN_IMAGE_SOURCE", source=source, path=relative_path)
        if not isinstance(relative_path, str) or not relative_path:
            raise ImageRootError("UNSAFE_IMAGE_PATH", source=source, path=str(relative_path))
        if "\\" in relative_path or "\x00" in relative_path:
            raise ImageRootError("UNSAFE_IMAGE_PATH", source=source, path=relative_path)
        if relative_path.startswith(("/", "//")) or re.match(r"^[A-Za-z]:", relative_path):
            raise ImageRootError("UNSAFE_IMAGE_PATH", source=source, path=relative_path)
        parts = relative_path.split("/")
        if not parts or any(part in {"", ".", ".."} for part in parts):
            raise ImageRootError("UNSAFE_IMAGE_PATH", source=source, path=relative_path)
        root = self._roots[source]
        resolved = (root / Path(*parts)).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ImageRootError("UNSAFE_IMAGE_PATH", source=source, path=relative_path) from exc
        if not resolved.exists():
            raise ImageRootError("IMAGE_MISSING", source=source, path=relative_path)
        if not resolved.is_file():
            raise ImageRootError("IMAGE_NOT_REGULAR_FILE", source=source, path=relative_path)
        return resolved

    def load_rgb(self, image_source: str, relative_path: str) -> Any:
        path = self.resolve(image_source, relative_path)
        try:
            from PIL import Image
            with Image.open(path) as image:
                image.load()
                if image.width <= 0 or image.height <= 0:
                    raise ImageRootError("IMAGE_DECODE_ERROR", source=image_source, path=relative_path)
                return image.convert("RGB")
        except ImageRootError:
            raise
        except Exception as exc:  # noqa: BLE001 - stable profile boundary
            raise ImageRootError("IMAGE_DECODE_ERROR", source=image_source, path=relative_path) from exc

