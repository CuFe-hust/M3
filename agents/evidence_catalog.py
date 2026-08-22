"""Versioned canonical category catalog shared by visual protocols.

The catalog owns canonical leaf identities, deterministic aliases, semantic
parent expansion, and per-task executable capability. It never selects a
backend or exposes physical model identity.
该目录统一管理 canonical 叶子身份、确定性 alias、语义父类展开和
按任务划分的可执行能力；它不选择 backend，也不暴露物理模型身份。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ModelCapability = Literal["yolo", "segformer"]

_CATEGORY_PATTERN = r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$"
_VERSION_PATTERN = r"^[a-z0-9][a-z0-9_.-]*$"
_BINDING_PATTERN = r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$"
_EXECUTABLE_TASKS = frozenset(
    {"counting", "fine_grained_counting", "general_vqa", "grounding"}
)
_PLACEHOLDER_PATTERN = re.compile(r"^label(?:[-_\s]*)\d+$", re.IGNORECASE)
_SENSITIVE_KEYS = frozenset(
    {
        "api_key", "apikey", "authorization", "access_token", "refresh_token",
        "private_key", "password", "credential", "token", "secret",
    }
)


class CatalogCategoryError(ValueError):
    """Stable catalog failure without file contents or host paths.
    不泄露文件内容或主机路径的稳定 catalog 错误。"""

    def __init__(
        self,
        code: str,
        *,
        categories: Sequence[str] = (),
        available: Sequence[str] = (),
    ) -> None:
        message = f"EVIDENCE_CATALOG:{code}"
        if categories:
            message += f": {', '.join(categories)}"
        super().__init__(message)
        self.code = code
        self.categories = tuple(categories)
        self.available = tuple(available)


class LeafCapabilities(BaseModel):
    """Verified raw model-label mappings for one canonical leaf."""

    model_config = ConfigDict(extra="forbid")

    yolo_labels: list[str] = Field(default_factory=list)
    segformer_labels: list[str] | None = None
    # Stable logical segmenter binding (e.g. segmenter_mitb2_001) — never a
    # checkpoint path, device, or backend. It expresses capability ownership
    # only; the composition root maps it to a verified logical client.
    # 稳定逻辑 segmenter binding（如 segmenter_mitb2_001）——绝不是 checkpoint
    # 路径、设备或 backend。它只表达能力归属；由组合根映射到已验证逻辑客户端。
    segformer_binding: str | None = None
    yolo_enabled: bool = False
    segformer_enabled: bool = False

    @model_validator(mode="after")
    def validate_capabilities(self) -> "LeafCapabilities":
        for label in self.yolo_labels:
            _validate_model_label(label, "yolo")
        if self.segformer_labels is not None:
            if not self.segformer_labels:
                raise ValueError("segformer_labels must be null or non-empty")
            for label in self.segformer_labels:
                _validate_model_label(label, "segformer")
        if self.yolo_enabled and not self.yolo_labels:
            raise ValueError("yolo_enabled requires verified yolo_labels")
        if self.segformer_enabled and not self.segformer_labels:
            raise ValueError("segformer_enabled requires verified segformer_labels")
        # SegFormer labels and the stable binding must exist together or be
        # absent together; a label without a binding cannot be routed to a
        # verified client, and a binding without labels is a dead capability.
        # SegFormer 标签与稳定 binding 必须同时存在或同时缺失；无 binding 的
        # 标签无法路由到已验证客户端，无标签的 binding 是死能力。
        if (self.segformer_labels is None) != (self.segformer_binding is None):
            raise ValueError(
                "segformer_labels and segformer_binding must be all-or-none"
            )
        if self.segformer_binding is not None:
            _validate_model_label(self.segformer_binding, "segformer_binding")
            if re.fullmatch(_BINDING_PATTERN, self.segformer_binding) is None:
                raise ValueError("segformer_binding is not a stable binding identifier")
        return self


class EvidenceCatalog:
    """Immutable canonical leaves, aliases, parents, and task capability."""

    def __init__(self, data: Mapping[str, Any]) -> None:
        if not isinstance(data, Mapping):
            raise TypeError("catalog data must be a mapping")
        _check_no_sensitive_keys(data)
        expected = {
            "catalog_version", "aliases", "parents", "leaves", "task_capabilities",
        }
        top_level = set(data)
        if top_level != expected:
            raise CatalogCategoryError(
                "INVALID_TOP_LEVEL_KEYS",
                categories=sorted(top_level.symmetric_difference(expected)),
            )

        version = data["catalog_version"]
        if (
            not isinstance(version, str)
            or not version.strip()
            or any(ord(character) < 32 for character in version)
            or re.fullmatch(_VERSION_PATTERN, version) is None
        ):
            raise CatalogCategoryError("INVALID_CATALOG_VERSION")
        self._version = version

        leaves_raw = _require_mapping(data["leaves"])
        leaves: dict[str, LeafCapabilities] = {}
        for name, capabilities in leaves_raw.items():
            _validate_canonical_name(name)
            if _PLACEHOLDER_PATTERN.fullmatch(name) or name == "background":
                raise CatalogCategoryError("INVALID_LEAF", categories=[str(name)])
            if not isinstance(capabilities, Mapping):
                raise CatalogCategoryError(
                    "INVALID_LEAF_CAPABILITIES", categories=[str(name)]
                )
            try:
                leaves[name] = LeafCapabilities.model_validate(dict(capabilities))
            except Exception as exc:
                raise CatalogCategoryError(
                    "INVALID_LEAF_CAPABILITIES", categories=[str(name)]
                ) from exc
        self._leaves = leaves

        parents_raw = _require_mapping(data["parents"])
        parents: dict[str, tuple[str, ...]] = {}
        for name, children in parents_raw.items():
            _validate_canonical_name(name)
            if name in leaves:
                raise CatalogCategoryError(
                    "PARENT_NAME_COLLIDES_WITH_LEAF", categories=[name]
                )
            if not isinstance(children, Sequence) or isinstance(children, (str, bytes)):
                raise CatalogCategoryError("INVALID_PARENT_CHILDREN", categories=[name])
            ordered: list[str] = []
            for child in children:
                if child not in leaves:
                    raise CatalogCategoryError(
                        "PARENT_TARGETS_UNKNOWN_LEAF", categories=[name, str(child)]
                    )
                if child in ordered:
                    raise CatalogCategoryError(
                        "PARENT_TARGETS_DUPLICATED", categories=[name, child]
                    )
                ordered.append(child)
            if not ordered:
                raise CatalogCategoryError("INVALID_PARENT_CHILDREN", categories=[name])
            parents[name] = tuple(ordered)
        self._parents = parents

        aliases_raw = _require_mapping(data["aliases"])
        aliases: dict[str, str] = {}
        semantic_names = set(leaves) | set(parents)
        for alias, target in aliases_raw.items():
            normalized_alias = _normalize_semantic(alias)
            if not isinstance(target, str) or target not in semantic_names:
                raise CatalogCategoryError(
                    "ALIAS_TARGET_UNKNOWN", categories=[str(alias), str(target)]
                )
            if normalized_alias in semantic_names:
                raise CatalogCategoryError(
                    "ALIAS_COLLIDES_WITH_CANONICAL", categories=[str(alias)]
                )
            owner = aliases.get(normalized_alias)
            if owner is not None and owner != target:
                raise CatalogCategoryError("ALIAS_CONFLICT", categories=[str(alias)])
            aliases[normalized_alias] = target
        self._aliases = MappingProxyType(aliases)

        capabilities_raw = _require_mapping(data["task_capabilities"])
        if set(capabilities_raw) != _EXECUTABLE_TASKS:
            raise CatalogCategoryError(
                "INVALID_TASK_CAPABILITIES", categories=sorted(capabilities_raw)
            )
        task_capabilities: dict[str, tuple[str, ...]] = {}
        for task, categories in capabilities_raw.items():
            if not isinstance(categories, Sequence) or isinstance(categories, (str, bytes)):
                raise CatalogCategoryError("INVALID_TASK_CAPABILITIES", categories=[task])
            ordered: list[str] = []
            for category in categories:
                if category not in leaves:
                    raise CatalogCategoryError(
                        "TASK_CAPABILITY_UNKNOWN_LEAF", categories=[task, str(category)]
                    )
                if category in ordered:
                    raise CatalogCategoryError(
                        "TASK_CAPABILITY_DUPLICATED", categories=[task, category]
                    )
                ordered.append(category)
            task_capabilities[task] = tuple(ordered)
        self._task_capabilities = task_capabilities

    @property
    def catalog_version(self) -> str:
        return self._version

    @property
    def leaf_categories(self) -> tuple[str, ...]:
        return tuple(self._leaves)

    @property
    def parent_categories(self) -> tuple[str, ...]:
        return tuple(self._parents)

    @property
    def aliases(self) -> Mapping[str, str]:
        return self._aliases

    @property
    def parent_expansions(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(self._parents)

    def is_leaf(self, value: str) -> bool:
        return isinstance(value, str) and value in self._leaves

    def is_parent(self, value: str) -> bool:
        return isinstance(value, str) and value in self._parents

    def canonicalize_alias(self, value: str) -> str:
        normalized = _normalize_semantic(value)
        return self._aliases.get(normalized, normalized)

    def expand_target(self, value: str) -> tuple[str, ...]:
        canonical = self.canonicalize_alias(value)
        if canonical in self._leaves:
            return (canonical,)
        return self._parents.get(canonical, ())

    def executable_leaves_for_task(self, task: str) -> tuple[str, ...]:
        try:
            return self._task_capabilities[task]
        except KeyError:
            raise CatalogCategoryError(
                "TASK_CAPABILITY_UNKNOWN", categories=[str(task)]
            ) from None

    def validate_plan_leaves(
        self,
        categories: Sequence[str],
        *,
        task: str,
    ) -> tuple[str, ...]:
        allowed = frozenset(self.executable_leaves_for_task(task))
        validated: list[str] = []
        for category in categories:
            if not isinstance(category, str) or category not in self._leaves:
                raise CatalogCategoryError(
                    "PLAN_CATEGORY_NOT_CANONICAL_LEAF",
                    categories=[str(category)],
                    available=tuple(allowed),
                )
            if category not in allowed:
                raise CatalogCategoryError(
                    "PLAN_CATEGORY_NOT_EXECUTABLE",
                    categories=[category],
                    available=tuple(allowed),
                )
            if category in validated:
                raise CatalogCategoryError(
                    "PLAN_CATEGORY_DUPLICATED", categories=[category]
                )
            validated.append(category)
        return tuple(validated)

    def executable_leaves_for_target(
        self,
        target: str,
        *,
        task: str,
    ) -> tuple[str, ...]:
        leaves = self.expand_target(target)
        allowed = frozenset(self.executable_leaves_for_task(task))
        if not leaves or any(leaf not in allowed for leaf in leaves):
            return ()
        return leaves

    def leaf_yolo_labels(self, leaf: str) -> tuple[str, ...]:
        return tuple(self._leaf(leaf).yolo_labels)

    def leaf_segformer_labels(self, leaf: str) -> tuple[str, ...] | None:
        labels = self._leaf(leaf).segformer_labels
        return None if labels is None else tuple(labels)

    def leaf_segformer_binding(self, leaf: str) -> str | None:
        """Return the stable logical segmenter binding of one leaf; unknown
        leaves keep the stable catalog failure. 返回单个叶子的稳定逻辑
        segmenter binding；未知叶子保持稳定 catalog 失败。"""
        return self._leaf(leaf).segformer_binding

    def capability_enabled(self, leaf: str, capability: ModelCapability) -> bool:
        spec = self._leaf(leaf)
        return spec.yolo_enabled if capability == "yolo" else spec.segformer_enabled

    def capability_identity(self, leaf: str, capability: ModelCapability) -> str:
        self._leaf(leaf)
        return f"{self._version}:{leaf}:{capability}"

    @classmethod
    def from_file(cls, path: Path | str) -> "EvidenceCatalog":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CatalogCategoryError("CATALOG_LOAD_FAILED") from exc
        return cls(payload)

    def _leaf(self, leaf: str) -> LeafCapabilities:
        try:
            return self._leaves[leaf]
        except KeyError:
            raise CatalogCategoryError(
                "UNKNOWN_LEAF", categories=[leaf], available=self.leaf_categories
            ) from None


def load_evidence_catalog(path: Path | str) -> EvidenceCatalog:
    return EvidenceCatalog.from_file(path)


def _normalize_semantic(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CatalogCategoryError("INVALID_CATEGORY_NAME", categories=[str(value)])
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise CatalogCategoryError("INVALID_CATEGORY_NAME", categories=[value])
    if "/" in value or "\\" in value or value.strip() in {".", ".."}:
        raise CatalogCategoryError("INVALID_CATEGORY_NAME", categories=[value])
    normalized = re.sub(
        r"-+", "-", re.sub(r"[_\s]+", "-", value.strip().casefold())
    ).strip("-")
    if re.fullmatch(_CATEGORY_PATTERN, normalized) is None:
        raise CatalogCategoryError("INVALID_CATEGORY_NAME", categories=[value])
    return normalized


def _validate_canonical_name(value: Any) -> None:
    if not isinstance(value, str) or re.fullmatch(_CATEGORY_PATTERN, value) is None:
        raise CatalogCategoryError("INVALID_CATEGORY_NAME", categories=[str(value)])


def _validate_model_label(label: Any, capability: str) -> None:
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"{capability} label must be a non-empty string")
    if any(ord(character) < 32 for character in label):
        raise ValueError(f"{capability} label contains control characters")
    if label.startswith(("/", "\\")) or "\\" in label or ":" in label:
        raise ValueError(f"{capability} label must not be path-like")
    if _PLACEHOLDER_PATTERN.fullmatch(label):
        raise ValueError(f"{capability} placeholder label is not verified")


def _require_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CatalogCategoryError("INVALID_CATALOG_STRUCTURE")
    return value


def _check_no_sensitive_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_").replace(" ", "_")
            if normalized in _SENSITIVE_KEYS:
                raise CatalogCategoryError(
                    "SENSITIVE_KEY_IN_CATALOG", categories=[str(key)]
                )
            _check_no_sensitive_keys(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _check_no_sensitive_keys(item)


__all__ = [
    "CatalogCategoryError", "EvidenceCatalog", "LeafCapabilities",
    "ModelCapability", "load_evidence_catalog",
]
