"""Versioned closed category catalog shared by VQA and Grounding protocols.

VQA/Grounding 共用的版本化封闭类别目录。本模块只维护组合类别展开、叶子类别
到两类模型输出标签的映射与逻辑能力身份等纯事实：不执行任何模型、不读取
Ground Truth、不保存物理路径、不导入重依赖。第一次 Qwen prompt、plan 校验、
组合类别展开、模型能力判断和结果筛选都必须读取同一目录版本；目录外或部分
非法类别的容错策略未被批准时严格失败。Grounding 只消费 YOLO mapping，VQA
可消费两类 mapping；未校准能力保持 disabled。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# The two model capabilities the catalog maps leaf categories to.
# 目录为叶子类别映射的两类模型能力。
ModelCapability = Literal["yolo", "segformer"]

_CATEGORY_PATTERN = r"^[a-z][a-z0-9_]*$"
_VERSION_PATTERN = r"^[a-z0-9][a-z0-9_.-]*$"

# Keys that must never appear anywhere in the catalog data; the catalog is a
# static asset, so a sensitive key indicates corruption, not configuration.
# 目录数据中任何位置都不得出现的键；目录是静态资产，敏感键说明数据损坏。
_SENSITIVE_KEYS = frozenset({
    "api_key", "apikey", "authorization", "access_token", "refresh_token",
    "private_key", "password", "credential", "token", "secret",
})


class CatalogCategoryError(ValueError):
    """Stable error for catalog validation failures; the public message
    carries only the stable code and category names, never file contents or
    machine paths. 目录校验失败的稳定错误；公共消息只携带稳定 code 与类别名，
    绝不携带文件内容或机器路径。"""

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
    """Verified output-label mappings for one leaf category. An enabled
    capability requires verified labels; labels without an enabled flag stay
    declared but uncalibrated (disabled). segformer_labels is None when no
    SegFormer mapping is declared at all.
    单个叶子类别的已验证输出标签映射。启用的能力必须有已验证标签；有标签但
    未启用表示已声明但未校准（禁用）。未声明任何 SegFormer 映射时
    segformer_labels 为 None。"""

    model_config = ConfigDict(extra="forbid")

    yolo_labels: list[str] = Field(default_factory=list)
    segformer_labels: list[str] | None = None
    yolo_enabled: bool = False
    segformer_enabled: bool = False

    @model_validator(mode="after")
    def validate_capabilities(self) -> "LeafCapabilities":
        """Validate every label and the enabled/labels linkage. Labels are
        exact model class names: non-empty, no control characters, never a
        local path. 校验每条标签及 enabled/labels 联动。标签是模型类别名的
        精确字符串：非空、无控制字符、绝不是一个本地路径。"""
        for label in self.yolo_labels:
            _validate_label(label, "yolo")
        if self.segformer_labels is not None:
            if not self.segformer_labels:
                raise ValueError("segformer_labels must be null or non-empty")
            for label in self.segformer_labels:
                _validate_label(label, "segformer")
        if self.yolo_enabled and not self.yolo_labels:
            raise ValueError("yolo_enabled requires verified yolo_labels")
        if self.segformer_enabled and not self.segformer_labels:
            raise ValueError("segformer_enabled requires verified segformer_labels")
        return self


def _validate_label(label: Any, capability: str) -> None:
    """One verified model output label; absolute-path-like strings are
    rejected so a physical path can never masquerade as a label.
    一条已验证模型输出标签；类绝对路径字符串被拒绝，使物理路径无法冒充标签。"""
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"{capability} label must be a non-empty string")
    if any(ord(character) < 32 for character in label):
        raise ValueError(f"{capability} label contains control characters")
    if label.startswith(("/", "\\")) or "\\" in label or ":" in label:
        raise ValueError(f"{capability} label must not be path-like: {label!r}")


class EvidenceCatalog:
    """Validated, versioned, closed category catalog. All consumers of the
    same instance share one version; out-of-catalog categories fail strictly
    until a tolerance policy is approved.
    校验后的版本化封闭类别目录。同一实例的所有消费者共享一个版本；目录外
    类别在容错策略获批前严格失败。"""

    def __init__(self, data: Mapping[str, Any]) -> None:
        if not isinstance(data, Mapping):
            raise TypeError("catalog data must be a mapping")
        _check_no_sensitive_keys(data)
        top_level = set(data)
        expected = {"catalog_version", "composites", "leaves"}
        if top_level != expected:
            raise CatalogCategoryError(
                "INVALID_TOP_LEVEL_KEYS", categories=sorted(top_level - expected)
            )
        version = data["catalog_version"]
        if not isinstance(version, str) or not version.strip():
            raise CatalogCategoryError("INVALID_CATALOG_VERSION")
        if any(ord(character) < 32 for character in version):
            raise CatalogCategoryError("INVALID_CATALOG_VERSION")
        if not _matches(version, _VERSION_PATTERN):
            raise CatalogCategoryError("INVALID_CATALOG_VERSION")
        self._version = version

        composites_raw = data["composites"]
        leaves_raw = data["leaves"]
        if not isinstance(composites_raw, Mapping) or not isinstance(leaves_raw, Mapping):
            raise CatalogCategoryError("INVALID_CATALOG_STRUCTURE")

        leaves: dict[str, LeafCapabilities] = {}
        for name, capabilities in leaves_raw.items():
            _validate_category_name(name)
            if not isinstance(capabilities, Mapping):
                raise CatalogCategoryError(
                    "INVALID_LEAF_CAPABILITIES", categories=[name]
                )
            try:
                leaves[name] = LeafCapabilities.model_validate(dict(capabilities))
            except Exception as exc:
                raise CatalogCategoryError(
                    "INVALID_LEAF_CAPABILITIES", categories=[name]
                ) from exc
        self._leaves = leaves

        composites: dict[str, tuple[str, ...]] = {}
        for name, targets in composites_raw.items():
            _validate_category_name(name)
            if name in leaves:
                raise CatalogCategoryError(
                    "COMPOSITE_NAME_COLLIDES_WITH_LEAF", categories=[name]
                )
            if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
                raise CatalogCategoryError(
                    "INVALID_COMPOSITE_TARGETS", categories=[name]
                )
            expanded: list[str] = []
            for target in targets:
                _validate_category_name(target)
                if target not in leaves:
                    raise CatalogCategoryError(
                        "COMPOSITE_TARGETS_UNKNOWN_LEAF", categories=[name, target]
                    )
                if target in expanded:
                    raise CatalogCategoryError(
                        "COMPOSITE_TARGETS_DUPLICATED", categories=[name, target]
                    )
                expanded.append(target)
            composites[name] = tuple(expanded)
        self._composites = composites

    # ── identity and shape / 身份与结构 ─────────────────────────────────

    @property
    def catalog_version(self) -> str:
        """The version every consumer of this instance must share.
        该实例所有消费者必须共享的版本。"""
        return self._version

    @property
    def composite_categories(self) -> tuple[str, ...]:
        """All composite categories in stable declaration order.
        按稳定声明顺序的全部组合类别。"""
        return tuple(self._composites)

    @property
    def leaf_categories(self) -> tuple[str, ...]:
        """All leaf categories in stable declaration order.
        按稳定声明顺序的全部叶子类别。"""
        return tuple(self._leaves)

    def is_composite(self, name: str) -> bool:
        """Return whether name is a declared composite category.
        返回 name 是否为已声明组合类别。"""
        return name in self._composites

    def is_leaf(self, name: str) -> bool:
        """Return whether name is a declared leaf category.
        返回 name 是否为已声明叶子类别。"""
        return name in self._leaves

    # ── expansion / 展开 ────────────────────────────────────────────────

    def composite_leaves(self, name: str) -> tuple[str, ...]:
        """Ordered leaf categories of one composite; unknown composites fail
        with a stable error. 一个组合类别的有序叶子类别；未知组合以稳定错误
        失败。"""
        if name not in self._composites:
            raise CatalogCategoryError(
                "UNKNOWN_COMPOSITE", categories=[name], available=self.composite_categories
            )
        return self._composites[name]

    def expand_composites(self, categories: Sequence[str]) -> tuple[str, ...]:
        """Expand requested composite categories to ordered leaf categories,
        deduplicated stably while preserving first occurrence. Every category
        must be a known composite; direct leaf names are rejected.
        将请求的组合类别展开为有序叶子类别，稳定去重并保留首次出现顺序。每个
        类别必须是已知组合；直接请求叶子名被拒绝。"""
        expanded: list[str] = []
        for category in categories:
            leaves = self.composite_leaves(category)
            for leaf in leaves:
                if leaf not in expanded:
                    expanded.append(leaf)
        return tuple(expanded)

    def validate_plan_categories(self, categories: Sequence[str]) -> None:
        """Strictly require every planned composite category to belong to this
        catalog version; no tolerance policy has been approved, so unknown or
        non-composite categories fail instead of being guessed.
        严格要求每个计划组合类别属于本目录版本；容错策略尚未批准，因此未知或
        非组合类别严格失败，绝不猜测。"""
        invalid = [
            category
            for category in categories
            if not isinstance(category, str) or category not in self._composites
        ]
        if invalid:
            raise CatalogCategoryError(
                "UNKNOWN_COMPOSITE",
                categories=[str(item) for item in invalid],
                available=self.composite_categories,
            )

    # ── capabilities / 能力 ─────────────────────────────────────────────

    def leaf_yolo_labels(self, leaf: str) -> tuple[str, ...]:
        """Verified YOLO output labels for one leaf (empty while uncalibrated).
        The catalog is a closed static asset, so callers get immutable tuples.
        单个叶子的已验证 YOLO 输出标签（未校准时为空）。目录是封闭静态资产，
        调用方获得不可变元组。"""
        return tuple(self._leaf(leaf).yolo_labels)

    def leaf_segformer_labels(self, leaf: str) -> tuple[str, ...] | None:
        """Verified SegFormer output labels for one leaf, or None when no
        SegFormer mapping is declared. 单个叶子的已验证 SegFormer 输出标签；
        未声明 SegFormer 映射时为 None。"""
        caps = self._leaf(leaf)
        if caps.segformer_labels is None:
            return None
        return tuple(caps.segformer_labels)

    def capability_enabled(self, leaf: str, capability: ModelCapability) -> bool:
        """Whether the approved model capability is calibrated and enabled for
        one leaf. Uncalibrated capabilities stay disabled.
        单个叶子已批准模型能力是否已校准并启用。未校准能力保持禁用。"""
        leaf_caps = self._leaf(leaf)
        if capability == "yolo":
            return leaf_caps.yolo_enabled
        return leaf_caps.segformer_enabled

    def capability_identity(self, leaf: str, capability: ModelCapability) -> str:
        """Stable logical capability identity: version + leaf + capability.
        Never a physical path or a model-specific name.
        稳定逻辑能力身份：版本 + 叶子 + 能力。绝不是一个物理路径或模型专属名。"""
        self._leaf(leaf)
        return f"{self._version}:{leaf}:{capability}"

    # ── loading / 加载 ──────────────────────────────────────────────────

    @classmethod
    def from_file(cls, path: Path | str) -> "EvidenceCatalog":
        """Load one catalog JSON asset; parse and validation errors stay
        stable and never leak file contents. 加载一个目录 JSON 资产；解析与
        校验错误保持稳定且绝不泄漏文件内容。"""
        try:
            text = Path(path).read_text(encoding="utf-8")
            data = json.loads(text)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CatalogCategoryError("CATALOG_LOAD_FAILED") from exc
        return cls(data)

    # ── internals / 内部 ────────────────────────────────────────────────

    def _leaf(self, leaf: str) -> LeafCapabilities:
        if leaf not in self._leaves:
            raise CatalogCategoryError(
                "UNKNOWN_LEAF", categories=[leaf], available=self.leaf_categories
            )
        return self._leaves[leaf]


def load_evidence_catalog(path: Path | str) -> EvidenceCatalog:
    """Convenience loader for the shared catalog asset. 共享目录资产的便利加载器。"""
    return EvidenceCatalog.from_file(path)


def _validate_category_name(name: Any) -> None:
    if not isinstance(name, str) or not _matches(name, _CATEGORY_PATTERN):
        raise CatalogCategoryError("INVALID_CATEGORY_NAME", categories=[str(name)])


def _matches(value: str, pattern: str) -> bool:
    return re.fullmatch(pattern, value) is not None


def _check_no_sensitive_keys(value: Any) -> None:
    """Reject sensitive key names anywhere in the catalog data.
    拒绝目录数据中任何位置的敏感键名。"""
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_").replace(" ", "_")
            if normalized in _SENSITIVE_KEYS:
                raise CatalogCategoryError("SENSITIVE_KEY_IN_CATALOG", categories=[str(key)])
            _check_no_sensitive_keys(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _check_no_sensitive_keys(item)
