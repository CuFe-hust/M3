"""Validated, data-driven capability catalog for counting experts.

This module only describes expert capabilities.  It never constructs models,
checks runtime availability, or selects a backend for execution.
"""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Literal, Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from agents.counting.schema import CountTargetSpec

ExpertKind = Literal[
    "yolo_obb",
    "semantic_segmentation",
    "quantity_proposal",
    "qwen_point",
]
CountingMode = Literal[
    "native_detection",
    "connected_components",
    "grounded_localization",
    "point_counting",
    "unsupported",
]
ExpertStatus = Literal["active", "blocked_unverified_class_map"]

_EXPERT_KIND_ORDER: dict[str, int] = {
    "yolo_obb": 0,
    "semantic_segmentation": 1,
    "quantity_proposal": 2,
    "qwen_point": 3,
}
_PLACEHOLDER_LABEL = re.compile(r"^label[_\-\s]*\d+$", re.IGNORECASE)


class ExpertCatalogError(ValueError):
    """A catalog could not be read or did not satisfy its contract."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CatalogTargetSpec(_FrozenModel):
    """One canonical target and its explicit, dataset-neutral metadata."""

    aliases: tuple[str, ...] = ()
    countable: bool
    hints: tuple[str, ...] = ()

    @field_validator("aliases", "hints")
    @classmethod
    def values_must_be_nonempty_and_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        stripped = tuple(value.strip() for value in values)
        if any(not value for value in stripped):
            raise ValueError("catalog target values must be non-empty")
        normalized = tuple(_normalize_label(value) for value in stripped)
        if len(normalized) != len(set(normalized)):
            raise ValueError("catalog target values must be unique after normalization")
        return stripped


class ExpertAssetSpec(_FrozenModel):
    """Portable references and immutable identity for one local asset."""

    model_dir: str
    class_map: str | None = None
    weights: str | None = None
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("model_dir", "class_map", "weights")
    @classmethod
    def paths_must_be_repository_relative(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().replace("\\", "/")
        path = PurePosixPath(normalized)
        if (
            not normalized
            or path.is_absolute()
            or re.match(r"^[a-zA-Z]:", normalized)
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("expert asset paths must be repository-relative")
        return path.as_posix()


class ClassMapVerificationSpec(_FrozenModel):
    class_map: Literal["verified", "unverified"]


class CountingPolicySpec(_FrozenModel):
    """Optional, backend-neutral thresholds declared by a capability."""

    min_component_area_px: int | None = Field(default=None, ge=1)
    min_mean_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ExpertTargetSupportSpec(_FrozenModel):
    """How one expert represents and counts one canonical target."""

    model_labels: tuple[str, ...] = Field(min_length=1)
    counting_mode: CountingMode
    policy: CountingPolicySpec = Field(default_factory=CountingPolicySpec)

    @field_validator("model_labels")
    @classmethod
    def model_labels_must_be_explicit(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        stripped = tuple(value.strip() for value in values)
        if any(not value for value in stripped):
            raise ValueError("model labels must be non-empty")
        if any(_PLACEHOLDER_LABEL.fullmatch(value) for value in stripped):
            raise ValueError("placeholder model labels are not capability mappings")
        folded = tuple(value.casefold() for value in stripped)
        if len(folded) != len(set(folded)):
            raise ValueError("model labels must be unique")
        return stripped


class ExpertSpec(_FrozenModel):
    """Validated declaration for one counting expert."""

    backend_name: str = Field(min_length=1)
    kind: ExpertKind
    logical_model_id: str = Field(min_length=1)
    enabled: bool
    status: ExpertStatus = "active"
    priority: int = Field(ge=0)
    asset: ExpertAssetSpec
    verification: ClassMapVerificationSpec
    supports: dict[str, ExpertTargetSupportSpec] = Field(default_factory=dict)

    @field_validator("backend_name", "logical_model_id")
    @classmethod
    def identifiers_must_be_trimmed(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("expert identifiers must be non-empty")
        return stripped

    @model_validator(mode="after")
    def validate_status_and_mode_contract(self) -> "ExpertSpec":
        if self.enabled and self.status != "active":
            raise ValueError("blocked experts cannot be enabled")
        if self.kind == "semantic_segmentation" and self.enabled:
            if self.verification.class_map != "verified" or self.asset.class_map is None:
                raise ValueError("enabled semantic experts require a verified class map")

        allowed_modes: dict[str, frozenset[str]] = {
            "yolo_obb": frozenset({"native_detection", "unsupported"}),
            "semantic_segmentation": frozenset({"connected_components", "unsupported"}),
            "quantity_proposal": frozenset({"grounded_localization", "unsupported"}),
            "qwen_point": frozenset({"point_counting", "unsupported"}),
        }
        if any(
            support.counting_mode not in allowed_modes[self.kind]
            for support in self.supports.values()
        ):
            raise ValueError("counting mode is incompatible with expert kind")
        for support in self.supports.values():
            if support.counting_mode == "connected_components" and (
                support.policy.min_component_area_px is None
                or support.policy.min_mean_confidence is None
            ):
                raise ValueError("connected-components support requires component thresholds")
        return self


class _CatalogDocument(_FrozenModel):
    schema_version: Literal[1]
    targets: dict[str, CatalogTargetSpec]
    experts: tuple[ExpertSpec, ...]

    @model_validator(mode="after")
    def validate_catalog_relations(self) -> "_CatalogDocument":
        aliases: dict[str, str] = {}
        for canonical, target in self.targets.items():
            if canonical != _normalize_label(canonical) or _PLACEHOLDER_LABEL.fullmatch(canonical):
                raise ValueError("target keys must be normalized canonical labels")
            for label in (canonical, *target.aliases):
                normalized = _normalize_label(label)
                if _PLACEHOLDER_LABEL.fullmatch(label):
                    raise ValueError("placeholder labels cannot define catalog targets")
                owner = aliases.get(normalized)
                if owner is not None and owner != canonical:
                    raise ValueError("target alias resolves to multiple canonical targets")
                aliases[normalized] = canonical

        backend_names: set[str] = set()
        for expert in self.experts:
            backend_key = expert.backend_name.casefold()
            if backend_key in backend_names:
                raise ValueError("duplicate expert backend name")
            backend_names.add(backend_key)
            unknown_targets = set(expert.supports) - set(self.targets)
            if unknown_targets:
                raise ValueError("expert supports an unknown canonical target")
            if any(target != _normalize_label(target) for target in expert.supports):
                raise ValueError("expert support keys must be normalized canonical labels")
        return self


class ExpertCatalog:
    """Immutable catalog view with explicit aliasing and stable ordering."""

    def __init__(self, document: _CatalogDocument) -> None:
        aliases: dict[str, str] = {}
        for canonical, target in document.targets.items():
            for label in (canonical, *target.aliases):
                aliases[_normalize_label(label)] = canonical
        self._targets: Mapping[str, CatalogTargetSpec] = MappingProxyType(dict(document.targets))
        self._aliases: Mapping[str, str] = MappingProxyType(aliases)
        self._experts = tuple(document.experts)
        self._by_backend: Mapping[str, ExpertSpec] = MappingProxyType(
            {expert.backend_name: expert for expert in document.experts}
        )

    @classmethod
    def load(cls, path: Path) -> "ExpertCatalog":
        """Load a catalog without exposing the host path in public errors."""

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError):
            raise ExpertCatalogError("expert catalog could not be read") from None
        except json.JSONDecodeError:
            raise ExpertCatalogError("expert catalog is not valid JSON") from None
        try:
            document = _CatalogDocument.model_validate(payload)
        except ValidationError:
            raise ExpertCatalogError("expert catalog validation failed") from None
        return cls(document)

    def target_hints(self, target: CountTargetSpec) -> dict[str, object]:
        """Return target-only metadata; unknown targets have no hints."""

        canonical = self._canonical_target(target)
        if canonical is None:
            return {}
        spec = self._targets[canonical]
        return {
            "canonical_label": canonical,
            "countable": spec.countable,
            "hints": list(spec.hints),
        }

    def candidates(
        self,
        target: CountTargetSpec,
        *,
        kinds: frozenset[str] | None = None,
        enabled_only: bool = True,
    ) -> tuple[ExpertSpec, ...]:
        """Return explicit supporters in kind/priority/name order."""

        canonical = self._canonical_target(target)
        if canonical is None or not self._targets[canonical].countable:
            return ()
        if kinds is not None:
            unknown_kinds = kinds - frozenset(_EXPERT_KIND_ORDER)
            if unknown_kinds:
                raise ValueError("unknown expert kind filter")
        candidates = (
            expert
            for expert in self._experts
            if (not enabled_only or expert.enabled)
            and (kinds is None or expert.kind in kinds)
            and canonical in expert.supports
            and expert.supports[canonical].counting_mode != "unsupported"
        )
        return tuple(
            sorted(
                candidates,
                key=lambda expert: (
                    _EXPERT_KIND_ORDER[expert.kind],
                    -expert.priority,
                    expert.backend_name,
                ),
            )
        )

    def expert(self, backend_name: str) -> ExpertSpec:
        """Return the exact backend declaration."""

        try:
            return self._by_backend[backend_name]
        except KeyError:
            raise KeyError("unknown expert backend") from None

    def _canonical_target(self, target: CountTargetSpec) -> str | None:
        return self._aliases.get(_normalize_label(target.canonical_label))


def _normalize_label(value: str) -> str:
    """Normalize separators and case without introducing fuzzy semantics."""

    folded = value.strip().casefold()
    return re.sub(r"-+", "-", re.sub(r"[_\s]+", "-", folded)).strip("-")


__all__ = [
    "CatalogTargetSpec",
    "CountingMode",
    "CountingPolicySpec",
    "ExpertAssetSpec",
    "ExpertCatalog",
    "ExpertCatalogError",
    "ExpertKind",
    "ExpertSpec",
    "ExpertTargetSupportSpec",
]
