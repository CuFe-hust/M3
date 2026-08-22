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
    "yolo_detect",
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
    "yolo_detect": 0,
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
    executable_leaf: bool = True
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


class ChangeSemanticSpec(_FrozenModel):
    """Verified, model-neutral semantic roles for Change evidence."""

    enabled: bool = False
    participation: Literal["core", "rescue", "diagnostic"] = "core"
    role: Literal["generic", "persistent_landcover", "object_semantic"] = "generic"
    neutral_model_labels: tuple[str, ...] = ()
    transient_model_labels: tuple[str, ...] = ()
    structural_model_labels: tuple[str, ...] = ()
    landcover_candidate_model_labels: tuple[str, ...] = ()

    rescue_model_labels: tuple[str, ...] = ()
    rescue_strategy: Literal["none", "building_footprint_delta", "edge_corner_building"] = "none"

    # Backward-compatible alias for older catalogs.  New code must not treat
    # this field as proof that every class flip is persistent.
    persistent_model_labels: tuple[str, ...] = ()

    @field_validator(
        "neutral_model_labels",
        "transient_model_labels",
        "structural_model_labels",
        "landcover_candidate_model_labels",
        "persistent_model_labels",
        "rescue_model_labels",
    )
    @classmethod
    def labels_are_explicit(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(value.strip() for value in values)
        if any(not value or _PLACEHOLDER_LABEL.fullmatch(value) for value in cleaned):
            raise ValueError("Change semantic labels must be verified explicit names")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("Change semantic labels must be unique")
        return cleaned

    @model_validator(mode="after")
    def roles_do_not_overlap(self) -> "ChangeSemanticSpec":
        groups = (
            set(self.neutral_model_labels),
            set(self.transient_model_labels),
            set(self.structural_model_labels),
            set(self.landcover_candidate_model_labels),
            set(self.persistent_model_labels),
        )
        # The legacy persistent list is a compatibility alias and may overlap
        # with the new structural/land-cover tiers.  Validate only the
        # mutually exclusive semantic roles themselves.
        semantic_groups = groups[:4]
        if any(
            semantic_groups[index] & semantic_groups[other]
            for index in range(len(semantic_groups))
            for other in range(index)
        ):
            raise ValueError("Change semantic label roles must not overlap")
        if self.participation == "rescue" and self.rescue_strategy == "none":
            raise ValueError("rescue semantic experts require an explicit rescue strategy")
        if self.participation != "rescue" and self.rescue_model_labels:
            raise ValueError("only rescue semantic experts may declare rescue labels")
        return self


class MorphologyPolicySpec(_FrozenModel):
    """Explicit morphology only; zero disables the operation.
    仅允许显式形态学配置；零表示关闭对应操作。"""

    open_kernel: int = Field(default=0, ge=0, le=31)
    close_kernel: int = Field(default=0, ge=0, le=31)

    @model_validator(mode="after")
    def kernels_are_disabled_or_odd(self) -> "MorphologyPolicySpec":
        if any(value and value % 2 == 0 for value in (self.open_kernel, self.close_kernel)):
            raise ValueError("morphology kernels must be zero or odd")
        return self


class CountingPolicySpec(_FrozenModel):
    """Thresholds explicitly declared by one expert capability.
    单项专家能力显式声明的阈值。"""

    min_component_area_px: int | None = Field(default=None, ge=1)
    max_component_area_ratio: float | None = Field(default=None, gt=0.0, le=1.0)
    min_mean_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    morphology: MorphologyPolicySpec = Field(default_factory=MorphologyPolicySpec)


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
    change_semantics: ChangeSemanticSpec | None = None

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
        if self.change_semantics is not None and self.change_semantics.enabled:
            if self.kind != "semantic_segmentation" or not self.enabled:
                raise ValueError("Change semantic experts must be enabled semantic experts")
            if self.verification.class_map != "verified":
                raise ValueError("Change semantic experts require a verified class map")

        allowed_modes: dict[str, frozenset[str]] = {
            "yolo_obb": frozenset({"native_detection", "unsupported"}),
            "yolo_detect": frozenset({"native_detection", "unsupported"}),
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
                or support.policy.max_component_area_ratio is None
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
            if any(
                not self.targets[target].executable_leaf for target in expert.supports
            ):
                raise ValueError("expert supports must contain canonical leaves only")
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
    def load(
        cls,
        path: Path,
        *,
        asset_root: Path | None = None,
    ) -> "ExpertCatalog":
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
        try:
            _validate_verified_semantic_labels(
                document,
                _catalog_asset_root(path, document, asset_root),
            )
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
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

    def experts(
        self,
        *,
        kinds: frozenset[str] | None = None,
        enabled_only: bool = True,
    ) -> tuple[ExpertSpec, ...]:
        """Enumerate immutable declarations in stable routing order.
        按稳定路由顺序枚举不可变专家声明。"""

        if kinds is not None:
            unknown_kinds = kinds - frozenset(_EXPERT_KIND_ORDER)
            if unknown_kinds:
                raise ValueError("unknown expert kind filter")
        experts = (
            expert
            for expert in self._experts
            if (not enabled_only or expert.enabled)
            and (kinds is None or expert.kind in kinds)
        )
        return tuple(
            sorted(
                experts,
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


def _catalog_asset_root(
    path: Path,
    document: _CatalogDocument,
    explicit: Path | None,
) -> Path:
    """Resolve the portable catalog asset root without leaking its path.
    解析可移植 catalog 资产根，且不在错误中泄漏路径。"""

    references = tuple(
        expert.asset.class_map
        for expert in document.experts
        if expert.enabled and expert.kind == "semantic_segmentation"
    )
    if explicit is not None:
        return explicit
    for candidate in (path.parent, *path.parents):
        if all(reference and (candidate / reference).is_file() for reference in references):
            return candidate
    raise ValueError("verified class-map assets are unavailable")


def _validate_verified_semantic_labels(
    document: _CatalogDocument,
    asset_root: Path,
) -> None:
    """Reject semantic capabilities outside their verified class map.
    拒绝超出 verified class map 的语义能力声明。"""

    for expert in document.experts:
        if not expert.enabled or expert.kind != "semantic_segmentation":
            continue
        if expert.asset.class_map is None:
            raise ValueError("verified semantic class map is missing")
        payload = json.loads(
            (asset_root / expert.asset.class_map).read_text(encoding="utf-8")
        )
        raw_labels = payload["id2name"]
        if not isinstance(raw_labels, dict) or not raw_labels:
            raise ValueError("verified semantic class map is invalid")
        labels = set(raw_labels.values())
        declared = {
            label
            for support in expert.supports.values()
            for label in support.model_labels
        }
        if not declared.issubset(labels):
            raise ValueError("semantic capability label is not verified")
        change_semantics = expert.change_semantics
        if change_semantics is not None and change_semantics.enabled:
            change_labels = {
                *change_semantics.neutral_model_labels,
                *change_semantics.transient_model_labels,
                *change_semantics.structural_model_labels,
                *change_semantics.landcover_candidate_model_labels,
                *change_semantics.persistent_model_labels,
            }
            if not change_labels.issubset(labels):
                raise ValueError("Change semantic label is not verified")


__all__ = [
    "CatalogTargetSpec",
    "CountingMode",
    "CountingPolicySpec",
    "ChangeSemanticSpec",
    "ExpertAssetSpec",
    "ExpertCatalog",
    "ExpertCatalogError",
    "ExpertKind",
    "ExpertSpec",
    "ExpertTargetSupportSpec",
]
