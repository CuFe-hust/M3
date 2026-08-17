"""Deterministic reconciliation of semantic counting targets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from agents.counting.expert_catalog import ExpertCatalog
from agents.counting.schema import CountTargetSpec
from agents.evidence_catalog import CatalogCategoryError, EvidenceCatalog

APPROVED_SCOPE_MODIFIERS = frozenset({"small", "large"})
APPROVED_COMMON_VARIANTS = {"vehical": "vehicle", "vehicals": "vehicle"}
_APPROVED_PLURALS = {
    "aircrafts": "aircraft", "buildings": "building", "cars": "car",
    "harbors": "harbor", "harbours": "harbour", "helicopters": "helicopter",
    "planes": "plane", "ships": "ship", "storage-tanks": "storage-tank",
    "vehicles": "vehicle",
}
_COUNT_QUESTION = re.compile(
    r"^(?:how\s+many|count\s+the\s+number\s+of)\s+(.+?)(?:[?!.]|$)", re.I
)


class CountTargetResolutionError(ValueError):
    """Stable, path-free target-resolution failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ResolvedCountTarget:
    target: CountTargetSpec
    executable_leaf_categories: tuple[str, ...]
    target_source: str
    validation_status: str
    verifier_source: str | None
    planner_target: str | None
    planner_object_categories: tuple[str, ...]


@dataclass(frozen=True)
class _Verifier:
    target: str
    source: str
    spec: CountTargetSpec | None


class CountTargetResolver:
    """Resolve planner proposals and approved hints without model calls."""

    def __init__(
        self, *, evidence_catalog: EvidenceCatalog,
        expert_catalog: ExpertCatalog | None = None,
    ) -> None:
        self._evidence_catalog = evidence_catalog
        self._expert_catalog = expert_catalog

    def resolve(
        self, *, question: str, planner_target: str | None,
        planner_object_categories: tuple[str, ...],
        count_target_hint: dict[str, Any] | str | None,
        legacy_metadata: dict[str, Any] | None,
    ) -> ResolvedCountTarget:
        verifier = self._resolve_verifier(
            question=question, count_target_hint=count_target_hint,
            legacy_metadata=legacy_metadata,
        )
        if planner_target is None:
            if planner_object_categories:
                raise CountTargetResolutionError("COUNT_TARGET_INVALID")
            if verifier is None:
                raise CountTargetResolutionError("COUNT_TARGET_SOURCE_REQUIRED")
            leaves = self._expand_semantic_target(verifier.target)
            return ResolvedCountTarget(
                target=self._build_target_spec(verifier.target, verifier.spec),
                executable_leaf_categories=leaves,
                target_source="explicit_hint", validation_status="explicit_hint",
                verifier_source=verifier.source, planner_target=None,
                planner_object_categories=(),
            )

        planner = self._normalize_semantic_target(
            planner_target, error_code="COUNT_TARGET_INVALID"
        )
        try:
            planner_leaves = self._evidence_catalog.validate_plan_leaves(
                planner_object_categories, task="counting"
            )
        except (CatalogCategoryError, TypeError):
            raise CountTargetResolutionError(
                "COUNT_TARGET_PLANNER_LEAVES_INVALID"
            ) from None
        expected = self._expand_semantic_target(planner)
        if expected and any(leaf not in expected for leaf in planner_leaves):
            raise CountTargetResolutionError(
                "COUNT_TARGET_PLANNER_LEAVES_INVALID"
            )

        final_target, status = self._reconcile(
            planner=planner, planner_leaves=planner_leaves,
            verifier=None if verifier is None else verifier.target,
        )
        return ResolvedCountTarget(
            target=self._build_target_spec(
                final_target, None if verifier is None else verifier.spec
            ),
            executable_leaf_categories=self._expand_semantic_target(final_target),
            target_source="visual_task_plan", validation_status=status,
            verifier_source=None if verifier is None else verifier.source,
            planner_target=planner, planner_object_categories=planner_leaves,
        )

    def _resolve_verifier(
        self, *, question: str,
        count_target_hint: dict[str, Any] | str | None,
        legacy_metadata: dict[str, Any] | None,
    ) -> _Verifier | None:
        if count_target_hint is not None:
            return self._verifier_from_hint(
                count_target_hint, source="normalization.count_target_hint", question=question
            )
        if isinstance(legacy_metadata, dict) and "count_target_hint" in legacy_metadata:
            return self._verifier_from_hint(
                legacy_metadata["count_target_hint"],
                source="legacy_metadata.count_target_hint",
                question=question,
            )
        return None

    def _verifier_from_hint(
        self, hint: Any, *, source: str, question: str,
    ) -> _Verifier:
        spec: CountTargetSpec | None = None
        if isinstance(hint, dict):
            try:
                spec = CountTargetSpec.model_validate(hint)
            except ValidationError:
                raise CountTargetResolutionError(
                    "COUNT_TARGET_VERIFIER_INVALID"
                ) from None
            raw_target = spec.canonical_label
        elif isinstance(hint, str) and hint.strip():
            match = _COUNT_QUESTION.match(hint.strip())
            raw_target = match.group(1) if match else hint
            raw_target = re.sub(
                r"\b(?:are|is)\s+(?:visible|there)\b.*$", "", raw_target, flags=re.I
            ).strip(" ,。")
        else:
            raise CountTargetResolutionError("COUNT_TARGET_VERIFIER_INVALID")
        target = self._normalize_semantic_target(
            raw_target, error_code="COUNT_TARGET_VERIFIER_INVALID"
        )
        return _Verifier(
            target=self._preserve_specificity(question, target), source=source, spec=spec
        )

    def _normalize_semantic_target(self, value: str, *, error_code: str) -> str:
        try:
            normalized = re.sub(
                r"-+", "-", re.sub(r"[_\s]+", "-", value.strip().casefold())
            ).strip("-")
            normalized = "-".join(
                APPROVED_COMMON_VARIANTS.get(part, part)
                for part in normalized.split("-")
            )
            normalized = _APPROVED_PLURALS.get(normalized, normalized)
            canonical = self._evidence_catalog.canonicalize_alias(normalized)
        except (AttributeError, CatalogCategoryError):
            raise CountTargetResolutionError(error_code) from None
        if not canonical:
            raise CountTargetResolutionError(error_code)
        return canonical

    def _preserve_specificity(self, question: str, target: str) -> str:
        """Narrow a broad vehicle hint only when approved scope is explicit."""
        if target != "vehicle" or not isinstance(question, str):
            return target
        words = [
            APPROVED_COMMON_VARIANTS.get(word, word)
            for word in re.findall(r"[a-z]+", question.casefold())
        ]
        for modifier in APPROVED_SCOPE_MODIFIERS:
            if any(
                words[index:index + 2] == [modifier, vehicle]
                for index in range(len(words) - 1)
                for vehicle in ("vehicle", "vehicles")
            ):
                candidate = f"{modifier}-vehicle"
                if self._expand_semantic_target(candidate):
                    return candidate
        return target

    def _expand_semantic_target(self, target: str) -> tuple[str, ...]:
        return self._evidence_catalog.executable_leaves_for_target(
            target, task="counting"
        )

    def _reconcile(
        self, *, planner: str, planner_leaves: tuple[str, ...],
        verifier: str | None,
    ) -> tuple[str, str]:
        expected = self._expand_semantic_target(planner)
        if verifier is None:
            if not expected:
                if planner_leaves:
                    raise CountTargetResolutionError(
                        "COUNT_TARGET_PLANNER_LEAVES_INVALID"
                    )
                return planner, "planner_only_no_visual_expert"
            if planner_leaves != expected:
                return planner, "incomplete_parent_expansion_corrected"
            return planner, "planner_only"
        verifier_leaves = self._expand_semantic_target(verifier)
        if planner == verifier:
            if expected and planner_leaves != expected:
                return planner, "incomplete_parent_expansion_corrected"
            return planner, "matched_parent_expansion" if len(expected) > 1 else "matched"
        if not expected or not verifier_leaves:
            raise CountTargetResolutionError("COUNT_TARGET_CONFLICT")
        planner_set, verifier_set = frozenset(expected), frozenset(verifier_leaves)
        if verifier_set < planner_set:
            return verifier, "planner_scope_broadened_corrected"
        if planner_set < verifier_set:
            return verifier, "planner_scope_narrowed_corrected"
        raise CountTargetResolutionError("COUNT_TARGET_CONFLICT")

    @staticmethod
    def _build_target_spec(
        final_target: str, verifier_spec: CountTargetSpec | None,
    ) -> CountTargetSpec:
        if verifier_spec is not None:
            return verifier_spec.model_copy(update={"canonical_label": final_target})
        return CountTargetSpec(
            canonical_label=final_target,
            inclusion_rule=(
                "Count only distinct visible instances matching the exact requested target."
            ),
            exclusion_rule=(
                "Exclude other categories, duplicate views, and unconfirmable fragments."
            ),
        )


__all__ = [
    "CountTargetResolutionError", "CountTargetResolver", "ResolvedCountTarget",
]
