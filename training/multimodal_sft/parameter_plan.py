"""Semantic tuning policies and exact parameter plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .contracts import AdapterContractError, ModelIdentity, ModelStructure, MultimodalModelAdapter


class ParameterPlanError(ValueError):
    """A tuning policy cannot be mapped to a safe concrete model plan."""


@dataclass(frozen=True)
class TuningPolicy:
    """Model-neutral policy expressed only in semantic roles."""

    name: str
    lora_roles: tuple[str, ...] = ("language_lora_targets",)
    full_train_roles: tuple[str, ...] = ()
    rank: int = 64
    alpha: int = 128
    dropout: float = 0.05

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "lora_roles": list(self.lora_roles),
            "full_train_roles": list(self.full_train_roles),
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
        }

    @classmethod
    def from_name(cls, name: str) -> "TuningPolicy":
        policies = {
            "lora_only": cls("lora_only"),
            "projector_only": cls("projector_only", lora_roles=(), full_train_roles=("vision_connectors",)),
            "lora_plus_projector": cls("lora_plus_projector", full_train_roles=("vision_connectors",)),
            "full_language_lora": cls("full_language_lora", lora_roles=("language_lora_targets",)),
        }
        try:
            return policies[name]
        except KeyError as exc:
            raise ParameterPlanError(f"unknown tuning policy: {name}") from exc


@dataclass(frozen=True)
class ParameterPlan:
    adapter_name: str
    policy: str
    language_backbone: str
    vision_backbone: str
    lora_module_paths: tuple[str, ...] = ()
    full_train_module_paths: tuple[str, ...] = ()
    parameter_names: tuple[str, ...] = ()
    full_train_parameter_names: tuple[str, ...] = ()
    frozen_parameter_names: tuple[str, ...] = ()
    model_identity: Mapping[str, Any] = field(default_factory=dict)
    structure: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "policy": self.policy,
            "language_backbone": self.language_backbone,
            "vision_backbone": self.vision_backbone,
            "lora_module_paths": list(self.lora_module_paths),
            "full_train_module_paths": list(self.full_train_module_paths),
            "parameter_names": list(self.parameter_names),
            "full_train_parameter_names": list(self.full_train_parameter_names),
            "frozen_parameter_names": list(self.frozen_parameter_names),
            "model_identity": dict(self.model_identity),
            "structure": dict(self.structure),
        }

    def validate(self) -> None:
        if not self.lora_module_paths and not self.full_train_module_paths:
            raise ParameterPlanError("parameter plan selected no trainable semantic modules")
        overlap = set(self.lora_module_paths) & set(self.full_train_module_paths)
        if overlap:
            raise ParameterPlanError(f"module paths cannot be both LoRA and full-train: {sorted(overlap)}")
        for path in (*self.lora_module_paths, *self.full_train_module_paths):
            if not path or path.startswith("/"):
                raise ParameterPlanError(f"invalid concrete module path: {path!r}")


def _named_parameter_names(model: Any) -> tuple[str, ...]:
    try:
        return tuple(name for name, _ in model.named_parameters())
    except AttributeError:
        return ()


def _parameters_under(module_paths: Iterable[str], parameter_names: Iterable[str]) -> tuple[str, ...]:
    roots = tuple(path + "." for path in module_paths)
    return tuple(name for name in parameter_names if name.startswith(roots))


def _identity_from_probe(probe: Any) -> dict[str, Any]:
    identity = getattr(probe, "identity", None)
    return identity.as_dict() if identity is not None and hasattr(identity, "as_dict") else {}


def build_parameter_plan(
    model: Any,
    adapter: MultimodalModelAdapter,
    policy: TuningPolicy | str = "lora_plus_projector",
    *,
    probe: Any | None = None,
) -> ParameterPlan:
    """Resolve semantic roles to exact paths and fail closed on ambiguity."""

    if isinstance(policy, str):
        policy = TuningPolicy.from_name(policy)
    try:
        structure: ModelStructure = adapter.discover_structure(model)
    except Exception as exc:
        if isinstance(exc, AdapterContractError):
            raise
        raise ParameterPlanError("adapter structure discovery failed") from exc
    lora_paths = tuple(path for role in policy.lora_roles for path in structure.paths_for(role))
    full_paths = tuple(path for role in policy.full_train_roles for path in structure.paths_for(role))
    if policy.lora_roles and not lora_paths:
        raise ParameterPlanError(f"policy {policy.name!r} has no discovered LoRA targets")
    if policy.full_train_roles and not full_paths:
        raise ParameterPlanError(f"policy {policy.name!r} has no discovered full-train modules")
    names = _named_parameter_names(model)
    selected = _parameters_under((*lora_paths, *full_paths), names)
    full_selected = _parameters_under(full_paths, names)
    frozen = tuple(name for name in names if name not in selected)
    plan = ParameterPlan(
        adapter_name=str(getattr(adapter, "name", type(adapter).__name__)),
        policy=policy.name,
        language_backbone=structure.language_backbone,
        vision_backbone=structure.vision_backbone,
        lora_module_paths=lora_paths,
        full_train_module_paths=full_paths,
        parameter_names=selected,
        full_train_parameter_names=full_selected,
        frozen_parameter_names=frozen,
        model_identity=_identity_from_probe(probe),
        structure=structure.as_dict(),
    )
    plan.validate()
    return plan
