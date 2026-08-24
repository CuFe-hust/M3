"""Model-family-agnostic multimodal SFT interfaces.

The package deliberately contains no model-family imports.  Concrete model
knowledge lives below :mod:`training.multimodal_sft.adapters` and is selected
through the registry.
"""

from .contracts import (
    AdapterContractError,
    AdapterProbe,
    CapabilityReport,
    CanonicalEpisode,
    DataProfile,
    ModelIdentity,
    ModelStructure,
    MultimodalModelAdapter,
)
from .parameter_plan import (
    ParameterPlan,
    ParameterPlanError,
    TuningPolicy,
    build_parameter_plan,
)
from .optimizer import OptimizerConfig, build_cosine_scheduler, build_optimizer_groups
from .registry import AdapterRegistry, UnsupportedModelAdapter, default_registry

__all__ = [
    "AdapterContractError",
    "AdapterProbe",
    "AdapterRegistry",
    "CapabilityReport",
    "CanonicalEpisode",
    "DataProfile",
    "ModelIdentity",
    "ModelStructure",
    "MultimodalModelAdapter",
    "ParameterPlan",
    "ParameterPlanError",
    "TuningPolicy",
    "OptimizerConfig",
    "UnsupportedModelAdapter",
    "build_parameter_plan",
    "default_registry",
    "build_cosine_scheduler",
    "build_optimizer_groups",
]

