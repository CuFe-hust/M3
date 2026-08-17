"""Change-domain contracts, settings, validation, and harmonization exports.
变化域契约、设置、校验与一致化导出。"""

from agents.change.difference_proposal import propose_changes, render_overlay
from agents.change.harmonizer import (
    HarmonizationCandidate,
    PairHarmonizer,
    compute_metrics,
    estimate_pif_mask,
)
from agents.change.pair_validator import PairValidator, ValidatedPair
from agents.change.registration import (
    GeometricRegistration,
    RegisteredPair,
    RegistrationError,
    register_pair,
)
from agents.change.perception import (
    PERCEPTION_VERSION,
    ChangePerceptionError,
    ChangePerceptionPipeline,
    ChangePerceptionResult,
)
from agents.change.preprocess import (
    ChangePreparedPair,
    prepare_pair,
    preprocess_pair,
    publish_change_proposals,
)
from agents.change.reviewer import review_result
from agents.change.semantic_transition import infer_semantic_transition
from agents.change.schema import (
    ChangePreprocessResult,
    ChangeProposal,
    HarmonizationDecision,
    HarmonizationMetrics,
    PairValidationReport,
    RegistrationDecision,
    RegistrationMetrics,
    RegistrationReport,
    SemanticTransition,
)
from agents.change.settings import (
    AgentChangeSettings,
    ChangeHarmonizationSettings,
    ChangeLearnedChangeSettings,
    ChangeProposalSettings,
    ChangeSemanticSettings,
    ChangeReviewSettings,
    ChangeRegistrationSettings,
    ChangeReliabilitySettings,
)

__all__ = [
    "AgentChangeSettings",
    "ChangeHarmonizationSettings",
    "ChangeLearnedChangeSettings",
    "ChangePerceptionError",
    "ChangePerceptionPipeline",
    "ChangePerceptionResult",
    "ChangePreparedPair",
    "ChangePreprocessResult",
    "ChangeProposal",
    "ChangeProposalSettings",
    "ChangeSemanticSettings",
    "ChangeReviewSettings",
    "ChangeRegistrationSettings",
    "ChangeReliabilitySettings",
    "HarmonizationCandidate",
    "HarmonizationDecision",
    "HarmonizationMetrics",
    "PairHarmonizer",
    "PERCEPTION_VERSION",
    "PairValidationReport",
    "RegistrationDecision",
    "RegistrationMetrics",
    "RegistrationReport",
    "SemanticTransition",
    "GeometricRegistration",
    "RegisteredPair",
    "RegistrationError",
    "PairValidator",
    "ValidatedPair",
    "compute_metrics",
    "estimate_pif_mask",
    "preprocess_pair",
    "prepare_pair",
    "propose_changes",
    "publish_change_proposals",
    "render_overlay",
    "review_result",
    "register_pair",
    "infer_semantic_transition",
]
