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
from agents.change.preprocess import preprocess_pair
from agents.change.reviewer import review_result
from agents.change.schema import (
    ChangePreprocessResult,
    ChangeProposal,
    HarmonizationDecision,
    HarmonizationMetrics,
    PairValidationReport,
)
from agents.change.settings import (
    AgentChangeSettings,
    ChangeHarmonizationSettings,
    ChangeProposalSettings,
    ChangeReviewSettings,
)

__all__ = [
    "AgentChangeSettings",
    "ChangeHarmonizationSettings",
    "ChangePreprocessResult",
    "ChangeProposal",
    "ChangeProposalSettings",
    "ChangeReviewSettings",
    "HarmonizationCandidate",
    "HarmonizationDecision",
    "HarmonizationMetrics",
    "PairHarmonizer",
    "PairValidationReport",
    "PairValidator",
    "ValidatedPair",
    "compute_metrics",
    "estimate_pif_mask",
    "preprocess_pair",
    "propose_changes",
    "render_overlay",
    "review_result",
]
