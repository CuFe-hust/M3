"""Change-detection agent — wraps the proven visual primitive.
变化检测 Agent — 封装已验证的视觉原语。
"""

from spacers_agent.agents.change.agent import ChangeAgent  # noqa: F401
from spacers_agent.agents.change.harmonizer import PairHarmonizer  # noqa: F401
from spacers_agent.agents.change.pair_validator import PairValidator  # noqa: F401
from spacers_agent.agents.change.schemas import (  # noqa: F401
    ChangePreprocessResult,
    ChangeProposal,
    HarmonizationDecision,
    HarmonizationMetrics,
    PairValidationReport,
)
