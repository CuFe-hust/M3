"""Counting-domain contracts and settings exports.
计数域契约与配置导出。"""

from agents.counting.schema import (
    CountTargetSpec,
    CountingDraft,
    CountingResult,
    GlobalPointObservation,
    IssueRecord,
    LocalPointObservation,
    PixelRect,
    PointProvenance,
    TileCountResponse,
    TileSpec,
)
from agents.counting.settings import (
    AgentCountingSettings,
    CountingSettings,
    YoloCountingSettings,
    YoloDetectorSettings,
)


def __getattr__(name: str):
    """Keep the public CountingAgent export without eagerly loading wiring."""
    if name == "CountingAgent":
        from agents.counting.agent import CountingAgent

        return CountingAgent
    raise AttributeError(name)

__all__ = [
    "AgentCountingSettings",
    "CountTargetSpec",
    "CountingAgent",
    "CountingDraft",
    "CountingResult",
    "CountingSettings",
    "GlobalPointObservation",
    "IssueRecord",
    "LocalPointObservation",
    "PixelRect",
    "PointProvenance",
    "TileCountResponse",
    "TileSpec",
    "YoloCountingSettings",
    "YoloDetectorSettings",
]
