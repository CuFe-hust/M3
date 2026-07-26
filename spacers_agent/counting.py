"""Deprecated compatibility exports for the point-counting pipeline.
点式计数流水线的弃用兼容导出。

The business implementation lives in
``spacers_agent.agents.counting.point_pipeline``.
真实业务实现位于 ``spacers_agent.agents.counting.point_pipeline``。
"""

from spacers_agent.agents.counting.point_pipeline import (
    BoundaryConflict,
    PointCountingOrchestrator,
    SeamDecision,
    TileCheckpoint,
    TileCheckpointStore,
    TileStatus,
    apply_acceptance_policy,
    finalize_representatives,
    find_boundary_conflicts,
)

__all__ = [
    "BoundaryConflict",
    "PointCountingOrchestrator",
    "SeamDecision",
    "TileCheckpoint",
    "TileCheckpointStore",
    "TileStatus",
    "apply_acceptance_policy",
    "finalize_representatives",
    "find_boundary_conflicts",
]
