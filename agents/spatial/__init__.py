"""Dataset-neutral spatial query contracts, geometry, review, and agent.
数据集无关的空间查询契约、几何、复核与 Agent。"""

from agents.spatial.agent import SpatialAgent
from agents.spatial.candidate_review import (
    SpatialCandidateReviewResult,
    SpatialCandidateReviewer,
)
from agents.spatial.evidence_merge import (
    is_corner_anchored_box,
    maximum_repair_severity,
    merge_visual_evidence,
    needs_candidate_review,
    same_visual_observation,
)
from agents.spatial.geometry import apply_spatial_geometry, canonical_answer
from agents.spatial.schema import SpatialOperation, SpatialQuerySpec, spatial_query_from_metadata

__all__ = [
    "SpatialAgent",
    "SpatialCandidateReviewResult",
    "SpatialCandidateReviewer",
    "SpatialOperation",
    "SpatialQuerySpec",
    "apply_spatial_geometry",
    "canonical_answer",
    "is_corner_anchored_box",
    "maximum_repair_severity",
    "merge_visual_evidence",
    "needs_candidate_review",
    "same_visual_observation",
    "spatial_query_from_metadata",
]
