"""Dataset-neutral spatial query contracts and geometry rules.
数据集无关的空间查询契约与几何规则。"""

from agents.spatial.geometry import apply_spatial_geometry, canonical_answer
from agents.spatial.schema import SpatialOperation, SpatialQuerySpec, spatial_query_from_metadata

__all__ = [
    "SpatialOperation",
    "SpatialQuerySpec",
    "apply_spatial_geometry",
    "canonical_answer",
    "spatial_query_from_metadata",
]
