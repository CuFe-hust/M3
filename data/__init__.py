"""Unified sample contracts for the data layer.
数据层统一样本契约。

This package only re-exports the stable data-layer types.
本包只重导出数据层稳定类型。
"""

from data.schema import (
    GroundTruth,
    ImageRef,
    ImageRole,
    JsonScalar,
    JsonValue,
    TaskName,
    TaskNormalization,
    UnifiedSample,
    ValidationIssue,
    stable_sample_id,
)

__all__ = [
    "GroundTruth",
    "ImageRef",
    "ImageRole",
    "JsonScalar",
    "JsonValue",
    "TaskName",
    "TaskNormalization",
    "UnifiedSample",
    "ValidationIssue",
    "stable_sample_id",
]
