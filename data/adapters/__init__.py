"""Dataset-agnostic adapter foundation exports.
与数据集无关的适配器基础层导出。

This package only re-exports the stable adapter-layer types.
本包只重导出适配器层稳定类型。
"""

from data.adapters.base import (
    AdapterProbe,
    DatasetAdapter,
    DatasetProbeError,
    read_json_rows,
    validate_manifest_mapping,
)

__all__ = [
    "AdapterProbe",
    "DatasetAdapter",
    "DatasetProbeError",
    "read_json_rows",
    "validate_manifest_mapping",
]
