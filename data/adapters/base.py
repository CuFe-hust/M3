"""Dataset-agnostic adapter foundation: probe, protocol, and read utilities.

与数据集无关的适配器基础层：Probe、Protocol 与只读读取工具。
本模块不知道任何具体数据集名称，不导入 agents / routing / workflows，
也不在 import 时访问文件系统。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from data.schema import UnifiedSample


class DatasetProbeError(ValueError):
    """Raised when an adapter cannot prove its declared layout.
    适配器无法证明其声明布局时抛出。"""


@dataclass(frozen=True)
class AdapterProbe:
    """Observed layout evidence returned before execution.
    运行前返回的已观察布局证据。"""

    dataset: str
    version: str
    sample_file: Path
    observed_fields: tuple[str, ...]
    sample_count: int


class DatasetAdapter(Protocol):
    """Read-only adapter contract with explicit probe and validation.
    具有显式探测与校验的只读适配器契约。"""

    name: str
    supported_tasks: set[str] | frozenset[str] | tuple[str, ...]

    def probe(self, root: Path) -> AdapterProbe: ...

    def iter_samples(self, root: Path, split: str, task: str) -> Iterator[UnifiedSample]: ...


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    """Read a small explicit JSON or JSONL manifest without network fallback.
    读取显式 JSON 或 JSONL 清单，不使用网络回退。只读，不修改源文件。"""
    if not path.is_file():
        raise DatasetProbeError(f"Declared samples_file does not exist: {path}")
    if path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and isinstance(payload.get("samples"), list):
            rows = payload["samples"]
        else:
            rows = []
    else:
        raise DatasetProbeError("samples_file must be .json or .jsonl")
    if not all(isinstance(row, dict) for row in rows):
        raise DatasetProbeError("All sample rows must be JSON objects")
    return list(rows)


def validate_manifest_mapping(
    manifest: Mapping[str, Any],
    *,
    dataset: str,
    version: str = "1",
    required_fields: Iterable[str] = ("id", "split", "task", "question", "images"),
) -> Mapping[str, str]:
    """Validate an explicit versioned mapping manifest and return its fields.
    校验显式版本化映射清单并返回字段映射。不推测字段名。"""
    if manifest.get("dataset") != dataset or manifest.get("version") != version:
        raise DatasetProbeError(
            f"Expected dataset={dataset!r} and version={version!r} in adapter manifest"
        )
    samples_value = manifest.get("samples_file")
    fields = manifest.get("fields")
    if not isinstance(samples_value, str) or not isinstance(fields, Mapping):
        raise DatasetProbeError("Adapter manifest requires string samples_file and object fields")
    missing = sorted(set(required_fields) - set(fields))
    if missing:
        raise DatasetProbeError(f"Adapter manifest misses required field mappings: {missing}")
    return fields
