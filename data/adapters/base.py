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

from data.schema import SampleDraft, UnifiedSample


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
    task: str | None = None
    available_tasks: tuple[str, ...] = ()


class DatasetAdapter(Protocol):
    """Read-only adapter contract with explicit probe and validation.
    具有显式探测与校验的只读适配器契约。"""

    name: str
    supported_tasks: set[str] | frozenset[str] | tuple[str, ...]

    def probe(self, root: Path, task: str | None = None) -> AdapterProbe: ...

    def iter_samples(self, root: Path, split: str, task: str) -> Iterator[UnifiedSample]: ...


class DraftDatasetAdapter(Protocol):
    """Read-only adapter contract for datasets without an explicit per-sample
    task; yields SampleDraft rows that the workflow resolves before
    materialization. 无显式逐样本 task 数据集的只读适配器契约；产出由工作流
    在物化前解析的 SampleDraft。"""

    name: str
    supported_tasks: set[str] | frozenset[str] | tuple[str, ...]

    def probe(self, root: Path, task: str | None = None) -> AdapterProbe: ...

    def iter_drafts(self, root: Path, split: str) -> Iterator[SampleDraft]: ...


JSON_RECORD_CONTAINER_KEYS = (
    "samples",
    "data",
    "annotations",
    "items",
    "images",
)


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    """Read a small explicit JSON or JSONL manifest without network fallback.

    Supported shapes: JSONL (one JSON object per non-empty line), a top-level
    JSON list, or a top-level object containing exactly one supported record
    container (samples/data/annotations/items/images). Unknown or ambiguous
    structures raise DatasetProbeError; errors carry file name and line number.
    读取显式 JSON 或 JSONL 清单，不使用网络回退。支持 JSONL、顶层 list 与
    单一受支持容器键（samples/data/annotations/items/images）；未知或歧义结构
    显式失败；错误信息包含文件名与行号。"""
    suffix = path.suffix.lower()
    if suffix not in {".json", ".jsonl"}:
        raise DatasetProbeError(
            f"{path.name}: samples_file must be .json or .jsonl, got {suffix or '<none>'}"
        )
    if not path.is_file():
        raise DatasetProbeError(f"Declared samples_file does not exist: {path}")
    if suffix == ".jsonl":
        return _read_jsonl_rows(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DatasetProbeError(
            f"{path.name}: invalid JSON: {type(error).__name__}: {error}"
        ) from error
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        present = [
            key for key in JSON_RECORD_CONTAINER_KEYS
            if isinstance(payload.get(key), list)
        ]
        if len(present) > 1:
            raise DatasetProbeError(
                f"{path.name}: ambiguous record containers: {present}"
            )
        if not present:
            raise DatasetProbeError(
                f"{path.name}: no supported record container "
                f"in {sorted(JSON_RECORD_CONTAINER_KEYS)}"
            )
        rows = payload[present[0]]
    else:
        raise DatasetProbeError(
            f"{path.name}: unsupported top-level JSON type {type(payload).__name__}"
        )
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise DatasetProbeError(
                f"{path.name}: row {index} is not a JSON object "
                f"({type(row).__name__})"
            )
    return list(rows)


def _read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise DatasetProbeError(
            f"{path.name}: encoding error: {error}"
        ) from error
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise DatasetProbeError(
                f"{path.name}:{line_number}: invalid JSON line: {error.msg}"
            ) from error
        if not isinstance(row, dict):
            raise DatasetProbeError(
                f"{path.name}:{line_number}: row is not a JSON object "
                f"({type(row).__name__})"
            )
        rows.append(row)
    return rows


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
