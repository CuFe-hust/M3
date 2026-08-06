"""Contract tests for the dataset-agnostic adapter foundation.

适配器基础层契约测试：Probe、Protocol、JSON/JSONL 只读读取、manifest 映射工具。
测试使用最小 FakeAdapter，不导入模型、不依赖任何数据集名。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Iterator

import pytest

from data.adapters import (
    AdapterProbe,
    DatasetAdapter,
    DatasetProbeError,
    read_json_rows,
    validate_manifest_mapping,
)
from data.schema import GroundTruth, ImageRef, UnifiedSample

REPO_ROOT = Path(__file__).resolve().parents[2]


class FakeAdapter:
    """Minimal offline adapter implementing the DatasetAdapter contract.
    实现 DatasetAdapter 契约的最小离线适配器。"""

    name = "fake"
    supported_tasks = {"general_vqa"}

    def probe(self, root: Path) -> AdapterProbe:
        rows = read_json_rows(root / "samples.json")
        fields = tuple(sorted({key for row in rows[:5] for key in row}))
        return AdapterProbe(
            dataset=self.name,
            version="test-v1",
            sample_file=root / "samples.json",
            observed_fields=fields,
            sample_count=len(rows),
        )

    def iter_samples(self, root: Path, split: str, task: str) -> Iterator[UnifiedSample]:
        if task not in self.supported_tasks:
            raise DatasetProbeError(f"fake does not support task={task!r}")
        probe = self.probe(root)
        for index, row in enumerate(read_json_rows(probe.sample_file)):
            if row.get("split") != split:
                continue
            yield UnifiedSample(
                sample_id=str(row["id"]),
                dataset=self.name,
                split=split,
                task=task,
                images=[ImageRef(image_id=f"{index}", path=str(row["image"]), role="image")],
                question=str(row["question"]),
                ground_truth=GroundTruth(answers=[str(row["answer"])]),
            )


def _write_samples(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    else:
        content = json.dumps(rows, ensure_ascii=False)
    path.write_text(content, encoding="utf-8")


# ── Protocol / Probe / 契约 ─────────────────────────────────────────────────


def test_fake_adapter_probe_returns_adapter_probe(tmp_path: Path) -> None:
    _write_samples(tmp_path / "samples.json", [{"id": "s1", "split": "test", "question": "Q", "image": "a.png", "answer": "yes"}])
    probe = FakeAdapter().probe(tmp_path)
    assert isinstance(probe, AdapterProbe)
    assert probe.dataset == "fake"
    assert probe.version == "test-v1"
    assert probe.sample_file == tmp_path / "samples.json"
    assert "id" in probe.observed_fields
    assert probe.sample_count == 1


def test_fake_adapter_iter_samples_yields_unified_samples(tmp_path: Path) -> None:
    _write_samples(tmp_path / "samples.json", [{"id": "s1", "split": "test", "question": "Q", "image": "a.png", "answer": "yes"}])
    samples = list(FakeAdapter().iter_samples(tmp_path, "test", "general_vqa"))
    assert len(samples) == 1
    sample = samples[0]
    assert isinstance(sample, UnifiedSample)
    assert sample.sample_id == "s1"
    assert sample.images[0].role == "image"
    assert sample.ground_truth is not None and sample.ground_truth.answers == ["yes"]


def test_fake_adapter_rejects_unsupported_task(tmp_path: Path) -> None:
    with pytest.raises(DatasetProbeError, match="not support"):
        list(FakeAdapter().iter_samples(tmp_path, "test", "counting"))


def test_dataset_probe_error_is_value_error() -> None:
    assert issubclass(DatasetProbeError, ValueError)


def test_adapter_probe_is_frozen() -> None:
    probe = AdapterProbe("d", "1", Path("f.json"), ("id",), 1)
    with pytest.raises(Exception):
        probe.dataset = "other"


# ── read_json_rows / 通用只读读取 ───────────────────────────────────────────


def test_read_json_rows_json_array(tmp_path: Path) -> None:
    path = tmp_path / "rows.json"
    path.write_text(json.dumps([{"a": 1}, {"a": 2}]), encoding="utf-8")
    assert read_json_rows(path) == [{"a": 1}, {"a": 2}]


def test_read_json_rows_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    _write_samples(path, [{"a": 1}, {"a": 2}])
    assert read_json_rows(path) == [{"a": 1}, {"a": 2}]


def test_read_json_rows_dict_with_samples_key(tmp_path: Path) -> None:
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"samples": [{"a": 1}]}), encoding="utf-8")
    assert read_json_rows(path) == [{"a": 1}]


def test_read_json_rows_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetProbeError, match="does not exist"):
        read_json_rows(tmp_path / "nope.json")


def test_read_json_rows_bad_suffix_raises(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("a,b\n", encoding="utf-8")
    with pytest.raises(DatasetProbeError, match=".json or .jsonl"):
        read_json_rows(path)


def test_read_json_rows_rejects_non_dict_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(DatasetProbeError, match="JSON objects"):
        read_json_rows(path)


def test_read_json_rows_is_read_only(tmp_path: Path) -> None:
    path = tmp_path / "rows.json"
    content = json.dumps([{"a": 1}])
    path.write_text(content, encoding="utf-8")
    read_json_rows(path)
    assert path.read_text(encoding="utf-8") == content


# ── validate_manifest_mapping / manifest 映射工具 ───────────────────────────


def test_validate_manifest_mapping_ok() -> None:
    manifest = {
        "dataset": "LEVIR-CC",
        "version": "1",
        "samples_file": "samples.json",
        "fields": {"id": "id", "split": "split", "task": "task", "question": "question", "images": "images"},
    }
    fields = validate_manifest_mapping(manifest, dataset="LEVIR-CC")
    assert fields["id"] == "id"


def test_validate_manifest_mapping_rejects_dataset_mismatch() -> None:
    manifest = {"dataset": "OTHER", "version": "1", "samples_file": "s.json", "fields": {}}
    with pytest.raises(DatasetProbeError, match="Expected dataset"):
        validate_manifest_mapping(manifest, dataset="LEVIR-CC")


def test_validate_manifest_mapping_rejects_version_mismatch() -> None:
    manifest = {"dataset": "LEVIR-CC", "version": "2", "samples_file": "s.json", "fields": {}}
    with pytest.raises(DatasetProbeError, match="Expected dataset"):
        validate_manifest_mapping(manifest, dataset="LEVIR-CC")


def test_validate_manifest_mapping_rejects_missing_required_fields() -> None:
    manifest = {
        "dataset": "LEVIR-CC", "version": "1", "samples_file": "s.json",
        "fields": {"id": "id"},
    }
    with pytest.raises(DatasetProbeError, match="misses required"):
        validate_manifest_mapping(manifest, dataset="LEVIR-CC")


# ── 架构边界 / architecture boundaries ──────────────────────────────────────


def test_adapter_base_knows_no_dataset_names() -> None:
    source = (REPO_ROOT / "data" / "adapters" / "base.py").read_text(encoding="utf-8")
    for name in ("VRSBench", "LEVIR-CC", "MME-RealWorld", "XLRS"):
        assert name not in source, f"adapter base must not know dataset name {name}"


def test_adapter_base_imports_only_data_and_stdlib() -> None:
    source = (REPO_ROOT / "data" / "adapters" / "base.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"agents", "routing", "workflows", "evaluation", "reporting",
                 "application", "models", "spacers_agent", "eval"}
    tops = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            tops.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            tops.add(node.module.split(".")[0])
    assert not (tops & forbidden), f"adapter base imports forbidden packages: {tops & forbidden}"


def test_adapter_base_has_no_import_time_file_access() -> None:
    source = (REPO_ROOT / "data" / "adapters" / "base.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        assert not isinstance(node, (ast.Call, ast.With, ast.Try)), (
            f"adapter base top-level {type(node).__name__} must not run at import"
        )
