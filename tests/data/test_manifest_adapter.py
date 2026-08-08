"""Contract tests for the data-layer run-manifest dataset-probe adapter.

数据层 run manifest dataset_probe 适配层契约测试：正常写回、缺失/损坏/
违反最小 schema 的 manifest 稳定失败、幂等覆盖、旧格式兼容升级、JSON-safe
序列化与敏感扫描、原子写入无临时文件残留、错误不回显内容。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.adapters.base import AdapterProbe
from data.adapters.manifest import ManifestAdapterError, update_manifest_probe
from workflows.run_store import RunStore


def _create_run(tmp_path: Path) -> tuple[Path, dict]:
    store = RunStore(tmp_path / "runs", tmp_path)
    manifest = store.create_run(
        config_payload={"models": {"qwen": "qwen-model"}},
        model_ids={"qwen": "qwen-model"},
        prompt_paths=[],
        run_id="probe-run",
    )
    run_dir = tmp_path / "runs" / "probe-run"
    return run_dir, json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


def _probe(**overrides: object) -> AdapterProbe:
    values = dict(
        dataset="parity",
        version="1",
        sample_file=Path("samples.jsonl"),
        observed_fields=("id", "question", "images"),
        sample_count=12,
    )
    values.update(overrides)
    return AdapterProbe(**values)  # type: ignore[arg-type]


def test_update_manifest_probe_writes_dataset_probe(tmp_path: Path) -> None:
    run_dir, original = _create_run(tmp_path)
    updated = update_manifest_probe(
        run_dir, _probe(task="counting", available_tasks=("counting", "caption"))
    )
    payload = updated["dataset_probe"]
    assert payload["dataset"] == "parity"
    assert payload["version"] == "1"
    assert payload["sample_file"] == "samples.jsonl"
    assert payload["observed_fields"] == ["id", "question", "images"]
    assert payload["sample_count"] == 12
    assert payload["task"] == "counting"
    assert payload["available_tasks"] == ["counting", "caption"]
    persisted = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["dataset_probe"] == payload
    # Other manifest fields are preserved. / 其余 manifest 字段保留。
    assert persisted["run_id"] == original["run_id"]
    assert persisted["config_hash"] == original["config_hash"]
    assert list(run_dir.glob("*.tmp")) == []


def test_update_manifest_probe_is_idempotent(tmp_path: Path) -> None:
    run_dir, _ = _create_run(tmp_path)
    update_manifest_probe(run_dir, _probe(sample_count=1))
    updated = update_manifest_probe(run_dir, _probe(sample_count=2))
    assert updated["dataset_probe"]["sample_count"] == 2
    persisted = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["dataset_probe"]["sample_count"] == 2


def test_optional_probe_fields_omitted_when_empty(tmp_path: Path) -> None:
    run_dir, _ = _create_run(tmp_path)
    updated = update_manifest_probe(run_dir, _probe())
    assert "task" not in updated["dataset_probe"]
    assert "available_tasks" not in updated["dataset_probe"]


def test_missing_manifest_fails_with_stable_code(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "missing"
    run_dir.mkdir(parents=True)
    with pytest.raises(ManifestAdapterError) as error:
        update_manifest_probe(run_dir, _probe())
    assert error.value.code == "MANIFEST_MISSING"
    assert "MANIFEST_ADAPTER_FAILED:MANIFEST_MISSING" in str(error.value)


def test_invalid_manifest_json_fails(tmp_path: Path) -> None:
    run_dir, _ = _create_run(tmp_path)
    (run_dir / "manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ManifestAdapterError) as error:
        update_manifest_probe(run_dir, _probe())
    assert error.value.code == "MANIFEST_INVALID"


def test_non_dict_manifest_fails(tmp_path: Path) -> None:
    run_dir, _ = _create_run(tmp_path)
    (run_dir / "manifest.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ManifestAdapterError) as error:
        update_manifest_probe(run_dir, _probe())
    assert error.value.code == "MANIFEST_INVALID"


def test_manifest_without_run_id_fails(tmp_path: Path) -> None:
    run_dir, _ = _create_run(tmp_path)
    (run_dir / "manifest.json").write_text('{"config_hash": "abc"}', encoding="utf-8")
    with pytest.raises(ManifestAdapterError) as error:
        update_manifest_probe(run_dir, _probe())
    assert error.value.code == "MANIFEST_SCHEMA"


def test_manifest_error_never_echoes_content(tmp_path: Path) -> None:
    run_dir, _ = _create_run(tmp_path)
    (run_dir / "manifest.json").write_text('{"run_id": 7, "boom": "secret-raw"}', encoding="utf-8")
    with pytest.raises(ManifestAdapterError) as error:
        update_manifest_probe(run_dir, _probe())
    assert "boom" not in str(error.value)
    assert "secret-raw" not in str(error.value)


def test_probe_secret_value_rejected_before_write(tmp_path: Path) -> None:
    run_dir, _ = _create_run(tmp_path)
    with pytest.raises(ValueError, match="sensitive"):
        update_manifest_probe(run_dir, _probe(dataset="sk-secret"))
    persisted = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "dataset_probe" not in persisted


def test_probe_secret_value_in_observed_fields_rejected(tmp_path: Path) -> None:
    run_dir, _ = _create_run(tmp_path)
    with pytest.raises(ValueError, match="sensitive"):
        update_manifest_probe(
            run_dir, _probe(dataset="d", observed_fields=("id", "sk-secret"))
        )
    persisted = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "dataset_probe" not in persisted


def test_legacy_manifest_without_probe_key_is_upgraded(tmp_path: Path) -> None:
    run_dir, legacy = _create_run(tmp_path)
    assert "dataset_probe" not in legacy  # RunStore never writes it
    updated = update_manifest_probe(run_dir, _probe())
    assert updated["dataset_probe"]["dataset"] == "parity"
