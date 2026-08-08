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
from data.adapters.manifest import (
    DatasetProbeError,
    ManifestDraftAdapter,
    ManifestAdapterError,
    iter_manifest_drafts,
    update_manifest_probe,
)
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


# ── manifest-driven draft adapter / manifest 驱动的 draft 适配器 ─────────────


def _make_dataset_root(root: Path, *, with_task: bool = True) -> Path:
    """Create a dataset root with a mapping manifest, a sample file, and real
    images. 创建带映射清单、样本文件与真实图像的 dataset root。"""
    root.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.new("RGB", (4, 4), (1, 2, 3)).save(root / "img.png", format="PNG")
    Image.new("RGB", (4, 4), (1, 2, 3)).save(root / "img2.png", format="PNG")
    fields = {
        "id": "id",
        "split": "split",
        "question": "question",
        "images": "images",
    }
    if with_task:
        fields["task"] = "task"
    (root / "spacers_adapter.json").write_text(
        json.dumps(
            {
                "dataset": "auto-demo",
                "version": "1",
                "samples_file": "samples.jsonl",
                "fields": fields,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return root


def _write_rows(root: Path, rows: list[dict]) -> None:
    (root / "samples.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def _draft_rows() -> list[dict]:
    return [
        {
            "id": "a1",
            "split": "test",
            "question": "Is there a road?",
            "images": ["img.png"],
            "task": "general_vqa",
        },
        {
            "id": "a2",
            "split": "test",
            "question": "Describe the scene.",
            "images": ["img.png", "img2.png"],
        },
    ]


def test_manifest_draft_adapter_yields_drafts_with_optional_task(tmp_path: Path) -> None:
    root = _make_dataset_root(tmp_path, with_task=True)
    _write_rows(root, _draft_rows())
    adapter = ManifestDraftAdapter("auto-demo", {"general_vqa", "caption"})
    drafts = list(adapter.iter_drafts(root, "test"))
    assert len(drafts) == 2
    assert drafts[0].sample_id == "a1"
    assert drafts[0].explicit_task == "general_vqa"
    assert drafts[1].explicit_task is None  # task column absent in the row
    assert [image.role for image in drafts[0].images] == ["image"]
    assert drafts[1].split == "test"
    assert drafts[1].question == "Describe the scene."


def test_manifest_draft_adapter_task_field_optional_in_mapping(tmp_path: Path) -> None:
    root = _make_dataset_root(tmp_path, with_task=False)
    _write_rows(root, _draft_rows())
    drafts = list(iter_manifest_drafts(root, dataset="auto-demo", split="test"))
    assert len(drafts) == 2
    assert all(draft.explicit_task is None for draft in drafts)


def test_manifest_drafts_filter_by_split(tmp_path: Path) -> None:
    root = _make_dataset_root(tmp_path, with_task=False)
    _write_rows(
        root,
        [
            {**row, "split": "train" if row["id"] == "a2" else "test"}
            for row in _draft_rows()
        ],
    )
    drafts = list(iter_manifest_drafts(root, dataset="auto-demo", split="test"))
    assert [draft.sample_id for draft in drafts] == ["a1"]


def test_manifest_probe_reports_layout_evidence(tmp_path: Path) -> None:
    root = _make_dataset_root(tmp_path, with_task=True)
    _write_rows(root, _draft_rows())
    adapter = ManifestDraftAdapter("auto-demo", {"general_vqa", "caption"})
    probe = adapter.probe(root, task=None)
    assert probe.dataset == "auto-demo"
    assert probe.version == "1"
    assert probe.sample_count == 2
    assert probe.sample_file == root / "samples.jsonl"
    assert probe.available_tasks == ("caption", "general_vqa")  # sorted / 字母序


def test_missing_manifest_fails_stable(tmp_path: Path) -> None:
    with pytest.raises(DatasetProbeError, match="spacers_adapter.json"):
        list(iter_manifest_drafts(tmp_path, dataset="auto-demo", split="test"))


def test_manifest_dataset_mismatch_fails(tmp_path: Path) -> None:
    root = _make_dataset_root(tmp_path, with_task=True)
    _write_rows(root, _draft_rows())
    with pytest.raises(DatasetProbeError, match="dataset"):
        list(iter_manifest_drafts(root, dataset="other-dataset", split="test"))


def test_manifest_missing_required_field_mapping_fails(tmp_path: Path) -> None:
    root = _make_dataset_root(tmp_path, with_task=True)
    (root / "spacers_adapter.json").write_text(
        json.dumps(
            {
                "dataset": "auto-demo",
                "version": "1",
                "samples_file": "samples.jsonl",
                "fields": {"id": "id", "split": "split"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DatasetProbeError, match="field"):
        list(iter_manifest_drafts(root, dataset="auto-demo", split="test"))


def test_manifest_invalid_image_path_fails(tmp_path: Path) -> None:
    root = _make_dataset_root(tmp_path, with_task=True)
    _write_rows(root, [{**_draft_rows()[0], "images": ["C:\\absolute\\img.png"]}])
    with pytest.raises(DatasetProbeError, match="image path"):
        list(iter_manifest_drafts(root, dataset="auto-demo", split="test"))


def test_manifest_missing_image_file_fails(tmp_path: Path) -> None:
    root = _make_dataset_root(tmp_path, with_task=True)
    _write_rows(root, [{**_draft_rows()[0], "images": ["missing.png"]}])
    with pytest.raises(DatasetProbeError, match="missing image"):
        list(iter_manifest_drafts(root, dataset="auto-demo", split="test"))


def test_manifest_invalid_task_value_fails(tmp_path: Path) -> None:
    root = _make_dataset_root(tmp_path, with_task=True)
    _write_rows(root, [{**_draft_rows()[0], "task": "not_a_task"}])
    with pytest.raises(DatasetProbeError, match="draft contract"):
        list(iter_manifest_drafts(root, dataset="auto-demo", split="test"))


def test_manifest_optional_ground_truth_fields(tmp_path: Path) -> None:
    root = _make_dataset_root(tmp_path, with_task=True)
    manifest = json.loads(
        (root / "spacers_adapter.json").read_text(encoding="utf-8")
    )
    manifest["fields"]["count"] = "count"
    manifest["fields"]["answers"] = "answers"
    (root / "spacers_adapter.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    _write_rows(
        root,
        [{**_draft_rows()[0], "count": 3, "answers": ["3"]}],
    )
    drafts = list(iter_manifest_drafts(root, dataset="auto-demo", split="test"))
    assert drafts[0].ground_truth is not None
    assert drafts[0].ground_truth.count == 3
    assert drafts[0].ground_truth.answers == ["3"]


def test_data_layer_never_imports_workflows_or_models() -> None:
    import ast

    for path in (
        Path(__file__).resolve().parents[2] / "data" / "adapters" / "manifest.py",
        Path(__file__).resolve().parents[2] / "data" / "schema.py",
    ):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(("workflows", "models")), alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(("workflows", "models")), node.module
