"""Contract tests for the data-layer manifest-driven draft adapter and
samples_file path containment.

数据层 manifest 驱动的 draft 适配器与 samples_file 路径包含契约测试：显式
字段映射、task 可选、JSON/JSONL、字段不猜测、稳定失败、路径绝不逃逸
dataset root。run manifest 的写回不属于数据层（见
workflows.artifact_writer.write_dataset_probe）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.adapters.base import DatasetProbeError, resolve_dataset_relative_path
from data.adapters.manifest import (
    ManifestDraftAdapter,
    iter_manifest_drafts,
)


# ── samples_file path containment / samples_file 路径包含 ──────────────────


@pytest.mark.parametrize(
    "relative",
    [
        "../outside.jsonl",
        "../../secret.json",
        r"C:\other\data.jsonl",
        "D:/other/data.json",
        r"\\server\share\data.jsonl",
        "//server/share/data.jsonl",
        "/var/tmp/data.jsonl",
        "nested/../outside.jsonl",
        "data/../x.jsonl",
        "a//b.jsonl",
        "nested/",
        "",
    ],
)
def test_resolve_dataset_relative_path_rejects_escape_candidates(
    tmp_path: Path, relative: str
) -> None:
    with pytest.raises(DatasetProbeError):
        resolve_dataset_relative_path(tmp_path, relative, field_name="samples_file")


@pytest.mark.parametrize("relative", ["samples.jsonl", "nested/samples.jsonl"])
def test_resolve_dataset_relative_path_accepts_nested_relative(
    tmp_path: Path, relative: str
) -> None:
    target = resolve_dataset_relative_path(tmp_path, relative, field_name="samples_file")
    assert target == tmp_path / relative
    assert target.resolve().is_relative_to(tmp_path.resolve())


def test_manifest_samples_file_escape_rejected(tmp_path: Path) -> None:
    root = _make_dataset_root(tmp_path, with_task=True)
    manifest = json.loads((root / "spacers_adapter.json").read_text(encoding="utf-8"))
    manifest["samples_file"] = "../outside.jsonl"
    (root / "spacers_adapter.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetProbeError, match="relative|dot-dot"):
        list(iter_manifest_drafts(root, dataset="auto-demo", split="test"))


def test_manifest_mapped_field_non_string_rejected(tmp_path: Path) -> None:
    root = _make_dataset_root(tmp_path, with_task=True)
    manifest = json.loads((root / "spacers_adapter.json").read_text(encoding="utf-8"))
    manifest["fields"]["id"] = 5
    (root / "spacers_adapter.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetProbeError, match="column name"):
        list(iter_manifest_drafts(root, dataset="auto-demo", split="test"))


def test_manifest_mapped_field_empty_string_rejected(tmp_path: Path) -> None:
    root = _make_dataset_root(tmp_path, with_task=True)
    manifest = json.loads((root / "spacers_adapter.json").read_text(encoding="utf-8"))
    manifest["fields"]["id"] = ""
    (root / "spacers_adapter.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetProbeError, match="column name"):
        list(iter_manifest_drafts(root, dataset="auto-demo", split="test"))


def test_manifest_empty_samples_file_rejected(tmp_path: Path) -> None:
    root = _make_dataset_root(tmp_path, with_task=True)
    manifest = json.loads((root / "spacers_adapter.json").read_text(encoding="utf-8"))
    manifest["samples_file"] = ""
    (root / "spacers_adapter.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DatasetProbeError, match="samples_file"):
        list(iter_manifest_drafts(root, dataset="auto-demo", split="test"))


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
