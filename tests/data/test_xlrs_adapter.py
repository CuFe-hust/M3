"""Contract tests for the offline-first XLRS-Bench adapter.

XLRS-Bench 适配器测试：离线优先（不触网）、三种任务（caption/grounding/
VQA-lite）产出 UnifiedSample、源框/尺寸/坐标来源保留、choices 与
multi-answer 提示、Registry 注册。datasets 库通过注入 loader 边界隔离，
本测试不依赖其安装。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from PIL import Image

from data.adapters.base import DatasetProbeError
from data.adapters.xlrs import XLRSAdapter
from data.registry import DatasetRegistry, register_default_adapters

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_image(path: Path, seed: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (seed, seed * 2, seed * 3)).save(path)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _caption_rows() -> list[dict]:
    return [
        {"question": "Describe the scene.", "caption": ["A harbor.", "A busy harbor."], "image": "img_1.png"},
    ]


def _grounding_rows() -> list[dict]:
    return [
        {"question": "Locate the plane.", "bbox": [10, 20, 110, 120],
         "image_width": 500, "image_height": 400, "image": "img_1.png"},
    ]


def _lite_rows() -> list[dict]:
    return [
        {"question": "What is the overall land use type?", "choices": ["A", "B", "C", "D"],
         "answer": "B", "image": "img_1.png"},
        {"question": "Which class is the target?", "A": "x", "B": "y", "C": "z", "D": "w",
         "label": "A", "image": "img_1.png"},
    ]


def _root_with_image(tmp_path: Path) -> Path:
    root = tmp_path / "xlrs"
    _make_image(root / "img_1.png")
    # Local-release marker so the injected loader path is exercised offline.
    # 本地发布标记，使注入 loader 路径在离线下被触发。
    _write_json(root / "dataset_dict.json", {"splits": {}})
    return root


def _loader_for(rows: list[dict]):
    def loader(root: Path, task: str) -> list[dict]:
        return rows
    return loader


# ── 三任务产出 / three tasks produce UnifiedSample ─────────────────────────


def test_caption_task_outputs_all_references(tmp_path: Path) -> None:
    root = _root_with_image(tmp_path)
    adapter = XLRSAdapter(dataset_loader=_loader_for(_caption_rows()))
    samples = list(adapter.iter_samples(root, "train", "caption"))
    assert len(samples) == 1
    assert samples[0].task == "caption"
    assert samples[0].ground_truth is not None
    assert samples[0].ground_truth.answers == ["A harbor.", "A busy harbor."]
    assert samples[0].question == "Describe the scene."


def test_grounding_task_preserves_box_size_and_coordinate_source(tmp_path: Path) -> None:
    root = _root_with_image(tmp_path)
    adapter = XLRSAdapter(dataset_loader=_loader_for(_grounding_rows()))
    sample = next(adapter.iter_samples(root, "test", "grounding"))
    assert sample.task == "grounding"
    assert sample.ground_truth is not None
    assert sample.ground_truth.boxes == [[10, 20, 110, 120]]
    assert sample.ground_truth.coordinate_frame == "source_pixels_top_left"
    assert sample.metadata["image_width"] == 500
    assert sample.metadata["image_height"] == 400


def test_vqa_lite_task_outputs_choices_and_multi_answer_hint(tmp_path: Path) -> None:
    root = _root_with_image(tmp_path)
    adapter = XLRSAdapter(dataset_loader=_loader_for(_lite_rows()))
    samples = list(adapter.iter_samples(root, "train", "multiple_choice_vqa"))
    assert len(samples) == 2
    assert all(sample.task == "multiple_choice_vqa" for sample in samples)
    assert samples[0].metadata["choices"] == ["A", "B", "C", "D"]
    assert samples[0].metadata["multi_answer"] is True
    assert samples[1].metadata["choices"] == ["x", "y", "z", "w"]  # A–E key fallback
    assert samples[1].metadata["multi_answer"] is False


def test_three_tasks_all_produce_unified_samples(tmp_path: Path) -> None:
    root = _root_with_image(tmp_path)
    rows_by_task = {
        "caption": _caption_rows(),
        "grounding": _grounding_rows(),
        "multiple_choice_vqa": _lite_rows(),
    }

    def loader(root: Path, task: str):
        return rows_by_task[task]

    adapter = XLRSAdapter(dataset_loader=loader)
    total = 0
    for task in ("caption", "grounding", "multiple_choice_vqa"):
        samples = list(adapter.iter_samples(root, "train", task))
        assert samples, task
        total += len(samples)
    assert total == 4


# ── 离线优先 / offline-first ───────────────────────────────────────────────


def test_offline_without_local_release_never_calls_loader(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    calls: list[str] = []

    def loader(root: Path, task: str):
        calls.append(task)
        return []

    adapter = XLRSAdapter(dataset_loader=loader)  # allow_download defaults to False
    assert adapter.allow_download is False
    with pytest.raises(DatasetProbeError, match="offline"):
        list(adapter.iter_samples(root, "train", "caption"))
    assert calls == [], "offline machine must not attempt any load"


def test_allow_download_true_invokes_loader(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    calls: list[str] = []

    def loader(root: Path, task: str):
        calls.append(task)
        return _caption_rows()

    _make_image(root / "img_1.png")
    adapter = XLRSAdapter(allow_download=True, dataset_loader=loader)
    samples = list(adapter.iter_samples(root, "train", "caption"))
    assert calls == ["caption"]
    assert len(samples) == 1


def test_local_release_is_preferred_without_download(tmp_path: Path) -> None:
    root = _root_with_image(tmp_path)
    _write_json(root / "dataset_dict.json", {"splits": {}})
    calls: list[str] = []

    def loader(root: Path, task: str):
        calls.append(task)
        return _caption_rows()

    adapter = XLRSAdapter(dataset_loader=loader)  # download disabled, local wins
    samples = list(adapter.iter_samples(root, "train", "caption"))
    assert calls == ["caption"]
    assert len(samples) == 1


def test_local_split_state_json_is_detected(tmp_path: Path) -> None:
    root = _root_with_image(tmp_path)
    _write_json(root / "train" / "state.json", {"_split": "train"})
    calls: list[str] = []

    def loader(root: Path, task: str):
        calls.append(task)
        return _caption_rows()

    adapter = XLRSAdapter(dataset_loader=loader)
    list(adapter.iter_samples(root, "train", "caption"))
    assert calls == ["caption"]


def test_probe_offline_without_local_fails(tmp_path: Path) -> None:
    with pytest.raises(DatasetProbeError, match="offline"):
        XLRSAdapter().probe(tmp_path / "empty")


# ── 显式失败 / explicit failures ───────────────────────────────────────────


def test_missing_image_fails(tmp_path: Path) -> None:
    root = tmp_path / "xlrs_missing_img"
    _write_json(root / "dataset_dict.json", {"splits": {}})
    adapter = XLRSAdapter(dataset_loader=_loader_for(_caption_rows()))
    with pytest.raises(DatasetProbeError, match="missing image"):
        list(adapter.iter_samples(root, "train", "caption"))


def test_broken_caption_row_fails(tmp_path: Path) -> None:
    root = _root_with_image(tmp_path)
    adapter = XLRSAdapter(dataset_loader=_loader_for([{"image": "img_1.png"}]))
    with pytest.raises(DatasetProbeError, match="question"):
        list(adapter.iter_samples(root, "train", "caption"))


def test_unsupported_task_fails(tmp_path: Path) -> None:
    root = _root_with_image(tmp_path)
    adapter = XLRSAdapter(dataset_loader=_loader_for([]))
    with pytest.raises(DatasetProbeError, match="not support"):
        list(adapter.iter_samples(root, "train", "general_vqa"))


# ── Registry / 边界 ────────────────────────────────────────────────────────


def test_registry_registers_xlrs_targets_and_aliases() -> None:
    registry = DatasetRegistry()
    register_default_adapters(registry)
    assert isinstance(registry.get("XLRS-Bench"), XLRSAdapter)
    assert isinstance(registry.get("XLRS-Bench-lite"), XLRSAdapter)
    assert registry.get("XLRS").name == "XLRS-Bench"
    assert registry.get("xlrs-lite").name == "XLRS-Bench-lite"
    assert {"XLRS-Bench", "XLRS-Bench-lite"} <= set(registry.names())


def test_no_datasets_import_at_module_level() -> None:
    source = (REPO_ROOT / "data" / "adapters" / "xlrs.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert "datasets" not in [alias.name for alias in node.names], "datasets must be lazy"
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            assert node.module.split(".")[0] != "datasets", "datasets must be lazy"
    forbidden = {"agents", "routing", "workflows", "evaluation", "reporting",
                 "application", "models", "spacers_agent", "eval"}
    tops = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            tops.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            tops.add(node.module.split(".")[0])
    assert not (tops & forbidden), f"adapter imports forbidden: {tops & forbidden}"
