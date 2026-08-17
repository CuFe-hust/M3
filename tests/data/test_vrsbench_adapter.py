"""Contract tests for the audited VRSBench adapter.

VRSBench 适配器测试：官方 caption/VQA/grounding 加载、任务规范化接入、
缺图/缺字段/多候选显式失败、源顺序与 sample_id 稳定、Registry 注册。
不导入 routing / agents；不拼接 Agent prompt。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from PIL import Image

from data.adapters.base import DatasetProbeError
from data.adapters.vrsbench.adapter import VRSBenchAdapter
from data.registry import REGISTRY, DatasetRegistry, register_default_adapters

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_image(path: Path, seed: int = 7) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (seed, seed * 2, seed * 3)).save(path)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _vqa_rows() -> list[dict]:
    return [
        {"image_id": "img_1.png", "question": "How many small vehicles are in the image?",
         "ground_truth": "3", "question_id": "vq_1", "type": "quantity"},
        {"image_id": "img_1.png", "question": "What category is the topmost vehicle?",
         "ground_truth": "small-vehicle", "question_id": "vq_2", "type": "object"},
        {"image_id": "img_1.png", "question": "What color is the building?",
         "ground_truth": "red", "question_id": "vq_3", "type": "attribute"},
    ]


def _build_vqa_root(tmp_path: Path) -> Path:
    root = tmp_path / "vrsbench"
    _make_image(root / "Images_val" / "img_1.png")
    _write_json(root / "VRSBench_EVAL_vqa.json", _vqa_rows())
    return root


# ── VQA 任务规范化 / VQA task normalization ────────────────────────────────


def test_vqa_questions_normalize_to_standard_tasks(tmp_path: Path) -> None:
    root = _build_vqa_root(tmp_path)
    samples = list(VRSBenchAdapter().iter_samples(root, "validation", "general_vqa"))
    assert [sample.task for sample in samples] == ["counting", "spatial_relation", "general_vqa"]


def test_vqa_not_all_general_vqa(tmp_path: Path) -> None:
    root = _build_vqa_root(tmp_path)
    tasks = {sample.task for sample in VRSBenchAdapter().iter_samples(root, "validation", "general_vqa")}
    assert tasks == {"counting", "spatial_relation", "general_vqa"}


def test_vqa_metadata_fields(tmp_path: Path) -> None:
    root = _build_vqa_root(tmp_path)
    samples = list(VRSBenchAdapter().iter_samples(root, "validation", "general_vqa"))
    first = samples[0]
    assert first.metadata["question_id"] == "vq_1"
    assert first.metadata["question_type"] == "quantity"
    assert first.metadata["source_index"] == 0
    assert first.metadata["source"] == "VRSBench"
    assert first.images[0].image_id == "img_1.png"
    assert first.split == "validation"
    assert first.normalization is not None
    assert first.normalization.normalized_task == "counting"
    assert first.ground_truth is not None
    assert first.ground_truth.raw["question_id"] == "vq_1"


def test_vqa_sample_id_is_stable_and_question_scoped(tmp_path: Path) -> None:
    root = _build_vqa_root(tmp_path)
    adapter = VRSBenchAdapter()
    first = list(adapter.iter_samples(root, "validation", "general_vqa"))[0]
    second = list(adapter.iter_samples(root, "validation", "general_vqa"))[0]
    assert first.sample_id == second.sample_id == "vq_1"


# ── Caption / grounding / 加载 ──────────────────────────────────────────────


def test_caption_release_loads(tmp_path: Path) -> None:
    root = tmp_path / "vrsbench_cap"
    _make_image(root / "Images_val" / "img_1.png")
    _write_json(root / "VRSBench_EVAL_Cap.json",
                [{"image_id": "img_1.png", "caption": "A harbor with ships."}])
    samples = list(VRSBenchAdapter().iter_samples(root, "validation", "caption"))
    assert len(samples) == 1
    assert samples[0].task == "caption"
    assert samples[0].question == ""
    assert samples[0].ground_truth is not None
    assert samples[0].ground_truth.answers == ["A harbor with ships."]


def test_grounding_release_loads_one_sample_per_object(tmp_path: Path) -> None:
    root = tmp_path / "vrsbench_det"
    _make_image(root / "Images_val" / "img_1.png")
    _write_json(root / "VRSBench_EVAL_Det.json", [
        {"image_id": "img_1.png", "objects": [
            {"name": "small-vehicle", "bbox": [10, 10, 40, 40]},
            {"name": "large-vehicle", "bbox": [50, 50, 90, 90]},
        ]},
    ])
    samples = list(VRSBenchAdapter().iter_samples(root, "validation", "grounding"))
    assert len(samples) == 2
    assert [sample.question for sample in samples] == [
        "Locate the small-vehicle.", "Locate the large-vehicle.",
    ]
    assert samples[0].ground_truth is not None
    assert samples[0].ground_truth.boxes == [[10, 10, 40, 40]]
    assert samples[0].ground_truth.labels == ["small-vehicle"]


def test_official_referring_release_preserves_normalized_polygon(tmp_path: Path) -> None:
    """Load the official VRSBench referring release without flattening its
    four-corner polygon into an unlabelled xyxy box.
    加载官方 VRSBench referring 发布，并保留四角 polygon，不把它静默压成
    无标签 xyxy 框。"""

    root = tmp_path / "vrsbench_referring"
    _make_image(root / "Images_val" / "img_1.png")
    _write_json(root / "VRSBench_EVAL_referring.json", [
        {
            "image_id": "img_1.png",
            "question": "The vehicle near the road.",
            "question_id": 7,
            "ground_truth": "{<25><40><33><60>}",
            "obj_corner": [0.10, 0.20, 0.30, 0.20, 0.30, 0.40, 0.10, 0.40],
            "obj_cls": "vehicle",
            "type": "ref",
        }
    ])

    samples = list(VRSBenchAdapter().iter_samples(root, "val", "grounding"))

    assert len(samples) == 1
    sample = samples[0]
    assert sample.ground_truth is not None
    assert sample.ground_truth.coordinate_frame == "normalized_0_1_top_left"
    assert sample.ground_truth.boxes == [[0.10, 0.20, 0.30, 0.20, 0.30, 0.40, 0.10, 0.40]]
    assert sample.ground_truth.labels == ["vehicle"]
    assert sample.ground_truth.raw["coordinate_frame"] == "normalized_0_1_top_left"


def test_probe_discovers_official_referring_grounding_file(tmp_path: Path) -> None:
    root = tmp_path / "vrsbench_referring_probe"
    _make_image(root / "Images_val" / "img_1.png")
    _write_json(root / "VRSBench_EVAL_referring.json", [
        {
            "image_id": "img_1.png",
            "question": "The vehicle.",
            "obj_corner": [0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.1, 0.4],
            "obj_cls": "vehicle",
        }
    ])

    probe = VRSBenchAdapter().probe(root, task="grounding")

    assert probe.sample_count == 1
    assert probe.sample_file.name == "VRSBench_EVAL_referring.json"


def test_source_order_is_preserved(tmp_path: Path) -> None:
    root = _build_vqa_root(tmp_path)
    sample_ids = [sample.sample_id for sample in VRSBenchAdapter().iter_samples(root, "validation", "general_vqa")]
    assert sample_ids == ["vq_1", "vq_2", "vq_3"]


# ── 显式失败 / explicit failures ────────────────────────────────────────────


def test_missing_image_fails(tmp_path: Path) -> None:
    root = tmp_path / "vrsbench_missing_img"
    _write_json(root / "VRSBench_EVAL_vqa.json", _vqa_rows())
    with pytest.raises(DatasetProbeError, match="image is missing"):
        list(VRSBenchAdapter().iter_samples(root, "validation", "general_vqa"))


def test_missing_required_field_fails(tmp_path: Path) -> None:
    root = tmp_path / "vrsbench_missing_field"
    _make_image(root / "Images_val" / "img_1.png")
    rows = _vqa_rows()
    del rows[0]["ground_truth"]
    _write_json(root / "VRSBench_EVAL_vqa.json", rows)
    with pytest.raises(DatasetProbeError, match="misses one of fields"):
        list(VRSBenchAdapter().iter_samples(root, "validation", "general_vqa"))


def test_multiple_annotation_candidates_fail(tmp_path: Path) -> None:
    root = tmp_path / "vrsbench_multi"
    _make_image(root / "Images_val" / "img_1.png")
    _write_json(root / "a" / "VRSBench_EVAL_vqa.json", _vqa_rows())
    _write_json(root / "b" / "VRSBench_EVAL_vqa.json", _vqa_rows())
    with pytest.raises(DatasetProbeError, match="Expected exactly one"):
        list(VRSBenchAdapter().iter_samples(root, "validation", "general_vqa"))


def test_unsupported_split_and_task_fail(tmp_path: Path) -> None:
    root = _build_vqa_root(tmp_path)
    with pytest.raises(DatasetProbeError, match="split"):
        list(VRSBenchAdapter().iter_samples(root, "train", "general_vqa"))
    with pytest.raises(DatasetProbeError, match="not support"):
        list(VRSBenchAdapter().iter_samples(root, "validation", "counting"))


# ── Probe / Registry / 边界 ─────────────────────────────────────────────────


def test_probe_reports_observed_fields_and_count(tmp_path: Path) -> None:
    root = tmp_path / "vrsbench"
    _make_image(root / "Images_val" / "img_1.png")
    _write_json(root / "VRSBench_EVAL_vqa.json", _vqa_rows())
    probe = VRSBenchAdapter().probe(root)
    assert probe.dataset == "VRSBench"
    assert probe.sample_count == 3
    assert "question_id" in probe.observed_fields


def test_registry_registers_real_vrsbench_adapter() -> None:
    registry = DatasetRegistry()
    register_default_adapters(registry)
    adapter = registry.get("VRSBench")
    assert isinstance(adapter, VRSBenchAdapter)
    assert "VRSBench" in registry.names()
    with pytest.raises(DatasetProbeError):
        registry.register("VRSBench", lambda: VRSBenchAdapter())  # duplicate


def test_default_registry_stays_empty_until_explicit_registration() -> None:
    assert REGISTRY.names() == ()


def test_adapter_imports_no_agents_or_routing() -> None:
    source = (REPO_ROOT / "data" / "adapters" / "vrsbench" / "adapter.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"agents", "routing", "workflows", "evaluation", "reporting",
                 "application", "models", "spacers_agent", "eval"}
    tops = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            tops.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            tops.add(node.module.split(".")[0])
    assert not (tops & forbidden), f"adapter imports forbidden: {tops & forbidden}"
    assert "system" not in source.casefold() or "prompt" not in source.casefold()
