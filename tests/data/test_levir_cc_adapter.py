"""Contract tests for the audited LEVIR-CC adapter.

LEVIR-CC 适配器测试：官方 captions 加载、t1/t2 严格映射、多参考 captions
保留、change_qa 条件支持、损坏行显式失败、Registry 注册与别名。
不运行一致化/差异检测；不生成模型 Prompt。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from PIL import Image

from data.adapters.base import DatasetProbeError
from data.adapters.levir_cc import LEVIRCCAdapter
from data.registry import DatasetRegistry, register_default_adapters

REPO_ROOT = Path(__file__).resolve().parents[2]


def _make_image(path: Path, seed: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (seed, seed * 3, seed * 5)).save(path)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _build_official_root(tmp_path: Path, split: str = "test") -> Path:
    root = tmp_path / "levir_cc"
    _make_image(root / "images" / split / "A" / "0001.png", 1)
    _make_image(root / "images" / split / "B" / "0001.png", 2)
    _write_json(root / "LevirCCcaptions.json", [
        {
            "split": split,
            "image_A": f"images/{split}/A/0001.png",
            "image_B": f"images/{split}/B/0001.png",
            "captions": [
                {"raw": "A building appeared."},
                {"raw": "New buildings were constructed."},
            ],
        },
    ])
    return root


# ── change_caption / 加载与顺序 ─────────────────────────────────────────────


def test_change_caption_loads_with_all_references(tmp_path: Path) -> None:
    root = _build_official_root(tmp_path)
    samples = list(LEVIRCCAdapter().iter_samples(root, "test", "change_caption"))
    assert len(samples) == 1
    sample = samples[0]
    assert sample.task == "change_caption"
    assert sample.ground_truth is not None
    assert sample.ground_truth.answers == ["A building appeared.", "New buildings were constructed."]


def test_t1_t2_strict_mapping_and_order(tmp_path: Path) -> None:
    root = _build_official_root(tmp_path)
    sample = next(LEVIRCCAdapter().iter_samples(root, "test", "change_caption"))
    roles = [image.role for image in sample.images]
    assert roles == ["t1", "t2"]
    t1, t2 = sample.images
    assert t1.path.name == "0001.png" and "A" in t1.path.parts
    assert t2.path.name == "0001.png" and "B" in t2.path.parts
    # The pair must never be swapped: A stays t1, B stays t2. / 绝不交换顺序。
    assert t1.path != t2.path


def test_sample_passes_temporal_validation(tmp_path: Path) -> None:
    from data.validation import validate_sample

    root = _build_official_root(tmp_path)
    sample = next(LEVIRCCAdapter().iter_samples(root, "test", "change_caption"))
    report = validate_sample(sample, root)
    assert report.ok is True, report.issues


def test_before_after_aliases_map_t1_t2(tmp_path: Path) -> None:
    root = tmp_path / "levir_cc_alias"
    _make_image(root / "a.png", 1)
    _make_image(root / "b.png", 2)
    _write_json(root / "LevirCCcaptions.json", [
        {"split": "test", "before": "a.png", "after": "b.png", "captions": ["changed"]},
    ])
    sample = next(LEVIRCCAdapter().iter_samples(root, "test", "change_caption"))
    assert sample.images[0].role == "t1" and sample.images[0].path.name == "a.png"
    assert sample.images[1].role == "t2" and sample.images[1].path.name == "b.png"


def test_filepath_derives_t2_from_t1(tmp_path: Path) -> None:
    root = tmp_path / "levir_cc_derive"
    _make_image(root / "images" / "test" / "A" / "0002.png", 3)
    _make_image(root / "images" / "test" / "B" / "0002.png", 4)
    _write_json(root / "LevirCCcaptions.json", [
        {"split": "test", "filepath": "images/test/A/0002.png", "captions": ["changed"]},
    ])
    sample = next(LEVIRCCAdapter().iter_samples(root, "test", "change_caption"))
    assert "A" in sample.images[0].path.parts
    assert "B" in sample.images[1].path.parts


def test_filename_split_mode_resolves_images(tmp_path: Path) -> None:
    root = tmp_path / "levir_cc_splitmode"
    _make_image(root / "images" / "test" / "A" / "0003.png", 5)
    _make_image(root / "images" / "test" / "B" / "0003.png", 6)
    _write_json(root / "LevirCCcaptions.json", [
        {"split": "test", "filepath": "test", "filename": "0003.png", "captions": ["changed"]},
    ])
    sample = next(LEVIRCCAdapter().iter_samples(root, "test", "change_caption"))
    assert "A" in sample.images[0].path.parts and "B" in sample.images[1].path.parts


def test_split_filtering(tmp_path: Path) -> None:
    root = tmp_path / "levir_cc_split"
    _make_image(root / "images" / "train" / "A" / "0001.png", 7)
    _make_image(root / "images" / "train" / "B" / "0001.png", 8)
    _write_json(root / "LevirCCcaptions.json", [
        {"split": "train", "image_A": "images/train/A/0001.png",
         "image_B": "images/train/B/0001.png", "captions": ["train change"]},
    ])
    assert list(LEVIRCCAdapter().iter_samples(root, "test", "change_caption")) == []
    assert len(list(LEVIRCCAdapter().iter_samples(root, "train", "change_caption"))) == 1


# ── change_qa / 条件支持 ────────────────────────────────────────────────────


def test_change_qa_requires_explicit_question(tmp_path: Path) -> None:
    root = _build_official_root(tmp_path)
    with pytest.raises(DatasetProbeError, match="no question"):
        list(LEVIRCCAdapter().iter_samples(root, "test", "change_qa"))
    _write_json(root / "LevirCCcaptions.json", [
        {"split": "test",
         "image_A": "images/test/A/0001.png",
         "image_B": "images/test/B/0001.png",
         "captions": ["A building appeared."],
         "question": "Did a building appear?"},
    ])
    sample = next(LEVIRCCAdapter().iter_samples(root, "test", "change_qa"))
    assert sample.task == "change_qa"
    assert sample.question == "Did a building appear?"


# ── 显式失败 / explicit failures ───────────────────────────────────────────


def test_missing_captions_never_skipped(tmp_path: Path) -> None:
    root = tmp_path / "levir_cc_bad_cap"
    _make_image(root / "images" / "test" / "A" / "0001.png", 1)
    _make_image(root / "images" / "test" / "B" / "0001.png", 2)
    _write_json(root / "LevirCCcaptions.json", [
        {"split": "test", "image_A": "images/test/A/0001.png", "image_B": "images/test/B/0001.png"},
    ])
    with pytest.raises(DatasetProbeError, match="captions"):
        list(LEVIRCCAdapter().iter_samples(root, "test", "change_caption"))


def test_empty_caption_list_raises(tmp_path: Path) -> None:
    root = tmp_path / "levir_cc_empty_cap"
    _make_image(root / "images" / "test" / "A" / "0001.png", 1)
    _make_image(root / "images" / "test" / "B" / "0001.png", 2)
    _write_json(root / "LevirCCcaptions.json", [
        {"split": "test", "image_A": "images/test/A/0001.png",
         "image_B": "images/test/B/0001.png", "captions": [""]},
    ])
    with pytest.raises(DatasetProbeError, match="no non-empty"):
        list(LEVIRCCAdapter().iter_samples(root, "test", "change_caption"))


def test_missing_image_file_raises(tmp_path: Path) -> None:
    root = tmp_path / "levir_cc_missing_img"
    _write_json(root / "LevirCCcaptions.json", [
        {"split": "test", "image_A": "images/test/A/0001.png",
         "image_B": "images/test/B/0001.png", "captions": ["changed"]},
    ])
    with pytest.raises(DatasetProbeError, match="missing images"):
        list(LEVIRCCAdapter().iter_samples(root, "test", "change_caption"))


def test_multiple_annotations_fail(tmp_path: Path) -> None:
    root = tmp_path / "levir_cc_multi"
    _write_json(root / "a" / "LevirCCcaptions.json", [{"captions": ["x"], "image_A": "a.png", "image_B": "b.png"}])
    _write_json(root / "b" / "LevirCCcaptions.json", [{"captions": ["x"], "image_A": "a.png", "image_B": "b.png"}])
    with pytest.raises(DatasetProbeError, match="Expected exactly one"):
        LEVIRCCAdapter().probe(root)


def test_unsupported_task_fails(tmp_path: Path) -> None:
    root = _build_official_root(tmp_path)
    with pytest.raises(DatasetProbeError, match="not support"):
        list(LEVIRCCAdapter().iter_samples(root, "test", "general_vqa"))


# ── Probe / Registry / 边界 ─────────────────────────────────────────────────


def test_probe_reports_layout(tmp_path: Path) -> None:
    root = _build_official_root(tmp_path)
    probe = LEVIRCCAdapter().probe(root)
    assert probe.dataset == "LEVIR-CC"
    assert probe.sample_count == 1
    assert "captions" in probe.observed_fields


def test_registry_registers_levir_cc_with_aliases() -> None:
    registry = DatasetRegistry()
    register_default_adapters(registry)
    adapter = registry.get("LEVIR-CC")
    assert isinstance(adapter, LEVIRCCAdapter)
    assert registry.get("levir-cc").name == "LEVIR-CC"
    assert registry.get("LEVIR").name == "LEVIR-CC"
    assert {"LEVIR-CC", "VRSBench"} <= set(registry.names())


def test_adapter_never_generates_prompts(tmp_path: Path) -> None:
    source = (REPO_ROOT / "data" / "adapters" / "levir_cc.py").read_text(encoding="utf-8")
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
    # The adapter must never build chat messages or prompt payloads.
    # 适配器绝不构造对话消息或 prompt 载荷。
    assert "prompt=" not in source and '"system"' not in source and "'system'" not in source
