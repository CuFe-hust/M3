"""Official-format adapter integration tests (Task 11.5 Phase F/G).

官方格式适配器集成测试：对 tests/fixtures/ 下的官方结构执行
probe → iter_samples → validate_sample 全链路；覆盖 VRSBench 三任务、
LEVIR 顶层结构、MME RS 子集、XLRS release 解析；网络隔离与源目录只读验证。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.adapters.base import DatasetProbeError
from data.adapters.levir_cc import LEVIRCCAdapter
from data.adapters.mme_realworld import MMERealWorldAdapter
from data.adapters.vrsbench.adapter import VRSBenchAdapter
from data.adapters.xlrs import XLRSAdapter
from data.validation import validate_sample

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def _validate_all(samples, root: Path) -> list:
    samples = list(samples)
    assert samples, "no samples produced"
    for sample in samples:
        report = validate_sample(sample, root)
        assert report.ok is True, (sample.sample_id, report.issues)
    return samples


# ── VRSBench / 三任务独立与完整 ─────────────────────────────────────────────


def test_vrsbench_vqa_only_probe_and_iterate() -> None:
    root = FIXTURES / "vrsbench" / "vqa_only"
    adapter = VRSBenchAdapter()
    probe = adapter.probe(root, task="general_vqa")
    assert probe.sample_count > 0
    samples = _validate_all(adapter.iter_samples(root, "validation", "general_vqa"), root)
    assert {sample.task for sample in samples} == {"counting", "spatial_relation", "general_vqa"}


def test_vrsbench_caption_only_probe_and_iterate() -> None:
    root = FIXTURES / "vrsbench" / "caption_only"
    adapter = VRSBenchAdapter()
    probe = adapter.probe(root, task="caption")
    assert probe.sample_count > 0
    samples = _validate_all(adapter.iter_samples(root, "validation", "caption"), root)
    assert samples[0].ground_truth is not None
    assert samples[0].ground_truth.answers == ["A harbor with ships.", "A busy harbor scene."]


def test_vrsbench_grounding_only_probe_and_iterate() -> None:
    root = FIXTURES / "vrsbench" / "grounding_only"
    adapter = VRSBenchAdapter()
    probe = adapter.probe(root, task="grounding")
    assert probe.sample_count > 0
    samples = _validate_all(adapter.iter_samples(root, "validation", "grounding"), root)
    assert len(samples) == 2
    assert samples[0].ground_truth is not None
    assert samples[0].ground_truth.label_binding == "boxes"
    assert samples[0].ground_truth.coordinate_frame == "source_pixels_top_left"
    assert len(samples[1].ground_truth.boxes[0]) == 8  # polygon / 8 值 polygon


def test_vrsbench_full_root_discovers_all_tasks() -> None:
    root = FIXTURES / "vrsbench" / "full"
    adapter = VRSBenchAdapter()
    probe = adapter.probe(root)
    assert set(probe.available_tasks) == {"caption", "general_vqa", "grounding"}
    for task in ("general_vqa", "caption", "grounding"):
        _validate_all(adapter.iter_samples(root, "validation", task), root)


def test_vrsbench_ambiguous_annotation_fails() -> None:
    from data.adapters.base import DatasetProbeError

    root = FIXTURES / "vrsbench" / "ambiguous"
    with pytest.raises(DatasetProbeError, match="Expected exactly one"):
        VRSBenchAdapter().probe(root, task="general_vqa")


def test_vrsbench_grounding_refs_and_top_level_structures(tmp_path: Path) -> None:
    from data.adapters.base import DatasetProbeError
    from PIL import Image

    root = tmp_path / "vrsbench_refs"
    (root / "Images_val").mkdir(parents=True)
    Image.new("RGB", (4, 4), (1, 2, 3)).save(root / "Images_val" / "x.png")
    (root / "VRSBench_EVAL_Det.json").write_text(json.dumps([
        # refs structure / refs 结构
        {"image_id": "x.png", "refs": [{"ref": "the ship near the port", "box": [10, 20, 30, 40]}]},
        # top-level single-object structure / 顶层单对象结构
        {"image_id": "x.png", "question": "the plane", "obj_corner": [1, 2, 3, 4], "obj_cls": "plane"},
    ]), encoding="utf-8")
    samples = _validate_all(VRSBenchAdapter().iter_samples(root, "validation", "grounding"), root)
    assert samples[0].question == "the ship near the port"  # referring preserved / 保留源 referring
    assert samples[1].ground_truth is not None
    assert samples[1].ground_truth.labels == ["plane"]


# ── LEVIR-CC / 官方顶层结构 ─────────────────────────────────────────────────


def test_levir_official_top_level_layout() -> None:
    root = FIXTURES / "levir_cc" / "full"
    adapter = LEVIRCCAdapter()
    probe = adapter.probe(root)
    assert probe.sample_count > 0
    samples = _validate_all(adapter.iter_samples(root, "test", "change_caption"), root)
    assert samples[0].images[0].role == "t1"
    assert samples[0].images[1].role == "t2"
    assert "A" in samples[0].images[0].path.parts
    assert "B" in samples[0].images[1].path.parts
    assert samples[0].ground_truth is not None
    assert samples[0].ground_truth.answers == [
        "A building was constructed.", "A new building appeared.",
    ]


# ── MME-RealWorld / RS 子集 ─────────────────────────────────────────────────


def test_mme_rs_only_loaded() -> None:
    root = FIXTURES / "mme_realworld" / "full"
    adapter = MMERealWorldAdapter()
    probe = adapter.probe(root)
    assert probe.sample_count == 1
    samples = _validate_all(adapter.iter_samples(root, "test", "multiple_choice_vqa"), root)
    assert [sample.sample_id for sample in samples] == ["rs_1"]
    assert "B" not in samples[0].question


# ── XLRS / release 解析与加载 ───────────────────────────────────────────────


def _xlrs_loader(rows):
    def loader(root: Path, task: str):
        return rows
    return loader


def test_xlrs_release_root_resolution_with_loader(tmp_path: Path) -> None:
    import shutil

    from PIL import Image

    unified = tmp_path / "xlrs"
    release = unified / "XLRS-Bench-lite"
    shutil.copytree(FIXTURES / "xlrs" / "XLRS-Bench-lite", release)
    Image.new("RGB", (4, 4), (9, 9, 9)).save(release / "img_1.png")
    rows = [{"question": "What is the overall land use type?",
             "choices": ["A", "B", "C", "D"], "answer": "B", "image": "img_1.png"}]
    adapter = XLRSAdapter(dataset_loader=_xlrs_loader(rows))
    probe = adapter.probe(unified, task="multiple_choice_vqa")
    assert probe.sample_count > 0
    samples = _validate_all(adapter.iter_samples(unified, "train", "multiple_choice_vqa"), release)
    assert samples[0].metadata["choices"] == ["A", "B", "C", "D"]


def test_xlrs_split_mismatch_fails_before_loading(tmp_path: Path) -> None:
    import shutil

    unified = tmp_path / "xlrs_mismatch"
    shutil.copytree(FIXTURES / "xlrs" / "XLRS-Bench-lite", unified / "XLRS-Bench-lite")
    calls: list[str] = []

    def loader(root: Path, task: str):
        calls.append(task)
        return []

    adapter = XLRSAdapter(dataset_loader=loader)
    with pytest.raises(DatasetProbeError, match="requires split='train'"):
        list(adapter.iter_samples(unified, "test", "multiple_choice_vqa"))
    assert calls == [], "split mismatch must fail before any load"


def test_xlrs_allow_download_false_never_calls_hub_loader(tmp_path: Path) -> None:
    calls: list[str] = []
    real_load_from_hub = XLRSAdapter._load_from_hub

    def spy_hub(release_root: Path, task: str):
        calls.append(task)
        raise AssertionError("hub loader must not be called")

    XLRSAdapter._load_from_hub = staticmethod(spy_hub)
    try:
        adapter = XLRSAdapter()  # allow_download=False by default
        with pytest.raises(DatasetProbeError, match="no local"):
            list(adapter.iter_samples(tmp_path / "empty", "train", "multiple_choice_vqa"))
        assert calls == []
    finally:
        XLRSAdapter._load_from_hub = staticmethod(real_load_from_hub)


# ── 只读验证 / read-only source verification ───────────────────────────────


def test_adapter_run_does_not_modify_source(tmp_path: Path) -> None:
    import shutil

    root = tmp_path / "vrsbench_ro"
    shutil.copytree(FIXTURES / "vrsbench" / "full", root)
    before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }
    samples = list(VRSBenchAdapter().iter_samples(root, "validation", "general_vqa"))
    assert samples
    after = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }
    assert before == after


# ── XLRS bytes/PIL 物化 / materialization ──────────────────────────────────


def test_xlrs_bytes_image_materializes_to_external_cache(tmp_path: Path) -> None:
    import hashlib
    import io
    import shutil

    from PIL import Image

    unified = tmp_path / "xlrs_bytes"
    release = unified / "XLRS-Bench-lite"
    shutil.copytree(FIXTURES / "xlrs" / "XLRS-Bench-lite", release)
    (release / "img_1.png").unlink()  # no path-backed image / 移除 path 图片
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (6, 6, 6)).save(buffer, format="PNG")
    rows = [{"question": "Which class is the target?", "choices": ["A", "B", "C", "D"],
             "answer": "A", "image": {"path": "missing.png", "bytes": buffer.getvalue()}}]
    cache = tmp_path / "cache"
    adapter = XLRSAdapter(dataset_loader=_xlrs_loader(rows), cache_root=cache)
    samples = list(adapter.iter_samples(unified, "train", "multiple_choice_vqa"))
    assert len(samples) == 1
    # Cache file exists, deterministic name. / cache 文件存在，确定性命名。
    cached = list(cache.glob("*.png"))
    assert len(cached) == 1
    assert cached[0].name == hashlib.sha256(buffer.getvalue()).hexdigest() + ".png"
    # Re-run reuses the same cache file. / 重跑复用同一 cache 文件。
    adapter2 = XLRSAdapter(dataset_loader=_xlrs_loader(rows), cache_root=cache)
    list(adapter2.iter_samples(unified, "train", "multiple_choice_vqa"))
    assert len(list(cache.glob("*.png"))) == 1
    # validate against the cache root (the sample's image_root_kind=cache).
    # 以 cache root（样本 image_root_kind=cache）执行校验。
    report = validate_sample(samples[0], cache)
    assert report.ok is True, report.issues
    # The dataset release root gained no new files (real snapshot).
    # dataset release 根未新增文件（真实快照）。
    release_files_before = {
        path.relative_to(release).as_posix(): path.read_bytes()
        for path in release.rglob("*") if path.is_file()
    }
    assert (release / "missing.png").exists() is False
    release_files_after = {
        path.relative_to(release).as_posix(): path.read_bytes()
        for path in release.rglob("*") if path.is_file()
    }
    assert release_files_after == release_files_before
    assert samples[0].ground_truth is not None
    assert samples[0].ground_truth.raw["source_row"]["image"]["image_present"] is True
    assert samples[0].metadata["image_root_kind"] == "cache"


def test_xlrs_pil_image_materializes_and_raw_stays_json_safe(tmp_path: Path) -> None:
    import shutil

    from PIL import Image

    unified = tmp_path / "xlrs_pil"
    release = unified / "XLRS-Bench-lite"
    shutil.copytree(FIXTURES / "xlrs" / "XLRS-Bench-lite", release)
    (release / "img_1.png").unlink()
    pil_image = Image.new("RGB", (4, 4), (8, 8, 8))
    rows = [{"question": "What is the land use?", "choices": ["A", "B", "C", "D"],
             "answer": "B", "image": pil_image}]
    cache = tmp_path / "cache_pil"
    adapter = XLRSAdapter(dataset_loader=_xlrs_loader(rows), cache_root=cache)
    samples = list(adapter.iter_samples(unified, "train", "multiple_choice_vqa"))
    assert len(samples) == 1
    assert len(list(cache.glob("*.png"))) == 1
    # raw must stay JSON-serializable (no PIL object). / raw 必须保持 JSON 可序列化。
    import json

    payload = samples[0].model_dump(mode="json")
    json.dumps(payload)
    assert payload["ground_truth"]["raw"]["source_row"]["image"]["image_source_type"] == "pil"
