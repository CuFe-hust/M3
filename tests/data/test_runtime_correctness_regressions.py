"""Runtime correctness regression tests (Task 11.6 Phase G3).

Task 11.6 运行时正确性回归矩阵：安全 source ID 不绕过路径校验、MME/VRSBench/
XLRS 协议一致性、混合图片根、Lite capability、multi-answer 规则、审计逃逸。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from data.adapters.base import DatasetProbeError
from data.adapters.levir_cc import LEVIRCCAdapter
from data.adapters.mme_realworld import MMERealWorldAdapter
from data.adapters.vrsbench.adapter import VRSBenchAdapter
from data.adapters.xlrs import XLRSAdapter
from data.registry import build_default_registry
from data.schema import stable_sample_id
from data.validation import audit_dataset_root, validate_sample

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def _make_image(path: Path, seed: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (seed, seed * 2, seed * 3)).save(path)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# ── Stable sample ID / 稳定样本 ID ──────────────────────────────────────────


def test_safe_source_id_does_not_bypass_absolute_path_validation() -> None:
    with pytest.raises(ValueError, match="relative"):
        stable_sample_id(
            dataset="D", split="s", source_id="qid-1",
            relative_image_paths=["/absolute/path.png"],
            question="Q", source_index=0,
        )
    with pytest.raises(ValueError, match="relative"):
        stable_sample_id(
            dataset="D", split="s", source_id="qid-1",
            relative_image_paths=[r"C:\absolute\path.png"],
            question="Q", source_index=0,
        )


def test_safe_source_id_does_not_bypass_escape_validation() -> None:
    for bad in ("../escape.png", "images/../../escape.png", "a/./b.png"):
        with pytest.raises(ValueError, match="segment"):
            stable_sample_id(
                dataset="D", split="s", source_id="qid-1",
                relative_image_paths=[bad],
                question="Q", source_index=0,
            )


def test_safe_source_id_does_not_bypass_empty_path_validation() -> None:
    with pytest.raises(ValueError, match="empty"):
        stable_sample_id(
            dataset="D", split="s", source_id="qid-1",
            relative_image_paths=[""],
            question="Q", source_index=0,
        )


# ── MME / task-aware probe ──────────────────────────────────────────────────


def _mme_root(tmp_path: Path, rows: list[dict]) -> Path:
    root = tmp_path / "mme"
    _make_image(root / "img_1.png")
    _write_json(root / "MME_RealWorld.json", rows)
    return root


def test_mme_probe_accepts_supported_task(tmp_path: Path) -> None:
    root = _mme_root(tmp_path, [{"Question_id": "rs_1", "Subtask": "Remote Sensing",
                                "Text": "Q", "Answer choices": ["A", "B", "C", "D"],
                                "Ground truth": "A", "image": "img_1.png"}])
    probe = MMERealWorldAdapter().probe(root, task="multiple_choice_vqa")
    assert probe.sample_count == 1
    assert probe.available_tasks == ("multiple_choice_vqa",)


def test_mme_probe_rejects_unsupported_task(tmp_path: Path) -> None:
    root = _mme_root(tmp_path, [{"Question_id": "rs_1", "Subtask": "Remote Sensing",
                                "Text": "Q", "Answer choices": ["A", "B", "C", "D"],
                                "Ground truth": "A", "image": "img_1.png"}])
    with pytest.raises(DatasetProbeError, match="not support"):
        MMERealWorldAdapter().probe(root, task="general_vqa")


def test_mme_probe_rejects_zero_rs_records(tmp_path: Path) -> None:
    root = _mme_root(tmp_path, [{"Subtask": "OCR", "Question_id": "ocr_1"}])
    with pytest.raises(DatasetProbeError, match="zero remote-sensing"):
        MMERealWorldAdapter().probe(root)


def test_mme_probe_task_none_available_tasks_exact(tmp_path: Path) -> None:
    root = _mme_root(tmp_path, [{"Question_id": "rs_1", "Subtask": "Remote Sensing",
                                "Text": "Q", "Answer choices": ["A", "B", "C", "D"],
                                "Ground truth": "A", "image": "img_1.png"}])
    probe = MMERealWorldAdapter().probe(root)
    assert probe.available_tasks == ("multiple_choice_vqa",)


# ── VRSBench / 图片字段、split、grounding ID ────────────────────────────────


def _vrsbench_row(tmp_path: Path, image_field: str) -> Path:
    root = tmp_path / f"vrsbench_{image_field.replace(' ', '_')}"
    _make_image(root / "Images_val" / "img_1.png")
    row = {"image_id": "img_1.png", "question": "What color is the building?",
           "ground_truth": "red", "question_id": "vq_1", "type": "attribute"}
    if image_field != "image_id":
        row["image_id"] = "img_1.png"
        row[image_field] = "img_1.png"
    _write_json(root / "VRSBench_EVAL_vqa.json", [row])
    return root


@pytest.mark.parametrize("field", ["image", "image_path", "file_name", "filename"])
def test_vrsbench_alternative_image_fields_load(tmp_path: Path, field: str) -> None:
    root = _vrsbench_row(tmp_path, field)
    adapter = VRSBenchAdapter()
    probe = adapter.probe(root, task="general_vqa")
    assert probe.sample_count > 0
    samples = list(adapter.iter_samples(root, "validation", "general_vqa"))
    assert samples
    assert validate_sample(samples[0], root).ok


def test_vrsbench_val_and_validation_ids_match(tmp_path: Path) -> None:
    root = FIXTURES / "vrsbench" / "vqa_only"
    adapter = VRSBenchAdapter()
    val = list(adapter.iter_samples(root, "val", "general_vqa"))
    validation = list(adapter.iter_samples(root, "validation", "general_vqa"))
    assert [s.sample_id for s in val] == [s.sample_id for s in validation]
    assert all(s.split == "validation" for s in val)


def test_vrsbench_grounding_twelve_boxes_no_id_collision(tmp_path: Path) -> None:
    root = tmp_path / "vrsbench_12box"
    _make_image(root / "Images_val" / "x.png")
    _write_json(root / "VRSBench_EVAL_Det.json", [
        {"image_id": "x.png", "objects": [
            {"name": "obj-a", "bbox": [i * 10, 0, i * 10 + 4, 4]} for i in range(12)
        ]},
        {"image_id": "x.png", "objects": [
            {"name": "obj-b", "bbox": [i * 10, 10, i * 10 + 4, 14]} for i in range(12)
        ]},
    ])
    samples = list(VRSBenchAdapter().iter_samples(root, "validation", "grounding"))
    ids = [s.sample_id for s in samples]
    assert len(ids) == 24
    assert len(set(ids)) == 24, "grounding sample IDs must not collide"
    # object 0 box 10 vs object 1 box 0 must differ.
    assert ids[10] != ids[12]


def test_vrsbench_missing_field_raises_probe_error_not_keyerror(tmp_path: Path) -> None:
    root = tmp_path / "vrsbench_missing"
    _make_image(root / "Images_val" / "img_1.png")
    _write_json(root / "VRSBench_EVAL_vqa.json", [
        {"image_id": "img_1.png", "question": "Q", "question_id": "vq_1", "type": "attribute"},
    ])
    with pytest.raises(DatasetProbeError, match="misses one of fields"):
        list(VRSBenchAdapter().iter_samples(root, "validation", "general_vqa"))


# ── XLRS / cache、capability、multi-answer、probe、零行 ─────────────────────


def _xlrs_release(tmp_path: Path) -> Path:
    import shutil

    release = tmp_path / "XLRS-Bench-lite"
    shutil.copytree(FIXTURES / "xlrs" / "XLRS-Bench-lite", release)
    return release


def _xlrs_loader(rows):
    def loader(root: Path, task: str):
        return rows
    return loader


def test_xlrs_default_cache_handles_bytes_and_pil(tmp_path: Path, monkeypatch) -> None:
    import io

    release = _xlrs_release(tmp_path)
    (release / "img_1.png").unlink()
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (5, 5, 5)).save(buffer, format="PNG")
    rows = [
        {"question": "Q1", "choices": ["A", "B", "C", "D"], "answer": "A",
         "image": {"path": "nope.png", "bytes": buffer.getvalue()}},
        {"question": "Q2", "choices": ["A", "B", "C", "D"], "answer": "B",
         "image": Image.new("RGB", (4, 4), (7, 7, 7))},
    ]
    fake_cache = tmp_path / "default_cache"
    adapter = XLRSAdapter(dataset_loader=_xlrs_loader(rows))
    monkeypatch.setattr(adapter, "_effective_cache_root", lambda: fake_cache)
    samples = list(adapter.iter_samples(tmp_path, "train", "multiple_choice_vqa"))
    assert len(samples) == 2
    assert len(list(fake_cache.glob("*.png"))) == 2
    assert all(s.metadata["image_root_kind"] == "cache" for s in samples)


def test_xlrs_mixed_path_and_bytes_row_shares_cache_root(tmp_path: Path) -> None:
    import io
    import shutil

    release = _xlrs_release(tmp_path)
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (6, 6, 6)).save(buffer, format="PNG")
    rows = [{"question": "Q", "choices": ["A", "B", "C", "D"], "answer": "A",
             "images": ["img_1.png", {"path": "nope.png", "bytes": buffer.getvalue()}]}]
    cache = tmp_path / "cache"
    adapter = XLRSAdapter(dataset_loader=_xlrs_loader(rows), cache_root=cache)
    samples = list(adapter.iter_samples(tmp_path, "train", "multiple_choice_vqa"))
    assert len(samples) == 1
    assert samples[0].metadata["image_root_kind"] == "cache"
    # The path-backed image was copied into the cache too.
    # path 图片也被复制进 cache（path_cached）。
    cached_paths = [p.name for p in cache.glob("*.png")]
    assert len(cached_paths) == 2
    report = validate_sample(samples[0], cache)
    assert report.ok is True, report.issues
    # The release root keeps its original file untouched. / release 根原文件未动。
    assert (release / "img_1.png").is_file()
    assert len(list(release.glob("*.png"))) == 1


def test_xlrs_two_samples_release_and_cache_do_not_pollute(tmp_path: Path) -> None:
    release = _xlrs_release(tmp_path)
    rows = [
        {"question": "Q1", "choices": ["A", "B", "C", "D"], "answer": "A",
         "image": "img_1.png"},  # release mode / release 模式
        {"question": "Q2", "choices": ["A", "B", "C", "D"], "answer": "B",
         "image": {"path": "nope.png", "bytes": b"\x89PNG\r\n\x1a\n" + b"\x00" * 16}},
    ]
    cache = tmp_path / "cache"
    adapter = XLRSAdapter(dataset_loader=_xlrs_loader(rows), cache_root=cache)
    samples = list(adapter.iter_samples(tmp_path, "train", "multiple_choice_vqa"))
    assert samples[0].metadata["image_root_kind"] == "release"
    assert samples[1].metadata["image_root_kind"] == "cache"
    assert validate_sample(samples[0], release).ok
    assert validate_sample(samples[1], cache).ok
    assert validate_sample(samples[0], cache).ok is False  # release image not in cache


def test_xlrs_lite_registry_capability_is_exact() -> None:
    adapter = build_default_registry().get("XLRS-Bench-lite")
    assert adapter.supported_tasks == frozenset({"multiple_choice_vqa"})
    with pytest.raises(DatasetProbeError, match="not support"):
        list(adapter.iter_samples(Path("x"), "train", "caption"))
    with pytest.raises(DatasetProbeError, match="not support"):
        list(adapter.iter_samples(Path("x"), "test", "grounding"))


def test_xlrs_multi_answer_does_not_scan_row_text(tmp_path: Path) -> None:
    release = _xlrs_release(tmp_path)
    rows = [
        {"question": "What is the overall land use type?", "choices": ["A", "B", "C", "D"],
         "answer": "B", "image": "img_1.png",
         "choices_extra": "overall land use in another field"},
    ]
    adapter = XLRSAdapter(dataset_loader=_xlrs_loader(rows))
    sample = next(adapter.iter_samples(tmp_path, "train", "multiple_choice_vqa"))
    assert sample.metadata["allow_multiple"] is False
    rows2 = [{"question": "Q", "choices": ["A", "B", "C", "D"], "answer": "B",
              "image": "img_1.png", "allow_multiple": True}]
    sample2 = next(XLRSAdapter(dataset_loader=_xlrs_loader(rows2)).iter_samples(
        tmp_path, "train", "multiple_choice_vqa"))
    assert sample2.metadata["allow_multiple"] is True


def test_xlrs_probe_reports_real_row_count(tmp_path: Path) -> None:
    release = _xlrs_release(tmp_path)
    rows = [
        {"question": f"Q{i}", "choices": ["A", "B", "C", "D"], "answer": "A",
         "image": "img_1.png"} for i in range(3)
    ]
    adapter = XLRSAdapter(dataset_loader=_xlrs_loader(rows))
    probe = adapter.probe(tmp_path, task="multiple_choice_vqa")
    assert probe.sample_count == 3


def test_xlrs_zero_rows_fail_explicitly(tmp_path: Path) -> None:
    release = _xlrs_release(tmp_path)
    adapter = XLRSAdapter(dataset_loader=_xlrs_loader([]))
    with pytest.raises(DatasetProbeError, match="zero XLRS records"):
        list(adapter.iter_samples(tmp_path, "train", "multiple_choice_vqa"))
    with pytest.raises(DatasetProbeError, match="zero XLRS records"):
        adapter.probe(tmp_path, task="multiple_choice_vqa")


def test_xlrs_probe_does_not_download(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    real_hub = XLRSAdapter._load_from_hub

    def spy_hub(release_root: Path, task: str):
        calls.append(task)
        raise AssertionError("hub loader must not be called")

    monkeypatch.setattr(XLRSAdapter, "_load_from_hub", staticmethod(spy_hub))
    try:
        adapter = XLRSAdapter(allow_download=True)  # even with download allowed
        with pytest.raises(DatasetProbeError):
            adapter.probe(tmp_path / "empty")
        assert calls == []
    finally:
        XLRSAdapter._load_from_hub = staticmethod(real_hub)


# ── Audit / 排序与逃逸 ──────────────────────────────────────────────────────


def test_audit_quick_repeat_is_deterministic(tmp_path: Path) -> None:
    root = tmp_path / "audit_order"
    for name in ("c.png", "a.png", "b.png"):
        _make_image(root / "images" / name, seed=1)
    _write_json(root / "ann.json", [
        {"id": f"s{i}", "image": f"images/{name}"} for i, name in enumerate(("c", "a", "b"))
    ])
    first = audit_dataset_root(root, image_sample_limit=2)
    second = audit_dataset_root(root, image_sample_limit=2)
    assert first.image_samples == second.image_samples
    assert first.candidate_manifests == second.candidate_manifests
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_audit_escaped_references_reported(tmp_path: Path) -> None:
    root = tmp_path / "audit_escaped"
    _write_json(root / "ann.json", [
        {"id": "s1", "image": "../outside.png"},
        {"id": "s2", "image": "images/../../outside.png"},
        {"id": "s3", "image": "/absolute/path.png"},
        {"id": "s4", "image": r"C:\absolute\path.png"},
    ])
    report = audit_dataset_root(root)
    assert set(report.escaped_referenced_images) == {
        "../outside.png", "images/../../outside.png", "/absolute/path.png", r"C:\absolute\path.png",
    }
    # A reference belongs to exactly one class. / 一条引用只属于一个类别。
    all_classes = (
        set(report.missing_referenced_images)
        | set(report.unresolved_referenced_images)
        | set(report.ambiguous_referenced_images)
        | set(report.escaped_referenced_images)
    )
    assert not (set(report.escaped_referenced_images) & (
        set(report.missing_referenced_images)
        | set(report.unresolved_referenced_images)
        | set(report.ambiguous_referenced_images)
    ))
    assert len(all_classes) == 4
