"""Contract tests for the change pair validator.

变化图对校验器契约测试：时相角色、数量、尺寸、对齐状态、图片只读、
拒绝原因可序列化。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from agents.change.pair_validator import PairValidator, ValidatedPair
from data.schema import GroundTruth, ImageRef, UnifiedSample


def _write_pair(
    root: Path,
    size_t1: tuple[int, int] = (64, 64),
    size_t2: tuple[int, int] = (64, 64),
    *,
    metadata: dict | None = None,
) -> UnifiedSample:
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size_t1, (10, 20, 30)).save(root / "t1.png", format="PNG")
    Image.new("RGB", size_t2, (40, 50, 60)).save(root / "t2.png", format="PNG")
    return UnifiedSample(
        sample_id="s1",
        dataset="parity",
        split="test",
        task="change_caption",
        images=[
            ImageRef(image_id="t1", path="t1.png", role="t1"),
            ImageRef(image_id="t2", path="t2.png", role="t2"),
        ],
        question="Describe the change.",
        ground_truth=GroundTruth(answers=["x"]),
        metadata=metadata or {},
    )


def test_valid_pair_with_metadata_alignment(tmp_path: Path) -> None:
    sample = _write_pair(tmp_path / "data", metadata={"geometry_aligned": True})
    pair = PairValidator().validate(sample, data_root=tmp_path / "data")
    assert isinstance(pair, ValidatedPair)
    assert pair.t1 is not None and pair.t2 is not None
    assert pair.report.valid is True
    assert pair.report.temporal_roles_valid is True
    assert pair.report.same_size is True
    assert pair.report.alignment_status == "metadata_aligned"
    assert pair.report.original_sizes == [[64, 64], [64, 64]]


def test_valid_pair_weak_alignment_without_metadata(tmp_path: Path) -> None:
    sample = _write_pair(tmp_path / "data")
    pair = PairValidator().validate(sample, data_root=tmp_path / "data")
    assert pair.report.valid is True
    assert pair.report.alignment_status == "weakly_aligned"
    assert any(
        record.code == "ALIGNMENT_ONLY_SIZE_MATCH" for record in pair.report.warnings
    )


def test_invalid_temporal_roles(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32)).save(root / "a.png")
    Image.new("RGB", (32, 32)).save(root / "b.png")
    sample = UnifiedSample(
        sample_id="s1",
        dataset="parity",
        split="test",
        task="general_vqa",
        images=[
            ImageRef(image_id="a", path="a.png", role="image"),
            ImageRef(image_id="b", path="b.png", role="context"),
        ],
        question="Q",
    )
    pair = PairValidator().validate(sample, data_root=root)
    assert pair.report.valid is False
    assert pair.report.temporal_roles_valid is False
    assert pair.t1 is None and pair.t2 is None
    assert any(record.code == "INVALID_TEMPORAL_ROLES" for record in pair.report.warnings)


def test_size_mismatch_invalidates(tmp_path: Path) -> None:
    sample = _write_pair(tmp_path / "data", size_t2=(32, 32))
    pair = PairValidator().validate(sample, data_root=tmp_path / "data")
    assert pair.report.same_size is False
    assert pair.report.alignment_status == "unreliable"
    assert pair.report.valid is False
    assert any(record.code == "SIZE_MISMATCH_NO_POLICY" for record in pair.report.warnings)


def test_missing_image_fails_cleanly(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32)).save(root / "t1.png")
    sample = UnifiedSample(
        sample_id="s1",
        dataset="parity",
        split="test",
        task="change_caption",
        images=[
            ImageRef(image_id="t1", path="t1.png", role="t1"),
            ImageRef(image_id="t2", path="missing.png", role="t2"),
        ],
        question="Q",
    )
    pair = PairValidator().validate(sample, data_root=root)
    assert pair.report.valid is False
    assert pair.t1 is None and pair.t2 is None
    assert any(record.code == "IMAGE_DECODE_FAILED" for record in pair.report.warnings)


def test_input_images_are_not_mutated(tmp_path: Path) -> None:
    """Decoding must never modify the source image files.
    解码绝不修改源图像文件。"""
    root = tmp_path / "data"
    sample = _write_pair(root, metadata={"geometry_aligned": True})
    before_t1 = (root / "t1.png").read_bytes()
    before_t2 = (root / "t2.png").read_bytes()
    PairValidator().validate(sample, data_root=root)
    assert (root / "t1.png").read_bytes() == before_t1
    assert (root / "t2.png").read_bytes() == before_t2


def test_report_is_serializable(tmp_path: Path) -> None:
    import json

    sample = _write_pair(tmp_path / "data", metadata={"registration_id": "r1"})
    pair = PairValidator().validate(sample, data_root=tmp_path / "data")
    payload = json.loads(pair.report.model_dump_json())
    assert payload["valid"] is True
    assert payload["alignment_status"] == "metadata_aligned"
    assert len(payload["warnings"]) == 0


def test_validator_has_no_dataset_branch() -> None:
    source = (Path(__file__).resolve().parents[3] / "agents" / "change" / "pair_validator.py").read_text(
        encoding="utf-8"
    )
    assert "VRSBench" not in source
    assert "dataset" not in source
    assert "spacers_agent" not in source
