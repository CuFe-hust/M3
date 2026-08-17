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
from agents.change.schema import (
    RegistrationDecision,
    RegistrationMetrics,
    RegistrationReport,
)
from agents.change.settings import (
    AgentChangeSettings,
    ChangeLearnedChangeSettings,
    ChangeRegistrationSettings,
)
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


def test_size_mismatch_is_structurally_valid_but_unreliable_alignment(
    tmp_path: Path,
) -> None:
    sample = _write_pair(tmp_path / "data", size_t2=(32, 32))
    pair = PairValidator().validate(sample, data_root=tmp_path / "data")
    assert pair.report.same_size is False
    assert pair.report.alignment_status == "unreliable"
    assert pair.report.valid is True
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


def test_registration_contract_is_json_safe_and_preserves_reason_codes() -> None:
    report = RegistrationReport(
        decision=RegistrationDecision(
            version="global_registration_v1",
            status="rejected",
            model="affine",
            reason_codes=["REGISTRATION_LOW_INLIER_RATIO", "RAW_FALLBACK_USED"],
        ),
        metrics=RegistrationMetrics(
            match_count=20,
            inlier_count=8,
            inlier_ratio=0.4,
            median_reprojection_error=1.2,
            p95_reprojection_error=2.5,
            overlap_ratio=0.8,
            scale_x=1.0,
            scale_y=1.0,
        ),
        transform_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        source_size_t1=[64, 64],
        source_size_t2=[64, 64],
        output_size=[64, 64],
    )
    payload = report.model_dump(mode="json")
    assert payload["decision"]["reason_codes"] == [
        "REGISTRATION_LOW_INLIER_RATIO",
        "RAW_FALLBACK_USED",
    ]
    assert payload["transform_matrix"] == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def test_registration_settings_reject_invalid_thresholds() -> None:
    with pytest.raises(ValueError):
        ChangeRegistrationSettings(ratio_test=1.0)
    with pytest.raises(ValueError):
        ChangeRegistrationSettings(min_overlap_ratio=-0.1)
    with pytest.raises(ValueError, match="min_inliers"):
        ChangeRegistrationSettings(min_matches=4, min_inliers=5)


def test_learned_change_hook_defaults_off_and_validates_settings() -> None:
    settings = AgentChangeSettings()
    assert settings.learned_change == ChangeLearnedChangeSettings()
    assert settings.learned_change.enabled is False
    assert settings.learned_change.fusion_weight == 0.0
    with pytest.raises(ValueError):
        ChangeLearnedChangeSettings(fusion_weight=-0.1)
    with pytest.raises(ValueError):
        ChangeLearnedChangeSettings(failure_policy="unsupported")


def test_registration_can_be_disabled_without_partial_settings() -> None:
    settings = AgentChangeSettings(
        registration=ChangeRegistrationSettings(enabled=False)
    )
    assert settings.registration.enabled is False
    assert settings.registration.quality_policy == "fallback_raw"
    assert settings.registration.matcher == "opencv"


def test_size_mismatch_is_structurally_valid_and_registration_eligible(
    tmp_path: Path,
) -> None:
    sample = _write_pair(tmp_path / "data", size_t2=(32, 32))
    pair = PairValidator().validate(sample, data_root=tmp_path / "data")
    assert pair.t1 is not None and pair.t2 is not None
    assert pair.report.registration_eligible is True
    assert pair.report.valid is True
    registered_pair = PairValidator().validate(
        sample,
        data_root=tmp_path / "data",
        registration_enabled=True,
    )
    assert registered_pair.report.valid is True


def _registration_scene(size: int = 160) -> np.ndarray:
    import cv2

    rng = np.random.default_rng(17)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    points = rng.integers(12, size - 12, size=(90, 2))
    for x, y in points:
        cv2.circle(image, (int(x), int(y)), 2, (220, 220, 220), -1)
    cv2.rectangle(image, (20, 22), (size - 28, size - 35), (120, 210, 70), 2)
    cv2.line(image, (15, size // 2), (size - 15, size // 2), (255, 80, 40), 2)
    return image


def _registration_settings(**overrides):
    from agents.change.settings import ChangeRegistrationSettings

    values = dict(min_matches=6, min_inliers=4, min_inlier_ratio=0.30)
    values.update(overrides)
    return ChangeRegistrationSettings(**values)


def test_registration_identity_translation_and_input_immutability() -> None:
    import cv2

    from agents.change.registration import register_pair

    first = _registration_scene()
    second = cv2.warpAffine(first, np.float32([[1, 0, 12], [0, 1, 7]]), first.shape[1::-1])
    first_before, second_before = first.copy(), second.copy()
    result = register_pair(first, second, settings=_registration_settings())
    assert result.report.decision.status == "applied"
    assert result.report.decision.model == "similarity"
    assert result.report.metrics is not None
    assert result.report.metrics.overlap_ratio > 0.75
    assert result.t1.shape == first.shape == result.t2.shape
    assert result.valid_overlap_mask.shape == first.shape[:2]
    assert np.array_equal(first, first_before)
    assert np.array_equal(second, second_before)


def test_registration_metadata_identity_is_auditable() -> None:
    from agents.change.registration import register_pair

    first = _registration_scene()
    result = register_pair(
        first,
        first.copy(),
        metadata={"geometry_aligned": True},
        settings=_registration_settings(),
    )
    assert result.report.decision.status == "skipped"
    assert result.report.decision.model == "identity"
    assert result.report.decision.used_for_comparison is True
    assert "METADATA_ALIGNMENT_USED" in result.report.decision.reason_codes
    assert bool(result.valid_overlap_mask.all())


def test_registration_scale_rotation_and_affine_are_global_models() -> None:
    import cv2

    from agents.change.registration import register_pair

    first = _registration_scene()
    center = (first.shape[1] / 2, first.shape[0] / 2)
    similarity = cv2.getRotationMatrix2D(center, 8.0, 1.08)
    second = cv2.warpAffine(first, similarity, first.shape[1::-1])
    result = register_pair(first, second, settings=_registration_settings())
    assert result.report.decision.model == "similarity"
    assert result.report.metrics is not None
    assert abs(result.report.metrics.rotation_deg) == pytest.approx(8.0, abs=1.0)

    affine = np.float32([[1.08, 0.18, -8], [0.03, 0.91, 10]])
    affine_second = cv2.warpAffine(first, affine, first.shape[1::-1])
    affine_result = register_pair(
        first,
        affine_second,
        settings=_registration_settings(max_median_reprojection_error=0.8),
    )
    assert affine_result.report.decision.model in {"affine", "homography"}


def test_registration_rejects_insufficient_matches_and_implausible_transform() -> None:
    import cv2

    from agents.change.registration import RegistrationError, register_pair

    blank = np.zeros((128, 128, 3), dtype=np.uint8)
    insufficient = register_pair(blank, blank.copy(), settings=_registration_settings())
    assert "REGISTRATION_INSUFFICIENT_MATCHES" in insufficient.report.decision.reason_codes
    assert "RAW_FALLBACK_USED" in insufficient.report.decision.reason_codes

    first = _registration_scene(192)
    extreme = cv2.resize(first, (384, 384), interpolation=cv2.INTER_LINEAR)
    settings = _registration_settings(max_scale_ratio=1.2, min_overlap_ratio=0.2)
    rejected = register_pair(first, extreme, settings=settings)
    assert rejected.report.decision.status == "rejected"
    assert "RAW_FALLBACK_USED" in rejected.report.decision.reason_codes
    with pytest.raises(RegistrationError):
        register_pair(
            first,
            extreme,
            settings=settings.model_copy(update={"quality_policy": "fail"}),
        )


def test_registration_low_overlap_and_different_sizes_never_stretch_raw() -> None:
    import cv2

    from agents.change.registration import register_pair

    first = _registration_scene()
    translated = cv2.warpAffine(
        first,
        np.float32([[1, 0, 82], [0, 1, 0]]),
        first.shape[1::-1],
    )
    low_overlap = register_pair(
        first,
        translated,
        settings=_registration_settings(
            max_translation_ratio=0.9,
            min_overlap_ratio=0.75,
        ),
    )
    assert "RAW_FALLBACK_USED" in low_overlap.report.decision.reason_codes

    smaller = cv2.resize(first, (128, 128), interpolation=cv2.INTER_AREA)
    different_size = register_pair(
        first,
        smaller,
        settings=_registration_settings(min_overlap_ratio=0.2),
    )
    assert different_size.t1.shape == first.shape
    assert different_size.t2.shape == first.shape
    assert different_size.report.output_size == [160, 160]


def test_registration_is_deterministic_for_same_inputs() -> None:
    import cv2

    from agents.change.registration import register_pair

    first = _registration_scene()
    second = cv2.warpAffine(first, np.float32([[1, 0, 9], [0, 1, -4]]), first.shape[1::-1])
    settings = _registration_settings()
    left = register_pair(first, second, settings=settings)
    right = register_pair(first, second, settings=settings)
    assert left.report.model_dump(mode="json") == right.report.model_dump(mode="json")
    assert np.array_equal(left.t2, right.t2)
    assert np.array_equal(left.valid_overlap_mask, right.valid_overlap_mask)


def test_registration_ransac_preserves_a_new_change_patch() -> None:
    import cv2

    from agents.change.registration import register_pair

    first = _registration_scene()
    second = cv2.warpAffine(
        first,
        np.float32([[1, 0, 9], [0, 1, 5]]),
        first.shape[1::-1],
    )
    cv2.rectangle(second, (102, 102), (124, 124), (255, 255, 255), -1)
    result = register_pair(first, second, settings=_registration_settings())

    assert result.report.decision.used_for_comparison is True
    residual = np.mean(
        np.abs(result.t1.astype(np.int16) - result.t2.astype(np.int16)),
        axis=2,
    )
    valid = result.valid_overlap_mask
    assert float(np.count_nonzero((residual > 40) & valid)) > 100.0
    assert float(np.mean(residual[valid])) < 35.0
