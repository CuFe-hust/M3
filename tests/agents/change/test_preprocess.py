"""Contract tests for change preprocessing orchestration.

变化预处理编排契约测试：组合 pair/harmonizer/proposal、只在 artifact_dir
写派生产物、源图片不被修改、关闭一致化/提议行为、写盘失败显式暴露、产物
相对路径。
"""

from __future__ import annotations

import json
from dataclasses import is_dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from agents.change.preprocess import (
    _local_roi_box,
    _marked_review_crop,
    prepare_pair,
    preprocess_pair,
    publish_change_proposals,
)
from agents.change.registration import RegisteredPair
from agents.change.schema import (
    ChangePreprocessResult,
    ChangeProposal,
    HarmonizationDecision,
    RegistrationDecision,
    RegistrationMetrics,
    RegistrationReport,
    StructuralRescueCandidate,
)
from agents.change.settings import (
    AgentChangeSettings,
    ChangeHarmonizationSettings,
    ChangeProposalSettings,
    ChangeSemanticSettings,
)
from data.schema import GroundTruth, ImageRef, UnifiedSample


def _write_pair(root: Path, *, metadata: dict[str, object] | None = None) -> UnifiedSample:
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (10, 20, 30)).save(root / "t1.png")
    Image.new("RGB", (64, 64), (40, 50, 60)).save(root / "t2.png")
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
        metadata=metadata if metadata is not None else {"geometry_aligned": True},
    )


def _write_change_pair(root: Path) -> UnifiedSample:
    root.mkdir(parents=True, exist_ok=True)
    first = np.zeros((64, 64, 3), dtype=np.uint8)
    first[:, :] = (10, 20, 30)
    second = first.copy()
    second[24:40, 24:40] = (240, 230, 220)
    Image.fromarray(first).save(root / "t1.png")
    Image.fromarray(second).save(root / "t2.png")
    return UnifiedSample(
        sample_id="change-s1",
        dataset="parity",
        split="test",
        task="change_caption",
        images=[
            ImageRef(image_id="t1", path="t1.png", role="t1"),
            ImageRef(image_id="t2", path="t2.png", role="t2"),
        ],
        question="Describe the change.",
        ground_truth=GroundTruth(answers=["x"]),
    )


def _settings(**overrides) -> AgentChangeSettings:
    values = dict(
        harmonization=ChangeHarmonizationSettings(enabled=False),
        proposals=ChangeProposalSettings(enabled=False),
    )
    values.update(overrides)
    return AgentChangeSettings(**values)


def test_preprocess_with_disabled_stages(tmp_path: Path) -> None:
    """Harmonization and proposals disabled produce explicit skips.
    一致化与提议关闭时产生显式跳过。"""
    root = tmp_path / "data"
    sample = _write_pair(root)
    result = preprocess_pair(sample, _settings(), tmp_path / "run", data_root=root)
    assert result.decision.status == "skipped"
    assert "SKIPPED_DISABLED" in result.decision.reason_codes
    assert result.proposals == []
    assert result.artifact_files["validation_report"].endswith("validation_report.json")
    # Only derived artifacts inside artifact_dir. / 仅 artifact_dir 内派生产物。
    for relative in result.artifact_files.values():
        assert (tmp_path / "run" / relative).is_file()
    assert (root / "t1.png").is_file() and (root / "t2.png").is_file()


def test_preprocess_with_proposals_enabled(tmp_path: Path) -> None:
    root = tmp_path / "data"
    sample = _write_pair(root)
    settings = _settings(proposals=ChangeProposalSettings(enabled=True))
    result = preprocess_pair(sample, settings, tmp_path / "run", data_root=root)
    files = result.artifact_files
    assert "difference_map" in files
    assert "proposal_overlay" in files
    assert (tmp_path / "run" / files["difference_map"]).is_file()
    assert (tmp_path / "run" / files["proposal_overlay"]).is_file()
    # Proposals JSON is published. / proposals JSON 已发布。
    assert (tmp_path / "run" / "change_preprocess" / "proposals.json").is_file()


def test_registration_disabled_same_size_preserves_legacy_valid_canvas(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    sample = _write_change_pair(root)
    settings = _settings(
        registration={"enabled": False},
        proposals=ChangeProposalSettings(enabled=True),
    )

    prepared = prepare_pair(sample, settings, tmp_path / "run", data_root=root)
    result = preprocess_pair(sample, settings, tmp_path / "run", data_root=root)

    assert bool(np.all(prepared.registration_valid_mask))
    assert result.proposals
    difference = np.asarray(
        Image.open(tmp_path / "run" / result.artifact_files["difference_map"])
    )
    assert bool(np.any(difference))


def test_enabled_registration_failure_does_not_trust_same_size_raw_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "data"
    sample = _write_pair(root, metadata={})
    report = RegistrationReport(
        decision=RegistrationDecision(
            version="global_registration_v1",
            status="rejected",
            model="none",
            reason_codes=["REGISTRATION_INSUFFICIENT_MATCHES"],
            used_for_comparison=False,
        ),
        metrics=RegistrationMetrics(),
        transform_matrix=None,
        source_size_t1=[64, 64],
        source_size_t2=[64, 64],
        output_size=[64, 64],
    )

    from agents.change import preprocess as preprocess_module

    monkeypatch.setattr(
        preprocess_module,
        "register_pair",
        lambda *args, **kwargs: RegisteredPair(
            t1=args[0].copy(),
            t2=args[1].copy(),
            valid_overlap_mask=np.zeros((64, 64), dtype=bool),
            report=report,
        ),
    )
    settings = _settings(
        registration={"enabled": True},
        proposals=ChangeProposalSettings(enabled=True),
    )

    prepared = prepare_pair(sample, settings, tmp_path / "run", data_root=root)

    assert not bool(np.any(prepared.registration_valid_mask))


def test_different_size_registration_uses_reference_and_registered_crops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (160, 160), (10, 20, 30)).save(root / "t1.png")
    Image.new("RGB", (128, 128), (40, 50, 60)).save(root / "t2.png")
    sample = UnifiedSample(
        sample_id="different-size",
        dataset="parity",
        split="test",
        task="change_caption",
        images=[
            ImageRef(image_id="t1", path="t1.png", role="t1"),
            ImageRef(image_id="t2", path="t2.png", role="t2"),
        ],
        question="Describe the change.",
    )
    report = RegistrationReport(
        decision=RegistrationDecision(
            version="global_registration_v1",
            status="applied",
            model="similarity",
            reason_codes=["REGISTRATION_APPLIED"],
            used_for_comparison=True,
        ),
        metrics=RegistrationMetrics(
            match_count=20,
            inlier_count=18,
            inlier_ratio=0.9,
            median_reprojection_error=1.0,
            overlap_ratio=0.95,
        ),
        transform_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        source_size_t1=[160, 160],
        source_size_t2=[128, 128],
        output_size=[160, 160],
    )
    from agents.change import preprocess as preprocess_module

    monkeypatch.setattr(
        preprocess_module,
        "register_pair",
        lambda *args, **kwargs: RegisteredPair(
            t1=np.asarray(args[0]).copy(),
            t2=np.zeros_like(args[0]),
            valid_overlap_mask=np.ones((160, 160), dtype=bool),
            report=report,
        ),
    )
    settings = _settings(
        harmonization=ChangeHarmonizationSettings(enabled=False),
        proposals=ChangeProposalSettings(enabled=False),
    )
    prepared = prepare_pair(sample, settings, tmp_path / "run", data_root=root)
    proposal = ChangeProposal(
        proposal_id="change_000",
        box=[750, 750, 937, 937],
        pixel_box=[120, 120, 150, 150],
        score=0.8,
        area_ratio=0.05,
        source="fused_change_v2",
        mask_filename="change_000_mask.png",
    )
    result = publish_change_proposals(
        prepared,
        score_map=np.ones((160, 160), dtype=np.float32),
        proposals=[proposal],
        artifact_dir=tmp_path / "run",
        settings=settings,
        component_masks={"change_000_mask.png": np.ones((30, 30), dtype=np.uint8)},
    )

    proposal_result = result.proposals[0]
    assert "change_preprocess/crops/change_000_registered_t2.png" in proposal_result.evidence_filenames
    assert "change_preprocess/crops/change_000_raw_t2.png" not in proposal_result.evidence_filenames
    crop_dir = tmp_path / "run" / "change_preprocess" / "crops"
    assert Image.open(crop_dir / "change_000_raw_t1.png").size == (30, 30)
    assert Image.open(crop_dir / "change_000_registered_t2.png").size == (30, 30)
    assert Image.open(crop_dir / "change_000_mask.png").size == (30, 30)
    assert Image.open(crop_dir / "change_000_mask_overlay.png").size == (30, 30)


def test_preprocess_with_harmonization_enabled(tmp_path: Path) -> None:
    root = tmp_path / "data"
    sample = _write_pair(root)
    settings = _settings(
        harmonization=ChangeHarmonizationSettings(enabled=True, save_artifacts=True),
        proposals=ChangeProposalSettings(enabled=False),
    )
    result = preprocess_pair(sample, settings, tmp_path / "run", data_root=root)
    if result.decision.status == "applied":
        for key in ("harmonized_t1", "harmonized_t2", "pif_mask"):
            assert (tmp_path / "run" / result.artifact_files[key]).is_file()
    else:
        assert "RAW_FALLBACK_USED" in result.decision.reason_codes


def test_invalid_pair_skips_early(tmp_path: Path) -> None:
    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64)).save(root / "t1.png")
    Image.new("RGB", (32, 32)).save(root / "t2.png")
    sample = UnifiedSample(
        sample_id="s1",
        dataset="parity",
        split="test",
        task="change_caption",
        images=[
            ImageRef(image_id="t1", path="t1.png", role="t1"),
            ImageRef(image_id="t2", path="t2.png", role="t2"),
        ],
        question="Q",
    )
    result = preprocess_pair(sample, _settings(), tmp_path / "run", data_root=root)
    assert result.decision.status == "skipped"
    assert "SKIPPED_INVALID_PAIR" in result.decision.reason_codes
    assert result.proposals == []


def test_source_images_are_never_modified(tmp_path: Path) -> None:
    root = tmp_path / "data"
    sample = _write_pair(root)
    before_t1 = (root / "t1.png").read_bytes()
    before_t2 = (root / "t2.png").read_bytes()
    preprocess_pair(
        sample,
        _settings(
            harmonization=ChangeHarmonizationSettings(enabled=True),
            proposals=ChangeProposalSettings(enabled=True),
        ),
        tmp_path / "run",
        data_root=root,
    )
    assert (root / "t1.png").read_bytes() == before_t1
    assert (root / "t2.png").read_bytes() == before_t2


def test_harmonization_exception_surfaces_in_decision(tmp_path: Path, monkeypatch) -> None:
    """A harmonizer failure is visible in the decision, not silent.
    一致化失败在决策中可见，而非静默。"""
    root = tmp_path / "data"
    sample = _write_pair(root)

    from agents.change import harmonizer as harmonizer_module

    def _boom(self, t1, t2):
        raise RuntimeError("harmonizer crash")

    monkeypatch.setattr(harmonizer_module.PairHarmonizer, "run", _boom)
    result = preprocess_pair(
        sample,
        _settings(harmonization=ChangeHarmonizationSettings(enabled=True)),
        tmp_path / "run",
        data_root=root,
    )
    assert result.decision.status == "failed"
    assert "FAILED_HARMONIZATION_EXCEPTION" in result.decision.reason_codes
    assert result.transform_summary["error_type"] == "RuntimeError"


def test_pif_ratio_gate_marks_dense_but_insufficient_mask_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "data"
    sample = _write_pair(root)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[:4, :] = 255  # 256 pixels: enough by count, insufficient by ratio.

    from agents.change import harmonizer as harmonizer_module

    monkeypatch.setattr(
        harmonizer_module,
        "estimate_pif_mask",
        lambda first, second, settings: mask.copy(),
    )
    settings = _settings(
        harmonization=ChangeHarmonizationSettings(
            enabled=True,
            min_pif_pixels=128,
            min_pif_ratio=0.25,
        )
    )

    prepared = prepare_pair(sample, settings, tmp_path / "run", data_root=root)

    assert np.count_nonzero(prepared.pif_mask) == 256
    assert prepared.pif_valid is False
    assert prepared.decision.status == "skipped"
    assert "SKIPPED_INSUFFICIENT_PIF" in prepared.decision.reason_codes


def test_write_failure_is_exposed(tmp_path: Path, monkeypatch) -> None:
    """Artifact write failures must propagate, never be swallowed.
    产物写盘失败必须向上传播，绝不吞掉。"""
    root = tmp_path / "data"
    sample = _write_pair(root)

    from agents.change import preprocess as preprocess_module

    def _broken_write_json(path, payload):
        raise OSError("disk full")

    monkeypatch.setattr(preprocess_module, "_write_json", _broken_write_json)
    with pytest.raises(OSError, match="disk full"):
        preprocess_pair(sample, _settings(), tmp_path / "run", data_root=root)


def test_artifact_files_are_relative_paths(tmp_path: Path) -> None:
    root = tmp_path / "data"
    sample = _write_pair(root)
    result = preprocess_pair(sample, _settings(), tmp_path / "run", data_root=root)
    for relative in result.artifact_files.values():
        assert not Path(relative).is_absolute()
        assert relative.startswith("change_preprocess/")


def test_result_is_serializable(tmp_path: Path) -> None:
    root = tmp_path / "data"
    sample = _write_pair(root)
    result = preprocess_pair(sample, _settings(), tmp_path / "run", data_root=root)
    payload = json.loads(result.model_dump_json())
    assert payload["decision"]["status"] == "skipped"
    assert payload["proposals"] == []


def test_building_rescue_roi_transform_and_review_resize_are_deterministic() -> None:
    local = _local_roi_box((40, 30, 90, 80), (20, 10, 120, 110))
    assert local == (20, 20, 70, 70)
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    marked, scaled, size, scale = _marked_review_crop(
        crop, local, min_short_side=256
    )
    assert size == (256, 256)
    assert marked.shape[:2] == (256, 256)
    assert scaled == (51, 51, 179, 179)
    assert scale == pytest.approx(2.56)


def test_building_rescue_roi_transform_clamps_to_crop() -> None:
    assert _local_roi_box((-10, 4, 140, 90), (0, 0, 100, 80)) == (0, 4, 100, 80)


def test_building_rescue_publishes_raw_and_marked_review_crops(tmp_path: Path) -> None:
    root = tmp_path / "data"
    sample = _write_pair(root)
    settings = _settings()
    prepared = prepare_pair(sample, settings, tmp_path / "run", data_root=root)
    candidate = StructuralRescueCandidate(
        candidate_id="oem:added:0",
        expert_id="oem",
        direction="added",
        box=(16, 16, 32, 32),
        normalized_box=(250, 250, 500, 500),
        score=0.95,
        target_mean_probability=0.95,
        target_p10_probability=0.9,
        target_p50_probability=0.95,
        source_mean_probability=0.01,
        source_p90_probability=0.02,
        source_p95_probability=0.02,
        source_max_probability=0.03,
        area_px=256,
        area_ratio=0.0625,
        registration_tolerance_px=1,
    )
    result = publish_change_proposals(
        prepared,
        score_map=np.zeros((64, 64), dtype=np.float32),
        proposals=[],
        artifact_dir=tmp_path / "run",
        settings=settings,
        rescue_candidates=[candidate],
    )
    published = result.rescue_candidates[0]
    assert len(published.artifact_files) == 2
    assert len(published.review_artifact_files) == 2
    raw_t1 = Image.open(tmp_path / "run" / published.artifact_files[0])
    review_t1 = Image.open(tmp_path / "run" / published.review_artifact_files[0])
    assert raw_t1.size == (26, 26)
    assert min(review_t1.size) >= 256
    assert published.context_crop_bbox is not None
    assert published.local_roi_bbox == (5, 5, 21, 21)
    assert published.review_local_roi_bbox is not None


def test_preprocess_never_calls_qwen() -> None:
    source = (Path(__file__).resolve().parents[3] / "agents" / "change" / "preprocess.py").read_text(
        encoding="utf-8"
    )
    assert "qwen" not in source.casefold()
    assert "complete_json" not in source


def test_legacy_preprocess_validates_and_harmonizes_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "data"
    sample = _write_pair(root)
    from agents.change import preprocess as preprocess_module

    validate_calls = 0
    harmonize_calls = 0
    original_validate = preprocess_module.PairValidator.validate
    original_run = preprocess_module.PairHarmonizer.run

    def counted_validate(self, *args, **kwargs):
        nonlocal validate_calls
        validate_calls += 1
        return original_validate(self, *args, **kwargs)

    def counted_run(self, *args, **kwargs):
        nonlocal harmonize_calls
        harmonize_calls += 1
        return original_run(self, *args, **kwargs)

    monkeypatch.setattr(preprocess_module.PairValidator, "validate", counted_validate)
    monkeypatch.setattr(preprocess_module.PairHarmonizer, "run", counted_run)
    preprocess_pair(
        sample,
        _settings(harmonization=ChangeHarmonizationSettings(enabled=True)),
        tmp_path / "run",
        data_root=root,
    )
    assert validate_calls == 1
    assert harmonize_calls == 1


def test_prepared_pair_keeps_runtime_arrays_outside_serializable_result(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    sample = _write_pair(root)
    settings = _settings()
    prepared = prepare_pair(sample, settings, tmp_path / "run", data_root=root)
    assert is_dataclass(prepared)
    assert isinstance(prepared.raw_t1, np.ndarray)
    assert isinstance(prepared.comparison_t1, np.ndarray)
    assert isinstance(prepared.pif_mask, np.ndarray)
    assert prepared.comparison_t1 is not prepared.raw_t1
    assert prepared.comparison_t2 is not prepared.raw_t2
    assert np.array_equal(prepared.comparison_t1, prepared.raw_t1)
    assert np.array_equal(prepared.comparison_t2, prepared.raw_t2)
    assert np.count_nonzero(prepared.pif_mask) == 0
    assert prepared.pif_valid is False
    assert (tmp_path / "run" / "change_preprocess" / "validation_report.json").is_file()
    assert (tmp_path / "run" / "change_preprocess" / "harmonization_report.json").is_file()

    result = preprocess_pair(sample, settings, tmp_path / "legacy", data_root=root)
    payload = json.loads(result.model_dump_json())
    assert "raw_t1" not in payload
    assert "comparison_t1" not in payload
    assert "pif_mask" not in payload
    with pytest.raises(ValueError):
        ChangePreprocessResult.model_validate(
            {
                **result.model_dump(mode="python"),
                "diagnostics": {"forbidden_array": prepared.pif_mask},
            }
        )


def test_semantic_settings_default_to_full_multiscale_and_validate_geometry() -> None:
    settings = AgentChangeSettings()
    assert settings.semantic.enabled is True
    assert settings.semantic.feature_stage == 1
    assert settings.semantic.feature_stages == (1, 2, 3)
    assert settings.semantic.feature_stage_weights == {
        1: pytest.approx(1 / 3),
        2: pytest.approx(1 / 3),
        3: pytest.approx(1 / 3),
    }
    assert settings.semantic.tile_size == 768
    assert settings.semantic.tile_overlap == 64
    assert settings.proposals.pif_threshold_k == 4.5
    with pytest.raises(ValueError, match="tile_overlap"):
        ChangeSemanticSettings(tile_size=128, tile_overlap=128)
    with pytest.raises(ValueError, match="feature_stages"):
        ChangeSemanticSettings(feature_stages=(1, 5))


def test_fusion_settings_validate_major_branch_weights_and_kernel() -> None:
    with pytest.raises(ValueError, match="fusion weights"):
        ChangeProposalSettings(
            fusion_low_level_weight=0.0,
            fusion_feature_weight=0.0,
            fusion_semantic_weight=0.0,
        )
    with pytest.raises(ValueError, match="must be odd"):
        ChangeProposalSettings(mask_close_kernel=4)


def test_change_proposal_schema_is_backward_compatible_and_accepts_v2() -> None:
    legacy = ChangeProposal.model_validate(
        {
            "proposal_id": "change_000",
            "box": [0, 0, 1, 1],
            "pixel_box": [0, 0, 4, 4],
            "score": 0.5,
            "area_ratio": 0.25,
        }
    )
    assert legacy.source == "difference_map_v1"
    assert legacy.component_scores == {}
    assert legacy.mask_filename is None

    v2 = legacy.model_copy(
        update={
            "source": "fused_change_v2",
            "component_scores": {"feature": 0.7, "semantic": 0.4},
            "mask_filename": "change_preprocess/v2_mask.png",
        }
    )
    restored = ChangeProposal.model_validate_json(v2.model_dump_json())
    assert restored.source == "fused_change_v2"
    assert restored.component_scores["feature"] == 0.7


def test_v1_publisher_parity_and_v2_component_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "data"
    sample = _write_pair(root)
    settings = _settings()
    prepared = prepare_pair(sample, settings, tmp_path / "run", data_root=root)
    legacy = ChangeProposal(
        proposal_id="change_000",
        box=[0, 0, 32, 32],
        pixel_box=[0, 0, 32, 32],
        score=0.8,
        area_ratio=0.25,
    )
    v1 = publish_change_proposals(
        prepared,
        score_map=np.zeros((64, 64), dtype=np.float32),
        proposals=[legacy],
        artifact_dir=tmp_path / "run",
        settings=settings,
    )
    assert len(v1.proposals) == 1
    assert v1.proposals[0].pixel_box == [0, 0, 32, 32]
    assert v1.proposals[0].source == "difference_map_v1"
    assert set(v1.artifact_files) == {
        "validation_report",
        "harmonization_report",
        "difference_map",
        "proposal_overlay",
        "proposals",
    }
    assert v1.proposals[0].evidence_filenames == [
        "change_preprocess/crops/change_000_raw_t1.png",
        "change_preprocess/crops/change_000_raw_t2.png",
    ]

    v2_proposal = legacy.model_copy(
        update={
            "source": "fused_change_v2",
            "component_scores": {"low_level": 0.2, "feature": 0.8},
            "mask_filename": "change_preprocess/v2_mask.png",
        }
    )
    prepared_v2 = prepare_pair(sample, settings, tmp_path / "v2", data_root=root)
    v2 = publish_change_proposals(
        prepared_v2,
        score_map=np.ones((64, 64), dtype=np.float32),
        proposals=[v2_proposal],
        artifact_dir=tmp_path / "v2",
        settings=settings,
        component_maps={"v2_mask": np.ones((64, 64), dtype=np.uint8) * 255},
    )
    assert "fused_change_map" in v2.artifact_files
    assert "difference_map" not in v2.artifact_files
    assert (tmp_path / "v2" / v2.artifact_files["v2_mask"]).is_file()


def test_rejected_transform_publishes_pif_when_v2_actually_uses_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    sample = _write_pair(root)
    settings = _settings(
        harmonization=ChangeHarmonizationSettings(
            enabled=True,
            save_artifacts=True,
        )
    )
    prepared = prepare_pair(sample, settings, tmp_path / "run", data_root=root)
    prepared = replace(
        prepared,
        comparison_t1=prepared.raw_t1.copy(),
        comparison_t2=prepared.raw_t2.copy(),
        pif_mask=np.ones((64, 64), dtype=np.uint8) * 255,
        pif_valid=True,
        decision=HarmonizationDecision(
            version=settings.harmonization.version,
            status="rejected",
            reason_codes=["REJECTED_UNSTABLE_TRANSFORM", "RAW_FALLBACK_USED"],
            metrics=None,
            used_for_proposal=False,
        ),
    )

    result = publish_change_proposals(
        prepared,
        score_map=np.zeros((64, 64), dtype=np.float32),
        proposals=[],
        artifact_dir=tmp_path / "run",
        settings=settings,
        diagnostics={
            "pif_valid": True,
            "pif_used_for_feature_alignment": True,
            "pif_used_for_threshold": True,
        },
    )

    assert result.artifact_files["pif_mask"] == "change_preprocess/pif_mask.png"
    assert (tmp_path / "run" / result.artifact_files["pif_mask"]).is_file()
    assert "harmonized_t1" not in result.artifact_files


def test_consumed_pif_is_mandatory_when_optional_artifacts_are_disabled(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    sample = _write_pair(root)
    settings = _settings(
        harmonization=ChangeHarmonizationSettings(
            enabled=True,
            save_artifacts=False,
        )
    )
    prepared = prepare_pair(sample, settings, tmp_path / "run", data_root=root)
    prepared = replace(
        prepared,
        comparison_t1=prepared.raw_t1.copy(),
        comparison_t2=prepared.raw_t2.copy(),
        pif_mask=np.ones((64, 64), dtype=np.uint8) * 255,
        pif_valid=True,
        decision=HarmonizationDecision(
            version=settings.harmonization.version,
            status="rejected",
            reason_codes=["REJECTED_UNSTABLE_TRANSFORM", "RAW_FALLBACK_USED"],
            metrics=None,
            used_for_proposal=False,
        ),
    )

    result = publish_change_proposals(
        prepared,
        score_map=np.zeros((64, 64), dtype=np.float32),
        proposals=[],
        artifact_dir=tmp_path / "run",
        settings=settings,
        diagnostics={
            "pif_valid": True,
            "pif_used_for_feature_alignment": True,
            "pif_used_for_threshold": True,
        },
    )

    assert result.artifact_files["pif_mask"] == "change_preprocess/pif_mask.png"
    assert (tmp_path / "run" / result.artifact_files["pif_mask"]).is_file()
    assert "harmonized_t1" not in result.artifact_files
    assert "harmonized_t2" not in result.artifact_files


def test_unused_valid_pif_respects_disabled_optional_artifacts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    sample = _write_pair(root)
    settings = _settings(
        harmonization=ChangeHarmonizationSettings(
            enabled=True,
            save_artifacts=False,
        )
    )
    prepared = prepare_pair(sample, settings, tmp_path / "run", data_root=root)
    prepared = replace(
        prepared,
        comparison_t1=prepared.raw_t1.copy(),
        comparison_t2=prepared.raw_t2.copy(),
        pif_mask=np.ones((64, 64), dtype=np.uint8) * 255,
        pif_valid=True,
        decision=HarmonizationDecision(
            version=settings.harmonization.version,
            status="rejected",
            reason_codes=["REJECTED_UNSTABLE_TRANSFORM", "RAW_FALLBACK_USED"],
            metrics=None,
            used_for_proposal=False,
        ),
    )

    result = publish_change_proposals(
        prepared,
        score_map=np.zeros((64, 64), dtype=np.float32),
        proposals=[],
        artifact_dir=tmp_path / "run",
        settings=settings,
        diagnostics={
            "pif_valid": True,
            "pif_used_for_feature_alignment": False,
            "pif_used_for_threshold": False,
        },
    )

    assert "pif_mask" not in result.artifact_files
    assert not (tmp_path / "run" / "change_preprocess" / "pif_mask.png").exists()


def test_invalid_unused_pif_is_not_published_as_v2_evidence(tmp_path: Path) -> None:
    root = tmp_path / "data"
    sample = _write_pair(root)
    settings = _settings(
        harmonization=ChangeHarmonizationSettings(
            enabled=True,
            save_artifacts=True,
        )
    )
    prepared = prepare_pair(sample, settings, tmp_path / "run", data_root=root)
    prepared = replace(
        prepared,
        pif_valid=False,
        decision=HarmonizationDecision(
            version=settings.harmonization.version,
            status="skipped",
            reason_codes=["SKIPPED_INSUFFICIENT_PIF", "RAW_FALLBACK_USED"],
            metrics=None,
            used_for_proposal=False,
        ),
    )

    result = publish_change_proposals(
        prepared,
        score_map=np.zeros((64, 64), dtype=np.float32),
        proposals=[],
        artifact_dir=tmp_path / "run",
        settings=settings,
        diagnostics={
            "pif_valid": False,
            "pif_used_for_feature_alignment": False,
            "pif_used_for_threshold": False,
        },
    )

    assert "pif_mask" not in result.artifact_files


def test_preparation_module_has_no_concrete_model_call() -> None:
    source = (
        Path(__file__).resolve().parents[3] / "agents" / "change" / "preprocess.py"
    ).read_text(encoding="utf-8")
    lowered = source.casefold()
    assert "segformer" not in lowered
    assert "denseSemantic".casefold() not in lowered
    assert "models." not in lowered
