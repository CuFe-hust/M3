"""Contract tests for change preprocessing orchestration.

变化预处理编排契约测试：组合 pair/harmonizer/proposal、只在 artifact_dir
写派生产物、源图片不被修改、关闭一致化/提议行为、写盘失败显式暴露、产物
相对路径。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from agents.change.preprocess import preprocess_pair
from agents.change.settings import AgentChangeSettings, ChangeHarmonizationSettings, ChangeProposalSettings
from data.schema import GroundTruth, ImageRef, UnifiedSample


def _write_pair(root: Path) -> UnifiedSample:
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
        metadata={"geometry_aligned": True},
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


def test_preprocess_never_calls_qwen() -> None:
    source = (Path(__file__).resolve().parents[3] / "agents" / "change" / "preprocess.py").read_text(
        encoding="utf-8"
    )
    assert "qwen" not in source.casefold()
    assert "complete_json" not in source
