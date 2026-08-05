"""Offline PairValidator coverage. / PairValidator 离线覆盖。"""

from pathlib import Path

from PIL import Image

from spacers_agent.agents.change.pair_validator import PairValidator
from spacers_agent.schemas import ImageRef, UnifiedSample


def _sample(first: Path, second: Path, dataset: str = "LEVIR-CC") -> UnifiedSample:
    return UnifiedSample(
        sample_id="pair", dataset=dataset, split="test", task="change_caption",
        images=[ImageRef(image_id="a", path=first, role="t1"), ImageRef(image_id="b", path=second, role="t2")],
        question="Describe change.",
    )


def test_levir_pair_is_rgb_and_dataset_aligned(tmp_path: Path) -> None:
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    Image.new("L", (24, 20), 100).save(first)
    Image.new("RGBA", (24, 20), (100, 100, 100, 255)).save(second)
    result = PairValidator().validate(_sample(first, second))
    assert result.report.valid
    assert result.report.alignment_status == "assumed_dataset_aligned"
    assert result.t1 is not None and result.t1.shape == (20, 24, 3)


def test_size_mismatch_is_visible_and_not_stretched(tmp_path: Path) -> None:
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    Image.new("RGB", (24, 20)).save(first)
    Image.new("RGB", (12, 20)).save(second)
    result = PairValidator().validate(_sample(first, second))
    assert not result.report.valid
    assert result.report.alignment_status == "unreliable"
    assert "SIZE_MISMATCH_NO_POLICY" in [warning.code for warning in result.report.warnings]

