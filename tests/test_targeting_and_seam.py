from __future__ import annotations

from PIL import Image

from spacers_agent.counting import apply_acceptance_policy
from spacers_agent.seam import build_seam_crop
from spacers_agent.schemas import GlobalPointObservation
from spacers_agent.targeting import _rule_target


def _point(point_id: str, x: int, y: int, *, confidence: float = 0.5, owned: bool = True) -> GlobalPointObservation:
    return GlobalPointObservation(global_id=point_id, target="building", source_tile_id="r0c0", local_id=point_id, local_x_norm=0, local_y_norm=0, local_radius_norm=0, global_x_px=x, global_y_px=y, global_x_norm=0, global_y_norm=0, radius_px=0, confidence=confidence, ownership_valid=owned, near_core_boundary=True, accepted=owned, short_evidence="roof")


def test_rule_target_parser_handles_chinese_and_english_without_answer_number() -> None:
    assert _rule_target("图中有多少栋建筑？").canonical_label == "栋建筑"
    assert _rule_target("How many airplanes?").canonical_label == "airplane"


def test_acceptance_policy_covers_owner_halo_and_threshold() -> None:
    assert apply_acceptance_policy(_point("a", 1, 1, confidence=0.2), min_confidence=0.2).accepted
    assert apply_acceptance_policy(_point("b", 1, 1, confidence=0.199), min_confidence=0.2).rejection_reason == "LOW_CONFIDENCE"
    assert apply_acceptance_policy(_point("c", 1, 1, owned=False), min_confidence=0.0).rejection_reason == "OUTSIDE_OWNER_CORE"


def test_seam_crop_clamps_edges_and_marks_normalized_points() -> None:
    crop = build_seam_crop(Image.new("RGB", (20, 10), "white"), _point("a", 0, 0), _point("b", 19, 9), margin_px=8, max_side=16)
    assert crop.rect.left == 0 and crop.rect.right == 20 and crop.touches_source_border
    assert all(0 <= value <= 999 for value in (*crop.first_norm, *crop.second_norm))
    assert len(crop.sha256) == 64
