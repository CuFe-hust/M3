"""Difference proposal extraction tests. / 差异候选提取测试。"""

import numpy as np

from spacers_agent.agents.change.difference_proposal import propose_changes
from spacers_agent.settings import ChangeProposalSettings


def test_identical_images_have_no_proposals() -> None:
    image = np.full((100, 100, 3), 80, dtype=np.uint8)
    score, proposals = propose_changes(image, image, ChangeProposalSettings())
    assert float(score.max()) == 0.0
    assert proposals == []


def test_inserted_rectangle_is_covered_and_normalized() -> None:
    first = np.full((100, 100, 3), 80, dtype=np.uint8)
    second = first.copy()
    second[30:60, 40:75] = 220
    _, proposals = propose_changes(first, second, ChangeProposalSettings(threshold_quantile=0.85))
    assert proposals
    candidate = proposals[0]
    assert candidate.pixel_box[0] <= 40 and candidate.pixel_box[2] >= 75
    assert candidate.pixel_box[1] <= 30 and candidate.pixel_box[3] >= 60
    assert all(0 <= value <= 999 for value in candidate.box)
