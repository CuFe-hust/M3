"""Public legacy-path compatibility checks.
公开旧路径兼容性检查。
"""

from spacers_agent.agents.counting.point_pipeline import PointCountingOrchestrator as NewPointCountingOrchestrator
from spacers_agent.agents.spatial.evidence_merge import merge_visual_evidence
from spacers_agent.counting import PointCountingOrchestrator
from spacers_agent.workflow import _merge_visual_evidence


def test_legacy_counting_export_uses_the_standalone_implementation() -> None:
    """The retired counting module must not fork the point pipeline.
    已弃用的计数模块不得分叉点计数流水线。
    """

    assert PointCountingOrchestrator is NewPointCountingOrchestrator


def test_legacy_spatial_helper_uses_the_pure_agent_helper() -> None:
    """The retired workflow helper must share spatial merge behavior.
    已弃用的工作流 helper 必须共享空间合并行为。
    """

    assert _merge_visual_evidence is merge_visual_evidence
