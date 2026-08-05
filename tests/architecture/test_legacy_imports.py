"""Freeze current public API — every symbol that must survive migration.
冻结当前公开 API — 迁移后必须继续可用的所有符号。
"""

from __future__ import annotations

import pytest

# ── spacers_agent.workflow / spacers_agent.workflow 公开导出 ──────────────


def test_workflow_public_symbols_importable():
    """All documented workflow symbols are importable."""
    from spacers_agent.workflow import (  # noqa: F401
        DatasetRunner,
        TargetParser,
        WorkflowService,
        ChangeExpert,
        GroundingExpert,
        SpatialExpert,
        GeneralVQAExpert,
        atomic_write_json,
        CountTargetParser,
    )


def test_workflow_vrsbench_helper_symbols():
    """Internal VRSBench helpers used by tests must remain importable."""
    from spacers_agent.workflow import (  # noqa: F401
        _is_tiny_border_fragment,
        _recover_count_proposal_header,
        _accepted_count_evidence,
        _merge_visual_evidence,
        _merge_count_evidence,
    )


def test_workflow_spatial_helper_is_a_compatibility_alias():
    """The old helper name must resolve to the standalone pure implementation.
    旧 helper 名称必须解析到独立纯函数实现。
    """

    from spacers_agent.agents.spatial.evidence_merge import merge_visual_evidence
    from spacers_agent.workflow import _merge_visual_evidence

    assert _merge_visual_evidence is merge_visual_evidence


def test_workflow_visual_expert_importable():
    """VisualExpert base class importable."""
    from spacers_agent.workflow import VisualExpert  # noqa: F401


# ── spacers_agent.routing 公开导出 ──────────────────────────────────────


def test_routing_public_symbols_importable():
    """All documented routing symbols are importable from the package."""
    from spacers_agent.routing import (  # noqa: F401
        TaskRouter,
        RoutingDecision,
        ExpertAssignment,
        CallBudget,
        CountingExpert,
        ROUTES,
        AgentName,
        normalize_agent_name,
    )


def test_routing_budget_helpers_importable():
    """CallBudget helpers importable."""
    from spacers_agent.routing import (  # noqa: F401
        attach_qwen_budget,
        make_budget_guard,
        CountingExpertAnswer,
        CallBudgetExceeded,
    )


def test_routing_routes_type_aliases():
    """Type aliases used by downstream code."""
    from spacers_agent.routing import (  # noqa: F401
        ExpertName,
        RoutableTask,
    )


# ── spacers_agent.counting 公开导出 ─────────────────────────────────────


def test_counting_public_symbols_importable():
    """All documented counting symbols are importable."""
    from spacers_agent.counting import (  # noqa: F401
        PointCountingOrchestrator,
        TileCheckpointStore,
        BoundaryConflict,
        SeamDecision,
        finalize_representatives,
        find_boundary_conflicts,
        apply_acceptance_policy,
    )


# ── spacers_agent.experts 公开导出 ──────────────────────────────────────


def test_experts_public_symbols_importable():
    """Expert Protocol and ExpertContext importable."""
    from spacers_agent.experts import Expert, ExpertContext  # noqa: F401


# ── spacers_agent (top-level) 公开导出 ──────────────────────────────────


def test_top_level_exports():
    """Top-level package exports."""
    from spacers_agent import (  # noqa: F401
        AppSettings,
        load_settings,
    )
    # New dataclass-based types are also exported
    from spacers_agent.agents import (  # noqa: F401
        Agent,
        AgentContext,
        AgentExecution,
        AgentName,
        AgentPayload,
        AgentRegistry,
        LEGACY_AGENT_NAME_ALIASES,
        normalize_agent_name,
    )


# ── spacers_agent.schemas 关键模型 ──────────────────────────────────────


def test_schemas_core_models_importable():
    """Core schema models importable."""
    from spacers_agent.schemas import (  # noqa: F401
        UnifiedSample,
        CountingResult,
        ExpertResult,
        CountTargetSpec,
        TileSpec,
        GlobalPointObservation,
        LocalPointObservation,
        VisualEvidence,
        PixelRect,
        ImageRef,
        GroundTruth,
        YoloCountingSettings,
        YoloDetectorSettings,
        PointProvenance,
        BackendConfig,
    )


# ── spacers_agent.clients ──────────────────────────────────────────────


def test_clients_importable():
    """Shared model base types and test/training clients importable.
    共享模型基础类型及测试/训练客户端可导入。
    """
    from models.base import (  # noqa: F401
        RequestMeta,
        VisionLanguageClient,
        JsonResponseCache,
        image_to_data_url,
        build_request_hash,
    )
    from spacers_agent.clients.deepseek import DeepSeekJudgeClient  # noqa: F401
    from spacers_agent.clients.mock import MockVisionClient  # noqa: F401


# ── spacers_agent.settings ─────────────────────────────────────────────


def test_settings_models_importable():
    """All settings models importable."""
    from spacers_agent.settings import (  # noqa: F401
        AppSettings,
        QwenSettings,
        DeepSeekSettings,
        CountingSettings,
        RunSettings,
        RouterSettings,
        PathSettings,
    )
