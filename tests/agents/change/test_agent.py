"""Offline unit tests for the auditable dual-path ChangeAgent.

可审计双路径 ChangeAgent 离线单测：三种有序 image manifest（raw_only /
harmonized_only / dual_path）、载荷（temporal roles / harmonization /
proposal summary）、Qwen budget 恰好一次、reviewer 复核降级、失败不伪装为
completed、无模型情况下以 fake client 完整测试。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from agents.base import AgentContext, AgentExecution
from agents.change.agent import ChangeAgent, resolve_input_mode
from agents.change.schema import (
    ChangeAdjudicationResult,
    ChangePreprocessResult,
    ChangeProposal,
    HarmonizationDecision,
    HarmonizationMetrics,
    PairValidationReport,
    RegistrationDecision,
    RegistrationMetrics,
    RegistrationReport,
    SemanticTransition,
)
from agents.change.settings import (
    AgentChangeSettings,
    ChangeHarmonizationSettings,
    ChangeProposalSettings,
)
from agents.errors import AgentExecutionError, AgentTaskMismatchError
from agents.schema import AgentResult
from agents.visual_base import PromptBinding
from data.schema import GroundTruth, ImageRef, UnifiedSample
from models.base import ModelCacheIdentity


class _FakeBudget:
    def __init__(self) -> None:
        self.qwen_calls = 0

    def reserve_qwen(self) -> None:
        self.qwen_calls += 1

    def reserve_deepseek(self) -> None:
        pass


class _RecordingClient:
    """Records messages and request meta; returns a stable AgentResult.
    记录消息与请求元数据；返回稳定的 AgentResult。"""

    def __init__(self, answer: str = "A building was removed.") -> None:
        self.calls: list[dict[str, Any]] = []
        self._answer = answer

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="fake-model",
            generation={"temperature": 0.0, "do_sample": False, "max_tokens": 128},
            client_version="1",
        )

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls.append({"messages": messages, "request_hash": request_meta.request_hash})
        if response_model.__name__ == "ChangeAdjudicationResult":
            payload = json.loads(messages[1]["content"][-1]["text"])
            return response_model.model_validate({
                "agent_name": "change_agent",
                "global_review": {"verdict": "no_persistent_change", "t1_state": "stable scene", "t2_state": "stable scene", "reason": "no persistent geometry change", "change_category": None},
                "candidate_reviews": [{"proposal_id": item["proposal_id"], "verdict": "appearance_only", "t1_state": "same structure", "t2_state": "same structure", "reason": "appearance differs only", "change_category": None} for item in payload["adjudication_candidates"]],
                "answer": "No significant semantic change detected.", "status": "completed",
            })
        return response_model.model_validate(
            {"agent_name": "change_agent", "answer": self._answer, "status": "completed"}
        )


class _NoIdentityClient(_RecordingClient):
    @property
    def cache_identity(self) -> None:
        return None


def _make_pair(root: Path, *, identical: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), (10, 20, 30)).save(root / "t1.png")
    if identical:
        Image.new("RGB", (64, 64), (10, 20, 30)).save(root / "t2.png")
    else:
        Image.new("RGB", (64, 64), (40, 50, 60)).save(root / "t2.png")


def _sample(
    root: Path, *, task: str = "change_caption", change: bool = True
) -> UnifiedSample:
    _make_pair(root)
    images = (
        [
            ImageRef(image_id="t1", path="t1.png", role="t1"),
            ImageRef(image_id="t2", path="t2.png", role="t2"),
        ]
        if change
        else [ImageRef(image_id="i1", path="t1.png", role="image")]
    )
    return UnifiedSample(
        sample_id="s1",
        dataset="parity",
        split="test",
        task=task,  # type: ignore[arg-type]
        images=images,
        question="Describe the change.",
        ground_truth=GroundTruth(answers=["x"]),
    )


def _agent(client: _RecordingClient | None = None, **kwargs) -> ChangeAgent:
    kwargs.setdefault(
        "prompt",
        PromptBinding(text="Raw T1/T2 evidence is authoritative.", version="v2"),
    )
    return ChangeAgent(client or _RecordingClient(), **kwargs)


def _context(root: Path, budget: _FakeBudget | None = None) -> AgentContext:
    return AgentContext(
        artifact_dir=root / "artifacts",
        qwen_client=None,
        call_budget=budget or _FakeBudget(),
        data_root=root,
    )


def _last_user_payload(client: _RecordingClient) -> dict[str, Any]:
    """Parse the JSON payload embedded in the last recorded user message.
    解析最后一条已记录 user 消息中的 JSON 载荷。"""
    messages = client.calls[-1]["messages"]
    user_content = messages[1]["content"]
    text = user_content[-1]["text"]
    return json.loads(text)


def _manifest_roles(client: _RecordingClient) -> list[str]:
    return [item["role"] for item in _last_user_payload(client)["image_manifest"]]


def _stub_preprocess(
    artifact_dir: Path,
    *,
    status: str = "applied",
    proposals: list[ChangeProposal] | None = None,
) -> ChangePreprocessResult:
    """A controlled preprocess result with real artifact files on disk.
    带真实产物文件的受控预处理结果。"""
    output = artifact_dir / "change_preprocess"
    output.mkdir(parents=True, exist_ok=True)
    if status == "applied":
        Image.new("RGB", (32, 32), (1, 2, 3)).save(output / "harmonized_t1.png")
        Image.new("RGB", (32, 32), (1, 2, 3)).save(output / "harmonized_t2.png")
    Image.new("RGB", (32, 32), (1, 2, 3)).save(output / "proposal_overlay.png")
    proposals = proposals or []
    for proposal in proposals:
        for relative in proposal.evidence_filenames:
            path = artifact_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 8), (1, 2, 3)).save(path)
    metrics = HarmonizationMetrics(
        pif_ratio=0.5,
        mad_full_before=10.0,
        mad_full_after=8.0,
        mad_pif_before=5.0,
        mad_pif_after=4.0,
        pct_diff_gt20_before=0.1,
        pct_diff_gt20_after=0.08,
        lapvar_t1_before=20.0,
        lapvar_t2_before=22.0,
        lapvar_t1_after=20.0,
        lapvar_t2_after=21.0,
    )
    files = {
        "validation_report": "change_preprocess/validation_report.json",
        "difference_map": "change_preprocess/difference_map.png",
        "proposal_overlay": "change_preprocess/proposal_overlay.png",
        "harmonization_report": "change_preprocess/harmonization_report.json",
    }
    if status == "applied":
        files.update(
            {
                "harmonized_t1": "change_preprocess/harmonized_t1.png",
                "harmonized_t2": "change_preprocess/harmonized_t2.png",
            }
        )
    if proposals:
        files["proposals"] = "change_preprocess/proposals.json"
    return ChangePreprocessResult(
        validation=PairValidationReport(
            valid=True,
            temporal_roles_valid=True,
            same_size=True,
            alignment_status="assumed_dataset_aligned",
        ),
        decision=HarmonizationDecision(
            version="pif_lab_midpoint_v1",
            status=status,  # type: ignore[arg-type]
            reason_codes=["PIF_MATCHED"] if status == "applied" else ["RAW_FALLBACK_USED"],
            metrics=metrics if status == "applied" else None,
            used_for_proposal=status == "applied",
        ),
        proposals=proposals,
        artifact_files=files,
        transform_summary={"sharpness_adjustment_used": status == "applied"},
    )


def _proposal(proposal_id: str = "change_000", score: float = 0.8) -> ChangeProposal:
    return ChangeProposal(
        proposal_id=proposal_id,
        box=[100, 100, 300, 300],
        pixel_box=[10, 10, 20, 20],
        score=score,
        area_ratio=0.05,
        evidence_filenames=[
            f"change_preprocess/crops/{proposal_id}_raw_t1.png",
            f"change_preprocess/crops/{proposal_id}_raw_t2.png",
        ],
    )


# ── 协议 / protocol ────────────────────────────────────────────────────────


def test_agent_identity_and_tasks() -> None:
    agent = _agent()
    assert agent.name == "change_agent"
    assert agent.supported_tasks == frozenset({"change_caption", "change_qa"})


def test_resolve_input_mode_policy() -> None:
    assert resolve_input_mode(
        AgentChangeSettings(harmonization=ChangeHarmonizationSettings(enabled=False))
    ) == "raw_only"
    assert resolve_input_mode(
        AgentChangeSettings(proposals=ChangeProposalSettings(enabled=False))
    ) == "harmonized_only"
    assert resolve_input_mode(AgentChangeSettings()) == "dual_path"


def test_unsupported_task_fails_before_any_io(tmp_path: Path) -> None:
    client = _RecordingClient()
    budget = _FakeBudget()
    with pytest.raises(AgentTaskMismatchError):
        asyncio.run(
            _agent(client).run(
                _sample(tmp_path, task="counting", change=False), _context(tmp_path, budget)
            )
        )
    assert budget.qwen_calls == 0
    assert client.calls == []


def test_invalid_cache_identity_fails_before_preprocess(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    def _boom(*args, **kwargs):
        calls.append("preprocess")
        raise AssertionError("preprocess must not run")

    monkeypatch.setattr(ChangeAgent, "_prepare_perception_and_publish", _boom)
    with pytest.raises(AgentExecutionError, match="cache_identity"):
        asyncio.run(_agent(_NoIdentityClient()).run(_sample(tmp_path), _context(tmp_path)))


def test_data_root_required(tmp_path: Path) -> None:
    context = AgentContext(
        artifact_dir=tmp_path / "artifacts",
        qwen_client=None,
        call_budget=_FakeBudget(),
        data_root=None,
    )
    with pytest.raises(AgentExecutionError, match="data_root"):
        asyncio.run(_agent().run(_sample(tmp_path), context))


# ── 三种 manifest 模式 / three manifest modes ──────────────────────────────


def test_run_raw_only_uses_real_preprocess(tmp_path: Path) -> None:
    client = _RecordingClient()
    budget = _FakeBudget()
    agent = _agent(
        client,
        settings=AgentChangeSettings(harmonization=ChangeHarmonizationSettings(enabled=False)),
    )
    execution = asyncio.run(agent.run(_sample(tmp_path), _context(tmp_path, budget)))
    assert isinstance(execution, AgentExecution)
    assert isinstance(execution.payload, AgentResult)
    assert execution.agent_name == "change_agent"
    assert execution.result_filename == "agent_result.json"
    assert _manifest_roles(client) == ["raw_full_t1", "raw_full_t2"]
    assert budget.qwen_calls == 1
    assert len(client.calls) == 1


def test_run_dual_path_with_identical_images(tmp_path: Path) -> None:
    """Identical images make the harmonizer apply deterministically; the
    dual-path manifest then carries raw + harmonized + overlay images.
    相同图像使一致化确定性 applied；双路径 manifest 携带 raw + harmonized +
    overlay 图像。"""
    client = _RecordingClient()
    execution = asyncio.run(_agent(client).run(_sample(tmp_path), _context(tmp_path)))
    payload = _last_user_payload(client)
    assert payload["input_mode"] == "dual_path"
    assert _manifest_roles(client) == ["raw_full_t1", "raw_full_t2"]
    assert execution.trace["harmonization_status"] == "applied"
    assert execution.trace["pif_ratio"] is not None
    assert execution.trace["proposal_count"] == 0


def test_run_harmonized_only(tmp_path: Path) -> None:
    client = _RecordingClient()
    agent = _agent(
        client,
        settings=AgentChangeSettings(proposals=ChangeProposalSettings(enabled=False)),
    )
    asyncio.run(agent.run(_sample(tmp_path), _context(tmp_path)))
    payload = _last_user_payload(client)
    assert payload["input_mode"] == "harmonized_only"
    assert _manifest_roles(client) == ["raw_full_t1", "raw_full_t2"]


def test_raw_authority_is_present_even_when_derived_evidence_is_available(
    tmp_path: Path,
) -> None:
    client = _RecordingClient()
    asyncio.run(_agent(client).run(_sample(tmp_path), _context(tmp_path)))
    payload = _last_user_payload(client)
    assert payload["image_manifest"][0:2] == [
        {"index": "0", "role": "raw_full_t1"},
        {"index": "1", "role": "raw_full_t2"},
    ]
    assert payload["evidence_audit"]["raw_authority_attached"] is True


def test_registered_global_evidence_has_explicit_role(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preprocess = _stub_preprocess(tmp_path / "artifacts")
    registered_path = tmp_path / "artifacts" / "change_preprocess" / "registered_t2.png"
    Image.new("RGB", (32, 32), (2, 3, 4)).save(registered_path)
    preprocess = preprocess.model_copy(
        update={
            "artifact_files": {
                **preprocess.artifact_files,
                "registered_t2": "change_preprocess/registered_t2.png",
            },
            "registration": RegistrationReport(
                decision=RegistrationDecision(
                    version="global_registration_v1",
                    status="applied",
                    model="affine",
                    reason_codes=["REGISTRATION_APPLIED"],
                    used_for_comparison=True,
                ),
                metrics=RegistrationMetrics(
                    match_count=20,
                    inlier_count=18,
                    inlier_ratio=0.9,
                    median_reprojection_error=1.2,
                    p95_reprojection_error=2.4,
                    overlap_ratio=0.95,
                ),
                transform_matrix=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                source_size_t1=[32, 32],
                source_size_t2=[32, 32],
                output_size=[32, 32],
            ),
        }
    )
    monkeypatch.setattr(
        ChangeAgent,
        "_prepare_perception_and_publish",
        lambda self, sample, context: preprocess,
    )
    client = _RecordingClient()
    asyncio.run(_agent(client).run(_sample(tmp_path), _context(tmp_path)))
    assert _manifest_roles(client)[0:2] == ["raw_full_t1", "raw_full_t2"]
    payload = _last_user_payload(client)
    assert payload["registration"]["quality"] == {
        "inlier_ratio": 0.9,
        "median_reprojection_error": 1.2,
        "overlap_ratio": 0.95,
    }


def test_run_dual_path_includes_crops(tmp_path: Path, monkeypatch) -> None:
    proposal = _proposal()
    preprocess = _stub_preprocess(tmp_path / "artifacts", proposals=[proposal])

    def _stub(self, sample, context):
        return preprocess

    monkeypatch.setattr(ChangeAgent, "_prepare_perception_and_publish", _stub)
    client = _RecordingClient()
    asyncio.run(_agent(client).run(_sample(tmp_path), _context(tmp_path)))
    assert _manifest_roles(client) == [
        "raw_full_t1", "raw_full_t2", "proposal_overlay",
        "change_000:reference_t1_crop", "change_000:t2_raw_fallback_crop",
    ]


def test_rejected_transform_keeps_raw_and_attaches_proposal_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proposal = _proposal()
    preprocess = _stub_preprocess(
        tmp_path / "artifacts",
        status="rejected",
        proposals=[proposal],
    )

    monkeypatch.setattr(
        ChangeAgent,
        "_prepare_perception_and_publish",
        lambda self, sample, context: preprocess,
    )
    client = _RecordingClient()
    budget = _FakeBudget()

    execution = asyncio.run(
        _agent(client).run(_sample(tmp_path), _context(tmp_path, budget))
    )

    assert _manifest_roles(client) == [
        "raw_full_t1",
        "raw_full_t2",
        "proposal_overlay",
        "change_000:reference_t1_crop",
        "change_000:t2_raw_fallback_crop",
    ]
    assert len(client.calls) == 1
    assert budget.qwen_calls == 1
    assert execution.trace["harmonized_evidence_available"] is False
    assert execution.trace["proposal_evidence_attached"] is True
    assert execution.trace["image_manifest_roles"] == _manifest_roles(client)


def test_manifest_ordering_is_stable_and_indexed(tmp_path: Path, monkeypatch) -> None:
    preprocess = _stub_preprocess(tmp_path / "artifacts", proposals=[_proposal()])

    def _stub(self, sample, context):
        return preprocess

    monkeypatch.setattr(ChangeAgent, "_prepare_perception_and_publish", _stub)
    client = _RecordingClient()
    asyncio.run(_agent(client).run(_sample(tmp_path), _context(tmp_path)))
    manifest = _last_user_payload(client)["image_manifest"]
    assert [item["index"] for item in manifest] == [str(i) for i in range(len(manifest))]
    roles = [item["role"] for item in manifest]
    assert roles[0] == "raw_full_t1" and roles[1] == "raw_full_t2"


def test_proposals_are_ranked_by_score_times_reliability_and_truncated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AgentChangeSettings()
    low_score_high_reliability = _proposal("change_000", 0.9).model_copy(
        update={"reliability": {"registration": 0.1}}
    )
    high_score_low_reliability = _proposal("change_001", 0.8).model_copy(
        update={"reliability": {"registration": 1.0}}
    )
    preprocess = _stub_preprocess(
        tmp_path / "artifacts",
        proposals=[low_score_high_reliability, high_score_low_reliability],
    )
    monkeypatch.setattr(
        ChangeAgent,
        "_prepare_perception_and_publish",
        lambda self, sample, context: preprocess,
    )
    client = _RecordingClient()
    asyncio.run(_agent(client, settings=settings).run(_sample(tmp_path), _context(tmp_path)))
    roles = _manifest_roles(client)
    assert "change_001:reference_t1_crop" in roles
    assert "change_000:reference_t1_crop" in roles
    payload = _last_user_payload(client)
    assert payload["evidence_audit"]["proposal_count_total"] == 2
    assert payload["evidence_audit"]["proposal_count_attached"] == 2
    assert {item["proposal_id"] for item in payload["proposals"]} == {
        "change_000",
        "change_001",
    }


# ── 载荷 / payload ─────────────────────────────────────────────────────────


def test_payload_contract(tmp_path: Path, monkeypatch) -> None:
    preprocess = _stub_preprocess(tmp_path / "artifacts", proposals=[_proposal()])

    def _stub(self, sample, context):
        return preprocess

    monkeypatch.setattr(ChangeAgent, "_prepare_perception_and_publish", _stub)
    client = _RecordingClient()
    asyncio.run(_agent(client).run(_sample(tmp_path), _context(tmp_path)))
    payload = _last_user_payload(client)
    assert payload["task"] == "change_caption"
    assert payload["temporal_roles"] == ["t1", "t2"]
    assert payload["coordinate_frame"] == "normalized_0_999_top_left"
    assert payload["harmonization"]["status"] == "applied"
    assert payload["harmonization"]["reason_codes"] == ["PIF_MATCHED"]
    assert payload["harmonization"]["used_for_proposal"] is True
    assert payload["proposals"][0]["proposal_id"] == "change_000"
    assert payload["proposals"][0]["score"] == 0.8
    assert set(payload["registration"]) >= {
        "registration_status",
        "registration_model",
        "quality",
    }
    assert "effective_weights" in payload["perception"]
    assert "semantic_support_note" in payload
    serialized = json.dumps(payload, ensure_ascii=False)
    assert str(tmp_path) not in serialized
    assert payload["perception"]["training_capture"] == {
        "enabled": False,
        "save_dense_features": False,
        "dense_features_saved": False,
    }


def test_vlm_payload_and_trace_are_compact_json_safe(tmp_path: Path, monkeypatch) -> None:
    proposal = _proposal().model_copy(
        update={
            "semantic_transition": SemanticTransition(
                from_class="vegetation",
                from_confidence=0.91,
                to_class="building",
                to_confidence=0.88,
                changed_class="building",
                support_ratio=0.94,
                transition_confidence=0.86,
            ),
            "component_scores": {"low_level": 0.4, "feature": 0.8},
            "reliability": {"registration": 0.9, "semantic": 0.8},
        }
    )
    preprocess = _stub_preprocess(tmp_path / "artifacts", proposals=[proposal])
    monkeypatch.setattr(
        ChangeAgent,
        "_prepare_perception_and_publish",
        lambda self, sample, context: preprocess,
    )
    client = _RecordingClient()
    execution = asyncio.run(_agent(client).run(_sample(tmp_path), _context(tmp_path)))

    payload = _last_user_payload(client)
    assert payload["proposals"][0]["semantic_support"]["from_class"] == "vegetation"
    assert payload["proposals"][0]["semantic_support"]["to_class"] == "building"
    trace_json = json.dumps(execution.trace, ensure_ascii=False)
    assert "base64" not in trace_json.casefold()
    assert str(tmp_path) not in trace_json
    assert "numpy" not in trace_json.casefold()
    assert "empty_proposals_instruction" in payload
    # Ground truth never leaks. / ground truth 绝不泄漏。
    assert "ground_truth" not in payload
    assert "answers" not in payload


# ── 复核与失败语义 / review and failure semantics ─────────────────────────


def test_review_warnings_downgrade_status_to_partial(tmp_path: Path, monkeypatch) -> None:
    preprocess = _stub_preprocess(
        tmp_path / "artifacts", proposals=[_proposal("change_000", 0.8), _proposal("change_001", 0.7)]
    )

    def _stub(self, sample, context):
        return preprocess

    monkeypatch.setattr(ChangeAgent, "_prepare_perception_and_publish", _stub)
    client = _RecordingClient(answer="No visible change.")
    execution = asyncio.run(_agent(client).run(_sample(tmp_path), _context(tmp_path)))
    assert execution.payload.status == "partial"
    assert "CHANGE_RESULT_CONFLICT" in execution.trace["review_warnings"]
    assert execution.trace["adjudication_used"] is False
    assert execution.trace["review_used"] is True


def test_clean_semantic_answer_stays_completed(tmp_path: Path, monkeypatch) -> None:
    preprocess = _stub_preprocess(tmp_path / "artifacts", proposals=[_proposal()])

    def _stub(self, sample, context):
        return preprocess

    monkeypatch.setattr(ChangeAgent, "_prepare_perception_and_publish", _stub)
    execution = asyncio.run(_agent(_RecordingClient()).run(_sample(tmp_path), _context(tmp_path)))
    assert execution.payload.status == "completed"
    assert execution.trace["review_warnings"] == []


def _adjudication_result(
    global_verdict: str, candidate_verdicts: list[str], *, answer: str = "A building was added."
) -> ChangeAdjudicationResult:
    return ChangeAdjudicationResult.model_validate({
        "agent_name": "change_agent",
        "global_review": {
            "verdict": global_verdict, "t1_state": "before", "t2_state": "after",
            "reason": "reviewed raw pair",
            "change_category": "building_structure" if global_verdict == "persistent_change" else None,
        },
        "candidate_reviews": [
            {
                "proposal_id": f"change_{index:03d}", "verdict": verdict,
                "t1_state": "before", "t2_state": "after", "reason": "reviewed crop",
                "change_category": "building_structure" if verdict == "persistent_change" else None,
            }
            for index, verdict in enumerate(candidate_verdicts)
        ],
        "answer": answer,
        "boxes": [[1, 2, 3, 4]], "evidence": ["raw_full_t1"],
        "status": "completed",
    })


def test_global_negative_overrides_local_insufficient_merge(tmp_path: Path) -> None:
    merged, outcome, provenance = _agent()._merge_adjudication(
        _adjudication_result("no_persistent_change", ["insufficient_visual_evidence"]),
        "change_caption", [],
    )
    assert outcome == "negative"
    assert merged.answer == "No significant semantic change detected."
    assert merged.status == "completed"
    assert merged.boxes == merged.evidence == merged.evidence_items == []
    assert provenance["final_rule"] == "GLOBAL_NEGATIVE_OVERRIDES_LOCAL_INSUFFICIENT"


def test_global_negative_with_mixed_nonpersistent_reviews_is_canonical(tmp_path: Path) -> None:
    merged, outcome, _ = _agent()._merge_adjudication(
        _adjudication_result("no_persistent_change", ["appearance_only", "transient", "registration_artifact", "insufficient_visual_evidence"]),
        "change_caption", [],
    )
    assert outcome == "negative"
    assert merged.answer == "No significant semantic change detected."
    assert merged.status == "completed"


def test_valid_candidate_positive_wins_over_global_negative(tmp_path: Path) -> None:
    merged, outcome, provenance = _agent()._merge_adjudication(
        _adjudication_result("no_persistent_change", ["persistent_change"]), "change_caption", []
    )
    assert outcome == "positive"
    assert merged.answer == "A building was added."
    assert provenance["final_rule"] == "VALID_PERSISTENT_POSITIVE"


def test_invalid_global_negative_with_insufficient_candidate_remains_partial(tmp_path: Path) -> None:
    merged, outcome, _ = _agent()._merge_adjudication(
        _adjudication_result("no_persistent_change", ["insufficient_visual_evidence"]),
        "change_caption", ["ADJUDICATION_INVALID_AGENT"],
    )
    assert outcome == "unresolved"
    assert merged.status == "partial"


def test_wrong_agent_name_fails_not_masked(tmp_path: Path, monkeypatch) -> None:
    preprocess = _stub_preprocess(tmp_path / "artifacts")

    def _stub(self, sample, context):
        return preprocess

    monkeypatch.setattr(ChangeAgent, "_prepare_perception_and_publish", _stub)

    class _WrongNameClient(_RecordingClient):
        async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
            return response_model.model_validate(
                {"agent_name": "caption_agent", "answer": "x", "status": "completed"}
            )

    with pytest.raises(AgentExecutionError, match="agent_name"):
        asyncio.run(_agent(_WrongNameClient()).run(_sample(tmp_path), _context(tmp_path)))


def test_missing_raw_image_fails_with_stable_error(tmp_path: Path, monkeypatch) -> None:
    preprocess = _stub_preprocess(tmp_path / "artifacts")

    def _stub(self, sample, context):
        return preprocess

    monkeypatch.setattr(ChangeAgent, "_prepare_perception_and_publish", _stub)
    root = tmp_path / "data"
    sample = _sample(root)
    (root / "t2.png").unlink()
    with pytest.raises(AgentExecutionError, match="image file does not exist"):
        asyncio.run(_agent().run(sample, _context(tmp_path)))


def test_model_call_error_propagates_unmasked(tmp_path: Path, monkeypatch) -> None:
    preprocess = _stub_preprocess(tmp_path / "artifacts")

    def _stub(self, sample, context):
        return preprocess

    monkeypatch.setattr(ChangeAgent, "_prepare_perception_and_publish", _stub)

    class _BoomClient(_RecordingClient):
        async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
            raise RuntimeError("model exploded")

    with pytest.raises(RuntimeError, match="model exploded"):
        asyncio.run(_agent(_BoomClient()).run(_sample(tmp_path), _context(tmp_path)))


def test_request_hash_changes_when_proposal_payload_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _stub_preprocess(tmp_path / "first" / "artifacts", proposals=[_proposal(score=0.8)])
    second = _stub_preprocess(tmp_path / "second" / "artifacts", proposals=[_proposal(score=0.7)])
    monkeypatch.setattr(
        ChangeAgent,
        "_prepare_perception_and_publish",
        lambda self, sample, context: first,
    )
    first_client = _RecordingClient()
    first_root = tmp_path / "first"
    asyncio.run(_agent(first_client).run(_sample(first_root), _context(first_root)))
    monkeypatch.setattr(
        ChangeAgent,
        "_prepare_perception_and_publish",
        lambda self, sample, context: second,
    )
    second_client = _RecordingClient()
    second_root = tmp_path / "second"
    asyncio.run(_agent(second_client).run(_sample(second_root), _context(second_root)))
    assert first_client.calls[0]["request_hash"] != second_client.calls[0]["request_hash"]


def test_request_hash_changes_when_registration_payload_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _stub_preprocess(tmp_path / "first" / "artifacts")
    second_base = _stub_preprocess(tmp_path / "second" / "artifacts")
    second = second_base.model_copy(
        update={
            "registration": RegistrationReport(
                decision=RegistrationDecision(
                    version="global_registration_v1",
                    status="rejected",
                    model="none",
                    reason_codes=["REGISTRATION_LOW_INLIER_RATIO"],
                    used_for_comparison=False,
                ),
                metrics=RegistrationMetrics(
                    match_count=4,
                    inlier_count=1,
                    inlier_ratio=0.25,
                    median_reprojection_error=8.0,
                    p95_reprojection_error=12.0,
                    overlap_ratio=0.8,
                ),
                source_size_t1=[64, 64],
                source_size_t2=[64, 64],
                output_size=[64, 64],
            )
        }
    )
    monkeypatch.setattr(
        ChangeAgent,
        "_prepare_perception_and_publish",
        lambda self, sample, context: first,
    )
    first_client = _RecordingClient()
    first_root = tmp_path / "first"
    asyncio.run(_agent(first_client).run(_sample(first_root), _context(first_root)))
    monkeypatch.setattr(
        ChangeAgent,
        "_prepare_perception_and_publish",
        lambda self, sample, context: second,
    )
    second_client = _RecordingClient()
    second_root = tmp_path / "second"
    asyncio.run(_agent(second_client).run(_sample(second_root), _context(second_root)))
    assert first_client.calls[0]["request_hash"] != second_client.calls[0]["request_hash"]


# ── trace 与边界 / trace and boundaries ────────────────────────────────────


def test_trace_contains_pif_mad_sharpness_review_artifacts(tmp_path: Path) -> None:
    execution = asyncio.run(_agent().run(_sample(tmp_path), _context(tmp_path)))
    trace = execution.trace
    assert set(trace) >= {
        "prompt_version",
        "request_hash",
        "image_sha256",
        "model",
        "image_roles",
        "input_mode",
        "harmonization_version",
        "harmonization_status",
        "pif_ratio",
        "mad_pif_before",
        "mad_pif_after",
        "raw_fallback_used",
        "sharpness_adjustment_used",
        "proposal_count",
        "proposal_source",
        "review_used",
        "review_warnings",
        "preprocess_artifacts",
    }
    assert trace["image_roles"] == ["t1", "t2"]
    assert trace["prompt_version"] == "v2"
    assert trace["preprocess_artifacts"]["validation_report"].startswith("change_preprocess/")


def test_agent_has_no_judge_evaluation_or_dataset_branch() -> None:
    source = (Path(__file__).resolve().parents[3] / "agents" / "change" / "agent.py").read_text(
        encoding="utf-8"
    )
    assert "judge" not in source.casefold()
    assert "vrsbench" not in source.casefold()
    assert "spacers_agent" not in source
    assert "ground_truth" not in source
    assert "apply_" not in source
    assert "torch" not in source.casefold()
    assert "post_training" not in source.casefold()


def test_import_agent_does_not_load_legacy_packages() -> None:
    import agents.change  # noqa: F401

    import sys

    for legacy in ("spacers_agent", "eval"):
        assert legacy not in sys.modules


# ── 无效时相图对 / invalid temporal pairs (33.5) ───────────────────────────


def _pair_sample(root: Path, images: list[tuple[str, str]]) -> UnifiedSample:
    """Build a change sample from (filename, role) pairs.
    从 (filename, role) 列表构建变化样本。"""
    root.mkdir(parents=True, exist_ok=True)
    refs = []
    for index, (filename, role) in enumerate(images):
        Image.new("RGB", (64, 64), (10, 20, 30)).save(root / filename)
        refs.append(ImageRef(image_id=f"i{index}", path=filename, role=role))  # type: ignore[arg-type]
    return UnifiedSample(
        sample_id="s1",
        dataset="parity",
        split="test",
        task="change_caption",
        images=refs,
        question="Describe the change.",
        ground_truth=GroundTruth(answers=["x"]),
    )


def test_invalid_pair_three_images_fails_before_any_model_call(tmp_path: Path) -> None:
    """A t1/t2/context triplet passes the schema but is an invalid pair for
    the PairValidator; the agent must fail before any model call. 三图
    t1/t2/context 能通过 schema，但对 PairValidator 是无效图对；Agent 必须
    在任何模型调用前失败。（单图/乱序/错误角色已在 data.schema 层被拒绝，
    无法到达 Agent。）"""
    from agents.errors import AgentExecutionError

    client = _RecordingClient()
    budget = _FakeBudget()
    images = [("a.png", "t1"), ("b.png", "t2"), ("c.png", "context")]
    with pytest.raises(AgentExecutionError, match="INVALID_CHANGE_PAIR"):
        asyncio.run(
            _agent(client).run(_pair_sample(tmp_path, images), _context(tmp_path, budget))
        )
    assert budget.qwen_calls == 0
    assert client.calls == []


def test_invalid_pair_size_mismatch_fails(tmp_path: Path) -> None:
    from agents.errors import AgentExecutionError

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
        ground_truth=GroundTruth(answers=["x"]),
    )
    client = _RecordingClient()
    budget = _FakeBudget()
    with pytest.raises(AgentExecutionError, match="INVALID_CHANGE_PAIR"):
        asyncio.run(_agent(client).run(sample, _context(root, budget)))
    assert budget.qwen_calls == 0
    assert client.calls == []


def test_invalid_pair_decode_failure_fails(tmp_path: Path) -> None:
    from agents.errors import AgentExecutionError

    root = tmp_path / "data"
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64)).save(root / "t1.png")
    (root / "t2.png").write_bytes(b"corrupt image bytes")
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
        ground_truth=GroundTruth(answers=["x"]),
    )
    client = _RecordingClient()
    budget = _FakeBudget()
    with pytest.raises(AgentExecutionError, match="INVALID_CHANGE_PAIR"):
        asyncio.run(_agent(client).run(sample, _context(root, budget)))
    assert budget.qwen_calls == 0
    assert client.calls == []
