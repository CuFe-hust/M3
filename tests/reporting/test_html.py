"""Contract tests for the offline HTML report renderer: escaping, no CDN /
Base64, deterministic output, and secret/path safety.

离线 HTML 报告渲染器契约测试：转义、无 CDN/Base64、确定性输出与密钥/路径
安全。直接构造持久化运行产物。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from agents.base import AgentExecution
from agents.schema import AgentResult
from data.schema import GroundTruth, ImageRef, UnifiedSample
from evaluation.records import EvaluationRecord, VQADeterministicMetrics
from reporting.builder import build_report
from reporting.html import build_html
from reporting.schema import (
    CountingReportDetail,
    FallbackTransitionView,
    GroundTruthView,
    ModelCallAuditView,
    ModelWeightView,
    ProcessReport,
    Report,
    ReportSample,
    RoutingAttemptView,
    RoutingView,
    StructuredArtifactView,
    TaskCandidateView,
    TaskRoutingView,
    ExecutionStepView,
    TaskSummary,
    VisualAssetView,
    WorkflowSequenceView,
    WorkflowStepView,
)
from workflows.artifact_writer import ArtifactWriter
from workflows.run_store import RunStore
from workflows.schema import SampleRunStatus


def _storage_key(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]


def _build_escaping_run(tmp_path: Path) -> Path:
    store = RunStore(tmp_path / "runs", tmp_path)
    store.create_run(
        config_payload={"k": "v"},
        model_ids={"qwen": "q"},
        prompt_paths=[],
        run_id="html-run",
    )
    run_dir = tmp_path / "runs" / "html-run"
    writer = ArtifactWriter()
    sample = UnifiedSample(
        sample_id="evil-1",
        dataset="parity",
        split="test",
        task="general_vqa",
        images=[ImageRef(image_id="i0", path="img.png", role="image")],
        question='Is there a <script>alert(1)</script> road? & more "quotes"',
        ground_truth=GroundTruth(answers=["yes"]),
    )
    sample_dir = run_dir / "tasks" / "general_vqa" / "samples" / _storage_key("evil-1")
    writer.write_sample(sample_dir, sample)
    status = SampleRunStatus(
        sample_id="evil-1",
        task="general_vqa",
        state="succeeded",
        result_path=Path("agent_result.json"),
        updated_at="2026-01-01T00:00:00Z",
    )
    writer.write_final_status(sample_dir, status)
    writer.write_trace(
        sample_dir,
        {
            "resolved_task": "general_vqa",
            "execution_task": "general_vqa",
            "execution_agent": "general_vqa_agent",
            "fallback_used": False,
            "judge_status": "not_requested",
            "inference_seconds": 0.5,
        },
    )
    writer.write_evaluation(
        sample_dir,
        EvaluationRecord(
            sample_id="evil-1",
            task="general_vqa",
            deterministic_metrics=VQADeterministicMetrics(exact_match=True),
            judge_status="not_requested",
        ),
        filename="vqa_evaluation.json",
    )
    writer.write_execution(
        sample_dir,
        AgentExecution(
            agent_name="general_vqa_agent",
            payload=AgentResult(
                agent_name="general_vqa_agent",
                answer="yes <b>bold</b> & <i>italic</i>",
            ),
            result_filename="agent_result.json",
        ),
    )
    writer.append_prediction(
        run_dir,
        sample_id="evil-1",
        run_task="general_vqa",
        task="general_vqa",
        status=status,
        result_path=f"tasks/general_vqa/samples/{_storage_key('evil-1')}/agent_result.json",
    )
    return run_dir


def test_html_escapes_user_and_model_text(tmp_path: Path) -> None:
    document = build_html(build_report(_build_escaping_run(tmp_path)))
    assert "<script>alert(1)</script>" not in document
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in document
    assert "<b>bold</b>" not in document
    assert "&lt;b&gt;bold&lt;/b&gt;" in document
    assert "&quot;quotes&quot;" in document  # attribute-safe quoting / 属性安全引号


def test_html_is_offline_without_cdn_or_base64(tmp_path: Path) -> None:
    document = build_html(build_report(_build_escaping_run(tmp_path)))
    assert "http://" not in document
    assert "https://" not in document
    assert "cdn" not in document.casefold()
    assert "data:image" not in document
    assert "base64," not in document
    assert "sk-" not in document
    assert str(tmp_path) not in document  # machine paths never rendered


def test_html_deterministic(tmp_path: Path) -> None:
    report = build_report(_build_escaping_run(tmp_path))
    assert build_html(report) == build_html(report)


def test_html_contains_stable_fields_only(tmp_path: Path) -> None:
    document = build_html(build_report(_build_escaping_run(tmp_path)))
    assert "html-run" in document
    assert "evil-1" in document
    assert "general_vqa_agent" in document
    assert "exact_match=True" in document


def test_html_displays_execution_path_and_submodel_outputs() -> None:
    report = Report(
        run_id="audit-run",
        total=1,
        succeeded=1,
        partial=0,
        failed=0,
        skipped=0,
        samples=[
            ReportSample(
                sample_id="audit-1",
                run_task="general_vqa",
                task="general_vqa",
                state="succeeded",
                execution_path=[
                    "workflows.task_resolver.TaskResolver",
                    "routing.router.TaskRouter.route",
                    "agents.general_vqa.agent.GeneralVQAAgent",
                ],
                model_calls=[
                    ModelCallAuditView(
                        request_id="audit-1:qwen",
                        prompt_version="v1",
                        raw_response='{"answer":"red"}',
                        parsed_response='{"answer":"red"}',
                    )
                ],
                structured_artifacts=[
                    StructuredArtifactView(
                        filename="vqa_evidence.json",
                        payload={"detector": "yolo", "boxes": [[1, 2, 3, 4]]},
                    )
                ],
            )
        ],
    )

    document = build_html(report)

    assert "Top-level execution path / 顶层执行路径" in document
    assert "workflows.task_resolver.TaskResolver" in document
    assert "All model/submodel outputs / 全部模型/子模型输出" in document
    assert "vqa_evidence.json" in document
    assert "&quot;answer&quot;" in document
    assert "red" in document


def test_html_displays_concrete_workflow_and_yolo_seg_weights() -> None:
    report = Report(
        run_id="process-run",
        total=2,
        succeeded=2,
        partial=0,
        failed=0,
        skipped=0,
        process_report=ProcessReport(
            sample_process_count=2,
            workflow_sequences=[WorkflowSequenceView(
                task="counting",
                sample_count=2,
                steps=[
                    WorkflowStepView(
                        order=1,
                        phase="routing",
                        component="routing.router.TaskRouter",
                        operation="route",
                    ),
                    WorkflowStepView(
                        order=2,
                        phase="backend",
                        component="agents.counting.CountingAgent",
                        operation="primary",
                        backend_name="isaid_yolo11s",
                        repeat_count=2,
                    ),
                ],
            )],
            model_weights=[
                ModelWeightView(
                    family="yolo",
                    backend_name="isaid_yolo11s",
                    backend_kind="yolo_obb",
                    logical_model_id="YOLO11s:iSAID:epoch111",
                    weights_file="isaid-yolo11s-best.pt",
                    weights_sha256="a" * 64,
                    source_dataset="iSAID",
                    use_count=2,
                    phases=["primary"],
                    statuses=["succeeded"],
                ),
                ModelWeightView(
                    family="segmentation",
                    backend_name="isaid_segformer",
                    backend_kind="semantic_segmentation",
                    logical_model_id="SegFormer-MiT-B2:iSAID:local",
                    weights_sha256="b" * 64,
                    model_revision="rev-2",
                    use_count=1,
                    phases=["zero_review"],
                    statuses=["succeeded"],
                ),
            ],
        ),
    )

    document = build_html(report)

    for text in (
        "具体流程 / Concrete execution workflow",
        "实际使用的 YOLO / Seg 权重",
        "观测到的执行顺序 / Observed execution order",
        "YOLO11s:iSAID:epoch111",
        "isaid-yolo11s-best.pt",
        "SegFormer-MiT-B2:iSAID:local",
        "routing.router.TaskRouter",
        "backend=isaid_yolo11s",
        "repeated ×2",
    ):
        assert text in document
    assert "C:/" not in document


def _semantic_report(*, complete: bool) -> Report:
    semantic = {
        "total": 100,
        "deterministic_exact_correct": 80,
        "eligible_mismatches": 20,
        "judged_mismatches": 20 if complete else 10,
        "semantic_equivalent_mismatches": 8 if complete else 6,
        "semantic_non_equivalent_mismatches": 12 if complete else 4,
        "judge_failures": 0 if complete else 2,
        "unresolved_mismatches": 0 if complete else 10,
        "coverage": 1.0 if complete else 0.5,
        "corrected_correct": 88 if complete else 86,
        "lower_bound_score": 0.88 if complete else 0.86,
        "complete": complete,
        "score": 0.88 if complete else None,
    }
    evaluation = EvaluationRecord(
        sample_id="semantic-1",
        task="general_vqa",
        deterministic_metrics=VQADeterministicMetrics(exact_match=False),
        judge_status="succeeded",
        judge_parsed={"score": 1, "concise_rationale": "same meaning"},
    )
    task = TaskSummary(
        run_task="general_vqa",
        total=100,
        succeeded=100,
        partial=0,
        failed=0,
        skipped=0,
        fallback_count=0,
        fallback_rate=0.0,
        metrics={
            "general_vqa": {
                "metric": "exact_match_accuracy",
                "correct": 80,
                "total": 100,
                "score": 0.8,
            }
        },
        judge_metrics={"vqa_semantic_equivalence": semantic},
    )
    return Report(
        run_id="semantic-run",
        total=1,
        succeeded=1,
        partial=0,
        failed=0,
        skipped=0,
        tasks=[task],
        samples=[
            ReportSample(
                sample_id="semantic-1",
                run_task="general_vqa",
                task="general_vqa",
                state="succeeded",
                judge_status="succeeded",
                evaluation=evaluation,
            )
        ],
    )


def test_html_marks_incomplete_semantic_score_as_lower_bound() -> None:
    document = build_html(_semantic_report(complete=False))
    assert "Exact-match accuracy: 0.800000" in document
    assert "Semantic judge coverage: 50.00%" in document
    assert "Semantic equivalent mismatches: 6" in document
    assert "Judge failures: 2" in document
    assert "Unresolved mismatches: 10" in document
    assert "Complete: false" in document
    assert "Judge-assisted semantic accuracy: incomplete" in document
    assert "Confirmed lower bound: 0.860000" in document
    assert "Judge-assisted semantic accuracy: 0.860000" not in document
    assert "exact_match=False judge_score=1" in document


def test_html_displays_complete_semantic_accuracy() -> None:
    document = build_html(_semantic_report(complete=True))
    assert "Semantic judge coverage: 100.00%" in document
    assert "Complete: true" in document
    assert "Judge-assisted semantic accuracy: 0.880000" in document
    assert "Judge-assisted semantic accuracy: incomplete" not in document
    assert "Confirmed lower bound:" not in document


def _caption_report(metrics: dict) -> Report:
    return Report(
        run_id="caption-run",
        total=1,
        succeeded=1,
        partial=0,
        failed=0,
        skipped=0,
        tasks=[
            TaskSummary(
                run_task="caption",
                total=1,
                succeeded=1,
                partial=0,
                failed=0,
                skipped=0,
                fallback_count=0,
                fallback_rate=0.0,
                metrics={"caption": metrics},
            )
        ],
    )


def test_html_displays_caption_corpus_metrics() -> None:
    document = build_html(
        _caption_report(
            {
                "metric_status": "ok",
                "total": 1,
                "BLEU_1": 0.5,
                "BLEU_2": 0.4,
                "BLEU_3": 0.3,
                "BLEU_4": 0.2,
                "METEOR": 0.42,
                "ROUGE_L": 0.43,
                "CIDEr": 0.44,
            }
        )
    )
    for text in (
        "Metrics: caption",
        "metric_status",
        "BLEU_1",
        "BLEU_4",
        "METEOR",
        "ROUGE_L",
        "CIDEr",
    ):
        assert text in document


def test_html_displays_caption_dependency_status() -> None:
    document = build_html(
        _caption_report(
            {
                "metric_status": "dependency_missing",
                "record_count": 1,
                "dependency": "pycocoevalcap",
            }
        )
    )
    assert "dependency_missing" in document
    assert "pycocoevalcap" in document
    assert "record_count" in document


def test_report_v2_dashboard_routing_filters_counting_and_relative_asset() -> None:
    sample = ReportSample(
        sample_id="count-a",
        run_task="counting",
        task="counting",
        state="partial",
        result_quality="incorrect",
        question="How many small vehicles?",
        prediction="12",
        ground_truth=GroundTruthView(count=13),
        inference_seconds=0.48,
        fallback_used=True,
        warnings=["SEMANTIC_TILE_INFERENCE_FAILED"],
        routing=RoutingView(
            resolved_task="counting",
            execution_agent="counting_agent",
            candidate_backends=["detector_obb_csl_001", "segmenter_mitb2_001"],
            attempted_backends=[
                RoutingAttemptView(
                    backend_name="detector_obb_csl_001",
                    backend_kind="yolo_obb",
                    status="unavailable",
                    reason_code="BACKEND_UNAVAILABLE",
                ),
                RoutingAttemptView(
                    backend_name="segmenter_mitb2_001",
                    backend_kind="semantic_segmentation",
                    status="partial",
                ),
            ],
            primary_backend="detector_obb_csl_001",
            primary_backend_kind="yolo_obb",
            final_backend="segmenter_mitb2_001",
            final_backend_kind="semantic_segmentation",
            fallback_used=True,
            fallback_history=[FallbackTransitionView(
                from_backend="detector_obb_csl_001",
                to_backend="segmenter_mitb2_001",
                reason_code="BACKEND_UNAVAILABLE",
            )],
        ),
        task_detail=CountingReportDetail(
            target="small-vehicle",
            predicted_count=12,
            gold_count=13,
            absolute_error=1,
            exact_match=False,
            accepted_point_count=12,
            rejected_point_count=2,
            merged_group_count=1,
            unresolved_conflict_count=1,
        ),
        visuals=[VisualAssetView(
            image_id="i0",
            role="image",
            original_asset="assets/abc-original.webp",
            overlay_asset="assets/abc-overlay.png",
            status="available",
        )],
    )
    report = Report(
        run_id="v2-run", total=1, succeeded=0, partial=1, failed=0, skipped=0,
        samples=[sample],
    )
    document = build_html(report)
    for section in ("Overview", "Tasks", "Expert Routing", "Samples", "Failures", "Runtime"):
        assert section in document
    for attribute in (
        "data-task=", "data-state=", "data-quality=", "data-backend=",
        "data-fallback=", "data-warning=",
    ):
        assert attribute in document
    for text in (
        "Candidate Chain", "detector_obb_csl_001", "segmenter_mitb2_001",
        "BACKEND_UNAVAILABLE", "Target", "Gold", "Prediction", "Absolute Error",
        "Accepted", "Rejected", "Merged", "Unresolved",
    ):
        assert text in document
    assert 'src="assets/abc-overlay.png"' in document
    assert document.count('src="assets/abc-original.webp"') >= 2
    assert document.count('src="assets/abc-overlay.png"') >= 2
    assert 'class="sample result-incorrect"' in document
    assert "✕ 错误 / Incorrect" in document
    assert ".sample summary.sample-preview{display:grid;grid-template-columns:236px minmax(0,1fr)" in document
    assert "@media(max-width:560px)" in document
    for class_name in ("run-meta", "sample-preview", "sample-hero", "table-scroll"):
        assert class_name in document
    for text in ("Question", "Prediction", "Ground Truth", "Final backend", "Latency"):
        assert text in document
    assert document.index('id="samples"') < document.index('id="overview"')
    assert "data:image" not in document


def test_legacy_unknown_quality_uses_persisted_exact_match() -> None:
    sample = ReportSample(
        sample_id="legacy-correct",
        run_task="counting",
        task="counting",
        state="succeeded",
        result_quality="unknown",
        prediction="2",
        ground_truth=GroundTruthView(count=2),
        task_detail=CountingReportDetail(
            predicted_count=2,
            gold_count=2,
            exact_match=True,
        ),
    )
    document = build_html(Report(
        run_id="legacy", total=1, succeeded=1, partial=0, failed=0, skipped=0,
        samples=[sample],
    ))
    assert 'data-quality="correct"' in document
    assert 'class="sample result-correct"' in document
    assert "✓ 正确 / Correct" in document
    assert "? 未知 / Unknown" not in document


def test_v21_model_call_raw_and_parsed_text_are_escaped() -> None:
    report = Report(
        run_id="model-call", total=1, succeeded=1, partial=0, failed=0, skipped=0,
        samples=[ReportSample(
            sample_id="call-1", run_task="general_vqa", task="general_vqa",
            state="succeeded", model_calls=[ModelCallAuditView(
                request_id="qwen-1", prompt_version="v1",
                raw_response="<script>alert(1)</script>",
                parsed_response='{"answer":"<b>yes</b>"}',
            )],
        )],
    )
    document = build_html(report)
    assert "<script>alert(1)</script>" not in document
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in document
    assert "<b>yes</b>" not in document
    assert "&lt;b&gt;yes&lt;/b&gt;" in document


def test_task_routing_and_execution_timeline_are_rendered() -> None:
    sample = ReportSample(
        sample_id="route-flow",
        run_task="auto",
        task="counting",
        state="succeeded",
        task_routing=TaskRoutingView(
            source_task="auto",
            resolved_task="counting",
            executed_task="counting",
            planning_mode="visual-task-plan-v4",
            resolution_source="visual-task-plan-v4",
            candidate_tasks=[TaskCandidateView(
                order=1,
                task="counting",
                agent_names=["counting_agent"],
                status="executed",
                selected=True,
                executed=True,
            )],
            primary_agent="counting_agent",
            fallback_agents=["general_vqa_agent"],
            executed_agent="counting_agent",
            execution_mode="fallback",
            primary_reason="ROUTE_POLICY",
            reason_codes=["TASK_COUNTING"],
        ),
        execution_steps=[
            ExecutionStepView(
                order=1,
                phase="routing",
                component="routing.router.TaskRouter",
                operation="route",
                status="selected",
                task="counting",
                agent_name="counting_agent",
                reason_code="TASK_COUNTING",
            ),
            ExecutionStepView(
                order=2,
                phase="model_call",
                component="models.structured_client",
                operation="complete_json",
                status="succeeded",
                request_id="route-flow:qwen",
                artifact_names=["visual_task_plan.json"],
            ),
        ],
        routing=RoutingView(final_backend="detector"),
    )
    document = build_html(Report(
        run_id="route-flow", total=1, succeeded=1, partial=0, failed=0, skipped=0,
        samples=[sample],
    ))
    for text in (
        "Task Routing / 任务路由", "visual-task-plan-v4", "counting_agent",
        "ROUTE_POLICY", "Task candidates", "Execution Process / 执行过程",
        "routing.router.TaskRouter", "request=route-flow:qwen",
        "artifacts=visual_task_plan.json",
    ):
        assert text in document
    assert "Detailed per-backend outputs were not persisted" not in document
    assert 'class="route-chain"' in document
    assert 'class="stage timeline-step"' in document


def test_report_v2_bundle_contains_offline_outputs_and_assets_directory(
    tmp_path: Path,
) -> None:
    from reporting.exporters import REPORT_SCHEMA_VERSION, persist_report_bundle

    run_dir = _build_escaping_run(tmp_path)
    report_dir = persist_report_bundle(run_dir, build_report(run_dir))
    for name in (
        "report.html", "report.json", "samples.csv", "samples.jsonl", "metadata.json",
    ):
        assert (report_dir / name).is_file()
    assert (report_dir / "assets").is_dir()
    metadata = json.loads((report_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == REPORT_SCHEMA_VERSION == "report-v2"
    text = "\n".join(
        (report_dir / name).read_text(encoding="utf-8-sig")
        for name in ("report.html", "report.json", "samples.csv", "samples.jsonl", "metadata.json")
    )
    assert str(tmp_path) not in text
    assert "dataset_root" not in text
    assert "data:image" not in text
