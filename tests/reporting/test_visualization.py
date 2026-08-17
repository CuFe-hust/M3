"""Contract tests for the counting overlay renderer.

计数标注图渲染器契约测试：输出保存、尺寸一致性、拒绝点标记、确定性输出与
尺寸不匹配稳定失败。绝不导入 CountingAgent/YOLO backend。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from agents.counting.schema import (
    CountingBackendAttemptAudit,
    CountingExecutionAudit,
    CountingResult,
    GlobalPointObservation,
    PointProvenance,
)
from agents.schema import AgentResult, VisualEvidence
from data.schema import GroundTruth, ImageRef, UnifiedSample
from reporting.schema import BackendStageView, Report, ReportSample, VisualAssetView
from reporting.visualization import materialize_report_assets, render_counting_overlay
from workflows.artifact_writer import ArtifactWriter
from workflows.schema import RunRequest


def _point(
    gid: str,
    *,
    x: int,
    y: int,
    accepted: bool,
    radius: float = 5.0,
) -> GlobalPointObservation:
    return GlobalPointObservation(
        global_id=gid,
        target="car",
        source_tile_id="t0",
        local_id=gid,
        local_x_norm=100,
        local_y_norm=100,
        local_radius_norm=5,
        global_x_px=x,
        global_y_px=y,
        global_x_norm=100,
        global_y_norm=100,
        radius_px=radius,
        confidence=0.9,
        ownership_valid=True,
        near_core_boundary=False,
        accepted=accepted,
        short_evidence="visible",
    )


def _result() -> CountingResult:
    return CountingResult(
        sample_id="s1",
        target="car",
        question="How many cars?",
        source_width=100,
        source_height=100,
        tile_count=1,
        succeeded_tiles=["t0"],
        failed_tiles=[],
        global_points=[
            _point("p0", x=10, y=10, accepted=True),
            _point("p1", x=50, y=50, accepted=True),
            _point("p2", x=90, y=90, accepted=False),
        ],
        merged_groups=[],
        unresolved_conflicts=[],
        final_count=2,
        status="completed",
    )


def test_render_counting_overlay_saves_png(tmp_path: Path) -> None:
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    output = tmp_path / "overlay.png"
    render_counting_overlay(image, result=_result(), output_path=output)
    assert output.is_file()
    rendered = Image.open(output)
    assert rendered.size == (100, 100)
    assert rendered.mode == "RGB"


def test_render_overlay_is_deterministic(tmp_path: Path) -> None:
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    render_counting_overlay(image, result=_result(), output_path=first)
    render_counting_overlay(image, result=_result(), output_path=second)
    assert first.read_bytes() == second.read_bytes()


def test_render_overlay_image_size_mismatch_fails_stably(tmp_path: Path) -> None:
    image = Image.new("RGB", (50, 50), (255, 255, 255))
    with pytest.raises(ValueError, match="does not match"):
        render_counting_overlay(image, result=_result(), output_path=tmp_path / "bad.png")
    assert not (tmp_path / "bad.png").exists()


def test_render_overlay_does_not_mutate_source_image(tmp_path: Path) -> None:
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    before = image.tobytes()
    render_counting_overlay(image, result=_result(), output_path=tmp_path / "overlay.png")
    assert image.tobytes() == before


def test_report_v2_point_rings_use_stable_colors_and_keep_center_clear(
    tmp_path: Path,
) -> None:
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    output = tmp_path / "rings.png"
    render_counting_overlay(image, result=_result(), output_path=output)
    rendered = Image.open(output)
    assert rendered.getpixel((15, 10)) == (34, 197, 94)
    assert rendered.getpixel((95, 90)) == (239, 68, 68)
    assert rendered.getpixel((10, 10)) == (255, 255, 255)
    assert rendered.getpixel((90, 90)) == (255, 255, 255)


def test_report_v2_obb_draws_true_polygon_not_enclosing_rectangle(
    tmp_path: Path,
) -> None:
    point = _point("obb", x=50, y=50, accepted=True, radius=3).model_copy(
        update={
            "provenance": PointProvenance(
                source="yolo_obb_center",
                obb_polygon_global_px=[[50, 20], [80, 50], [50, 80], [20, 50]],
            )
        }
    )
    result = _result().model_copy(
        update={"global_points": [point], "final_count": 1}
    )
    output = tmp_path / "obb.png"
    render_counting_overlay(
        Image.new("RGB", (100, 100), "white"), result=result, output_path=output
    )
    rendered = Image.open(output)
    assert rendered.getpixel((50, 20)) == (34, 197, 94)
    assert rendered.getpixel((20, 20)) == (255, 255, 255)


def test_v21_obb_does_not_draw_pipeline_radius_ring(tmp_path: Path) -> None:
    point = _point("obb", x=150, y=150, accepted=True, radius=100).model_copy(
        update={"provenance": PointProvenance(
            source="yolo_obb_center",
            obb_polygon_global_px=[[120, 120], [180, 120], [180, 180], [120, 180]],
        )}
    )
    result = _result().model_copy(update={
        "source_width": 300, "source_height": 300,
        "global_points": [point], "final_count": 1,
    })
    output = tmp_path / "obb-no-radius.png"
    render_counting_overlay(
        Image.new("RGB", (300, 300), "white"), result=result, output_path=output,
    )
    rendered = Image.open(output)
    assert rendered.getpixel((120, 150)) == (34, 197, 94)
    assert rendered.getpixel((250, 150)) == (255, 255, 255)


def test_v21_point_only_uses_fixed_small_ring(tmp_path: Path) -> None:
    point = _point("point", x=150, y=150, accepted=True, radius=100)
    result = _result().model_copy(update={
        "source_width": 300, "source_height": 300,
        "global_points": [point], "final_count": 1,
    })
    output = tmp_path / "fixed-ring.png"
    render_counting_overlay(
        Image.new("RGB", (300, 300), "white"), result=result, output_path=output,
    )
    rendered = Image.open(output)
    assert rendered.getpixel((155, 150)) == (34, 197, 94)
    assert rendered.getpixel((160, 150)) == (255, 255, 255)


def test_report_v2_visual_budget_prioritizes_failed_then_incorrect(
    tmp_path: Path,
) -> None:
    samples = [
        ReportSample(
            sample_id=name,
            run_task="general_vqa",
            task="general_vqa",
            state=state,
            result_quality=quality,  # type: ignore[arg-type]
            fallback_used=fallback,
            visuals=[VisualAssetView(image_id="i0", role="image")],
        )
        for name, state, quality, fallback in [
            ("normal-a", "succeeded", "correct", False),
            ("fallback", "succeeded", "correct", True),
            ("wrong", "succeeded", "incorrect", False),
            ("failed", "failed", "unknown", False),
            ("normal-b", "succeeded", "correct", False),
        ]
    ]
    report = Report(
        run_id="budget",
        total=5,
        succeeded=4,
        partial=0,
        failed=1,
        skipped=0,
        samples=samples,
    )
    updated = materialize_report_assets(
        tmp_path / "budget", report, tmp_path / "budget" / "report",
        max_visual_samples=2,
    )
    statuses = {sample.sample_id: sample.visuals[0].status for sample in updated.samples}
    assert statuses["failed"] == "missing_source"
    assert statuses["wrong"] == "missing_source"
    assert statuses["normal-a"] == "omitted_by_budget"
    assert statuses["fallback"] == "omitted_by_budget"


def test_report_v2_dimension_mismatch_keeps_preview_without_forced_overlay(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "private-dataset"
    dataset_root.mkdir()
    Image.new("RGB", (20, 20), "white").save(dataset_root / "source.png")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    request = RunRequest(
        dataset="demo",
        dataset_root=dataset_root.as_posix(),
        split="test",
        task_mode="explicit",
        tasks=["counting"],
    )
    (run_dir / "run_request.json").write_text(request.model_dump_json(), encoding="utf-8")
    sample = UnifiedSample(
        sample_id="dimension",
        dataset="demo",
        split="test",
        task="counting",
        images=[ImageRef(image_id="i0", path="source.png", role="image")],
        question="How many?",
        ground_truth=GroundTruth(count=0),
    )
    key = hashlib.sha256(sample.sample_id.encode()).hexdigest()[:24]
    sample_dir = run_dir / "tasks" / "counting" / "samples" / key
    ArtifactWriter().write_sample(sample_dir, sample)
    result = CountingResult(
        sample_id="dimension",
        target="object",
        question="How many?",
        source_width=30,
        source_height=30,
        tile_count=0,
        final_count=0,
        status="completed",
    )
    (sample_dir / "counting_result.json").write_text(result.model_dump_json(), encoding="utf-8")
    report = Report(
        run_id="run", total=1, succeeded=1, partial=0, failed=0, skipped=0,
        samples=[ReportSample(
            sample_id="dimension", run_task="counting", task="counting",
            state="succeeded", visuals=[VisualAssetView(image_id="i0", role="image")],
        )],
    )
    updated = materialize_report_assets(run_dir, report, run_dir / "report")
    visual = updated.samples[0].visuals[0]
    assert visual.status == "dimension_mismatch"
    assert visual.original_asset is not None
    assert (run_dir / "report" / visual.original_asset).is_file()
    assert visual.overlay_asset is None
    assert dataset_root.as_posix() not in updated.model_dump_json()


def test_v21_persisted_attempt_materializes_hash_safe_stage_overlay(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "stage-images"
    dataset_root.mkdir()
    Image.new("RGB", (100, 100), "white").save(dataset_root / "source.png")
    run_dir = tmp_path / "stage-run"
    run_dir.mkdir()
    (run_dir / "run_request.json").write_text(RunRequest(
        dataset="demo", dataset_root=dataset_root.as_posix(), split="test",
        task_mode="explicit", tasks=["counting"],
    ).model_dump_json(), encoding="utf-8")
    sample = UnifiedSample(
        sample_id="stage", dataset="demo", split="test", task="counting",
        images=[ImageRef(image_id="i0", path="source.png", role="image")],
        question="How many?",
    )
    key = hashlib.sha256(sample.sample_id.encode()).hexdigest()[:24]
    sample_dir = run_dir / "tasks" / "counting" / "samples" / key
    ArtifactWriter().write_sample(sample_dir, sample)
    result = _result().model_copy(update={"sample_id": "stage"})
    (sample_dir / "counting_result.json").write_text(result.model_dump_json(), encoding="utf-8")
    audit = CountingExecutionAudit(
        sample_id="stage", target="car",
        attempts=[CountingBackendAttemptAudit(
            backend_name="qwen_point", backend_kind="qwen_point",
            phase="primary", status="succeeded", counting=result,
        )],
    )
    (sample_dir / "counting_attempts.json").write_text(audit.model_dump_json(), encoding="utf-8")
    report = Report(
        run_id="stage-run", total=1, succeeded=1, partial=0, failed=0, skipped=0,
        samples=[ReportSample(
            sample_id="stage", run_task="counting", task="counting", state="succeeded",
            visuals=[VisualAssetView(image_id="i0", role="image")],
            backend_stages=[BackendStageView(
                order=1, backend_name="qwen_point", backend_kind="qwen_point",
                phase="primary", status="succeeded",
            )],
        )],
    )
    updated = materialize_report_assets(run_dir, report, run_dir / "report")
    asset = updated.samples[0].backend_stages[0].overlay_asset
    assert asset is not None and asset.startswith("assets/") and "stage-01.png" in asset
    assert (run_dir / "report" / asset).is_file()


def test_report_v2_multi_image_evidence_binds_strictly_by_image_id(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "change-images"
    dataset_root.mkdir()
    for name in ("t1.png", "t2.png"):
        Image.new("RGB", (100, 100), "white").save(dataset_root / name)
    run_dir = tmp_path / "change-run"
    run_dir.mkdir()
    request = RunRequest(
        dataset="demo", dataset_root=dataset_root.as_posix(), split="test",
        task_mode="explicit", tasks=["change_qa"],
    )
    (run_dir / "run_request.json").write_text(request.model_dump_json(), encoding="utf-8")
    sample = UnifiedSample(
        sample_id="change",
        dataset="demo",
        split="test",
        task="change_qa",
        images=[
            ImageRef(image_id="t1", path="t1.png", role="t1"),
            ImageRef(image_id="t2", path="t2.png", role="t2"),
        ],
        question="What changed?",
    )
    key = hashlib.sha256(sample.sample_id.encode()).hexdigest()[:24]
    sample_dir = run_dir / "tasks" / "change_qa" / "samples" / key
    ArtifactWriter().write_sample(sample_dir, sample)
    result = AgentResult(
        agent_name="change_agent",
        answer="two local changes",
        evidence_items=[
            VisualEvidence(label="t1 evidence", point=[100, 100], image_id="t1"),
            VisualEvidence(label="t2 evidence", point=[800, 800], image_id="t2"),
        ],
    )
    (sample_dir / "agent_result.json").write_text(result.model_dump_json(), encoding="utf-8")
    report = Report(
        run_id="change-run", total=1, succeeded=1, partial=0, failed=0, skipped=0,
        samples=[ReportSample(
            sample_id="change", run_task="change_qa", task="change_qa", state="succeeded",
            visuals=[
                VisualAssetView(image_id="t1", role="t1"),
                VisualAssetView(image_id="t2", role="t2"),
            ],
        )],
    )
    updated = materialize_report_assets(run_dir, report, run_dir / "report")
    first, second = updated.samples[0].visuals
    assert first.status == second.status == "available"
    first_image = Image.open(run_dir / "report" / str(first.overlay_asset))
    second_image = Image.open(run_dir / "report" / str(second.overlay_asset))
    green = (34, 197, 94)
    assert any(first_image.getpixel((x, y)) == green for x in range(5, 15) for y in range(5, 15))
    assert all(first_image.getpixel((x, y)) != green for x in range(75, 85) for y in range(75, 85))
    assert any(second_image.getpixel((x, y)) == green for x in range(75, 85) for y in range(75, 85))
    assert all(second_image.getpixel((x, y)) != green for x in range(5, 15) for y in range(5, 15))

    unbound = result.model_copy(update={
        "evidence_items": [VisualEvidence(label="unbound", point=[500, 500])]
    })
    (sample_dir / "agent_result.json").write_text(unbound.model_dump_json(), encoding="utf-8")
    unbound_report = materialize_report_assets(run_dir, report, run_dir / "unbound-report")
    assert [item.status for item in unbound_report.samples[0].visuals] == [
        "unsupported_geometry", "unsupported_geometry"
    ]


def test_report_v2_grounding_prediction_and_gt_are_thin_distinct_outlines(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "grounding-images"
    dataset_root.mkdir()
    Image.new("RGB", (100, 100), "white").save(dataset_root / "source.png")
    run_dir = tmp_path / "grounding-run"
    run_dir.mkdir()
    request = RunRequest(
        dataset="demo", dataset_root=dataset_root.as_posix(), split="test",
        task_mode="explicit", tasks=["grounding"],
    )
    (run_dir / "run_request.json").write_text(request.model_dump_json(), encoding="utf-8")
    sample = UnifiedSample(
        sample_id="grounding",
        dataset="demo",
        split="test",
        task="grounding",
        images=[ImageRef(image_id="i0", path="source.png", role="image")],
        question="Locate it",
        ground_truth=GroundTruth(
            boxes=[[600, 600, 800, 800]],
            coordinate_frame="normalized_0_999_top_left",
        ),
    )
    key = hashlib.sha256(sample.sample_id.encode()).hexdigest()[:24]
    sample_dir = run_dir / "tasks" / "grounding" / "samples" / key
    ArtifactWriter().write_sample(sample_dir, sample)
    result = AgentResult(
        agent_name="grounding_agent", answer="located", boxes=[[100, 100, 300, 300]]
    )
    (sample_dir / "agent_result.json").write_text(result.model_dump_json(), encoding="utf-8")
    report = Report(
        run_id="grounding-run", total=1, succeeded=1, partial=0, failed=0, skipped=0,
        samples=[ReportSample(
            sample_id="grounding", run_task="grounding", task="grounding", state="succeeded",
            visuals=[VisualAssetView(image_id="i0", role="image")],
        )],
    )
    updated = materialize_report_assets(run_dir, report, run_dir / "report")
    visual = updated.samples[0].visuals[0]
    overlay = Image.open(run_dir / "report" / str(visual.overlay_asset))
    green, cyan, white = (34, 197, 94), (56, 189, 248), (255, 255, 255)
    assert any(overlay.getpixel((x, y)) == green for x in range(8, 32) for y in range(8, 32))
    assert any(overlay.getpixel((x, y)) == cyan for x in range(58, 82) for y in range(58, 82))
    assert overlay.getpixel((20, 20)) == white
    assert overlay.getpixel((70, 70)) == white


def test_report_v2_grounding_renders_official_normalized_polygon_gt(
    tmp_path: Path,
) -> None:
    """The official VRSBench referring polygon is rendered in its declared
    [0, 1] frame without converting it to xyxy in the sample contract.
    官方 VRSBench referring polygon 按声明的 [0, 1] 坐标系渲染，不在样本
    契约中静默转换为 xyxy。"""

    dataset_root = tmp_path / "official-grounding-images"
    dataset_root.mkdir()
    Image.new("RGB", (100, 100), "white").save(dataset_root / "source.png")
    run_dir = tmp_path / "official-grounding-run"
    run_dir.mkdir()
    request = RunRequest(
        dataset="VRSBench", dataset_root=dataset_root.as_posix(), split="val",
        task_mode="explicit", tasks=["grounding"],
    )
    (run_dir / "run_request.json").write_text(request.model_dump_json(), encoding="utf-8")
    sample = UnifiedSample(
        sample_id="official-grounding",
        dataset="VRSBench",
        split="validation",
        task="grounding",
        images=[ImageRef(image_id="i0", path="source.png", role="image")],
        question="The object.",
        ground_truth=GroundTruth(
            boxes=[[0.60, 0.60, 0.80, 0.60, 0.80, 0.80, 0.60, 0.80]],
            coordinate_frame="normalized_0_1_top_left",
        ),
    )
    key = hashlib.sha256(sample.sample_id.encode()).hexdigest()[:24]
    sample_dir = run_dir / "tasks" / "grounding" / "samples" / key
    ArtifactWriter().write_sample(sample_dir, sample)
    result = AgentResult(
        agent_name="grounding_agent", answer="located", boxes=[[100, 100, 300, 300]]
    )
    (sample_dir / "agent_result.json").write_text(result.model_dump_json(), encoding="utf-8")
    report = Report(
        run_id="official-grounding-run", total=1, succeeded=1, partial=0, failed=0, skipped=0,
        samples=[ReportSample(
            sample_id="official-grounding", run_task="grounding", task="grounding", state="succeeded",
            visuals=[VisualAssetView(image_id="i0", role="image")],
        )],
    )

    updated = materialize_report_assets(run_dir, report, run_dir / "report")
    visual = updated.samples[0].visuals[0]
    overlay = Image.open(run_dir / "report" / str(visual.overlay_asset))

    assert visual.status == "available"
    assert any(
        overlay.getpixel((x, y)) == (56, 189, 248)
        for x in range(58, 83)
        for y in range(58, 83)
    )
    assert any(
        overlay.getpixel((x, y)) == (34, 197, 94)
        for x in range(8, 33)
        for y in range(8, 33)
    )
