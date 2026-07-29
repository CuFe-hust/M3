from __future__ import annotations

from pathlib import Path

from PIL import Image

from spacers_agent.cli import main
from spacers_agent.evaluation import DeepSeekJudgeResult, EvaluationRecord, count_deterministic_metrics
from spacers_agent.reporting import summarize_evaluations
from spacers_agent.schemas import CountingResult, GlobalPointObservation
from spacers_agent.imaging import build_core_halo_tiles
from spacers_agent.visualization import render_counting_overlay


def _result() -> CountingResult:
    accepted = GlobalPointObservation(
        global_id="sample:r000_c000:p1",
        target="building",
        source_tile_id="r000_c000",
        local_id="p1",
        local_x_norm=500,
        local_y_norm=500,
        local_radius_norm=0,
        global_x_px=8,
        global_y_px=8,
        global_x_norm=500,
        global_y_norm=500,
        radius_px=0,
        confidence=0.9,
        ownership_valid=True,
        near_core_boundary=False,
        accepted=True,
        short_evidence="roof",
    )
    rejected = accepted.model_copy(
        update={"global_id": "sample:r000_c000:p2", "local_id": "p2", "global_x_px": 2, "accepted": False, "rejection_reason": "POINT_OUTSIDE_CORE"}
    )
    return CountingResult(
        sample_id="sample",
        target="building",
        question="count buildings",
        source_width=20,
        source_height=20,
        tile_count=1,
        succeeded_tiles=["r000_c000"],
        global_points=[accepted, rejected],
        final_count=1,
        status="completed",
    )


def test_render_counting_overlay_and_local_cli(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    result_path = tmp_path / "result.json"
    direct_output = tmp_path / "direct.png"
    cli_output = tmp_path / "cli.png"
    image = Image.new("RGB", (20, 20), "white")
    image.save(image_path)
    result = _result()
    result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    tiles = build_core_halo_tiles(20, 20, core_size=896, halo_size=128, model_max_side=1280)

    render_counting_overlay(image, result=result, tiles=tiles, output_path=direct_output)
    assert direct_output.is_file()
    assert Image.open(direct_output).getpixel((5, 8)) != (255, 255, 255)

    assert main(["render-count", "--image", str(image_path), "--result", str(result_path), "--output", str(cli_output)]) == 0
    assert cli_output.is_file()


def test_evaluation_summary_and_offline_cli_keep_metric_families_separate(tmp_path: Path) -> None:
    judge = DeepSeekJudgeResult(
        judge_scope="text_and_structured_evidence_only",
        can_verify_visual_truth=False,
        semantic_correctness=0.5,
        answer_evidence_consistency=0.7,
        constraint_following=0.8,
        clarity=0.9,
        verdict="mostly_correct",
        concise_rationale="Text is internally consistent.",
    )
    records = [
        EvaluationRecord(sample_id="a", task="counting", deterministic_metrics=count_deterministic_metrics(1, 1), judge_status="succeeded", judge_parsed=judge),
        EvaluationRecord(sample_id="b", task="counting", deterministic_metrics=count_deterministic_metrics(3, 1), judge_status="failed", judge_error="timeout"),
    ]
    summary = summarize_evaluations(records)
    input_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "summary.json"
    input_path.write_text("\n".join(record.model_dump_json() for record in records) + "\n", encoding="utf-8")

    assert summary.exact_match_rate == 0.5
    assert summary.mean_absolute_error == 1.0
    assert summary.judge_succeeded == 1
    assert summary.judge_failed == 1
    assert main(["summarize-evaluations", "--input", str(input_path), "--output", str(output_path)]) == 0
    assert '"mean_semantic_correctness": 0.5' in output_path.read_text(encoding="utf-8")
