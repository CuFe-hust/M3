"""Counting HTML report coverage for Qwen and YOLO traces.
覆盖 Qwen 与 YOLO 轨迹的计数 HTML 报告测试。
"""

import json
from pathlib import Path

from PIL import Image

from spacers_agent.counting_report import build_multiagent_counting_report
from spacers_agent.schemas import CountingResult, GlobalPointObservation, ImageRef, UnifiedSample
from spacers_agent.settings import QwenSettings


def _point() -> GlobalPointObservation:
    return GlobalPointObservation(global_id="p1", target="ship", source_tile_id="t0", local_id="l0", local_x_norm=500, local_y_norm=500, local_radius_norm=0, global_x_px=4, global_y_px=4, global_x_norm=500, global_y_norm=500, radius_px=1, confidence=.9, ownership_valid=True, near_core_boundary=False, accepted=True, short_evidence="ship")


def test_counting_report_includes_yolo_trace_and_qwen_only_sample(tmp_path: Path) -> None:
    for index, trace in enumerate((
        {"executed_backend": "yolo26s_dota_obb", "primary_backend": "yolo26s_dota_obb", "yolo": {"attempted": True, "used_for_final": True, "detector_name": "yolo26s_dota_obb", "model_id": "test", "source_dataset": "DOTAv1", "resolved_target_classes": ["ship"]}},
        {"executed_backend": "qwen_point", "primary_backend": "qwen_point", "yolo": {"attempted": False, "used_for_final": False}},
    ), start=1):
        sample_dir = tmp_path / "samples" / str(index)
        sample_dir.mkdir(parents=True)
        image_path = sample_dir / "image.png"
        Image.new("RGB", (8, 8)).save(image_path)
        sample = UnifiedSample(sample_id=str(index), dataset="fixture", split="test", task="counting", images=[ImageRef(image_id="image", path=image_path, role="image")], question="How many ships?")
        result = CountingResult(sample_id=str(index), target="ship", question=sample.question, source_width=8, source_height=8, tile_count=1, succeeded_tiles=["t0"], global_points=[_point()], final_count=1, status="completed")
        (sample_dir / "sample.json").write_text(sample.model_dump_json(), encoding="utf-8")
        (sample_dir / "counting_result.json").write_text(result.model_dump_json(), encoding="utf-8")
        (sample_dir / "agent_trace.json").write_text(json.dumps(trace), encoding="utf-8")
        (sample_dir / "status.json").write_text(json.dumps({"state": "succeeded"}), encoding="utf-8")
    report = build_multiagent_counting_report(tmp_path, qwen=QwenSettings(model="local"))
    assert report is not None and report.is_file()
    html = report.read_text(encoding="utf-8")
    assert "yolo26s_dota_obb" in html
    assert "not called" in html
    csv_text = (tmp_path / "counting.report" / "samples.csv").read_text(encoding="utf-8-sig")
    assert "yolo_attempted" in csv_text and "yolo_detector" in csv_text
