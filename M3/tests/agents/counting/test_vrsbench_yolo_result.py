from spacers_agent.agents.counting.agent import CountingAgent
from spacers_agent.schemas import CountingResult, GlobalPointObservation, PointProvenance


def test_vrsbench_yolo_count_adapts_to_reportable_agent_result() -> None:
    point = GlobalPointObservation(
        global_id="s:t:p", target="small-vehicle", source_tile_id="t", local_id="p",
        local_x_norm=500, local_y_norm=600, local_radius_norm=10,
        global_x_px=50, global_y_px=60, global_x_norm=500, global_y_norm=600,
        radius_px=5.0, confidence=0.9, ownership_valid=True, near_core_boundary=False,
        accepted=True, short_evidence="YOLO OBB small vehicle",
        provenance=PointProvenance(
            source="yolo_obb_center", backend_name="yolo", model_id="model",
            source_class="small vehicle", detector_confidence=0.9,
            obb_polygon_local_px=[[1, 1], [2, 1], [2, 2], [1, 2]],
            obb_polygon_global_px=[[1, 1], [2, 1], [2, 2], [1, 2]],
            detector_task="obb", detector_source_dataset="DOTAv1", weights_sha256="0" * 64,
        ),
    )
    counting = CountingResult(
        sample_id="s", target="small-vehicle", question="How many?", source_width=100,
        source_height=100, tile_count=1, initial_tile_count=1, leaf_tile_count=1,
        succeeded_tiles=["t"], failed_tiles=[], global_points=[point], merged_groups=[],
        unresolved_conflicts=[], warnings=[], final_count=1, status="completed",
    )
    result = CountingAgent._vrsbench_agent_result(counting, "image.png")
    assert result.answer == "1"
    assert result.agent_name == "counting_agent"
    assert result.status == "completed"
    assert result.evidence_items[0].point == [500, 600]
    assert result.evidence_items[0].label == "small vehicle"
