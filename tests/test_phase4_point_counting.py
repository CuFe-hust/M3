from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from spacers_agent.clients.base import RequestMeta
from spacers_agent.counting import PointCountingOrchestrator, finalize_representatives, find_boundary_conflicts
from spacers_agent.imaging import build_core_halo_tiles, should_tile_image, split_tile_owner_core
from spacers_agent.schemas import CountTargetSpec, GlobalPointObservation, TileCountResponse
from spacers_agent.settings import CountingSettings, QwenSettings


class RecordingTileClient:
    """Return deterministic tile responses without opening a network connection.
    在不建立网络连接的情况下返回确定性 tile 响应。
    """

    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self.responses = responses
        self.calls: list[RequestMeta] = []

    async def complete_json(
        self,
        *,
        messages: list[dict[str, Any]],
        response_model: type[TileCountResponse],
        request_meta: RequestMeta,
    ) -> TileCountResponse:
        assert len(messages) == 2
        assert len(messages[1]["content"]) == 2
        self.calls.append(request_meta)
        return response_model.model_validate(self.responses[request_meta.tile_id or ""])


def _target() -> CountTargetSpec:
    return CountTargetSpec(
        canonical_label="building",
        aliases=["house"],
        inclusion_rule="Count each distinct building.",
        exclusion_rule="Do not count shadows.",
    )


def _orchestrator(client: RecordingTileClient, run_dir: Path, **counting: Any) -> PointCountingOrchestrator:
    values: dict[str, Any] = {
        "tile_core_size": 32,
        "halo_size": 4,
        "model_max_side": 64,
        "boundary_band_px": 3,
        "min_core_size": 8,
        "max_recursive_depth": 1,
        "max_points_per_tile": 50,
    }
    values.update(counting)
    return PointCountingOrchestrator(
        client,
        counting=CountingSettings(**values),
        qwen=QwenSettings(model="mock-qwen"),
        system_prompt="count points",
        run_dir=run_dir,
    )


@pytest.mark.asyncio
async def test_sequential_counting_uses_points_and_resume_avoids_second_call(tmp_path: Path) -> None:
    response = {
        "r000_c000": {
            "target": "building",
            "tile_id": "r000_c000",
            "points": [
                {
                    "local_id": "p001",
                    "x": 500,
                    "y": 500,
                    "confidence": 0.9,
                    "radius": 10,
                    "short_evidence": "roof centre",
                }
            ],
            "reported_count": 1,
        }
    }
    image = Image.new("RGB", (20, 12), "white")
    first_client = RecordingTileClient(response)
    first = await _orchestrator(first_client, tmp_path).count_image(
        image, sample_id="sample", question="count buildings", target=_target()
    )

    assert first.final_count == 1
    assert first.status == "completed"
    assert len(first_client.calls) == 1
    assert (tmp_path / "tiles" / "r000_c000" / "spec.json").is_file()
    assert (tmp_path / "tiles" / "r000_c000" / "conversion_validation.json").is_file()

    resumed_client = RecordingTileClient({})
    resumed = await _orchestrator(resumed_client, tmp_path).count_image(
        image, sample_id="sample", question="count buildings", target=_target()
    )
    assert resumed.final_count == 1
    assert resumed_client.calls == []


@pytest.mark.asyncio
async def test_tile_mismatch_is_visible_as_partial_failure(tmp_path: Path) -> None:
    client = RecordingTileClient(
        {
            "r000_c000": {
                "target": "building",
                "tile_id": "wrong-tile",
                "points": [],
                "reported_count": 0,
            }
        }
    )

    result = await _orchestrator(client, tmp_path).count_image(
        Image.new("RGB", (10, 10)), sample_id="sample", question="count buildings", target=_target()
    )

    assert result.status == "failed"
    assert result.failed_tiles == ["r000_c000"]
    checkpoint = (tmp_path / "tiles" / "r000_c000" / "checkpoint.json").read_text(encoding="utf-8")
    assert '"status": "failed"' in checkpoint


@pytest.mark.asyncio
async def test_needs_split_replaces_parent_with_children_and_resumes_children(tmp_path: Path) -> None:
    parent = {
        "target": "building",
        "tile_id": "r000_c000",
        "points": [],
        "reported_count": 0,
        "needs_split": True,
    }
    responses = {"r000_c000": parent}
    for index in range(4):
        tile_id = f"r000_c000_d1_{index}"
        responses[tile_id] = {"target": "building", "tile_id": tile_id, "points": [], "reported_count": 0}
    first_client = RecordingTileClient(responses)
    image = Image.new("RGB", (16, 16))
    first = await _orchestrator(first_client, tmp_path, tile_core_size=16).count_image(
        image, sample_id="sample", question="count buildings", target=_target()
    )

    assert first.succeeded_tiles == [f"r000_c000_d1_{index}" for index in range(4)]
    assert len(first_client.calls) == 5
    assert '"status": "superseded_by_children"' in (
        tmp_path / "tiles" / "r000_c000" / "checkpoint.json"
    ).read_text(encoding="utf-8")

    resumed_client = RecordingTileClient({})
    resumed = await _orchestrator(resumed_client, tmp_path, tile_core_size=16).count_image(
        image, sample_id="sample", question="count buildings", target=_target()
    )
    assert resumed.succeeded_tiles == first.succeeded_tiles
    assert resumed_client.calls == []


def _global_point(point_id: str, tile_id: str, x: int, y: int, confidence: float = 0.8) -> GlobalPointObservation:
    return GlobalPointObservation(
        global_id=point_id,
        target="building",
        source_tile_id=tile_id,
        local_id=point_id.rsplit(":", 1)[-1],
        local_x_norm=0,
        local_y_norm=0,
        local_radius_norm=0,
        global_x_px=x,
        global_y_px=y,
        global_x_norm=0,
        global_y_norm=0,
        radius_px=0,
        confidence=confidence,
        ownership_valid=True,
        near_core_boundary=True,
        accepted=True,
        short_evidence="roof",
    )


def test_boundary_conflicts_are_local_and_explicit_merges_control_final_count() -> None:
    tiles = build_core_halo_tiles(20, 10, core_size=10, halo_size=2, model_max_side=64)
    left = _global_point("sample:r000_c000:p1", "r000_c000", 9, 5, 0.9)
    right = _global_point("sample:r000_c001:p1", "r000_c001", 10, 5, 0.7)
    conflicts = find_boundary_conflicts([left, right], tiles)

    assert len(conflicts) == 1
    final, groups = finalize_representatives([left, right], [(left.global_id, right.global_id)])
    assert sum(point.accepted for point in final) == 1
    assert groups == [[left.global_id, right.global_id]]


def test_tiling_decision_and_recursive_geometry_preserve_core_coverage() -> None:
    assert not should_tile_image(100, 100, model_max_side=1280, max_pixels_without_tiling=1_600_000)
    assert should_tile_image(1281, 10, model_max_side=1280, max_pixels_without_tiling=1_600_000)
    parent = build_core_halo_tiles(16, 16, core_size=16, halo_size=2, model_max_side=64)[0]
    children = split_tile_owner_core(parent, halo_size=2, model_max_side=64)

    assert len(children) == 4
    for x in range(16):
        for y in range(16):
            assert sum(child.owner_core_global.left <= x < child.owner_core_global.right and child.owner_core_global.top <= y < child.owner_core_global.bottom for child in children) == 1
