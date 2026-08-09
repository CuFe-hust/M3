"""Contract tests for semantic connected-component counting."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest
from PIL import Image

from agents.counting.backends.base import CountingRequest
from agents.counting.backends.semantic_segmentation import (
    SemanticSegmentationBackendError,
    SemanticSegmentationCountingBackend,
)
from agents.counting.expert_catalog import ExpertSpec
from agents.counting.schema import CountTargetSpec
from agents.counting.settings import CountingSettings
from data.schema import GroundTruth, ImageRef, UnifiedSample


_DIGEST = "a" * 64
_TARGET = CountTargetSpec(
    canonical_label="small-vehicle",
    inclusion_rule="count each visible vehicle",
    exclusion_rule="exclude ambiguous fragments",
)


class _FakeClient:
    def __init__(self, outputs: list[object]) -> None:
        self.outputs = list(outputs)
        self.calls = 0

    def predict(self, image: Image.Image) -> object:
        self.calls += 1
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        if callable(output):
            return output(image)
        return output


def _expert(
    *,
    enabled: bool = True,
    counting_mode: str = "connected_components",
    min_area: int = 1,
    max_area_ratio: float = 0.9,
    min_confidence: float = 0.5,
) -> ExpertSpec:
    return ExpertSpec.model_validate(
        {
            "backend_name": "segmenter_test_001",
            "kind": "semantic_segmentation",
            "logical_model_id": "segformer-test-local",
            "enabled": enabled,
            "priority": 100,
            "asset": {
                "model_dir": "models/test-segformer",
                "class_map": "models/test-segformer/classes.json",
                "sha256": _DIGEST,
            },
            "verification": {"class_map": "verified"},
            "supports": {
                "small-vehicle": {
                    "model_labels": ["Small_Vehicle"],
                    "counting_mode": counting_mode,
                    **(
                        {
                            "policy": {
                                "min_component_area_px": min_area,
                                "max_component_area_ratio": max_area_ratio,
                                "min_mean_confidence": min_confidence,
                                "morphology": {
                                    "open_kernel": 0,
                                    "close_kernel": 0,
                                },
                            }
                        }
                        if counting_mode == "connected_components"
                        else {}
                    ),
                }
            },
        }
    )


def _settings(*, tiled: bool = False) -> CountingSettings:
    return CountingSettings(
        tile_core_size=4,
        halo_size=2,
        model_max_side=64,
        max_pixels_without_tiling=1 if tiled else 10_000,
        boundary_band_px=8,
        min_confidence=0.2,
    )


def _sample() -> UnifiedSample:
    return UnifiedSample(
        sample_id="semantic-s1",
        dataset="parity",
        split="test",
        task="counting",
        images=[ImageRef(image_id="i1", path="img.png", role="image")],
        question="How many small vehicles?",
        ground_truth=GroundTruth(answers=["2"]),
    )


def _request(tmp_path: Path, *, width: int = 8, height: int = 8) -> CountingRequest:
    return CountingRequest(
        sample=_sample(),
        image=Image.new("RGB", (width, height), "white"),
        target=_TARGET,
        artifact_dir=tmp_path / "artifacts",
    )


def _output(
    width: int,
    height: int,
    blobs: list[tuple[int, int, int, int, float]] | None = None,
) -> SimpleNamespace:
    mask = np.zeros((height, width), dtype=np.int64)
    confidence = np.full((height, width), 0.99, dtype=np.float32)
    for left, top, right, bottom, score in blobs or []:
        mask[top:bottom, left:right] = 1
        confidence[top:bottom, left:right] = score
    return SimpleNamespace(
        width=width,
        height=height,
        mask=mask,
        confidence_map=confidence,
        id_to_label={0: "background", 1: "Small_Vehicle"},
        logical_model_id="segformer-test-local",
        model_revision="test-revision",
        weights_sha256=_DIGEST,
    )


def _run(
    client: _FakeClient,
    tmp_path: Path,
    *,
    expert: ExpertSpec | None = None,
    settings: CountingSettings | None = None,
    width: int = 8,
    height: int = 8,
):
    backend = SemanticSegmentationCountingBackend(
        client,
        expert or _expert(),
        settings or _settings(),
    )
    return asyncio.run(backend.count(_request(tmp_path, width=width, height=height), object()))


def test_two_separate_blobs_produce_two_points(tmp_path: Path) -> None:
    outcome = _run(
        _FakeClient([_output(8, 8, [(1, 1, 3, 3, 0.9), (5, 5, 7, 7, 0.8)])]),
        tmp_path,
    )

    assert outcome.counting.final_count == 2
    assert sum(point.accepted for point in outcome.counting.global_points) == 2
    assert {point.provenance.source for point in outcome.counting.global_points} == {
        "semantic_component_centroid"
    }


def test_touching_blobs_remain_one_semantic_instance_approximation(tmp_path: Path) -> None:
    outcome = _run(
        _FakeClient([_output(8, 8, [(1, 1, 3, 3, 0.9), (3, 1, 5, 3, 0.9)])]),
        tmp_path,
    )

    assert outcome.counting.final_count == 1
    assert outcome.trace["semantic_instance_approximation"] is True
    assert outcome.trace["touching_objects_may_undercount"] is True


def test_low_confidence_component_is_rejected(tmp_path: Path) -> None:
    outcome = _run(
        _FakeClient([_output(8, 8, [(1, 1, 4, 4, 0.2)])]),
        tmp_path,
    )

    assert outcome.counting.final_count == 0
    assert outcome.trace["confidence_rejected_count"] == 1


def test_tiny_component_is_rejected_by_label_policy(tmp_path: Path) -> None:
    outcome = _run(
        _FakeClient([_output(8, 8, [(2, 2, 3, 3, 0.9)])]),
        tmp_path,
        expert=_expert(min_area=2),
    )

    assert outcome.counting.final_count == 0
    assert outcome.trace["area_rejected_count"] == 1


def test_large_connected_region_is_rejected_by_label_policy(tmp_path: Path) -> None:
    outcome = _run(
        _FakeClient([_output(8, 8, [(1, 1, 7, 7, 0.9)])]),
        tmp_path,
        expert=_expert(max_area_ratio=0.25),
    )

    assert outcome.counting.final_count == 0
    assert outcome.trace["area_rejected_count"] == 1


def test_border_component_is_kept_when_centroid_belongs_to_owner_core(
    tmp_path: Path,
) -> None:
    outcome = _run(
        _FakeClient([_output(8, 8, [(0, 2, 2, 5, 0.9)])]),
        tmp_path,
    )

    assert outcome.counting.final_count == 1
    assert outcome.counting.global_points[0].accepted is True


def test_centroid_outside_owner_core_is_rejected(tmp_path: Path) -> None:
    first = lambda image: _output(image.width, image.height, [(4, 1, 6, 3, 0.9)])
    empty = lambda image: _output(image.width, image.height)
    outcome = _run(
        _FakeClient([first, empty]),
        tmp_path,
        settings=_settings(tiled=True),
        width=8,
        height=4,
    )

    assert outcome.counting.final_count == 0
    assert outcome.trace["ownership_rejected_count"] == 1
    assert outcome.counting.global_points[0].rejection_reason == "OUTSIDE_OWNER_CORE"


def test_neighbouring_tile_duplicate_is_merged(tmp_path: Path) -> None:
    first = lambda image: _output(image.width, image.height, [(3, 1, 4, 3, 0.9)])
    second = lambda image: _output(image.width, image.height, [(2, 1, 3, 3, 0.8)])
    outcome = _run(
        _FakeClient([first, second]),
        tmp_path,
        settings=_settings(tiled=True),
        width=8,
        height=4,
    )

    assert outcome.counting.final_count == 1
    assert outcome.trace["merged_duplicate_count"] == 1
    assert len(outcome.counting.merged_groups) == 1


def test_partial_tile_failure_is_visible(tmp_path: Path) -> None:
    empty: Callable[[Image.Image], object] = lambda image: _output(image.width, image.height)
    outcome = _run(
        _FakeClient([RuntimeError("C:/secret/checkpoint"), empty]),
        tmp_path,
        settings=_settings(tiled=True),
        width=8,
        height=4,
    )

    assert outcome.counting.status == "partial"
    assert len(outcome.counting.failed_tiles) == 1
    assert "secret" not in outcome.counting.warnings[0].message


def test_all_tile_failures_raise_stable_backend_error(tmp_path: Path) -> None:
    client = _FakeClient([RuntimeError("first"), OSError("second")])

    with pytest.raises(
        SemanticSegmentationBackendError,
        match="ALL_SEMANTIC_SEGMENTATION_TILES_FAILED",
    ):
        _run(
            client,
            tmp_path,
            settings=_settings(tiled=True),
            width=8,
            height=4,
        )


def test_unsupported_target_is_not_supported() -> None:
    backend = SemanticSegmentationCountingBackend(_FakeClient([]), _expert(), _settings())
    target = _TARGET.model_copy(update={"canonical_label": "ship"})

    assert backend.supports(target) is False


def test_disabled_expert_is_not_supported() -> None:
    backend = SemanticSegmentationCountingBackend(
        _FakeClient([]), _expert(enabled=False), _settings()
    )

    assert backend.supports(_TARGET) is False


def test_wrong_counting_mode_does_not_execute(tmp_path: Path) -> None:
    client = _FakeClient([_output(8, 8)])
    backend = SemanticSegmentationCountingBackend(
        client,
        _expert(counting_mode="unsupported"),
        _settings(),
    )

    assert backend.supports(_TARGET) is False
    with pytest.raises(ValueError, match="unsupported target"):
        asyncio.run(backend.count(_request(tmp_path), object()))
    assert client.calls == 0


def test_final_count_always_equals_accepted_points(tmp_path: Path) -> None:
    outcome = _run(
        _FakeClient([_output(8, 8, [(1, 1, 3, 3, 0.9), (5, 5, 7, 7, 0.9)])]),
        tmp_path,
    )

    assert outcome.counting.final_count == sum(
        point.accepted for point in outcome.counting.global_points
    )


def test_trace_contains_no_dense_arrays_or_absolute_paths(tmp_path: Path) -> None:
    outcome = _run(
        _FakeClient([_output(8, 8, [(1, 1, 3, 3, 0.9)])]),
        tmp_path,
    )
    serialized = json.dumps(outcome.trace)

    assert "mask" not in serialized
    assert "confidence_map" not in serialized
    assert "C:\\" not in serialized
    assert outcome.trace["counting_mode"] == "connected_components"
