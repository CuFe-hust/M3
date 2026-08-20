"""Qwen point-counting backend wrapping the generic point pipeline."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable
from typing import Any

from PIL import Image, ImageDraw

from agents.base import CallBudget
from agents.counting.backends.base import (
    BackendKind,
    CountingBackendOutcome,
    CountingRequest,
    require_model_cache_identity,
)
from agents.counting.point_pipeline import PointCountingOrchestrator
from agents.counting.schema import (
    CountTargetSpec,
    DisagreementReview,
    GlobalPointObservation,
    PixelRect,
    SeamDecision,
    TileCountResponse,
    TileSpec,
)
from agents.counting.settings import CountingSettings, CountingTargetStrategy
from models.base import RequestMeta, VisionLanguageClient, build_request_hash
from models.images import image_to_data_url


class QwenPointCountingBackend:
    """Default tile-based point counting via an injected vision client."""

    name = "qwen_point"
    kind: BackendKind = "qwen_point"
    priority = 0

    def __init__(
        self,
        client: VisionLanguageClient,
        *,
        counting: CountingSettings,
        system_prompt: str,
        prompt_version: str | None = None,
        empty_review_prompt: str | None = None,
        empty_review_prompt_version: str | None = None,
        seam_prompt: str | None = None,
        seam_prompt_version: str | None = None,
        disagreement_prompt: str | None = None,
        disagreement_prompt_version: str | None = None,
        strategy_resolver: Callable[[CountTargetSpec], CountingTargetStrategy]
        | None = None,
    ) -> None:
        self._client = client
        self._counting = counting
        self._system_prompt = system_prompt
        self._prompt_version = prompt_version or counting.prompt_version
        self._empty_review_prompt = empty_review_prompt or system_prompt
        self._empty_review_prompt_version = (
            empty_review_prompt_version or self._prompt_version
        )
        self._seam_prompt = seam_prompt
        self._seam_prompt_version = seam_prompt_version
        self._disagreement_prompt = disagreement_prompt or self._empty_review_prompt
        self._disagreement_prompt_version = disagreement_prompt_version or self._prompt_version
        self.last_disagreement_review_trace: dict[str, object] = {}
        self._strategy_resolver = strategy_resolver

    def is_enabled(self) -> bool:
        return True

    def is_available(self) -> bool:
        return True

    def supports(self, target: CountTargetSpec, hints: Any | None = None) -> bool:
        return True

    async def count(
        self,
        request: CountingRequest,
        context: object,
    ) -> CountingBackendOutcome:
        # This must fail before the pipeline can turn the error into a tile
        # failure; a real cache identity is mandatory for every model call.
        require_model_cache_identity(self._client, component="qwen_point")
        strategy = (
            self._strategy_resolver(request.target)
            if self._strategy_resolver is not None
            else CountingTargetStrategy()
        )
        minimum_scan_depth = (
            self._counting.small_object_min_scan_depth
            if strategy.small_object
            else 0
        )
        verify_empty = self._counting.verify_empty_tiles and strategy.verify_empty
        callback = _PipelineTileCallback(
            self._client,
            system_prompt=self._system_prompt,
            prompt_version=self._prompt_version,
            empty_review_prompt=self._empty_review_prompt,
            empty_review_prompt_version=self._empty_review_prompt_version,
            counting=self._counting,
            budget=getattr(context, "call_budget", None),
            artifact_root=request.artifact_dir,
            sample_id=request.sample.sample_id,
            upscale_max_side=(
                self._counting.small_object_upscale_max_side
                if strategy.small_object
                else None
            ),
        )
        seam_callback = (
            _QwenSeamReviewCallback(
                self._client,
                system_prompt=self._seam_prompt,
                prompt_version=self._seam_prompt_version or "seam-review-v1",
                budget=getattr(context, "call_budget", None),
                artifact_root=request.artifact_dir,
                sample_id=request.sample.sample_id,
            )
            if self._seam_prompt is not None
            and self._counting.seam_review_enabled
            else None
        )
        orchestrator = PointCountingOrchestrator(
            callback,
            counting=self._counting,
            empty_tile_reviewer=callback if verify_empty else None,
            seam_reviewer=seam_callback,
        )
        counting = await orchestrator.count_image(
            request.image,
            sample_id=request.sample.sample_id,
            question=request.sample.question,
            target=request.target,
            minimum_scan_depth=minimum_scan_depth,
        )
        original_size, transmitted_size = callback.transmission_summary()
        return CountingBackendOutcome(
            counting=counting,
            trace={
                "backend": self.name,
                "pipeline": "point_pipeline.count_image",
                "prompt_version": self._prompt_version,
                "minimum_scan_depth": minimum_scan_depth,
                "strategy": strategy.model_dump(mode="json"),
                "empty_review_enabled": verify_empty,
                "empty_review_attempt_count": callback.empty_review_attempt_count,
                "empty_review_positive_count": callback.empty_review_positive_count,
                "empty_review_failure_count": callback.empty_review_failure_count,
                "upscale_used": callback.upscale_used,
                "original_size": original_size,
                "transmitted_size": transmitted_size,
                "seam_review_enabled": seam_callback is not None,
                "seam_review_attempt_count": (
                    seam_callback.attempt_count if seam_callback is not None else 0
                ),
                "seam_review_same_count": (
                    seam_callback.same_count if seam_callback is not None else 0
                ),
                "seam_review_different_count": (
                    seam_callback.different_count if seam_callback is not None else 0
                ),
                "seam_review_uncertain_count": (
                    seam_callback.uncertain_count if seam_callback is not None else 0
                ),
                "seam_review_failure_count": (
                    seam_callback.failure_count if seam_callback is not None else 0
                ),
            },
        )

    async def review_disagreements(
        self,
        *,
        request: CountingRequest,
        conflicts: list[dict[str, Any]],
        context: object,
    ) -> DisagreementReview:
        """Review a bounded batch of unresolved detector conflicts once."""

        require_model_cache_identity(self._client, component="qwen_disagreement")
        ordered = sorted(conflicts, key=lambda item: str(item.get("conflict_id", "")))
        selected = ordered[: self._counting.max_disagreement_regions]
        truncated = ordered[self._counting.max_disagreement_regions :]
        content: list[dict[str, Any]] = []
        image_digests: list[str] = []
        for conflict in selected:
            crop, crop_hash, annotations = _disagreement_crop(
                request.image,
                conflict,
                padding_ratio=self._counting.disagreement_context_padding_ratio,
            )
            image_digests.append(crop_hash)
            content.append(
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "conflict_id": conflict.get("conflict_id"),
                            "target": request.target.canonical_label,
                            "candidate_ids": conflict.get("candidate_ids", []),
                            "candidate_annotations": annotations,
                        },
                        ensure_ascii=False,
                    ),
                }
            )
            with io.BytesIO() as buffer:
                crop.save(buffer, format="JPEG", quality=95)
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(buffer.getvalue())},
                    }
                )
        messages = [
            {"role": "system", "content": self._disagreement_prompt},
            {"role": "user", "content": content},
        ]
        digest = hashlib.sha256("|".join(image_digests).encode("utf-8")).hexdigest()
        request_hash = build_request_hash(
            model=require_model_cache_identity(self._client, component="qwen_disagreement").model,
            generation=require_model_cache_identity(self._client, component="qwen_disagreement").generation_payload(),
            prompt_version=self._disagreement_prompt_version,
            messages=messages,
            image_sha256=digest,
            target_spec=request.target.model_dump(mode="json"),
            response_schema=DisagreementReview.model_json_schema(),
            client_version=require_model_cache_identity(self._client, component="qwen_disagreement").client_version,
        )
        budget = getattr(context, "call_budget", None)
        if budget is not None:
            budget.reserve_qwen()
        result = await self._client.complete_json(
            messages=messages,
            response_model=DisagreementReview,
            request_meta=RequestMeta(
                request_id=f"{request.sample.sample_id}:detector-disagreement-review",
                request_hash=request_hash,
                prompt_version=self._disagreement_prompt_version,
                sample_id=request.sample.sample_id,
                artifact_dir=request.artifact_dir / "disagreement_review",
            ),
        )
        self.last_disagreement_review_trace = {
            "disagreement_review_triggered": True,
            "review_backend": self.name,
            "requested_conflict_ids": [str(item.get("conflict_id")) for item in ordered],
            "reviewed_conflict_ids": [str(item.get("conflict_id")) for item in selected],
            "truncated_conflict_ids": [str(item.get("conflict_id")) for item in truncated],
            "review_request_hash": request_hash,
        }
        return DisagreementReview.model_validate(result)


def _disagreement_crop(
    image: Image.Image,
    conflict: dict[str, Any],
    *,
    padding_ratio: float,
) -> tuple[Image.Image, str, list[dict[str, object]]]:
    """Render one bounded conflict region with crop-local candidate markers."""

    boxes: list[tuple[float, float, float, float]] = []
    centers: list[tuple[float, float]] = []
    for point in conflict.get("candidate_points", []):
        if not isinstance(point, dict):
            continue
        centers.append((float(point.get("global_x_px", 0)), float(point.get("global_y_px", 0))))
        provenance = point.get("provenance")
        if isinstance(provenance, dict):
            raw_box = provenance.get("bbox_xyxy_global_px")
            if isinstance(raw_box, list) and len(raw_box) == 4:
                boxes.append(tuple(float(value) for value in raw_box))
    if boxes:
        left = min(box[0] for box in boxes)
        top = min(box[1] for box in boxes)
        right = max(box[2] for box in boxes)
        bottom = max(box[3] for box in boxes)
    elif centers:
        left = min(center[0] for center in centers) - 32
        top = min(center[1] for center in centers) - 32
        right = max(center[0] for center in centers) + 32
        bottom = max(center[1] for center in centers) + 32
    else:
        left, top, right, bottom = 0, 0, image.width, image.height
    width = max(1.0, right - left)
    height = max(1.0, bottom - top)
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    bounds = (
        max(0, int(left - pad_x)),
        max(0, int(top - pad_y)),
        min(image.width, int(right + pad_x + 1)),
        min(image.height, int(bottom + pad_y + 1)),
    )
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        bounds = (0, 0, image.width, image.height)
    crop = image.crop(bounds)
    draw = ImageDraw.Draw(crop)
    point_by_id = {
        str(point.get("global_id")): point
        for point in conflict.get("candidate_points", [])
        if isinstance(point, dict) and point.get("global_id")
    }
    annotations: list[dict[str, object]] = []
    colors = ((255, 64, 64), (64, 224, 96), (64, 144, 255), (255, 192, 64))
    for index, candidate_id in enumerate(conflict.get("candidate_ids", [])):
        candidate_id = str(candidate_id)
        point = point_by_id.get(candidate_id, {})
        marker = chr(65 + index) if index < 26 else f"A{index - 25}"
        provenance = point.get("provenance") if isinstance(point, dict) else None
        geometry: dict[str, object] = {}
        if isinstance(provenance, dict):
            polygon = provenance.get("obb_polygon_global_px")
            if isinstance(polygon, list) and len(polygon) >= 3:
                local_polygon = [
                    [float(vertex[0]) - bounds[0], float(vertex[1]) - bounds[1]]
                    for vertex in polygon
                    if isinstance(vertex, list) and len(vertex) >= 2
                ]
                if len(local_polygon) >= 3:
                    geometry = {"type": "obb_polygon", "points": local_polygon}
                    draw.line([tuple(vertex) for vertex in local_polygon + [local_polygon[0]]], fill=colors[index % len(colors)], width=3)
            if not geometry:
                bbox = provenance.get("bbox_xyxy_global_px")
                if isinstance(bbox, list) and len(bbox) == 4:
                    local_bbox = [
                        float(bbox[0]) - bounds[0],
                        float(bbox[1]) - bounds[1],
                        float(bbox[2]) - bounds[0],
                        float(bbox[3]) - bounds[1],
                    ]
                    geometry = {"type": "bbox", "xyxy": local_bbox}
                    draw.rectangle(local_bbox, outline=colors[index % len(colors)], width=3)
        center = (
            float(point.get("global_x_px", (bounds[0] + bounds[2]) / 2)) - bounds[0],
            float(point.get("global_y_px", (bounds[1] + bounds[3]) / 2)) - bounds[1],
        )
        if not geometry:
            radius = max(4.0, float(point.get("radius_px", 4.0)))
            geometry = {"type": "point", "xy": [center[0], center[1]], "radius": radius}
            draw.ellipse(
                (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius),
                outline=colors[index % len(colors)],
                width=3,
            )
        label_box = (center[0] + 4, center[1] + 4, center[0] + 20, center[1] + 20)
        draw.rectangle(label_box, fill=colors[index % len(colors)])
        draw.text((center[0] + 7, center[1] + 5), marker, fill=(0, 0, 0))
        annotations.append(
            {
                "marker": marker,
                "candidate_id": candidate_id,
                "geometry": geometry,
            }
        )
    with io.BytesIO() as buffer:
        crop.save(buffer, format="JPEG", quality=95)
        digest = hashlib.sha256(buffer.getvalue()).hexdigest()
    return crop, digest, annotations


class _PipelineTileCallback:
    """Adapt the vision client to initial and independent-review callbacks."""

    def __init__(
        self,
        client: VisionLanguageClient,
        *,
        system_prompt: str,
        prompt_version: str,
        counting: CountingSettings,
        budget: CallBudget | None,
        artifact_root: Any,
        sample_id: str,
        empty_review_prompt: str | None = None,
        empty_review_prompt_version: str | None = None,
        upscale_max_side: int | None = None,
    ) -> None:
        self._client = client
        self._system_prompt = system_prompt
        self._prompt_version = prompt_version
        self._empty_review_prompt = empty_review_prompt or system_prompt
        self._empty_review_prompt_version = (
            empty_review_prompt_version or prompt_version
        )
        self._counting = counting
        self._budget = budget
        self._artifact_root = artifact_root
        self._sample_id = sample_id
        self._upscale_max_side = upscale_max_side
        self._transmissions: list[tuple[tuple[int, int], tuple[int, int]]] = []
        self.empty_review_attempt_count = 0
        self.empty_review_positive_count = 0
        self.empty_review_failure_count = 0

    async def count_tile(
        self,
        *,
        tile: TileSpec,
        image: Image.Image,
        target: CountTargetSpec,
    ) -> TileCountResponse:
        return await self._complete_tile(
            tile=tile,
            image=image,
            target=target,
            review=False,
        )

    async def review_empty_tile(
        self,
        *,
        tile: TileSpec,
        image: Image.Image,
        target: CountTargetSpec,
    ) -> TileCountResponse:
        self.empty_review_attempt_count += 1
        try:
            response = await self._complete_tile(
                tile=tile,
                image=image,
                target=target,
                review=True,
            )
        except Exception:
            self.empty_review_failure_count += 1
            raise
        if response.points:
            self.empty_review_positive_count += 1
        return response

    async def _complete_tile(
        self,
        *,
        tile: TileSpec,
        image: Image.Image,
        target: CountTargetSpec,
        review: bool,
    ) -> TileCountResponse:
        transmitted = _upscale_image(image, self._upscale_max_side)
        self._transmissions.append((image.size, transmitted.size))
        messages, request_hash, image_hash = self._build_request(
            tile, transmitted, target, review=review
        )
        if self._budget is not None:
            self._budget.reserve_qwen()
        prompt_version = (
            self._empty_review_prompt_version if review else self._prompt_version
        )
        return await self._client.complete_json(
            messages=messages,
            response_model=TileCountResponse,
            request_meta=RequestMeta(
                request_id=(
                    f"{self._sample_id}:{tile.tile_id}:empty-review"
                    if review
                    else f"{self._sample_id}:{tile.tile_id}"
                ),
                request_hash=request_hash,
                prompt_version=prompt_version,
                sample_id=self._sample_id,
                tile_id=tile.tile_id,
                image_sha256=image_hash,
                artifact_dir=(
                    self._artifact_root / "tiles" / tile.tile_id / "empty_review"
                    if review
                    else self._artifact_root / "tiles" / tile.tile_id
                ),
            ),
        )

    def _build_request(
        self,
        tile: TileSpec,
        image: Image.Image,
        target: CountTargetSpec,
        *,
        review: bool = False,
    ) -> tuple[list[dict[str, Any]], str, str]:
        with io.BytesIO() as buffer:
            image.save(buffer, format="JPEG", quality=95)
            image_bytes = buffer.getvalue()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        request_text = json.dumps(
            {
                "target_spec": target.model_dump(mode="json"),
                "tile_id": tile.tile_id,
                "owner_core_normalized": _owner_core_prompt_bounds(tile),
                "transmitted_image_size": list(image.size),
                "scan_pass": "independent_empty_review" if review else "initial",
                "instruction": (
                    "Independently rescan the complete owner core and return exactly one "
                    "point per supported instance; do not assume the earlier empty result "
                    "was correct. Halo is context only."
                    if review
                    else "Return exactly one point per instance whose centre is in the owner "
                    "core. Halo is context only; do not output halo-owned instances."
                ),
            },
            ensure_ascii=False,
        )
        prompt = self._empty_review_prompt if review else self._system_prompt
        prompt_version = (
            self._empty_review_prompt_version if review else self._prompt_version
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image_bytes)},
                    },
                    {"type": "text", "text": request_text},
                ],
            },
        ]
        identity = require_model_cache_identity(
            self._client, component="qwen_point"
        )
        request_hash = build_request_hash(
            model=identity.model,
            generation=identity.generation_payload(),
            prompt_version=prompt_version,
            messages=messages,
            image_sha256=image_hash,
            tile_geometry=tile.model_dump(mode="json"),
            target_spec=target.model_dump(mode="json"),
            response_schema=TileCountResponse.model_json_schema(),
            client_version=identity.client_version,
            model_revision=identity.revision,
        )
        return messages, request_hash, image_hash

    @property
    def upscale_used(self) -> bool:
        return any(
            original != transmitted
            for original, transmitted in self._transmissions
        )

    def transmission_summary(self) -> tuple[list[int], list[int]]:
        """Return the largest transmitted crop pair for compact public trace."""

        if not self._transmissions:
            return [0, 0], [0, 0]
        original, transmitted = max(
            self._transmissions,
            key=lambda pair: pair[1][0] * pair[1][1],
        )
        return list(original), list(transmitted)


class _QwenSeamReviewCallback:
    """Review one ambiguous pair from a local crop without recounting."""

    def __init__(
        self,
        client: VisionLanguageClient,
        *,
        system_prompt: str,
        prompt_version: str,
        budget: CallBudget | None,
        artifact_root: Any,
        sample_id: str,
    ) -> None:
        self._client = client
        self._system_prompt = system_prompt
        self._prompt_version = prompt_version
        self._budget = budget
        self._artifact_root = artifact_root
        self._sample_id = sample_id
        self.attempt_count = 0
        self.same_count = 0
        self.different_count = 0
        self.uncertain_count = 0
        self.failure_count = 0

    async def review(
        self,
        *,
        conflict_id: str,
        image: Image.Image,
        crop_global: PixelRect,
        first: GlobalPointObservation,
        second: GlobalPointObservation,
    ) -> SeamDecision:
        self.attempt_count += 1
        try:
            messages, request_hash, image_hash = self._build_request(
                conflict_id=conflict_id,
                image=image,
                crop_global=crop_global,
                first=first,
                second=second,
            )
            if self._budget is not None:
                self._budget.reserve_qwen()
            decision = await self._client.complete_json(
                messages=messages,
                response_model=SeamDecision,
                request_meta=RequestMeta(
                    request_id=f"{self._sample_id}:seam:{request_hash[:16]}",
                    request_hash=request_hash,
                    prompt_version=self._prompt_version,
                    sample_id=self._sample_id,
                    image_sha256=image_hash,
                    artifact_dir=(
                        self._artifact_root / "seams" / request_hash
                    ),
                ),
            )
            decision = SeamDecision.model_validate(decision)
        except Exception:
            self.failure_count += 1
            raise
        if decision.decision == "same_instance":
            self.same_count += 1
        elif decision.decision == "different_instances":
            self.different_count += 1
        else:
            self.uncertain_count += 1
        return decision

    def _build_request(
        self,
        *,
        conflict_id: str,
        image: Image.Image,
        crop_global: PixelRect,
        first: GlobalPointObservation,
        second: GlobalPointObservation,
    ) -> tuple[list[dict[str, Any]], str, str]:
        with io.BytesIO() as buffer:
            image.save(buffer, format="JPEG", quality=95)
            image_bytes = buffer.getvalue()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        geometry = {
            "conflict_id": conflict_id,
            "crop_global": crop_global.model_dump(mode="json"),
            "first": _seam_point_geometry(first, crop_global),
            "second": _seam_point_geometry(second, crop_global),
        }
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image_bytes)},
                    },
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                **geometry,
                                "instruction": (
                                    "Judge only this pair as same_instance, "
                                    "different_instances, or uncertain."
                                ),
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
            },
        ]
        identity = require_model_cache_identity(
            self._client, component="qwen_seam_review"
        )
        request_hash = build_request_hash(
            model=identity.model,
            generation=identity.generation_payload(),
            prompt_version=self._prompt_version,
            messages=messages,
            image_sha256=image_hash,
            tile_geometry=geometry,
            response_schema=SeamDecision.model_json_schema(),
            client_version=identity.client_version,
            model_revision=identity.revision,
        )
        return messages, request_hash, image_hash


def _seam_point_geometry(
    point: GlobalPointObservation,
    crop_global: PixelRect,
) -> dict[str, Any]:
    width = max(1, crop_global.width - 1)
    height = max(1, crop_global.height - 1)
    return {
        "global_id": point.global_id,
        "target": point.target,
        "crop_x_norm": round(
            (point.global_x_px - crop_global.left) / width * 999
        ),
        "crop_y_norm": round(
            (point.global_y_px - crop_global.top) / height * 999
        ),
        "radius_px": point.radius_px,
        "confidence": point.confidence,
        "source": (
            point.provenance.source if point.provenance is not None else None
        ),
    }


def _upscale_image(image: Image.Image, max_side: int | None) -> Image.Image:
    if max_side is None or max(image.size) >= max_side:
        return image
    scale = max_side / max(image.size)
    size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    return image.resize(size, Image.Resampling.LANCZOS)


def _owner_core_prompt_bounds(tile: TileSpec) -> list[int]:
    """Express owner core bounds in the crop's normalized coordinates."""

    local = tile.owner_core_local
    width, height = tile.crop_global.width, tile.crop_global.height
    return [
        round(local.left / max(1, width - 1) * 999),
        round(local.top / max(1, height - 1) * 999),
        round((local.right - 1) / max(1, width - 1) * 999),
        round((local.bottom - 1) / max(1, height - 1) * 999),
    ]
