"""Qwen point-counting backend wrapping the generic point pipeline."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable
from typing import Any

from PIL import Image

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
