"""Qwen point-counting backend — wraps the point-counting pipeline.

Qwen 点式计数后端 — 封装点式计数 pipeline。通过内部 TileCountCallback 将
VisionLanguageClient 适配到 pipeline；不做数据来源判断、不实现 Agent 回退、
不自行创建提示词目录（所有 prompt text/version 由构造参数注入）。
"""

from __future__ import annotations

import hashlib
import io
import json
from typing import Any

from PIL import Image

from agents.counting.backends.base import (
    CountingBackendOutcome,
    CountingRequest,
)
from agents.counting.point_pipeline import PointCountingOrchestrator
from agents.counting.schema import CountTargetSpec, TileCountResponse, TileSpec
from agents.counting.settings import CountingSettings
from models.base import RequestMeta, VisionLanguageClient, build_request_hash
from models.images import image_to_data_url


class QwenPointCountingBackend:
    """Default tile-based point counting via Qwen. / 通过 Qwen 的默认 tile 点式计数。"""

    name = "qwen_point"
    priority = 0  # lowest — default backend / 最低 — 默认后端

    def __init__(
        self,
        client: VisionLanguageClient,
        *,
        counting: CountingSettings,
        system_prompt: str,
        prompt_version: str | None = None,
        minimum_scan_depth: int = 0,
    ) -> None:
        self._client = client
        self._counting = counting
        self._system_prompt = system_prompt
        self._prompt_version = prompt_version or counting.prompt_version
        self._minimum_scan_depth = minimum_scan_depth

    def is_available(self) -> bool:
        return True  # always available / 始终可用

    def supports(self, target: CountTargetSpec, hints: Any | None = None) -> bool:
        return True  # handles all counting targets / 处理所有计数目标

    async def count(
        self,
        request: CountingRequest,
        context: object,
    ) -> CountingBackendOutcome:
        callback = _PipelineTileCallback(
            self._client,
            system_prompt=self._system_prompt,
            prompt_version=self._prompt_version,
            counting=self._counting,
            budget=getattr(context, "call_budget", None),
            artifact_root=request.artifact_dir,
            sample_id=request.sample.sample_id,
        )
        orchestrator = PointCountingOrchestrator(callback, counting=self._counting)
        counting = await orchestrator.count_image(
            request.image,
            sample_id=request.sample.sample_id,
            question=request.sample.question,
            target=request.target,
            minimum_scan_depth=self._minimum_scan_depth,
        )
        return CountingBackendOutcome(
            counting=counting,
            trace={
                "backend": self.name,
                "pipeline": "point_pipeline.count_image",
                "prompt_version": self._prompt_version,
                "minimum_scan_depth": self._minimum_scan_depth,
            },
        )


class _PipelineTileCallback:
    """Adapts the injected VisionLanguageClient to the pipeline's
    TileCountCallback protocol. 将注入的 VisionLanguageClient 适配为 pipeline
    的 TileCountCallback 协议。"""

    def __init__(
        self,
        client: VisionLanguageClient,
        *,
        system_prompt: str,
        prompt_version: str,
        counting: CountingSettings,
        budget: Any,
        artifact_root: Any,
        sample_id: str,
    ) -> None:
        self._client = client
        self._system_prompt = system_prompt
        self._prompt_version = prompt_version
        self._counting = counting
        self._budget = budget
        self._artifact_root = artifact_root
        self._sample_id = sample_id

    async def count_tile(
        self,
        *,
        tile: TileSpec,
        image: Image.Image,
        target: CountTargetSpec,
    ) -> TileCountResponse:
        messages, request_hash, image_hash = self._build_request(tile, image, target)
        if self._budget is not None:
            self._budget.reserve_qwen()
        return await self._client.complete_json(
            messages=messages,
            response_model=TileCountResponse,
            request_meta=RequestMeta(
                request_id=f"{self._sample_id}:{tile.tile_id}",
                request_hash=request_hash,
                prompt_version=self._prompt_version,
                sample_id=self._sample_id,
                tile_id=tile.tile_id,
                image_sha256=image_hash,
                artifact_dir=self._artifact_root / "tiles" / tile.tile_id,
            ),
        )

    def _build_request(
        self,
        tile: TileSpec,
        image: Image.Image,
        target: CountTargetSpec,
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
                "instruction": (
                    "Return exactly one point per instance whose centre is in the owner core. "
                    "Halo is context only; do not output halo-owned instances."
                ),
            },
            ensure_ascii=False,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_bytes)}},
                    {"type": "text", "text": request_text},
                ],
            },
        ]
        identity = getattr(self._client, "cache_identity", None)
        model = identity.model if identity is not None else "qwen_point"
        generation = (
            identity.generation_payload() if identity is not None else {"temperature": 0.0}
        )
        client_version = identity.client_version if identity is not None else "1"
        request_hash = build_request_hash(
            model=model,
            generation=generation,
            prompt_version=self._prompt_version,
            messages=messages,
            image_sha256=image_hash,
            tile_geometry=tile.model_dump(mode="json"),
            target_spec=target.model_dump(mode="json"),
            client_version=client_version,
            model_revision=identity.revision if identity is not None else None,
        )
        return messages, request_hash, image_hash


def _owner_core_prompt_bounds(tile: TileSpec) -> list[int]:
    """Express owner core bounds in the tile crop's normalized coordinates.
    在切片 crop 的归一化坐标中表达 owner core 边界。"""
    local = tile.owner_core_local
    width, height = tile.crop_global.width, tile.crop_global.height
    return [
        round(local.left / max(1, width - 1) * 999),
        round(local.top / max(1, height - 1) * 999),
        round((local.right - 1) / max(1, width - 1) * 999),
        round((local.bottom - 1) / max(1, height - 1) * 999),
    ]
