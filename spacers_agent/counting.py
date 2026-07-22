"""Sequential, resumable point-counting orchestration built on owner-core tiles.
基于 owner core 切片的顺序、可恢复点式计数编排。
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from spacers_agent.clients.base import RequestMeta, VisionLanguageClient, build_request_hash, image_to_data_url
from spacers_agent.imaging import (
    build_core_halo_tiles,
    convert_local_point_to_global,
    crop_for_tile,
    split_tile_owner_core,
)
from spacers_agent.schemas import CountTargetSpec, CountingDraft, CountingResult, GlobalPointObservation, IssueRecord, TileCountResponse, TileSpec
from spacers_agent.settings import CountingSettings, QwenSettings


TileStatus = Literal[
    "pending",
    "running",
    "succeeded",
    "succeeded_with_repair",
    "needs_split",
    "superseded_by_children",
    "failed",
]


class TileCheckpoint(BaseModel):
    """Durable state for one tile request without image payloads or credentials.
    不包含图像载荷或凭据的单个 tile 请求持久状态。
    """

    model_config = ConfigDict(extra="forbid")

    tile_id: str = Field(min_length=1)
    request_hash: str = Field(min_length=1)
    status: TileStatus
    error_type: str | None = None
    error_message: str | None = None


class BoundaryConflict(BaseModel):
    """A possible duplicated instance restricted to neighbouring tile boundaries.
    仅限相邻切片边界处的潜在重复实例记录。
    """

    model_config = ConfigDict(extra="forbid")

    conflict_id: str
    first_global_id: str
    second_global_id: str
    threshold_px: float = Field(gt=0.0)
    distance_px: float = Field(ge=0.0)


class SeamDecision(BaseModel):
    """Structured seam-verifier result using points and local evidence only.
    仅使用点和局部证据的结构化 seam 核验结果。
    """

    model_config = ConfigDict(extra="forbid")

    conflict_id: str
    decision: Literal["same_instance", "different_instances", "uncertain"]
    canonical_point: tuple[int, int] | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    short_reason: str = Field(max_length=240)


@dataclass(frozen=True)
class _TileOutcome:
    points: list[GlobalPointObservation]
    succeeded_tile_ids: list[str]
    failed_tile_ids: list[str]
    warnings: list[str]
    processed_tiles: list[TileSpec]


class TileCheckpointStore:
    """Persist tile inputs and validated outputs for safe local resume behavior.
    持久化 tile 输入和已校验输出，以支持安全的本地恢复。
    """

    def __init__(self, run_dir: Path) -> None:
        self.root = run_dir / "tiles"

    def load_success(self, tile: TileSpec, request_hash: str) -> TileCountResponse | None:
        """Return a matching successful response without issuing another model call.
        返回匹配的成功响应且不再发起模型调用。
        """

        directory = self._directory(tile)
        checkpoint_path = directory / "checkpoint.json"
        response_path = directory / "parsed.json"
        if not checkpoint_path.is_file() or not response_path.is_file():
            return None
        checkpoint = TileCheckpoint.model_validate_json(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.request_hash != request_hash or checkpoint.status not in {"succeeded", "succeeded_with_repair"}:
            return None
        return TileCountResponse.model_validate_json(response_path.read_text(encoding="utf-8"))

    def write_pending(self, tile: TileSpec, request_hash: str) -> Path:
        """Write tile geometry before a request so interrupted work remains auditable.
        在请求前写入 tile 几何信息，使中断任务仍可审计。
        """

        directory = self._directory(tile)
        directory.mkdir(parents=True, exist_ok=True)
        self._write_json(directory / "spec.json", tile.model_dump(mode="json"))
        self._write_checkpoint(directory, tile.tile_id, request_hash, "running")
        return directory

    def write_success(
        self,
        tile: TileSpec,
        request_hash: str,
        response: TileCountResponse,
        points: Sequence[GlobalPointObservation],
        *,
        status: TileStatus = "succeeded",
        warnings: Sequence[str] = (),
    ) -> None:
        """Save parsed output, conversion validation, and a successful checkpoint.
        保存解析输出、坐标换算校验和成功检查点。
        """

        directory = self._directory(tile)
        directory.mkdir(parents=True, exist_ok=True)
        self._write_json(directory / "parsed.json", response.model_dump(mode="json"))
        self._write_json(
            directory / "conversion_validation.json",
            {
                "accepted_point_ids": [point.global_id for point in points if point.accepted],
                "rejected_points": [
                    {"global_id": point.global_id, "reason": point.rejection_reason}
                    for point in points
                    if not point.accepted
                ],
                "warnings": list(warnings),
            },
        )
        self._write_checkpoint(directory, tile.tile_id, request_hash, status)

    def write_failure(self, tile: TileSpec, request_hash: str, error: Exception) -> None:
        """Record a visible failure instead of silently discarding a tile.
        记录可见失败，绝不静默丢弃 tile。
        """

        directory = self._directory(tile)
        directory.mkdir(parents=True, exist_ok=True)
        self._write_checkpoint(
            directory,
            tile.tile_id,
            request_hash,
            "failed",
            error_type=type(error).__name__,
            error_message=str(error),
        )

    def load_superseded(self, tile: TileSpec, request_hash: str) -> list[TileSpec] | None:
        """Load recorded child geometry so resume does not call a parent tile again.
        加载已记录的子切片几何，使恢复时不再调用父 tile。
        """

        directory = self._directory(tile)
        checkpoint_path = directory / "checkpoint.json"
        children_path = directory / "children.json"
        if not checkpoint_path.is_file() or not children_path.is_file():
            return None
        checkpoint = TileCheckpoint.model_validate_json(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.status != "superseded_by_children" or checkpoint.request_hash != request_hash:
            return None
        raw_children = json.loads(children_path.read_text(encoding="utf-8"))
        return [TileSpec.model_validate(value) for value in raw_children]

    def mark_superseded(self, tile: TileSpec, request_hash: str, children: Sequence[TileSpec]) -> None:
        """Mark a parent tile as replaced by child owner cores.
        将父 tile 标记为已由子 owner core 替代。
        """

        directory = self._directory(tile)
        self._write_json(directory / "children.json", [child.model_dump(mode="json") for child in children])
        self._write_checkpoint(directory, tile.tile_id, request_hash, "superseded_by_children")

    def _directory(self, tile: TileSpec) -> Path:
        return self.root / tile.tile_id

    def _write_checkpoint(
        self,
        directory: Path,
        tile_id: str,
        request_hash: str,
        status: TileStatus,
        *,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        self._write_json(
            directory / "checkpoint.json",
            TileCheckpoint(
                tile_id=tile_id,
                request_hash=request_hash,
                status=status,
                error_type=error_type,
                error_message=error_message,
            ).model_dump(mode="json"),
        )

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)


class PointCountingOrchestrator:
    """Count one image sequentially while treating accepted points as the sole truth.
顺序计数单张图像，并将接受点作为唯一事实来源。
    """

    def __init__(
        self,
        client: VisionLanguageClient,
        *,
        counting: CountingSettings,
        qwen: QwenSettings,
        system_prompt: str,
        run_dir: Path,
        before_qwen_call: Callable[[], None] | None = None,
        seam_prompt: str | None = None,
    ) -> None:
        self.client = client
        self.counting = counting
        self.qwen = qwen
        self.system_prompt = system_prompt
        self.checkpoints = TileCheckpointStore(run_dir)
        self.before_qwen_call = before_qwen_call
        self.seam_prompt = seam_prompt

    async def count_image(
        self,
        image: Image.Image,
        *,
        sample_id: str,
        question: str,
        target: CountTargetSpec,
    ) -> CountingResult:
        """Process initial tiles row-major and aggregate only final accepted points.
按行优先处理初始切片，并且只聚合最终接受点。
        """

        draft = await self.collect_points(image, sample_id=sample_id, question=question, target=target)
        conflicts = [BoundaryConflict.model_validate(value) for value in draft.boundary_conflicts]
        decisions, seam_warnings = await self._verify_seams(image.convert("RGB"), conflicts, draft.raw_global_points, sample_id)
        final_points, merged_groups = finalize_representatives(draft.raw_global_points, [
            (conflict.first_global_id, conflict.second_global_id)
            for conflict in conflicts if decisions.get(conflict.conflict_id) == "same_instance"
        ])
        unresolved = [conflict.conflict_id for conflict in conflicts if decisions.get(conflict.conflict_id) != "same_instance"]
        warnings = draft.warnings + [IssueRecord(code="SEAM_VERIFY_FAILED", message=value) for value in seam_warnings]
        status: Literal["completed", "completed_with_warnings", "partial", "failed"] = "partial" if draft.failed_tiles and draft.succeeded_tiles else "failed" if draft.failed_tiles else "completed_with_warnings" if unresolved or warnings else "completed"
        return CountingResult(sample_id=sample_id, target=target.canonical_label, question=question, source_width=draft.source_width, source_height=draft.source_height, tile_count=len(draft.processed_tiles), initial_tile_count=draft.initial_tile_count, leaf_tile_count=len(draft.processed_tiles), succeeded_tiles=draft.succeeded_tiles, failed_tiles=draft.failed_tiles, global_points=final_points, merged_groups=merged_groups, unresolved_conflicts=unresolved, warnings=warnings, final_count=sum(point.accepted for point in final_points), status=status)

    async def collect_points(
        self, image: Image.Image, *, sample_id: str, question: str, target: CountTargetSpec
    ) -> CountingDraft:
        """Collect accepted-policy tile points without seam finalization. / 收集经过接受策略的 tile 点但不做 seam 最终化。"""

        normalized = image.convert("RGB")
        tiles = build_core_halo_tiles(
            *normalized.size,
            core_size=self.counting.tile_core_size,
            halo_size=self.counting.halo_size,
            model_max_side=self.counting.model_max_side,
        )
        outcomes: list[_TileOutcome] = []
        for tile in tiles:
            outcomes.append(await self._process_tile(normalized, tile, sample_id, target))
        points = [point for outcome in outcomes for point in outcome.points]
        failed_tiles = [tile_id for outcome in outcomes for tile_id in outcome.failed_tile_ids]
        succeeded_tiles = [tile_id for outcome in outcomes for tile_id in outcome.succeeded_tile_ids]
        processed_tiles = [tile for outcome in outcomes for tile in outcome.processed_tiles]
        conflicts = find_boundary_conflicts(points, processed_tiles)
        warning_records = [IssueRecord(code="TILE_WARNING", message=warning) for outcome in outcomes for warning in outcome.warnings]
        warning_records.extend(IssueRecord(code="UNRESOLVED_SEAM_CONFLICT", message="Seam candidate requires explicit verification.", tile_ids=[point.source_tile_id for point in points if point.global_id in {conflict.first_global_id, conflict.second_global_id}], point_ids=[conflict.first_global_id, conflict.second_global_id]) for conflict in conflicts)
        return CountingDraft(
            sample_id=sample_id,
            target=target.canonical_label,
            question=question,
            source_width=normalized.width,
            source_height=normalized.height,
            initial_tile_count=len(tiles),
            succeeded_tiles=succeeded_tiles,
            failed_tiles=failed_tiles,
            raw_global_points=points,
            processed_tiles=processed_tiles,
            boundary_conflicts=[conflict.model_dump(mode="json") for conflict in conflicts],
            warnings=warning_records,
        )

    async def _process_tile(
        self,
        image: Image.Image,
        tile: TileSpec,
        sample_id: str,
        target: CountTargetSpec,
    ) -> _TileOutcome:
        messages, request_hash, image_hash = self._build_request(image, tile, target)
        superseded_children = self.checkpoints.load_superseded(tile, request_hash)
        if superseded_children is not None:
            resumed_children = [await self._process_tile(image, child, sample_id, target) for child in superseded_children]
            return _TileOutcome(
                [point for outcome in resumed_children for point in outcome.points],
                [tile_id for outcome in resumed_children for tile_id in outcome.succeeded_tile_ids],
                [tile_id for outcome in resumed_children for tile_id in outcome.failed_tile_ids],
                [warning for outcome in resumed_children for warning in outcome.warnings],
                [child_tile for outcome in resumed_children for child_tile in outcome.processed_tiles],
            )
        response = self.checkpoints.load_success(tile, request_hash)
        if response is None:
            artifact_dir = self.checkpoints.write_pending(tile, request_hash)
            try:
                if self.before_qwen_call is not None:
                    self.before_qwen_call()
                response = await self.client.complete_json(
                    messages=messages,
                    response_model=TileCountResponse,
                    request_meta=RequestMeta(
                        request_id=f"{sample_id}:{tile.tile_id}",
                        request_hash=request_hash,
                        prompt_version=self.counting.prompt_version,
                        sample_id=sample_id,
                        tile_id=tile.tile_id,
                        image_sha256=image_hash,
                        artifact_dir=artifact_dir,
                    ),
                )
            except Exception as error:
                self.checkpoints.write_failure(tile, request_hash, error)
                return _TileOutcome([], [], [tile.tile_id], [f"{tile.tile_id}: {type(error).__name__}"], [tile])

        try:
            self._validate_tile_response(response, tile, target)
        except Exception as error:
            self.checkpoints.write_failure(tile, request_hash, error)
            return _TileOutcome([], [], [tile.tile_id], [f"{tile.tile_id}: {type(error).__name__}"], [tile])

        points = [
            convert_local_point_to_global(
                point,
                tile,
                sample_id=sample_id,
                target=target.canonical_label,
                boundary_band_px=self.counting.boundary_band_px,
            )
            for point in response.points
        ]
        points = [apply_acceptance_policy(point, min_confidence=self.counting.min_confidence) for point in points]
        should_split = self._should_split(response)
        if should_split and self._can_split(tile):
            children = split_tile_owner_core(
                tile,
                halo_size=self.counting.halo_size,
                model_max_side=self.counting.model_max_side,
            )
            self.checkpoints.write_success(tile, request_hash, response, points, status="needs_split")
            self.checkpoints.mark_superseded(tile, request_hash, children)
            child_outcomes: list[_TileOutcome] = []
            for child in children:
                child_outcomes.append(await self._process_tile(image, child, sample_id, target))
            return _TileOutcome(
                [point for outcome in child_outcomes for point in outcome.points],
                [tile_id for outcome in child_outcomes for tile_id in outcome.succeeded_tile_ids],
                [tile_id for outcome in child_outcomes for tile_id in outcome.failed_tile_ids],
                [warning for outcome in child_outcomes for warning in outcome.warnings],
                [child_tile for outcome in child_outcomes for child_tile in outcome.processed_tiles],
            )

        warnings = ["RECURSIVE_SPLIT_LIMIT"] if should_split else []
        self.checkpoints.write_success(tile, request_hash, response, points, warnings=warnings)
        return _TileOutcome(points, [tile.tile_id], [], warnings, [tile])

    def _build_request(
        self,
        image: Image.Image,
        tile: TileSpec,
        target: CountTargetSpec,
    ) -> tuple[list[dict[str, Any]], str, str]:
        crop = crop_for_tile(image, tile)
        with io.BytesIO() as buffer:
            crop.save(buffer, format="JPEG", quality=95)
            image_bytes = buffer.getvalue()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        owner_core = _owner_core_prompt_bounds(tile)
        request_text = json.dumps(
            {
                "target_spec": target.model_dump(mode="json"),
                "tile_id": tile.tile_id,
                "owner_core_normalized": owner_core,
                "instruction": (
                    "Return exactly one point per instance whose centre is in the owner core. "
                    "Halo is context only; do not output halo-owned instances."
                ),
            },
            ensure_ascii=False,
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_bytes)}},
                    {"type": "text", "text": request_text},
                ],
            },
        ]
        request_hash = build_request_hash(
            model=self.qwen.model,
            generation={"temperature": self.qwen.temperature, "max_tokens": self.qwen.max_tokens},
            prompt_version=self.counting.prompt_version,
            messages=messages,
            image_sha256=image_hash,
            tile_geometry=tile.model_dump(mode="json"),
            target_spec=target.model_dump(mode="json"),
        )
        return messages, request_hash, image_hash

    def _validate_tile_response(self, response: TileCountResponse, tile: TileSpec, target: CountTargetSpec) -> None:
        if response.tile_id != tile.tile_id:
            raise ValueError(f"tile_id mismatch: expected {tile.tile_id}, got {response.tile_id}")
        accepted_labels = {target.canonical_label.casefold(), *(alias.casefold() for alias in target.aliases)}
        if response.target.casefold() not in accepted_labels:
            raise ValueError(f"target mismatch: expected {target.canonical_label}, got {response.target}")

    def _should_split(self, response: TileCountResponse) -> bool:
        uncertainties = {value.casefold() for value in response.uncertainty}
        return (
            response.needs_split
            or response.reported_count >= self.counting.max_points_per_tile
            or bool({"dense", "too_small"}.intersection(uncertainties))
        )

    def _can_split(self, tile: TileSpec) -> bool:
        core = tile.owner_core_global
        return (
            self.counting.recursive_split_enabled
            and tile.recursive_depth < self.counting.max_recursive_depth
            and core.width >= self.counting.min_core_size * 2
            and core.height >= self.counting.min_core_size * 2
        )

    async def _verify_seams(
        self,
        image: Image.Image,
        conflicts: Sequence[BoundaryConflict],
        points: Sequence[GlobalPointObservation],
        sample_id: str,
    ) -> tuple[dict[str, str], list[str]]:
        """Ask Qwen only for explicit local seam relations. / 仅为显式局部 seam 关系请求 Qwen。"""

        if not conflicts or not self.counting.seam_verify or not self.seam_prompt:
            return {}, []
        by_id = {point.global_id: point for point in points}
        decisions: dict[str, str] = {}
        warnings: list[str] = []
        for conflict in conflicts:
            first, second = by_id[conflict.first_global_id], by_id[conflict.second_global_id]
            margin = self.counting.seam_crop_margin_px
            left, top = max(0, min(first.global_x_px, second.global_x_px) - margin), max(0, min(first.global_y_px, second.global_y_px) - margin)
            right, bottom = min(image.width, max(first.global_x_px, second.global_x_px) + margin + 1), min(image.height, max(first.global_y_px, second.global_y_px) + margin + 1)
            crop = image.crop((left, top, right, bottom))
            with io.BytesIO() as buffer:
                crop.save(buffer, format="JPEG", quality=95)
                image_bytes = buffer.getvalue()
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": self.seam_prompt},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_to_data_url(image_bytes)}}, {"type": "text", "text": json.dumps({"conflict_id": conflict.conflict_id, "first_point": [first.global_x_px - left, first.global_y_px - top], "second_point": [second.global_x_px - left, second.global_y_px - top]}, ensure_ascii=False)}]},
            ]
            request_hash = build_request_hash(model=self.qwen.model, generation={"temperature": self.qwen.temperature, "max_tokens": self.qwen.max_tokens}, prompt_version="seam-verify-v1", messages=messages, image_sha256=hashlib.sha256(image_bytes).hexdigest())
            try:
                if self.before_qwen_call is not None:
                    self.before_qwen_call()
                decision = await self.client.complete_json(messages=messages, response_model=SeamDecision, request_meta=RequestMeta(request_id=f"{sample_id}:seam:{conflict.conflict_id}", request_hash=request_hash, prompt_version="seam-verify-v1", sample_id=sample_id, artifact_dir=self.run_dir / "seams" / hashlib.sha256(conflict.conflict_id.encode()).hexdigest()[:16]))
                if decision.conflict_id != conflict.conflict_id:
                    raise ValueError("seam conflict_id mismatch")
                decisions[conflict.conflict_id] = decision.decision
            except Exception as error:
                warnings.append(f"SEAM_VERIFY_FAILED:{conflict.conflict_id}:{type(error).__name__}")
        return decisions, warnings


def find_boundary_conflicts(
    points: Sequence[GlobalPointObservation],
    tiles: Sequence[TileSpec],
) -> list[BoundaryConflict]:
    """Find only adjacent-core, near-boundary duplicate candidates without clustering.
仅查找相邻 core 边界附近的重复候选，不执行全图聚类。
    """

    tile_by_id = {tile.tile_id: tile for tile in tiles}
    accepted = [point for point in points if point.accepted and point.near_core_boundary]
    conflicts: list[BoundaryConflict] = []
    for index, first in enumerate(accepted):
        first_tile = tile_by_id.get(first.source_tile_id)
        if first_tile is None:
            continue
        for second in accepted[index + 1 :]:
            second_tile = tile_by_id.get(second.source_tile_id)
            if second_tile is None or first.target.casefold() != second.target.casefold():
                continue
            if not _cores_are_neighbours(first_tile, second_tile):
                continue
            threshold = _conflict_threshold(first, second, first_tile)
            distance = ((first.global_x_px - second.global_x_px) ** 2 + (first.global_y_px - second.global_y_px) ** 2) ** 0.5
            if distance <= threshold:
                conflicts.append(
                    BoundaryConflict(
                        conflict_id=f"{first.global_id}|{second.global_id}",
                        first_global_id=first.global_id,
                        second_global_id=second.global_id,
                        threshold_px=threshold,
                        distance_px=distance,
                    )
                )
    return conflicts


def apply_acceptance_policy(
    point: GlobalPointObservation, *, min_confidence: float
) -> GlobalPointObservation:
    """Apply owner-core and confidence acceptance without geometry coupling. / 在不耦合几何的情况下应用 owner-core 与置信度接受策略。"""

    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")
    if not point.ownership_valid:
        return point.model_copy(update={"accepted": False, "rejection_reason": "OUTSIDE_OWNER_CORE"})
    if point.confidence < min_confidence:
        return point.model_copy(update={"accepted": False, "rejection_reason": "LOW_CONFIDENCE"})
    return point.model_copy(update={"accepted": True, "rejection_reason": None})


def finalize_representatives(
    points: Sequence[GlobalPointObservation],
    same_instance_pairs: Iterable[tuple[str, str]],
) -> tuple[list[GlobalPointObservation], list[list[str]]]:
    """Apply explicit seam merges and derive the final accepted point set.
应用显式 seam 合并，并导出最终接受点集合。
    """

    point_by_id = {point.global_id: point for point in points}
    parent = {point_id: point_id for point_id, point in point_by_id.items() if point.accepted}

    def find(point_id: str) -> str:
        while parent[point_id] != point_id:
            parent[point_id] = parent[parent[point_id]]
            point_id = parent[point_id]
        return point_id

    for first, second in same_instance_pairs:
        if first not in parent or second not in parent:
            continue
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[root_second] = root_first

    groups: dict[str, list[str]] = {}
    for point_id in parent:
        groups.setdefault(find(point_id), []).append(point_id)
    representatives: dict[str, str] = {}
    for root, group in groups.items():
        representatives[root] = min(group, key=lambda point_id: (-point_by_id[point_id].confidence, point_id))

    final_points: list[GlobalPointObservation] = []
    for point in points:
        if not point.accepted:
            final_points.append(point)
            continue
        representative = representatives[find(point.global_id)]
        if point.global_id == representative:
            final_points.append(point)
        else:
            final_points.append(point.model_copy(update={"accepted": False, "rejection_reason": "MERGED_AT_SEAM"}))
    merged_groups = [sorted(group) for group in groups.values() if len(group) > 1]
    return final_points, sorted(merged_groups)


def _owner_core_prompt_bounds(tile: TileSpec) -> list[int]:
    local = tile.owner_core_local
    width, height = tile.crop_global.width, tile.crop_global.height
    return [
        round(local.left / max(1, width - 1) * 999),
        round(local.top / max(1, height - 1) * 999),
        round((local.right - 1) / max(1, width - 1) * 999),
        round((local.bottom - 1) / max(1, height - 1) * 999),
    ]


def _cores_are_neighbours(first: TileSpec, second: TileSpec) -> bool:
    a, b = first.owner_core_global, second.owner_core_global
    horizontal_touch = a.right == b.left or b.right == a.left
    vertical_touch = a.bottom == b.top or b.bottom == a.top
    horizontal_overlap = max(a.left, b.left) < min(a.right, b.right)
    vertical_overlap = max(a.top, b.top) < min(a.bottom, b.bottom)
    corner_touch = (a.right == b.left or b.right == a.left) and (a.bottom == b.top or b.bottom == a.top)
    return (horizontal_touch and vertical_overlap) or (vertical_touch and horizontal_overlap) or corner_touch


def _conflict_threshold(first: GlobalPointObservation, second: GlobalPointObservation, tile: TileSpec) -> float:
    base = 6.0
    if first.radius_px > 0 and second.radius_px > 0:
        return min(max(min(first.radius_px, second.radius_px), base), 64.0)
    return max(base, 0.01 * min(tile.owner_core_global.width, tile.owner_core_global.height))
