"""Point-counting pipeline — tiles, model callback, recursion, final count.

点式计数流水线 — 切片、模型回调、递归分割、局部转全局、失败收集与最终
计数。pipeline 只组合纯几何与模型回调协议，不调用 Qwen/YOLO、不实现后端
选择器、不输出网页文档。最终结果完全由 accepted global points 导出。
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from PIL import Image

from agents.counting.geometry import (
    build_core_halo_tiles,
    convert_local_point_to_global,
    cores_are_neighbours,
    crop_for_tile,
    should_tile_image,
    split_tile_owner_core,
)
from agents.counting.schema import (
    CountTargetSpec,
    CountingDraft,
    CountingResult,
    GlobalPointObservation,
    IssueRecord,
    PixelRect,
    TileCountResponse,
    TileSpec,
)
from agents.counting.settings import CountingSettings


class TileCountCallback(Protocol):
    """Per-tile model callback: returns one validated point-count response for
    a tile crop. The pipeline never constructs model requests itself.
    单 tile 模型回调：为一块切片 crop 返回一条经校验的点计数响应。pipeline
    自身绝不构造模型请求。"""

    async def count_tile(
        self,
        *,
        tile: TileSpec,
        image: Image.Image,
        target: CountTargetSpec,
    ) -> TileCountResponse: ...


@dataclass(frozen=True)
class _TileOutcome:
    points: list[GlobalPointObservation]
    succeeded_tile_ids: list[str]
    failed_tile_ids: list[str]
    warnings: list[str]
    processed_tiles: list[TileSpec]


class PointCountingOrchestrator:
    """Count one image while treating accepted points as the sole truth.
    计数单张图像，并将接受点作为唯一事实来源。"""

    def __init__(
        self,
        callback: TileCountCallback,
        *,
        counting: CountingSettings,
    ) -> None:
        self.callback = callback
        self.counting = counting

    async def count_image(
        self,
        image: Image.Image,
        *,
        sample_id: str,
        question: str,
        target: CountTargetSpec,
        minimum_scan_depth: int = 0,
    ) -> CountingResult:
        """Process initial tiles and aggregate only final accepted points;
        final_count is derived from accepted global points and never from an
        independent model field.
        按行优先处理初始切片并只聚合最终接受点；final_count 由接受点导出，
        绝不来自独立的模型字段。"""
        draft = await self.collect_points(
            image,
            sample_id=sample_id,
            question=question,
            target=target,
            minimum_scan_depth=minimum_scan_depth,
        )
        warnings = list(draft.warnings)
        if self.counting.seam_verify:
            conflicts = find_boundary_conflicts(
                draft.raw_global_points, draft.processed_tiles
            )
            pairs, unresolved = decide_seam_pairs(conflicts)
            final_points, merged_groups = finalize_representatives(
                draft.raw_global_points, pairs
            )
            if merged_groups:
                warnings.append(
                    IssueRecord(
                        code="COUNTING_SEAM_DUPLICATE_MERGED",
                        message=(
                            f"Merged {len(merged_groups)} strongly matching "
                            "seam duplicate groups."
                        ),
                        point_ids=[
                            point_id for group in merged_groups for point_id in group
                        ],
                    )
                )
            for first, second in unresolved:
                warnings.append(
                    IssueRecord(
                        code="COUNTING_SEAM_CONFLICT_UNRESOLVED",
                        message=(
                            f"Seam duplicate candidate retained for review: "
                            f"{first}|{second}"
                        ),
                        point_ids=[first, second],
                    )
                )
            unresolved_conflicts = [f"{first}|{second}" for first, second in unresolved]
        else:
            # Seam finalization disabled by configuration; record it visibly.
            # seam 最终化被配置禁用；显式记录。
            final_points = list(draft.raw_global_points)
            merged_groups: list[list[str]] = []
            unresolved_conflicts: list[str] = []
            warnings.append(
                IssueRecord(
                    code="SEAM_VERIFY_DISABLED",
                    message="seam_verify=False; seam conflicts are not finalized.",
                )
            )
        if draft.failed_tiles and draft.succeeded_tiles:
            status: Literal["completed", "completed_with_warnings", "partial", "failed"] = "partial"
        elif draft.failed_tiles:
            status = "failed"
        elif warnings:
            status = "completed_with_warnings"
        else:
            status = "completed"
        return CountingResult(
            sample_id=sample_id,
            target=target.canonical_label,
            question=question,
            source_width=draft.source_width,
            source_height=draft.source_height,
            tile_count=len(draft.processed_tiles),
            initial_tile_count=draft.initial_tile_count,
            leaf_tile_count=len(draft.processed_tiles),
            succeeded_tiles=draft.succeeded_tiles,
            failed_tiles=draft.failed_tiles,
            global_points=final_points,
            merged_groups=merged_groups,
            unresolved_conflicts=unresolved_conflicts,
            warnings=warnings,
            final_count=sum(point.accepted for point in final_points),
            status=status,
        )

    async def collect_points(
        self,
        image: Image.Image,
        *,
        sample_id: str,
        question: str,
        target: CountTargetSpec,
        minimum_scan_depth: int = 0,
    ) -> CountingDraft:
        """Collect accepted-policy tile points without seam finalization.
        收集经过接受策略的 tile 点（不做 seam 最终化）。"""
        normalized = image.convert("RGB")
        if should_tile_image(
            normalized.width,
            normalized.height,
            model_max_side=self.counting.model_max_side,
            max_pixels_without_tiling=self.counting.max_pixels_without_tiling,
        ):
            tiles = build_core_halo_tiles(
                normalized.width,
                normalized.height,
                core_size=self.counting.tile_core_size,
                halo_size=self.counting.halo_size,
                model_max_side=self.counting.model_max_side,
            )
        else:
            tiles = [_whole_image_tile(normalized.width, normalized.height)]
        outcomes = await self._run_tiles(
            tiles,
            normalized,
            sample_id=sample_id,
            target=target,
            minimum_scan_depth=minimum_scan_depth,
        )
        points = [point for outcome in outcomes for point in outcome.points]
        failed_tiles = [
            tile_id for outcome in outcomes for tile_id in outcome.failed_tile_ids
        ]
        succeeded_tiles = [
            tile_id for outcome in outcomes for tile_id in outcome.succeeded_tile_ids
        ]
        processed_tiles = [tile for outcome in outcomes for tile in outcome.processed_tiles]
        warning_records = [
            IssueRecord(
                code="TILE_FAILURE" if warning.startswith("TILE_FAILURE:") else "TILE_WARNING",
                message=warning,
            )
            for outcome in outcomes
            for warning in outcome.warnings
        ]
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
            warnings=warning_records,
        )

    async def _run_tiles(
        self,
        tiles: Sequence[TileSpec],
        image: Image.Image,
        *,
        sample_id: str,
        target: CountTargetSpec,
        minimum_scan_depth: int,
    ) -> list[_TileOutcome]:
        """Run tiles sequentially by default; honour the configured
        concurrency when sequential is disabled. 默认顺序执行；sequential
        关闭时按配置并发执行。"""
        if self.counting.sequential or self.counting.concurrency <= 1:
            return [
                await self._process_tile(
                    image,
                    tile,
                    sample_id=sample_id,
                    target=target,
                    minimum_scan_depth=minimum_scan_depth,
                )
                for tile in tiles
            ]
        semaphore = asyncio.Semaphore(self.counting.concurrency)

        async def _guarded(tile: TileSpec) -> _TileOutcome:
            async with semaphore:
                return await self._process_tile(
                    image,
                    tile,
                    sample_id=sample_id,
                    target=target,
                    minimum_scan_depth=minimum_scan_depth,
                )

        return list(await asyncio.gather(*(_guarded(tile) for tile in tiles)))

    async def _process_tile(
        self,
        image: Image.Image,
        tile: TileSpec,
        *,
        sample_id: str,
        target: CountTargetSpec,
        minimum_scan_depth: int,
    ) -> _TileOutcome:
        """Process one tile: model callback → validation → conversion →
        acceptance → optional recursive split. Callback failures are recorded,
        never silently dropped. 处理一块切片：模型回调 → 校验 → 换算 →
        接受策略 → 可选递归分割。回调失败会被记录，绝不静默丢弃。"""
        try:
            response = await self.callback.count_tile(
                tile=tile,
                image=crop_for_tile(image, tile),
                target=target,
            )
        except Exception as error:
            return _TileOutcome(
                [],
                [],
                [tile.tile_id],
                [f"TILE_FAILURE:{tile.tile_id}:{type(error).__name__}"],
                [tile],
            )
        try:
            self._validate_tile_response(response, tile, target)
        except Exception as error:
            return _TileOutcome(
                [],
                [],
                [tile.tile_id],
                [f"TILE_FAILURE:{tile.tile_id}:{type(error).__name__}"],
                [tile],
            )
        points = [
            apply_acceptance_policy(
                convert_local_point_to_global(
                    point,
                    tile,
                    sample_id=sample_id,
                    target=target.canonical_label,
                    boundary_band_px=self.counting.boundary_band_px,
                ),
                min_confidence=self.counting.min_confidence,
            )
            for point in response.points
        ]
        should_split = self._should_split(response, tile, minimum_scan_depth)
        if should_split and self._can_split(tile):
            children = split_tile_owner_core(
                tile,
                halo_size=_child_halo_size(
                    self.counting.halo_size, tile.recursive_depth + 1
                ),
                model_max_side=self.counting.model_max_side,
            )
            child_outcomes = await self._run_tiles(
                children,
                image,
                sample_id=sample_id,
                target=target,
                minimum_scan_depth=minimum_scan_depth,
            )
            return _TileOutcome(
                [point for outcome in child_outcomes for point in outcome.points],
                [tile_id for outcome in child_outcomes for tile_id in outcome.succeeded_tile_ids],
                [tile_id for outcome in child_outcomes for tile_id in outcome.failed_tile_ids],
                [warning for outcome in child_outcomes for warning in outcome.warnings],
                [child_tile for outcome in child_outcomes for child_tile in outcome.processed_tiles],
            )
        warnings = ["RECURSIVE_SPLIT_LIMIT"] if should_split else []
        return _TileOutcome(points, [tile.tile_id], [], warnings, [tile])

    @staticmethod
    def _validate_tile_response(
        response: TileCountResponse,
        tile: TileSpec,
        target: CountTargetSpec,
    ) -> None:
        """Reject tile/target mismatches; the response schema already enforces
        reported_count == len(points) and unique local ids.
        拒绝 tile/target 不匹配；reported_count == len(points) 与 local_id
        唯一性已由响应 Schema 强制。"""
        if response.tile_id != tile.tile_id:
            raise ValueError(
                f"tile_id mismatch: expected {tile.tile_id}, got {response.tile_id}"
            )
        accepted_labels = {
            _normalize_target_label(target.canonical_label),
            *(_normalize_target_label(alias) for alias in target.aliases),
        }
        if _normalize_target_label(response.target) not in accepted_labels:
            raise ValueError(
                f"target mismatch: expected {target.canonical_label}, got {response.target}"
            )

    def _should_split(
        self,
        response: TileCountResponse,
        tile: TileSpec,
        minimum_scan_depth: int,
    ) -> bool:
        uncertainties = {value.casefold() for value in response.uncertainty}
        return (
            tile.recursive_depth < minimum_scan_depth
            or response.needs_split
            or response.reported_count >= self.counting.max_points_per_tile
            or bool({"dense", "too_small", "zero_unconfirmed"}.intersection(uncertainties))
        )

    def _can_split(self, tile: TileSpec) -> bool:
        core = tile.owner_core_global
        return (
            self.counting.recursive_split_enabled
            and tile.recursive_depth < self.counting.max_recursive_depth
            and core.width >= self.counting.min_core_size * 2
            and core.height >= self.counting.min_core_size * 2
        )


def apply_acceptance_policy(
    point: GlobalPointObservation,
    *,
    min_confidence: float,
) -> GlobalPointObservation:
    """Apply owner-core and confidence acceptance without geometry coupling.
    在不耦合几何的情况下应用 owner-core 与置信度接受策略。"""

    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")
    if not point.ownership_valid:
        return point.model_copy(
            update={"accepted": False, "rejection_reason": "OUTSIDE_OWNER_CORE"}
        )
    if point.confidence < min_confidence:
        return point.model_copy(
            update={"accepted": False, "rejection_reason": "LOW_CONFIDENCE"}
        )
    return point.model_copy(update={"accepted": True, "rejection_reason": None})


def decide_seam_pairs(
    conflicts: Sequence[Any],
    *,
    merge_distance_factor: float = 0.5,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Conservatively decide which boundary conflicts are strong enough to
    auto-merge and which stay unresolved for review. Only adjacent-core,
    near-boundary candidates reach this point; merging additionally requires
    a very small centre distance. 保守决定哪些边界冲突足够强可自动合并、哪些
    保留为待复核未解决。只有相邻 core 的边界候选能到达此处；合并还要求中心
    距离非常小。"""
    merged: list[tuple[str, str]] = []
    unresolved: list[tuple[str, str]] = []
    for conflict in conflicts:
        first = conflict["first_global_id"]
        second = conflict["second_global_id"]
        distance = float(conflict["distance_px"])
        threshold = float(conflict["threshold_px"])
        if distance <= threshold * merge_distance_factor:
            merged.append((first, second))
        else:
            unresolved.append((first, second))
    return merged, unresolved


def finalize_representatives(
    points: Sequence[GlobalPointObservation],
    same_instance_pairs: Any,
) -> tuple[list[GlobalPointObservation], list[list[str]]]:
    """Apply explicit same-instance merges and derive the final accepted
    point set via union-find; each group keeps its highest-confidence point.
    通过并查集应用显式同实例合并并导出最终接受点集合；每组保留置信度最高
    的点。"""
    point_by_id = {point.global_id: point for point in points}
    parent = {
        point_id: point_id
        for point_id, point in point_by_id.items()
        if point.accepted
    }

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
        representatives[root] = min(
            group, key=lambda point_id: (-point_by_id[point_id].confidence, point_id)
        )

    final_points: list[GlobalPointObservation] = []
    for point in points:
        if not point.accepted:
            final_points.append(point)
            continue
        representative = representatives[find(point.global_id)]
        if point.global_id == representative:
            final_points.append(point)
        else:
            final_points.append(
                point.model_copy(
                    update={"accepted": False, "rejection_reason": "MERGED_AT_SEAM"}
                )
            )
    merged_groups = [sorted(group) for group in groups.values() if len(group) > 1]
    return final_points, sorted(merged_groups)


def find_boundary_conflicts(
    points: Sequence[GlobalPointObservation],
    tiles: Sequence[TileSpec],
) -> list[Any]:
    """Find only adjacent-core, near-boundary duplicate candidates without
    clustering. 仅查找相邻 core 边界附近的重复候选，不执行全图聚类。"""
    tile_by_id = {tile.tile_id: tile for tile in tiles}
    accepted = [
        point for point in points if point.accepted and point.near_core_boundary
    ]
    conflicts: list[Any] = []
    for index, first in enumerate(accepted):
        first_tile = tile_by_id.get(first.source_tile_id)
        if first_tile is None:
            continue
        for second in accepted[index + 1:]:
            second_tile = tile_by_id.get(second.source_tile_id)
            if second_tile is None or first.target.casefold() != second.target.casefold():
                continue
            if not _cores_are_neighbours(first_tile, second_tile):
                continue
            threshold = _conflict_threshold(first, second, first_tile)
            distance = (
                (first.global_x_px - second.global_x_px) ** 2
                + (first.global_y_px - second.global_y_px) ** 2
            ) ** 0.5
            if distance <= threshold:
                conflicts.append(
                    {
                        "conflict_id": f"{first.global_id}|{second.global_id}",
                        "first_global_id": first.global_id,
                        "second_global_id": second.global_id,
                        "threshold_px": threshold,
                        "distance_px": distance,
                    }
                )
    return conflicts


def _whole_image_tile(width: int, height: int) -> TileSpec:
    """Build the single no-tiling tile covering the whole image.
    构建覆盖整图的不切片单 tile。"""
    whole = PixelRect(left=0, top=0, right=width, bottom=height)
    return TileSpec(
        tile_id="whole",
        row=0,
        col=0,
        crop_global=whole,
        owner_core_global=whole,
        owner_core_local=whole,
        source_width=width,
        source_height=height,
        model_input_width=width,
        model_input_height=height,
    )


def _child_halo_size(base_halo: int, child_depth: int) -> int:
    """Reduce halo as recursive crops become smaller so each pass gains
    detail. 随递归裁剪缩小 halo，使每轮复查获得真正更细的局部细节。"""
    return max(0, base_halo // (2 ** max(0, child_depth)))


def _normalize_target_label(value: str) -> str:
    """Normalize harmless target spelling variants before strict identity
    checks. 在严格类别一致性检查前规范化无害的目标拼写差异。"""
    import re

    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    parts = normalized.split("-")
    if parts and parts[-1].endswith("s") and len(parts[-1]) > 1:
        parts[-1] = parts[-1][:-1]
    return "-".join(parts)


def _cores_are_neighbours(first: TileSpec, second: TileSpec) -> bool:
    return cores_are_neighbours(first, second)


def _conflict_threshold(
    first: GlobalPointObservation,
    second: GlobalPointObservation,
    tile: TileSpec,
) -> float:
    base = 6.0
    if first.radius_px > 0 and second.radius_px > 0:
        return min(max(min(first.radius_px, second.radius_px), base), 64.0)
    return max(
        base, 0.01 * min(tile.owner_core_global.width, tile.owner_core_global.height)
    )
