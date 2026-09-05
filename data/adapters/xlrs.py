"""Offline-first XLRS-Bench adapter (caption / grounding / VQA-lite).

离线优先的 XLRS-Bench 适配器（caption / grounding / VQA-lite）。
- 官方 release 目录可从统一 XLRS 根解析；官方 split 被强制；
- 本地优先：HF disk 布局走 load_from_disk；仅 allow_download=True 才走下载；
  probe 绝不隐式下载、绝不使用假 sample_count；
- 惰性流式：加载器只返回可 len()/迭代的容器（datasets.Dataset），绝不转成
  list 驻留内存；probe 只物化前 20 行 + len()，iter_samples 逐行流式 yield；
- 每行图片统一解析根：全部 path-backed → release root；任一图片需要物化 →
  整行统一物化/复制到外部 cache（内容哈希命名、原子写、不写 dataset root）；
- 每样本 metadata.image_root_kind 明确解析根，不依赖迭代器状态；
- XLRS-Bench-lite 能力收窄；multi-answer 只认显式字段或审计类型。
"""

from __future__ import annotations

import base64
import binascii
import itertools
import json
import re
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from data.adapters.base import AdapterProbe, DatasetProbeError
from data.adapters.xlrs_image_cache import (
    ImageMaterializationError,
    cache_existing_path,
    materialize_image,
)
from data.schema import (
    GroundTruth,
    ImageRef,
    TaskNormalization,
    UnifiedSample,
    stable_sample_id,
)

ADAPTER_VERSION = "hf-disk-v1"
_VQA_HF_ADAPTER_VERSION = "hf-disk-v2-question-with-choices"
_JSONL_ADAPTER_VERSION = "sharded-jsonl-base64-v2-question-with-choices"
# Official Hugging Face releases per task. / 各任务的官方 Hugging Face 发布。
HF_REPOS = {
    "caption": "initiacms/XLRS-Bench_caption_en",
    "grounding": "initiacms/XLRS-Bench_visual_grounding_en",
    "multiple_choice_vqa": "initiacms/XLRS-Bench-lite",
}
# Official release directory names under a unified XLRS root. / 统一根下的官方目录名。
RELEASE_DIRS = {
    "multiple_choice_vqa": "XLRS-Bench-lite",
    "caption": "XLRS-Bench_caption_en",
    "grounding": "XLRS-Bench_visual_grounding_en",
}
# Official release splits; caller splits are normalized, never stored verbatim.
# 官方 release split；调用者 split 被规范化，绝不原样写入样本。
RELEASE_SPLITS = {
    "caption": "train",
    "grounding": "test",
    "multiple_choice_vqa": "train",
}
_SPLIT_ALIASES = {
    "train": "train",
    "test": "test",
    "testing": "test",
}
SUPPORTED_TASKS = frozenset(HF_REPOS)
_IMAGE_KEYS = ("image", "Image", "image_a", "image_b", "before", "after", "images")
_CHOICE_KEYS = ("choices", "options", "answer_choices", "multi-choice options")
_OPTION_LETTERS = ("A", "B", "C", "D", "E")
_GROUNDING_BOX_KEYS = ("bbox", "box", "boxes", "polygon", "answer")
_GROUNDING_LABEL_KEYS = ("name", "label", "class", "category")
_EXPLICIT_MULTI_ANSWER_KEYS = ("allow_multiple", "multi_answer")
# Values proven from the audited official release only.
# 仅包含从官方发布中审计确认的值。
_AUDITED_MULTI_ANSWER_TYPES = frozenset({"overall land use classification"})
_ANSWER_SEPARATOR = re.compile(r"[\s,，、]+")
_CAPTION_TEXT_KEYS = ("caption", "text", "raw")
_JSONL_PART_PATTERN = "XLRS-Bench-lite_part*.jsonl"
_JSONL_PART_NUMBER = re.compile(r"XLRS-Bench-lite_part(\d+)\.jsonl\Z")
_LABELED_CHOICE_LINE = re.compile(r"^\s*\(([A-E])\)\s*(\S.*)\s*$")
_CHOICE_PREFIX = re.compile(r"^\s*(?:\([A-E]\)|[A-E][.)、:-])\s*", re.IGNORECASE)


class _CaptionJsonRows:
    """Rows from the extracted ``train/captions.json`` release.

    The JSON annotations contain paths and text only, never decoded image
    payloads. Keeping this small annotation table in memory is bounded by the
    annotation file, while image loading remains per-row and on demand.
    从解压后的 ``train/captions.json`` 发布读取行。JSON 标注只含路径和文本，
    不含解码后的图片载荷；内存占用受标注文件大小约束，图片仍按行按需加载。
    """

    def __init__(self, path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise DatasetProbeError(
                f"invalid XLRS caption annotations at {path}"
            ) from error
        if not isinstance(payload, list) or any(
            not isinstance(row, dict) for row in payload
        ):
            raise DatasetProbeError(
                f"XLRS caption annotations at {path} must be a JSON array of objects"
            )
        self._rows = payload

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._rows)


class _ShardedJsonlRows:
    """Lazy rows from the VLM-exported XLRS-lite JSONL partitions.

    Only one JSON object and its decoded image are resident at a time. The
    source files remain read-only; decoded bytes flow through the existing
    content-addressed external image cache.
    从 VLM 导出的 XLRS-lite JSONL 分片惰性读取行。内存中一次只保留一个
    JSON 对象及其解码图片；源文件保持只读，解码字节进入既有外部内容寻址缓存。
    """

    def __init__(self, root: Path) -> None:
        parts: list[tuple[int, Path]] = []
        for path in root.glob(_JSONL_PART_PATTERN):
            match = _JSONL_PART_NUMBER.fullmatch(path.name)
            if match is not None and path.is_file():
                parts.append((int(match.group(1)), path))
        self._parts = tuple(path for _, path in sorted(parts))
        if not self._parts:
            raise DatasetProbeError(f"no XLRS-lite JSONL partitions under {root}")
        self._length: int | None = None

    def __len__(self) -> int:
        if self._length is None:
            self._length = sum(_count_nonempty_lines(path) for path in self._parts)
        return self._length

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for path in self._parts:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except (UnicodeError, json.JSONDecodeError) as error:
                        raise DatasetProbeError(
                            f"invalid XLRS-lite JSONL row at {path.name}:{line_number}"
                        ) from error
                    if not isinstance(row, dict):
                        raise DatasetProbeError(
                            f"XLRS-lite JSONL row at {path.name}:{line_number} must be an object"
                        )
                    yield _decode_jsonl_images(row, path.name, line_number)


def _count_nonempty_lines(path: Path) -> int:
    """Count records without parsing or retaining the large Base64 payloads.
    不解析、不保留大型 Base64 载荷，仅统计记录数。"""
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def _decode_jsonl_images(
    row: dict[str, Any], part_name: str, line_number: int
) -> dict[str, Any]:
    """Decode only the declared JSONL image list into HF-style byte features.
    仅将声明的 JSONL 图片列表解码为 HF 风格 bytes feature。"""
    values = row.get("image")
    if not isinstance(values, list) or not values:
        raise DatasetProbeError(
            f"XLRS-lite JSONL row at {part_name}:{line_number} has no image list"
        )
    decoded: list[dict[str, bytes]] = []
    for position, value in enumerate(values):
        if not isinstance(value, str) or not value:
            raise DatasetProbeError(
                f"XLRS-lite JSONL image {position} at {part_name}:{line_number} "
                "must be a non-empty Base64 string"
            )
        try:
            payload = base64.b64decode(value, validate=True)
        except (ValueError, binascii.Error) as error:
            raise DatasetProbeError(
                f"invalid XLRS-lite Base64 image at {part_name}:{line_number}"
            ) from error
        if not payload:
            raise DatasetProbeError(
                f"empty XLRS-lite Base64 image at {part_name}:{line_number}"
            )
        decoded.append({"bytes": payload})
    safe = dict(row)
    safe["image"] = decoded
    source_index = row.get("index")
    if source_index is not None:
        safe["source_id"] = str(source_index)
    safe["source_partition"] = part_name
    return safe

class LazyRows(Protocol):
    """Minimal lazy row container: cheap len() plus streaming iteration.
    datasets.Dataset satisfies this structurally; lists do too (tests inject
    lists). Never materialize the whole container into a list — XLRS rows
    carry image bytes that would explode memory.
    最小惰性行容器：廉价 len() 与流式迭代。datasets.Dataset 结构满足该协议，
    list 也满足（测试注入 list）。绝不把整个容器物化成 list——XLRS 行携带
    图片 bytes，整体物化会导致内存爆炸。"""

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[dict[str, Any]]: ...


RowLoader = Callable[[Path, str], LazyRows]


@dataclass(frozen=True)
class _ResolvedImage:
    """One resolved image: local file path plus its JSON-safe descriptor.
    一条已解析图片：本地文件路径与其 JSON 安全描述符。"""

    path: Path
    descriptor: dict[str, Any]


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _first_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _normalize_split(split: str) -> str:
    canonical = _SPLIT_ALIASES.get(split.strip().lower())
    if canonical is None:
        raise DatasetProbeError(
            f"unknown XLRS split {split!r}; supported={sorted(set(_SPLIT_ALIASES))}"
        )
    return canonical


class XLRSAdapter:
    """Read-only, offline-first adapter over the XLRS-Bench releases.
    只读、离线优先的 XLRS-Bench 发布适配器。"""

    name = "XLRS-Bench"
    supported_tasks = SUPPORTED_TASKS

    def __init__(
        self,
        *,
        name: str | None = None,
        allow_download: bool = False,
        dataset_loader: RowLoader | None = None,
        cache_root: Path | None = None,
        supported_tasks: frozenset[str] | None = None,
    ) -> None:
        """allow_download defaults to False; downloading requires an explicit opt-in.
        allow_download 默认 False；下载必须显式开启。cache_root 用于 bytes/PIL
        图片物化（默认系统临时目录下的稳定子目录）；supported_tasks 可收窄能力。"""
        self.name = name or self.name
        self.allow_download = allow_download
        self._loader = dataset_loader
        self._cache_root = cache_root
        self.supported_tasks = (
            frozenset(supported_tasks) if supported_tasks is not None else SUPPORTED_TASKS
        )

    def _effective_cache_root(self) -> Path:
        """Stable default user cache; created only when bytes/PIL are handled.
        稳定默认用户 cache；仅处理 bytes/PIL 时才创建。"""
        if self._cache_root is not None:
            return self._cache_root
        return Path(tempfile.gettempdir()) / "m3-xlrs-image-cache"

    # ── probe / 探测 ────────────────────────────────────────────────────────

    def probe(self, root: Path, task: str | None = None) -> AdapterProbe:
        """Report verified local evidence; never downloads, never fake counts.
        报告经本地验证的证据；绝不下载、绝不使用假计数。"""
        if task is not None:
            if task not in self.supported_tasks:
                raise DatasetProbeError(
                    f"XLRS-Bench does not support task={task!r}; "
                    f"supported={sorted(self.supported_tasks)}"
                )
            release_root = self._resolve_release_root(root, task)
            rows = self._rows_local_only(release_root, task)
            if not rows:
                raise DatasetProbeError(
                    f"zero XLRS records for task={task!r} under {release_root}"
                )
            # Only the first 20 rows are materialized for field discovery;
            # the container itself is never turned into a list.
            # 字段发现只物化前 20 行；容器本身绝不转成 list。
            observed = tuple(
                sorted({key for row in itertools.islice(rows, 20) for key in row})
            )
            version = ADAPTER_VERSION
            if task == "multiple_choice_vqa":
                version = (
                    _JSONL_ADAPTER_VERSION
                    if _has_jsonl_parts(release_root)
                    else _VQA_HF_ADAPTER_VERSION
                )
            return AdapterProbe(
                dataset=self.name,
                version=version,
                sample_file=release_root / "dataset_dict.json"
                if (release_root / "dataset_dict.json").is_file()
                else release_root,
                observed_fields=observed,
                sample_count=len(rows),
                task=task,
                available_tasks=(task,),
            )
        available: list[str] = []
        counts: dict[str, int] = {}
        for candidate in sorted(self.supported_tasks):
            try:
                release_root = self._resolve_release_root(root, candidate)
            except DatasetProbeError:
                continue
            rows = self._rows_local_only(release_root, candidate)
            if not rows:
                raise DatasetProbeError(
                    f"zero XLRS records for task={candidate!r} under {release_root}"
                )
            available.append(candidate)
            counts[candidate] = len(rows)
        if not available:
            raise DatasetProbeError(
                f"offline: no local XLRS release under {root}; "
                "pass allow_download=True to download from Hugging Face"
            )
        primary = available[0]
        return AdapterProbe(
            dataset=self.name,
            version=ADAPTER_VERSION,
            sample_file=root,
            observed_fields=("local",),
            sample_count=counts[primary],
            available_tasks=tuple(available),
        )

    # ── iter_samples / 样本迭代 ────────────────────────────────────────────

    def iter_samples(self, root: Path, split: str, task: str) -> Iterator[UnifiedSample]:
        """Yield unified samples for one task; official splits are enforced.
        产出某一任务的统一样本；官方 split 被强制；零行显式失败。"""
        if task not in self.supported_tasks:
            raise DatasetProbeError(
                f"XLRS-Bench does not support task={task!r}; "
                f"supported={sorted(self.supported_tasks)}"
            )
        canonical_split = _normalize_split(split)
        official_split = RELEASE_SPLITS[task]
        if canonical_split != official_split:
            raise DatasetProbeError(
                f"XLRS {task} requires split={official_split!r}, got {split!r}"
            )
        release_root = self._resolve_release_root(root, task)
        rows = self._rows(release_root, task)
        if not rows:
            raise DatasetProbeError(
                f"zero XLRS records for task={task!r} under {release_root}"
            )
        for index, row in enumerate(rows):
            safe_row, refs, image_root_kind = self._sanitize_row(
                row, release_root=release_root, index=index
            )
            image_root = (
                self._effective_cache_root()
                if image_root_kind == "cache"
                else release_root
            )
            images = self._image_refs(refs, index, image_root=image_root)
            if task == "caption":
                yield self._caption_sample(
                    safe_row, images, official_split, index,
                    image_root_kind=image_root_kind,
                )
            elif task == "grounding":
                yield self._grounding_sample(
                    safe_row, images, official_split, index,
                    image_root_kind=image_root_kind,
                )
            else:
                yield self._vqa_lite_sample(
                    safe_row, images, official_split, index,
                    image_root_kind=image_root_kind,
                )

    def resolve_image_root(self, dataset_root: Path, task: str) -> Path:
        """Return the task-level root used to resolve emitted ImageRef paths.

        Sharded JSONL VQA rows are uniformly materialized in the external
        cache; path-backed releases resolve against their concrete
        release directory. 返回解析该任务所产出 ImageRef 的任务级根目录。
        JSONL/HF bytes VQA 行统一物化到外部 cache；路径型发布相对其具体
        release 目录解析。
        """
        release_root = self._resolve_release_root(dataset_root, task)
        if task == "multiple_choice_vqa" and _has_jsonl_parts(release_root):
            return self._effective_cache_root()
        return release_root

    # ── loading strategy / 加载策略 ─────────────────────────────────────────

    def _has_local(self, root: Path, task: str) -> bool:
        split = RELEASE_SPLITS[task]
        if (root / "dataset_dict.json").is_file() or (
            root / split / "state.json"
        ).is_file():
            return True
        if task == "multiple_choice_vqa" and any(
            root.glob(_JSONL_PART_PATTERN)
        ):
            return True
        return task == "caption" and (
            root / split / "captions.json"
        ).is_file()

    def _resolve_release_root(self, root: Path, task: str) -> Path:
        if self._has_local(root, task):
            return root
        candidate = root / RELEASE_DIRS[task]
        if self._has_local(candidate, task):
            return candidate
        if self.allow_download:
            # Without local data the caller-supplied root acts as the logical
            # release root; the hub loader is used in _rows().
            # 无本地数据时，调用者 root 作为逻辑 release 根；_rows() 使用 hub loader。
            return root
        raise DatasetProbeError(
            f"no local XLRS {task} release under {root} "
            f"(tried {root} and {root / RELEASE_DIRS[task]})"
        )

    def _rows(self, release_root: Path, task: str) -> LazyRows:
        if self._has_local(release_root, task):
            loader = self._loader or self._load_from_disk
            return loader(release_root, task)
        if not self.allow_download:
            raise DatasetProbeError(
                f"offline: no local XLRS {task} release under {release_root}; "
                "pass allow_download=True to download from Hugging Face"
            )
        loader = self._loader or self._load_from_hub
        return loader(release_root, task)

    def _rows_local_only(self, release_root: Path, task: str) -> LazyRows:
        """Local-only row loading used by probe(); probe never downloads and
        never materializes the full container. probe 使用的仅本地加载；probe
        绝不下载，也绝不物化整个容器。"""
        if not self._has_local(release_root, task):
            raise DatasetProbeError(
                f"offline: no local XLRS {task} release under {release_root}"
            )
        loader = self._loader or self._load_from_disk
        return loader(release_root, task)

    @staticmethod
    def _load_from_disk(release_root: Path, task: str) -> LazyRows:
        """Load an extracted caption release or local HF layout.

        Extracted caption releases use ``train/captions.json`` plus
        ``train/images`` and need no optional dependency. HF releases remain
        lazy and import datasets only when selected.
        加载解压后的 caption 发布或本地 HF 布局。解压发布使用
        ``train/captions.json`` 与 ``train/images``，无需可选依赖；HF 发布仍
        保持惰性，并仅在选中该布局时导入 datasets。

        Returns the datasets.Dataset itself (cheap len(), per-row streaming
        iteration) instead of a materialized list — rows carry image bytes
        that would explode memory if the whole table were converted to dicts.
        惰性加载本地 HF release 布局；datasets 延迟导入。直接返回
        datasets.Dataset 本身（廉价 len()、逐行流式迭代），不再转成 list——
        行内图片 bytes 若整体转 dict 会使内存爆炸。"""
        split = RELEASE_SPLITS[task]
        annotations = release_root / split / "captions.json"
        if task == "caption" and annotations.is_file():
            return _CaptionJsonRows(annotations)
        if task == "multiple_choice_vqa" and any(
            release_root.glob(_JSONL_PART_PATTERN)
        ):
            return _ShardedJsonlRows(release_root)
        try:
            from datasets import Image as HFImage, load_from_disk
        except ImportError as error:
            raise RuntimeError(
                "Install the datasets package to load local XLRS releases."
            ) from error
        if (release_root / "dataset_dict.json").is_file():
            dataset = load_from_disk(release_root)[split]
        else:
            dataset = load_from_disk(release_root / split)
        # Keep Arrow image payloads compressed until a selected sample reaches
        # the planner. Decoding every 10000x10000 image to PIL and re-encoding
        # it as PNG during sample-id selection is prohibitively expensive.
        if "image" in dataset.column_names:
            dataset = dataset.cast_column("image", HFImage(decode=False))
        return dataset

    @staticmethod
    def _load_from_hub(release_root: Path, task: str) -> LazyRows:
        """Download from Hugging Face; only reachable with allow_download=True.
        Returns the datasets.Dataset lazily like the disk loader.
        从 Hugging Face 下载；仅 allow_download=True 时可达。与 disk 加载器
        一样惰性返回 datasets.Dataset。"""
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError(
                "Install the datasets package to download XLRS releases."
            ) from error
        dataset = load_dataset(HF_REPOS[task], split=RELEASE_SPLITS[task])
        return dataset

    # ── row sanitation / 行清洗 ─────────────────────────────────────────────

    def _sanitize_row(
        self,
        row: dict[str, Any],
        *,
        release_root: Path,
        index: int,
    ) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]], str]:
        """Strip non-JSON values and materialize image features. Every image of
        one row shares a single resolution root decided from the ACTUAL
        materialization results (descriptor.image_source_type), never from the
        raw Python type of the input value.
        去除非 JSON 值并物化图片特征。一行内所有图片共享单一解析根——
        该根由实际物化结果（descriptor.image_source_type）决定，绝不按输入
        值的 Python 类型判断。"""
        image_groups: dict[str, list[Any]] = {}
        for key in _IMAGE_KEYS:
            value = row.get(key)
            if value is not None:
                image_groups[key] = value if isinstance(value, list) else [value]

        all_resolved: list[tuple[str, _ResolvedImage]] = []
        for key in _IMAGE_KEYS:
            for item in image_groups.get(key, []):
                resolved = self._resolve_row_images(
                    [(key, item)], release_root=release_root, index=index
                )
                all_resolved.append((key, resolved[0]))
        unified, image_root_kind = self._unify_row_image_root(
            [resolved for _, resolved in all_resolved],
            release_root=release_root,
            index=index,
        )
        # Re-group unified images back to their source keys, preserving order.
        # 将统一后的图片按源键分组，保持顺序。
        unified_by_key: dict[str, list[_ResolvedImage]] = {}
        for (key, _original), resolved in zip(all_resolved, unified):
            unified_by_key.setdefault(key, []).append(resolved)

        safe: dict[str, Any] = {}
        refs: list[tuple[Path, dict[str, Any]]] = []
        excluded: list[str] = []
        for key, value in row.items():
            if key in image_groups:
                descriptors = [
                    resolved.descriptor for resolved in unified_by_key.get(key, [])
                ]
                if isinstance(value, list):
                    safe[key] = descriptors
                else:
                    safe[key] = descriptors[0] if descriptors else {"image_present": False}
                refs.extend(
                    (resolved.path, resolved.descriptor)
                    for resolved in unified_by_key.get(key, [])
                )
            elif _is_json_safe(value):
                safe[key] = value
            else:
                excluded.append(key)
        if excluded:
            safe["excluded_fields"] = sorted(excluded)
        return safe, refs, image_root_kind

    def _resolve_row_images(
        self,
        image_values: list[tuple[str, Any]],
        *,
        release_root: Path,
        index: int,
    ) -> list[_ResolvedImage]:
        """Resolve every image through materialize_image(); errors become
        DatasetProbeError. 通过 materialize_image() 解析每张图片；错误统一
        转换为 DatasetProbeError。"""
        resolved: list[_ResolvedImage] = []
        for _key, item in image_values:
            try:
                path, descriptor = materialize_image(
                    item,
                    release_root=release_root,
                    cache_root=self._effective_cache_root(),
                    index=index,
                )
            except ImageMaterializationError as error:
                raise DatasetProbeError(str(error)) from error
            resolved.append(_ResolvedImage(path=path, descriptor=descriptor))
        return resolved

    def _unify_row_image_root(
        self,
        resolved: list[_ResolvedImage],
        *,
        release_root: Path,
        index: int,
    ) -> tuple[list[_ResolvedImage], str]:
        """Decide the row image root from actual sources: any bytes/pil image
        forces the whole row into the cache root; path images are then copied
        via cache_existing_path(). 根据实际来源决定整行图片根：任一 bytes/pil
        图片使整行进入 cache 根；path 图片随后经 cache_existing_path() 复制。"""
        needs_cache = any(
            resolved_image.descriptor["image_source_type"] in {"bytes", "pil"}
            for resolved_image in resolved
        )
        if not needs_cache:
            return resolved, "release"
        unified: list[_ResolvedImage] = []
        for resolved_image in resolved:
            if resolved_image.descriptor["image_source_type"] in {"bytes", "pil"}:
                unified.append(resolved_image)
                continue
            try:
                cached_path, cached_descriptor = cache_existing_path(
                    resolved_image.path,
                    cache_root=self._effective_cache_root(),
                    index=index,
                )
            except ImageMaterializationError as error:
                raise DatasetProbeError(str(error)) from error
            unified.append(_ResolvedImage(path=cached_path, descriptor=cached_descriptor))
        return unified, "cache"

    def _image_refs(
        self,
        refs: list[tuple[Path, dict[str, Any]]],
        index: int,
        *,
        image_root: Path,
    ) -> list[ImageRef]:
        if not refs:
            raise DatasetProbeError(f"XLRS row {index} has no image field")
        images: list[ImageRef] = []
        for position, (path, _descriptor) in enumerate(refs):
            try:
                relative = path.relative_to(image_root)
            except ValueError as error:
                raise DatasetProbeError(
                    f"XLRS row {index} image {path} is outside image root {image_root}"
                ) from error
            role = "image" if position == 0 else "context"
            images.append(
                ImageRef(
                    image_id=f"xlrs-{index}-{position}",
                    path=relative.as_posix(),
                    role=role,
                )
            )
        return images

    # ── per-task mapping / 分任务映射 ───────────────────────────────────────

    def _caption_sample(
        self,
        row: dict[str, Any],
        images: list[ImageRef],
        split: str,
        index: int,
        *,
        image_root_kind: str,
    ) -> UnifiedSample:
        question = _first_text(row, ("question",))
        if not question:
            raise DatasetProbeError(f"XLRS caption row {index} has no question text")
        answers = _caption_texts(row, index)
        source_id = _first_text(row, ("id", "question_id", "source_id"))
        return UnifiedSample(
            sample_id=stable_sample_id(
                dataset=self.name, split=split, source_id=source_id,
                relative_image_paths=[image.path for image in images],
                question=question, source_index=index,
            ),
            dataset=self.name,
            split=split,
            task="caption",
            images=images,
            question=question,
            ground_truth=GroundTruth(
                answers=answers,
                raw={"adapter_version": ADAPTER_VERSION, "source_row": row},
            ),
            metadata={
                "source": "XLRS-Bench",
                "release": HF_REPOS["caption"],
                "release_split": RELEASE_SPLITS["caption"],
                "source_index": index,
                "image_root_kind": image_root_kind,
                "adapter_version": ADAPTER_VERSION,
            },
        )

    def _grounding_sample(
        self,
        row: dict[str, Any],
        images: list[ImageRef],
        split: str,
        index: int,
        *,
        image_root_kind: str,
    ) -> UnifiedSample:
        question = _first_text(row, ("question", "ref", "referring", "text"))
        box_value = _first_value(row, _GROUNDING_BOX_KEYS)
        if not question or box_value is None:
            raise DatasetProbeError(f"XLRS grounding row {index} is missing text or box")
        boxes = _parse_boxes(box_value, index)
        width = _first_value(row, ("image_width",))
        height = _first_value(row, ("image_height",))
        if width is None or height is None or float(width) <= 0 or float(height) <= 0:
            raise DatasetProbeError(
                f"XLRS grounding row {index} is missing positive image dimensions"
            )
        boxes = [_normalize_grounding_box_to_m3(box, index=index) for box in boxes]
        label = _first_text(row, _GROUNDING_LABEL_KEYS)
        labels = [label] if label else []
        source_id = _first_text(row, ("id", "question_id", "source_id"))
        return UnifiedSample(
            sample_id=stable_sample_id(
                dataset=self.name, split=split, source_id=source_id,
                relative_image_paths=[image.path for image in images],
                question=question, source_index=index,
            ),
            dataset=self.name,
            split=split,
            task="grounding",
            images=images,
            question=question,
            ground_truth=GroundTruth(
                boxes=boxes,
                labels=labels,
                label_binding="boxes" if labels else None,
                coordinate_frame="normalized_0_999_top_left",
                raw={
                    "adapter_version": ADAPTER_VERSION,
                    "box_source_field": _box_source_field(row),
                    "source_row": row,
                },
            ),
            metadata={
                "source": "XLRS-Bench",
                "release": HF_REPOS["grounding"],
                "release_split": RELEASE_SPLITS["grounding"],
                "source_index": index,
                "image_root_kind": image_root_kind,
                "image_width": float(width) if width is not None else None,
                "image_height": float(height) if height is not None else None,
                "coordinate_frame": "normalized_0_999_top_left",
                "adapter_version": ADAPTER_VERSION,
            },
        )

    def _vqa_lite_sample(
        self,
        row: dict[str, Any],
        images: list[ImageRef],
        split: str,
        index: int,
        *,
        image_root_kind: str,
    ) -> UnifiedSample:
        question = _first_text(row, ("question", "text", "query"))
        choices = _choices(row)
        if not question or not isinstance(choices, list) or not choices:
            raise DatasetProbeError(f"XLRS Lite row {index} is missing VQA fields")
        answer = _first_text(row, ("answer", "label", "ground_truth"))
        allow_multiple = _multi_answer_hint(row)
        parts = _vqa_answer_parts(answer, allow_multiple=allow_multiple)
        if answer is not None and len(choices) <= 5:
            allowed_set = set("ABCDE"[: len(choices)])
            if any(len(part) != 1 or part.upper() not in allowed_set for part in parts):
                raise DatasetProbeError(
                    f"XLRS Lite row {index} has invalid answer {answer!r} "
                    f"for {len(choices)} choices"
                )
            if len(parts) > 1:
                if not allow_multiple:
                    raise DatasetProbeError(
                        f"XLRS Lite row {index} has multiple answers but "
                        "allow_multiple is false"
                    )
                if len({part.upper() for part in parts}) != len(parts):
                    raise DatasetProbeError(
                        f"XLRS Lite row {index} repeats an answer letter: {answer!r}"
                    )
        source_id = _first_text(row, ("id", "question_id", "source_id"))
        adapter_version = (
            _JSONL_ADAPTER_VERSION
            if "source_partition" in row
            else _VQA_HF_ADAPTER_VERSION
        )
        canonical_question = _canonical_vqa_question(question, choices)
        versioned_source_id = (
            f"{source_id}-{adapter_version}" if source_id is not None else None
        )
        canonical_answers = (
            [", ".join(part.upper() for part in parts)]
            if answer is not None and allow_multiple and len(parts) > 1
            else [answer] if answer else []
        )
        return UnifiedSample(
            sample_id=stable_sample_id(
                dataset=self.name, split=split, source_id=versioned_source_id,
                relative_image_paths=[image.path for image in images],
                question=canonical_question, source_index=index,
            ),
            dataset=self.name,
            split=split,
            task="multiple_choice_vqa",
            images=images,
            question=canonical_question,
            ground_truth=GroundTruth(
                answers=canonical_answers,
                raw={"adapter_version": adapter_version, "source_row": row},
            ),
            metadata={
                "source": "XLRS-Bench-lite",
                "release": HF_REPOS["multiple_choice_vqa"],
                "release_split": RELEASE_SPLITS["multiple_choice_vqa"],
                "source_index": index,
                "image_root_kind": image_root_kind,
                "adapter_version": adapter_version,
            },
            normalization=TaskNormalization(
                source_task="xlrs_vqa_lite",
                normalized_task="multiple_choice_vqa",
                normalizer="xlrs_vqa_lite_adapter",
                version=adapter_version,
                choices=[str(choice) for choice in choices],
                allow_multiple=allow_multiple,
            ),
        )


def _multi_answer_hint(row: dict[str, Any]) -> bool:
    """Explicit fields win with strict bool parsing; otherwise only audited
    question types count. Never scans the whole row string.
    显式字段优先（严格 bool 解析）；否则仅已审计问题类型生效。
    绝不扫描整个 row 字符串。"""
    for key in _EXPLICIT_MULTI_ANSWER_KEYS:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes"}:
                return True
            if lowered in {"false", "0", "no"}:
                return False
            raise DatasetProbeError(
                f"invalid boolean value {value!r} for {key}"
            )
        raise DatasetProbeError(
            f"invalid boolean value {value!r} for {key}"
        )
    question_type = _first_text(
        row, ("question_type", "type", "subtask", "task_type", "l2-category")
    )
    if question_type is None:
        return False
    return question_type.casefold() in _AUDITED_MULTI_ANSWER_TYPES


def _vqa_answer_parts(answer: str | None, *, allow_multiple: bool) -> list[str]:
    """Parse separated answers and the audited compact multi-label form.
    解析分隔答案以及已审计的紧凑多标签形式。"""
    if answer is None:
        return []
    parts = [part.strip() for part in _ANSWER_SEPARATOR.split(answer) if part.strip()]
    compact = answer.strip().upper()
    if allow_multiple and len(parts) == 1 and len(compact) > 1 and compact.isalpha():
        return list(compact)
    return parts


def _caption_texts(row: dict[str, Any], index: int) -> list[str]:
    """Extract all non-empty reference captions; strings, list[str], or
    list[dict] with explicit text keys. Other structures fail.
    提取全部非空参考 caption；支持字符串、list[str] 或带明确文本键的
    list[dict]；其他结构失败。"""
    answer_value = _first_value(row, ("caption", "text", "answer", "description"))
    if answer_value is None:
        raise DatasetProbeError(f"XLRS caption row {index} has no caption field")
    if isinstance(answer_value, str):
        texts = [answer_value.strip()] if answer_value.strip() else []
    elif isinstance(answer_value, list):
        texts = []
        for item in answer_value:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    texts.append(stripped)
            elif isinstance(item, dict):
                text = _first_text(item, _CAPTION_TEXT_KEYS)
                if text is None:
                    raise DatasetProbeError(
                        f"XLRS caption row {index} has a dict caption without a text key"
                    )
                texts.append(text)
            else:
                raise DatasetProbeError(
                    f"XLRS caption row {index} has an unsupported caption item "
                    f"of type {type(item).__name__}"
                )
    else:
        raise DatasetProbeError(
            f"XLRS caption row {index} has an unsupported caption value "
            f"of type {type(answer_value).__name__}"
        )
    if not texts:
        raise DatasetProbeError(f"XLRS caption row {index} has no non-empty caption")
    return texts


def _parse_boxes(value: Any, index: int) -> list[list[float]]:
    """Parse 4/8 numeric boxes, nested one level, or numeric strings.
    解析 4/8 数值框、一层嵌套或可解析数值字符串。"""
    if isinstance(value, str):
        try:
            value = [float(part) for part in value.replace("[", "").replace("]", "").split(",")]
        except ValueError as error:
            raise DatasetProbeError(f"XLRS row {index} has unparseable box {value!r}") from error
    if not isinstance(value, (list, tuple)):
        raise DatasetProbeError(f"XLRS row {index} has invalid box structure")
    if value and isinstance(value[0], (list, tuple)):
        return [_single_box(item, index) for item in value]
    return [_single_box(value, index)]


def _single_box(value: Any, index: int) -> list[float]:
    if isinstance(value, str):
        try:
            value = [float(part) for part in value.replace("[", "").replace("]", "").split(",")]
        except ValueError as error:
            raise DatasetProbeError(f"XLRS row {index} has unparseable box {value!r}") from error
    if not isinstance(value, (list, tuple)):
        raise DatasetProbeError(f"XLRS row {index} has invalid box value")
    box = [float(part) for part in value]
    if len(box) not in (4, 8):
        raise DatasetProbeError(f"XLRS row {index} box must have 4 or 8 coordinates, got {len(box)}")
    return box


def _normalize_grounding_box_to_m3(
    box: list[float], *, index: int
) -> list[int]:
    """Convert the official XLRS Arrow bbox from [0, 1] to M3 0-999 xyxy."""
    if any(value < 0 or value > 1 for value in box):
        raise DatasetProbeError(
            f"XLRS grounding row {index} bbox must be normalized to [0, 1]"
        )
    return [max(0, min(999, int(round(value * 999)))) for value in box]


def _box_source_field(row: dict[str, Any]) -> str | None:
    for key in _GROUNDING_BOX_KEYS:
        if key in row and row[key] not in (None, ""):
            return key
    return None


def _choices(row: dict[str, Any]) -> list[Any] | None:
    value = _first_value(row, _CHOICE_KEYS)
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed: list[str] = []
        expected = ord("A")
        for line in value.splitlines():
            match = _LABELED_CHOICE_LINE.fullmatch(line)
            if match is None:
                continue
            label, text = match.groups()
            if ord(label) != expected:
                raise DatasetProbeError(
                    f"XLRS Lite choices have non-sequential label {label!r}"
                )
            parsed.append(f"({label}) {text.strip()}")
            expected += 1
        if len(parsed) < 2:
            raise DatasetProbeError("XLRS Lite choices string has fewer than two options")
        return parsed
    option_values = [row[key] for key in _OPTION_LETTERS if row.get(key) not in (None, "")]
    return option_values or None


def _canonical_vqa_question(question: str, choices: list[Any]) -> str:
    """Append labeled choices to the raw stem for planner and final-agent use.
    将带标签选项追加到原始题干，供 Planner 与最终 Agent 使用。"""
    rendered: list[str] = []
    for index, value in enumerate(choices):
        text = str(value).strip()
        if not text:
            raise DatasetProbeError("XLRS Lite choice text must not be empty")
        rendered.append(
            text if _CHOICE_PREFIX.match(text) else _label_choice(index, text)
        )
    return f"{question}\n\nChoices:\n" + "\n".join(rendered)


def _label_choice(index: int, text: str) -> str:
    """Attach one deterministic positional option label. / 添加确定性位置标签。"""
    return f"({chr(ord('A') + index)}) {text}"


def _has_jsonl_parts(root: Path) -> bool:
    """Return whether the audited sharded JSONL layout is present.
    返回是否存在已审计的分片 JSONL 布局。"""
    return any(root.glob(_JSONL_PART_PATTERN))


def _is_json_safe(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_is_json_safe(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_safe(item) for key, item in value.items())
    return False
