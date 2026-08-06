"""Offline-first XLRS-Bench adapter (caption / grounding / VQA-lite).

离线优先的 XLRS-Bench 适配器（caption / grounding / VQA-lite）。
- 官方 release 目录（XLRS-Bench-lite / XLRS-Bench_caption_en /
  XLRS-Bench_visual_grounding_en）可从统一 XLRS 根解析，或直接传入单个
  release 根；零候选失败、多候选歧义失败；
- 官方 split 被强制（train/test 及明确 alias），不匹配在加载前失败；
- 本地优先：HF disk 布局走 load_from_disk；仅 allow_download=True 才走
  Hugging Face 下载；datasets 只在实际加载时延迟导入；
- 图片特征（path/{path,bytes}/PIL）确定性物化到外部 cache，不写 dataset root；
- source row 清洗为 JSON-safe，不保存 bytes/PIL；
- XLRS-Bench-lite 注册项只声明 multiple_choice_vqa。
"""

from __future__ import annotations

import re
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from data.adapters.base import AdapterProbe, DatasetProbeError
from data.adapters.xlrs_image_cache import ImageMaterializationError, materialize_image
from data.schema import GroundTruth, ImageRef, UnifiedSample, stable_sample_id

ADAPTER_VERSION = "hf-disk-v1"
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
_ANSWER_SEPARATOR = re.compile(r"[\s,，、]+")

RowLoader = Callable[[Path, str], list[dict[str, Any]]]


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

    # ── probe / 探测 ────────────────────────────────────────────────────────

    def probe(self, root: Path, task: str | None = None) -> AdapterProbe:
        """Report the local release layout; never touches the network.
        报告本地 release 布局；绝不触网。"""
        if task is not None:
            if task not in self.supported_tasks:
                raise DatasetProbeError(
                    f"XLRS-Bench does not support task={task!r}; "
                    f"supported={sorted(self.supported_tasks)}"
                )
            release_root = self._resolve_release_root(root, task)
            return AdapterProbe(
                dataset=self.name,
                version=ADAPTER_VERSION,
                sample_file=release_root / "dataset_dict.json"
                if (release_root / "dataset_dict.json").is_file()
                else release_root,
                observed_fields=("local",),
                sample_count=1,
                task=task,
                available_tasks=(task,),
            )
        local_tasks = [task for task in sorted(self.supported_tasks) if self._has_local(root, task)]
        if not local_tasks and not self.allow_download:
            raise DatasetProbeError(
                f"offline: no local XLRS release under {root}; "
                "pass allow_download=True to download from Hugging Face"
            )
        available = tuple(local_tasks) if local_tasks else tuple(sorted(self.supported_tasks))
        return AdapterProbe(
            dataset=self.name,
            version=ADAPTER_VERSION,
            sample_file=root,
            observed_fields=("local",) if local_tasks else ("remote",),
            sample_count=len(local_tasks) if local_tasks else 1,
            available_tasks=available,
        )

    # ── iter_samples / 样本迭代 ────────────────────────────────────────────

    def _effective_cache_root(self) -> Path:
        """Stable default user cache; created only when bytes/PIL are handled.
        稳定默认用户 cache；仅处理 bytes/PIL 时才创建。"""
        if self._cache_root is not None:
            return self._cache_root
        return Path(tempfile.gettempdir()) / "m3-xlrs-image-cache"

    def iter_samples(self, root: Path, split: str, task: str) -> Iterator[UnifiedSample]:
        """Yield unified samples for one task; official splits are enforced.
        产出某一任务的统一样本；官方 split 被强制。"""
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
        for index, row in enumerate(rows):
            safe_row, refs, cache_used = self._sanitize_row(
                row, release_root=release_root, index=index
            )
            image_root = self._effective_cache_root() if cache_used else release_root
            images = self._image_refs(safe_row, refs, index, image_root=image_root)
            if task == "caption":
                yield self._caption_sample(safe_row, images, release_root, official_split, index)
            elif task == "grounding":
                yield self._grounding_sample(safe_row, images, release_root, official_split, index)
            else:
                yield self._vqa_lite_sample(safe_row, images, release_root, official_split, index)
        self._last_image_root = self._effective_cache_root() if False else release_root

    @property
    def image_root(self) -> Path | None:
        """The resolution root for ImageRef paths (release root or external cache).
        ImageRef 路径的解析根（release root 或外部 cache）。"""
        return getattr(self, "_last_image_root", None)

    # ── loading strategy / 加载策略 ─────────────────────────────────────────

    def _has_local(self, root: Path, task: str) -> bool:
        split = RELEASE_SPLITS[task]
        return (root / "dataset_dict.json").is_file() or (root / split / "state.json").is_file()

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

    def _rows(self, release_root: Path, task: str) -> list[dict[str, Any]]:
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

    @staticmethod
    def _load_from_disk(release_root: Path, task: str) -> list[dict[str, Any]]:
        """Load a local HF release layout; datasets is imported lazily.
        加载本地 HF release 布局；datasets 延迟导入。"""
        try:
            from datasets import load_from_disk
        except ImportError as error:
            raise RuntimeError(
                "Install the datasets package to load local XLRS releases."
            ) from error
        split = RELEASE_SPLITS[task]
        if (release_root / "dataset_dict.json").is_file():
            dataset = load_from_disk(release_root)[split]
        else:
            dataset = load_from_disk(release_root / split)
        return [dict(row) for row in dataset]

    @staticmethod
    def _load_from_hub(release_root: Path, task: str) -> list[dict[str, Any]]:
        """Download from Hugging Face; only reachable with allow_download=True.
        从 Hugging Face 下载；仅 allow_download=True 时可达。"""
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise RuntimeError(
                "Install the datasets package to download XLRS releases."
            ) from error
        dataset = load_dataset(HF_REPOS[task], split=RELEASE_SPLITS[task])
        return [dict(row) for row in dataset]

    # ── row sanitation / 行清洗 ─────────────────────────────────────────────

    def _sanitize_row(
        self,
        row: dict[str, Any],
        *,
        release_root: Path,
        index: int,
    ) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]], bool]:
        """Strip non-JSON values and materialize image features.
        去除非 JSON 值并物化图片特征。"""
        safe: dict[str, Any] = {}
        refs: list[tuple[str, dict[str, Any]]] = []
        used_cache = False
        excluded: list[str] = []
        for key, value in row.items():
            if key in _IMAGE_KEYS:
                is_list = isinstance(value, list)
                values = value if is_list else [value]
                descriptors = []
                for item in values:
                    try:
                        path, descriptor = materialize_image(
                            item, release_root=release_root,
                            cache_root=self._cache_root, index=index,
                        )
                    except ImageMaterializationError as error:
                        raise DatasetProbeError(str(error)) from error
                    descriptors.append(descriptor)
                    refs.append((str(path), descriptor))
                    if descriptor["image_source_type"] in {"bytes", "pil"}:
                        used_cache = True
                if is_list:
                    safe[key] = descriptors
                else:
                    safe[key] = descriptors[0] if descriptors else {"image_present": False}
            elif _is_json_safe(value):
                safe[key] = value
            else:
                excluded.append(key)
        if excluded:
            safe["excluded_fields"] = sorted(excluded)
        return safe, refs, used_cache

    def _image_refs(
        self,
        row: dict[str, Any],
        refs: list[tuple[str, dict[str, Any]]],
        index: int,
        *,
        image_root: Path,
    ) -> list[ImageRef]:
        if not refs:
            raise DatasetProbeError(f"XLRS row {index} has no image field")
        images: list[ImageRef] = []
        for position, (raw_path, _descriptor) in enumerate(refs):
            path = Path(raw_path)
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
        release_root: Path,
        split: str,
        index: int,
    ) -> UnifiedSample:
        question = _first_text(row, ("question",))
        if not question:
            raise DatasetProbeError(f"XLRS caption row {index} has no question text")
        answer_value = _first_value(row, ("caption", "text", "answer", "description"))
        answers = (
            [str(answer) for answer in answer_value]
            if isinstance(answer_value, list)
            else [str(answer_value or "")]
        )
        if not answers or not answers[0]:
            raise DatasetProbeError(f"XLRS caption row {index} has no caption field")
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
                "adapter_version": ADAPTER_VERSION,
            },
        )

    def _grounding_sample(
        self,
        row: dict[str, Any],
        images: list[ImageRef],
        release_root: Path,
        split: str,
        index: int,
    ) -> UnifiedSample:
        question = _first_text(row, ("question", "ref", "referring", "text"))
        box_value = _first_value(row, _GROUNDING_BOX_KEYS)
        if not question or box_value is None:
            raise DatasetProbeError(f"XLRS grounding row {index} is missing text or box")
        boxes = _parse_boxes(box_value, index)
        width = _first_value(row, ("image_width",))
        height = _first_value(row, ("image_height",))
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
                coordinate_frame="source_pixels_top_left",
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
                "image_width": float(width) if width is not None else None,
                "image_height": float(height) if height is not None else None,
                "adapter_version": ADAPTER_VERSION,
            },
        )

    def _vqa_lite_sample(
        self,
        row: dict[str, Any],
        images: list[ImageRef],
        release_root: Path,
        split: str,
        index: int,
    ) -> UnifiedSample:
        question = _first_text(row, ("question", "text", "query"))
        choices = _choices(row)
        if not question or not isinstance(choices, list) or not choices:
            raise DatasetProbeError(f"XLRS Lite row {index} is missing VQA fields")
        answer = _first_text(row, ("answer", "label", "ground_truth"))
        allowed_letters = "ABCDE"[: len(choices)]
        if answer is not None and len(choices) <= 5:
            allowed_set = set(allowed_letters)
            parts = [part.strip() for part in _ANSWER_SEPARATOR.split(answer) if part.strip()]
            if any(len(part) != 1 or part.upper() not in allowed_set for part in parts):
                raise DatasetProbeError(
                    f"XLRS Lite row {index} has invalid answer {answer!r} "
                    f"for {len(choices)} choices"
                )
        multi_answer = _multi_answer_hint(row)
        source_id = _first_text(row, ("id", "question_id", "source_id"))
        return UnifiedSample(
            sample_id=stable_sample_id(
                dataset=self.name, split=split, source_id=source_id,
                relative_image_paths=[image.path for image in images],
                question=question, source_index=index,
            ),
            dataset=self.name,
            split=split,
            task="multiple_choice_vqa",
            images=images,
            question=question,
            ground_truth=GroundTruth(
                answers=[answer] if answer else [],
                raw={"adapter_version": ADAPTER_VERSION, "source_row": row},
            ),
            metadata={
                "source": "XLRS-Bench-lite",
                "release": HF_REPOS["multiple_choice_vqa"],
                "release_split": RELEASE_SPLITS["multiple_choice_vqa"],
                "source_index": index,
                "choices": [str(choice) for choice in choices],
                "allow_multiple": multi_answer,
                "adapter_version": ADAPTER_VERSION,
            },
        )


def _multi_answer_hint(row: dict[str, Any]) -> bool:
    """Explicit fields win; otherwise the audited official question-type rule.
    显式字段优先；否则使用已审计的官方问题类型规则。"""
    for key in ("allow_multiple", "multi_answer"):
        value = row.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "1", "yes"}:
            return True
        if isinstance(value, str) and value.strip().lower() in {"false", "0", "no"}:
            return False
    return "overall land use" in str(row).lower()


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


def _box_source_field(row: dict[str, Any]) -> str | None:
    for key in _GROUNDING_BOX_KEYS:
        if key in row and row[key] not in (None, ""):
            return key
    return None


def _choices(row: dict[str, Any]) -> list[Any] | None:
    value = _first_value(row, _CHOICE_KEYS)
    if isinstance(value, list):
        return value
    option_values = [row[key] for key in _OPTION_LETTERS if row.get(key) not in (None, "")]
    return option_values or None


def _is_json_safe(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(_is_json_safe(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_safe(item) for key, item in value.items())
    return False
