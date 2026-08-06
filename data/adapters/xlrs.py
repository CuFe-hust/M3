"""Offline-first XLRS-Bench adapter (caption / grounding / VQA-lite).

离线优先的 XLRS-Bench 适配器（caption / grounding / VQA-lite）。
- 本地优先：磁盘上存在 HF release 布局（dataset_dict.json 或 <split>/state.json）
  时用 load_from_disk；否则只有调用者显式 allow_download=True 才走 Hugging
  Face 下载，默认 allow_download=False，离线机器绝不尝试网络；
- datasets 库只在需要加载时延迟导入，import 本模块不加载 datasets；
- caption 输出全部参考；grounding 保留源框、图像尺寸与坐标来源；
  VQA-lite 输出 multiple_choice_vqa、原始 choices 与 multi-answer 提示。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from data.adapters.base import AdapterProbe, DatasetProbeError
from data.schema import GroundTruth, ImageRef, UnifiedSample, stable_sample_id

ADAPTER_VERSION = "hf-disk-v1"
# Official Hugging Face releases per task. / 各任务的官方 Hugging Face 发布。
HF_REPOS = {
    "caption": "initiacms/XLRS-Bench_caption_en",
    "grounding": "initiacms/XLRS-Bench_visual_grounding_en",
    "multiple_choice_vqa": "initiacms/XLRS-Bench-lite",
}
RELEASE_SPLITS = {
    "caption": "train",
    "grounding": "test",
    "multiple_choice_vqa": "train",
}
SUPPORTED_TASKS = frozenset(HF_REPOS)
_IMAGE_KEYS = ("image", "Image", "image_a", "image_b", "before", "after", "images")
_CHOICE_KEYS = ("choices", "options", "answer_choices", "multi-choice options")
_OPTION_LETTERS = ("A", "B", "C", "D", "E")

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
    ) -> None:
        """allow_download defaults to False; downloading requires an explicit opt-in.
        allow_download 默认 False；下载必须显式开启。name 覆盖实例注册名；
        dataset_loader 供测试注入。"""
        self.name = name or self.name
        self.allow_download = allow_download
        self._loader = dataset_loader

    def probe(self, root: Path) -> AdapterProbe:
        """Report the local release layout; never touches the network.
        报告本地 release 布局；绝不触网。"""
        local_tasks = [task for task in SUPPORTED_TASKS if self._has_local(root, task)]
        if not local_tasks and not self.allow_download:
            raise DatasetProbeError(
                f"offline: no local XLRS release under {root}; "
                "pass allow_download=True to download from Hugging Face"
            )
        return AdapterProbe(
            dataset=self.name,
            version=ADAPTER_VERSION,
            sample_file=root / "dataset_dict.json" if (root / "dataset_dict.json").is_file() else root,
            observed_fields=("local",) if local_tasks else ("remote",),
            sample_count=len(local_tasks),
        )

    def iter_samples(self, root: Path, split: str, task: str) -> Iterator[UnifiedSample]:
        """Yield unified samples for one task from the local or remote rows.
        从本地或远程行产出某一任务的统一样本。"""
        if task not in SUPPORTED_TASKS:
            raise DatasetProbeError(
                f"XLRS-Bench does not support task={task!r}; supported={sorted(SUPPORTED_TASKS)}"
            )
        rows = self._rows(root, task)
        release_split = RELEASE_SPLITS[task]
        for index, row in enumerate(rows):
            images = self._image_refs(row, root, index)
            if task == "caption":
                yield self._caption_sample(row, images, root, split, index)
            elif task == "grounding":
                yield self._grounding_sample(row, images, root, split, index)
            else:
                yield self._vqa_lite_sample(row, images, root, split, index)

    # ── loading strategy / 加载策略 ─────────────────────────────────────────

    def _has_local(self, root: Path, task: str) -> bool:
        split = RELEASE_SPLITS[task]
        return (root / "dataset_dict.json").is_file() or (root / split / "state.json").is_file()

    def _rows(self, root: Path, task: str) -> list[dict[str, Any]]:
        if self._has_local(root, task):
            loader = self._loader or self._load_from_disk
            return loader(root, task)
        if not self.allow_download:
            raise DatasetProbeError(
                f"offline: no local XLRS {task} release under {root}; "
                "pass allow_download=True to download from Hugging Face"
            )
        loader = self._loader or self._load_from_hub
        return loader(root, task)

    @staticmethod
    def _load_from_disk(root: Path, task: str) -> list[dict[str, Any]]:
        """Load a local HF release layout; datasets is imported lazily.
        加载本地 HF release 布局；datasets 延迟导入。"""
        try:
            from datasets import load_from_disk
        except ImportError as error:
            raise RuntimeError(
                "Install the datasets package to load local XLRS releases."
            ) from error
        split = RELEASE_SPLITS[task]
        if (root / "dataset_dict.json").is_file():
            dataset = load_from_disk(root)[split]
        else:
            dataset = load_from_disk(root / split)
        return [dict(row) for row in dataset]

    @staticmethod
    def _load_from_hub(root: Path, task: str) -> list[dict[str, Any]]:
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

    # ── per-task mapping / 分任务映射 ───────────────────────────────────────

    def _caption_sample(
        self,
        row: dict[str, Any],
        images: list[ImageRef],
        root: Path,
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
        return UnifiedSample(
            sample_id=stable_sample_id(
                dataset=self.name, split=split, source_id=None,
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
                raw={"adapter_version": ADAPTER_VERSION, "source_row": dict(row)},
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
        root: Path,
        split: str,
        index: int,
    ) -> UnifiedSample:
        question = _first_text(row, ("question",))
        box = _first_value(row, ("bbox", "box", "boxes", "polygon", "answer"))
        if not question or box is None:
            raise DatasetProbeError(f"XLRS grounding row {index} is missing text or box")
        width = _first_value(row, ("image_width",))
        height = _first_value(row, ("image_height",))
        return UnifiedSample(
            sample_id=stable_sample_id(
                dataset=self.name, split=split, source_id=None,
                relative_image_paths=[image.path for image in images],
                question=question, source_index=index,
            ),
            dataset=self.name,
            split=split,
            task="grounding",
            images=images,
            question=question,
            ground_truth=GroundTruth(
                boxes=[list(box)],
                coordinate_frame="source_pixels_top_left",
                raw={"adapter_version": ADAPTER_VERSION, "source_row": dict(row)},
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
        root: Path,
        split: str,
        index: int,
    ) -> UnifiedSample:
        question = _first_text(row, ("question", "text", "query"))
        choices = _choices(row)
        if not question or not isinstance(choices, list) or not choices:
            raise DatasetProbeError(f"XLRS Lite row {index} is missing VQA fields")
        answer = _first_text(row, ("answer", "label", "ground_truth")) or ""
        multi_answer = "overall land use" in str(row).lower()
        return UnifiedSample(
            sample_id=stable_sample_id(
                dataset=self.name, split=split, source_id=None,
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
                raw={"adapter_version": ADAPTER_VERSION, "source_row": dict(row)},
            ),
            metadata={
                "source": "XLRS-Bench-lite",
                "release": HF_REPOS["multiple_choice_vqa"],
                "release_split": RELEASE_SPLITS["multiple_choice_vqa"],
                "source_index": index,
                "choices": [str(choice) for choice in choices],
                "multi_answer": multi_answer,
                "adapter_version": ADAPTER_VERSION,
            },
        )

    # ── row helpers / 行辅助 ────────────────────────────────────────────────

    def _image_refs(self, row: dict[str, Any], root: Path, index: int) -> list[ImageRef]:
        values: list[Any] = []
        for key in _IMAGE_KEYS:
            value = row.get(key)
            if value is None:
                continue
            values.extend(value if isinstance(value, list) else [value])
        if not values:
            raise DatasetProbeError(f"XLRS row {index} has no image field")
        refs: list[ImageRef] = []
        for position, value in enumerate(values):
            if isinstance(value, dict):
                value = value.get("path") or value.get("file_name")
            if not isinstance(value, str) or not value:
                raise DatasetProbeError(f"XLRS row {index} has an invalid image value")
            image_path = root / value
            if not image_path.is_file():
                raise DatasetProbeError(f"XLRS row {index} references missing image: {value}")
            role = "image" if position == 0 else "context"
            refs.append(ImageRef(image_id=f"xlrs-{index}-{position}", path=image_path.relative_to(root), role=role))
        return refs


def _choices(row: dict[str, Any]) -> list[Any] | None:
    value = _first_value(row, _CHOICE_KEYS)
    if isinstance(value, list):
        return value
    option_values = [row[key] for key in _OPTION_LETTERS if row.get(key) not in (None, "")]
    return option_values or None
