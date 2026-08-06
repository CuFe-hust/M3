"""Audited LEVIR-CC adapter for the official captions release.

经审计的 LEVIR-CC 适配器：加载官方 LevirCCcaptions.json（或显式
spacers_adapter.json 由调用方先行校验布局）。
- 严格映射 before/A → t1、after/B → t2，绝不交换图片顺序；
- 保留全部参考 captions（多参考答案）到 GroundTruth.answers；
- change_qa 仅在标注明确含 question 字段时支持；
- 损坏行（缺 captions、缺图片对、图片文件缺失）显式失败，绝不静默跳过；
- 只读源数据；不运行一致化/差异检测；不生成任何模型 Prompt。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from data.adapters.base import AdapterProbe, DatasetProbeError, read_json_rows
from data.schema import GroundTruth, ImageRef, UnifiedSample, stable_sample_id

ANNOTATION_NAME = "LevirCCcaptions.json"
SUPPORTED_TASKS = frozenset({"change_caption", "change_qa"})

# First image keys map to t1; second image keys map to t2. / 首图键映射 t1，次图键映射 t2。
_T1_KEYS = ("image_A", "A", "image1", "before")
_T2_KEYS = ("image_B", "B", "image2", "after")
_CAPTION_KEYS = ("captions", "caption", "sentences", "description")
_CAPTION_TEXT_KEYS = ("raw", "caption", "text", "sentence")


def _first_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


class LEVIRCCAdapter:
    """Read-only adapter over the official LEVIR-CC caption release.
    只读加载官方 LEVIR-CC caption 发布的适配器。"""

    name = "LEVIR-CC"
    supported_tasks = SUPPORTED_TASKS

    def probe(self, root: Path) -> AdapterProbe:
        """Locate and validate the unique official annotation before execution.
        运行前定位并校验唯一的官方标注。"""
        annotation = self._annotation_path(root)
        rows = read_json_rows(annotation)
        self._validate_rows(rows, annotation)
        observed = tuple(sorted({key for row in rows[:20] for key in row}))
        return AdapterProbe(
            dataset=self.name,
            version="official-captions-v1",
            sample_file=annotation,
            observed_fields=observed,
            sample_count=len(rows),
        )

    def iter_samples(self, root: Path, split: str, task: str) -> Iterator[UnifiedSample]:
        """Yield schema-validated unified samples in source order.
        按源顺序产出具 schema 校验的统一样本。"""
        if task not in SUPPORTED_TASKS:
            raise DatasetProbeError(
                f"LEVIR-CC does not support task={task!r}; supported={sorted(SUPPORTED_TASKS)}"
            )
        annotation = self._annotation_path(root)
        rows = read_json_rows(annotation)
        self._validate_rows(rows, annotation)
        dataset_root = annotation.parent
        for index, row in enumerate(rows):
            row_split = str(row.get("split", row.get("Split", "test"))).lower()
            if row_split != split.lower():
                continue
            image_a, image_b = self._image_pair(row, dataset_root)
            if not image_a.is_file() or not image_b.is_file():
                raise DatasetProbeError(
                    f"LEVIR-CC row {index} references missing images: "
                    f"{image_a} / {image_b}"
                )
            answers = self._caption_texts(row, index)
            question = ""
            if task == "change_qa":
                if "question" not in row or not str(row.get("question", "")).strip():
                    raise DatasetProbeError(
                        f"LEVIR-CC row {index} has no question for change_qa"
                    )
                question = str(row["question"])
            else:
                question = str(row.get("question", ""))
            yield UnifiedSample(
                sample_id=stable_sample_id(
                    dataset=self.name,
                    split=split,
                    source_id=None,
                    relative_image_paths=[
                        image_a.relative_to(root),
                        image_b.relative_to(root),
                    ],
                    question=question,
                    source_index=index,
                ),
                dataset=self.name,
                split=split,
                task=task,
                images=[
                    ImageRef(image_id=f"t1-{index}", path=image_a.relative_to(root), role="t1"),
                    ImageRef(image_id=f"t2-{index}", path=image_b.relative_to(root), role="t2"),
                ],
                question=question,
                ground_truth=GroundTruth(
                    answers=answers,
                    raw={
                        "adapter_version": "official-captions-v1",
                        "source_row": dict(row),
                    },
                ),
                metadata={
                    "source": "LEVIR-CC",
                    "source_index": index,
                    "adapter_version": "official-captions-v1",
                },
            )

    # ── helpers / 辅助 ──────────────────────────────────────────────────────

    def _annotation_path(self, root: Path) -> Path:
        direct = root / ANNOTATION_NAME
        matches = [direct] if direct.is_file() else []
        if root.is_dir():
            matches += [path for path in sorted(root.rglob(ANNOTATION_NAME)) if path != direct]
        if len(matches) != 1:
            raise DatasetProbeError(
                f"Expected exactly one {ANNOTATION_NAME}; observed {len(matches)} under {root}"
            )
        return matches[0]

    def _validate_rows(self, rows: list[dict[str, Any]], annotation: Path) -> None:
        for index, row in enumerate(rows):
            missing = []
            if _first_value(row, _T1_KEYS) is None and _first_value(row, ("filepath", "file_name", "filename", "image", "image_path")) is None:
                missing.append("image pair")
            if _first_value(row, _CAPTION_KEYS) is None:
                missing.append("captions")
            if missing:
                raise DatasetProbeError(
                    f"LEVIR-CC row {index} in {annotation.name} misses {', '.join(missing)}"
                )

    @staticmethod
    def _image_pair(row: dict[str, Any], dataset_root: Path) -> tuple[Path, Path]:
        first = _first_value(row, _T1_KEYS)
        second = _first_value(row, _T2_KEYS)
        if first is not None and second is not None:
            return dataset_root / str(first), dataset_root / str(second)
        filepath = _first_value(row, ("filepath", "file_name", "filename", "image", "image_path"))
        if filepath is None:
            raise DatasetProbeError("LEVIR-CC row has no image-pair field")
        filename = _first_value(row, ("filename", "file_name"))
        if filename is not None and str(filepath) in {"train", "val", "validation", "test"}:
            image_a = dataset_root / "images" / str(filepath) / "A" / str(filename)
            image_b = dataset_root / "images" / str(filepath) / "B" / str(filename)
            return image_a, image_b
        first_path = str(filepath)
        second_path = first_path.replace("/A/", "/B/").replace("\\A\\", "\\B\\")
        if second_path == first_path:
            raise DatasetProbeError(f"Cannot derive LEVIR-CC post-change image from {first_path!r}")
        return dataset_root / first_path, dataset_root / second_path

    @staticmethod
    def _caption_texts(row: dict[str, Any], index: int) -> list[str]:
        value = _first_value(row, _CAPTION_KEYS)
        values = value if isinstance(value, list) else [value]
        texts: list[str] = []
        for item in values:
            if isinstance(item, dict):
                text = _first_value(item, _CAPTION_TEXT_KEYS)
                text = str(text).strip() if text is not None else ""
            else:
                text = str(item).strip()
            if text:
                texts.append(text)
        if not texts:
            raise DatasetProbeError(f"LEVIR-CC row {index} has no non-empty reference caption")
        return texts
