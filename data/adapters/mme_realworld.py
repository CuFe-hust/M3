"""Audited MME-RealWorld remote-sensing adapter.

经审计的 MME-RealWorld 遥感子集适配器：
- 读取唯一 MME_RealWorld.json；
- 只保留 Remote Sensing 子任务（subtask 或 question_id 匹配），非遥感记录
  不进入样本流；
- 保留原始选项与问题文本事实，不拼接字母选项 Prompt、不构造 system
  instruction、不把正确答案写进 question；
- 输出 multiple_choice_vqa，metadata 保存 allow_multiple、source subtask 与
  原始 choices；答案必须存在且属于合法选项字母格式。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Pattern

import re

from data.adapters.base import AdapterProbe, DatasetProbeError, read_json_rows
from data.schema import GroundTruth, ImageRef, UnifiedSample, stable_sample_id

ANNOTATION_NAME = "MME_RealWorld.json"
SUPPORTED_TASKS = frozenset({"multiple_choice_vqa"})
ADAPTER_VERSION = "official-v1"
# 合法答案格式：一个或多个选项字母（A–E，大小写），逗号/空格/顿号分隔。
_ANSWER_PATTERN: Pattern[str] = re.compile(r"^[A-Ea-e](?:[,\s，、]+[A-Ea-e])*$")
_IMAGE_KEYS = ("image", "Image", "image_path", "img_path", "img", "image_name", "file_name", "filename", "image_id")


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


class MMERealWorldAdapter:
    """Read-only adapter over the official MME-RealWorld release (RS subset).
    只读加载官方 MME-RealWorld 发布（遥感子集）的适配器。"""

    name = "MME-RealWorld"
    supported_tasks = SUPPORTED_TASKS

    def probe(self, root: Path) -> AdapterProbe:
        """Locate and validate the unique official annotation before execution.
        运行前定位并校验唯一的官方标注。"""
        annotation = self._annotation_path(root)
        rows = read_json_rows(annotation)
        rs_rows = [row for row in rows if self._is_remote_sensing(row)]
        for index, row in enumerate(rows):
            if not self._is_remote_sensing(row):
                continue
            self._validate_row(row, index)
        observed = tuple(sorted({key for row in rs_rows[:20] for key in row}))
        return AdapterProbe(
            dataset=self.name,
            version=ADAPTER_VERSION,
            sample_file=annotation,
            observed_fields=observed,
            sample_count=len(rs_rows),
        )

    def iter_samples(self, root: Path, split: str, task: str) -> Iterator[UnifiedSample]:
        """Yield remote-sensing unified samples in source order.
        按源顺序产出遥感子集统一样本。"""
        if task not in SUPPORTED_TASKS:
            raise DatasetProbeError(
                f"MME-RealWorld does not support task={task!r}; supported={sorted(SUPPORTED_TASKS)}"
            )
        annotation = self._annotation_path(root)
        rows = read_json_rows(annotation)
        for index, row in enumerate(rows):
            if not self._is_remote_sensing(row):
                continue
            self._validate_row(row, index)
            question = _first_text(row, ("Text", "text", "question"))
            choices = row.get("Answer choices", row.get("answer_choices", []))
            if not isinstance(choices, list) or not choices:
                raise DatasetProbeError(f"MME-RealWorld RS row {index} has invalid choices")
            ground_truth = _first_text(row, ("Ground truth", "ground_truth", "answer"))
            assert ground_truth is not None  # guaranteed by _validate_row
            image_value = _first_value(row, _IMAGE_KEYS)
            assert image_value is not None  # guaranteed by _validate_row
            image_path = root / str(image_value)
            if not image_path.is_file():
                raise DatasetProbeError(
                    f"MME-RealWorld RS row {index} references missing image: {image_value}"
                )
            subtask = self._subtask(row)
            allow_multiple = _as_bool(
                row.get("allow_multiple", row.get("multi_answer", False))
            )
            yield UnifiedSample(
                sample_id=stable_sample_id(
                    dataset=self.name,
                    split=split,
                    source_id=str(row.get("Question_id", row.get("question_id", ""))) or None,
                    relative_image_paths=[image_path.relative_to(root)],
                    question=question,
                    source_index=index,
                ),
                dataset=self.name,
                split=split,
                task="multiple_choice_vqa",
                images=[ImageRef(image_id=f"mme-{index}", path=image_path.relative_to(root), role="image")],
                question=question,
                ground_truth=GroundTruth(
                    answers=[ground_truth],
                    raw={
                        "adapter_version": ADAPTER_VERSION,
                        "choices": [str(choice) for choice in choices],
                        "source_row": dict(row),
                    },
                ),
                metadata={
                    "source": "MME-RealWorld",
                    "source_index": index,
                    "subtask": subtask,
                    "allow_multiple": allow_multiple,
                    "choices": [str(choice) for choice in choices],
                    "adapter_version": ADAPTER_VERSION,
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

    @staticmethod
    def _subtask(row: dict[str, Any]) -> str:
        return str(row.get("Subtask", row.get("subtask", ""))).lower().replace("_", " ")

    def _is_remote_sensing(self, row: dict[str, Any]) -> bool:
        subtask = self._subtask(row)
        question_id = str(row.get("Question_id", row.get("question_id", ""))).lower().replace("_", " ")
        return "remote sensing" in subtask or "remote sensing" in question_id

    def _validate_row(self, row: dict[str, Any], index: int) -> None:
        question = _first_text(row, ("Text", "text", "question"))
        if not question:
            raise DatasetProbeError(f"MME-RealWorld RS row {index} has no question text")
        choices = row.get("Answer choices", row.get("answer_choices", []))
        if not isinstance(choices, list) or not choices:
            raise DatasetProbeError(f"MME-RealWorld RS row {index} has no answer choices")
        ground_truth = _first_text(row, ("Ground truth", "ground_truth", "answer"))
        if ground_truth is None:
            raise DatasetProbeError(f"MME-RealWorld RS row {index} has no ground truth")
        if not _ANSWER_PATTERN.match(ground_truth):
            raise DatasetProbeError(
                f"MME-RealWorld RS row {index} has invalid answer format: {ground_truth!r}"
            )
        if _first_value(row, _IMAGE_KEYS) is None:
            raise DatasetProbeError(f"MME-RealWorld RS row {index} has no image field")


def _first_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None
