"""Explicit, read-only dataset adapters and a registry.
显式、只读的数据集适配器及其注册表。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from spacers_agent.schemas import GroundTruth, ImageRef, UnifiedSample, stable_sample_id


class DatasetProbeError(ValueError):
    """Raised when an adapter cannot prove its declared layout. / 适配器无法证明声明布局时抛出。"""


@dataclass(frozen=True)
class AdapterProbe:
    """Observed layout evidence returned before execution. / 运行前返回的已观察布局证据。"""

    dataset: str
    version: str
    sample_file: Path
    observed_fields: tuple[str, ...]
    sample_count: int


class DatasetAdapter(Protocol):
    """Read-only adapter contract with explicit probe and validation. / 具有显式探测和验证的只读适配器契约。"""

    name: str

    def probe(self, root: Path) -> AdapterProbe: ...

    def iter_samples(self, root: Path, split: str, task: str) -> Iterator[UnifiedSample]: ...


class ManifestDatasetAdapter:
    """Adapter requiring a versioned, user-provided mapping manifest. / 要求用户提供版本化映射清单的适配器。"""

    manifest_name = "spacers_adapter.json"

    def __init__(self, name: str, supported_tasks: set[str]) -> None:
        self.name = name
        self.supported_tasks = supported_tasks

    def probe(self, root: Path) -> AdapterProbe:
        """Validate an explicit mapping instead of inferring field names. / 验证显式映射而不推测字段名称。"""

        manifest_path = root / self.manifest_name
        if not manifest_path.is_file():
            raise DatasetProbeError(
                f"{self.name} requires {self.manifest_name}; observed root entries: "
                f"{sorted(path.name for path in root.iterdir())[:20] if root.is_dir() else []}"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise DatasetProbeError(f"Invalid {self.manifest_name}: {error}") from error
        if manifest.get("dataset") != self.name or manifest.get("version") != "1":
            raise DatasetProbeError(f"Expected dataset={self.name!r} and version='1' in {self.manifest_name}")
        samples_value = manifest.get("samples_file")
        fields = manifest.get("fields")
        if not isinstance(samples_value, str) or not isinstance(fields, Mapping):
            raise DatasetProbeError("Adapter manifest requires string samples_file and object fields")
        required = {"id", "split", "task", "question", "images"}
        missing = sorted(required - set(fields))
        if missing:
            raise DatasetProbeError(f"Adapter manifest misses required field mappings: {missing}")
        sample_path = root / samples_value
        rows = _read_rows(sample_path)
        observed = tuple(sorted({key for row in rows[:20] for key in row}))
        mapped = tuple(str(value) for value in fields.values())
        absent = sorted(set(mapped) - set(observed))
        if absent:
            raise DatasetProbeError(f"Mapped fields absent from sample rows: {absent}; observed fields: {list(observed)}")
        return AdapterProbe(self.name, "1", sample_path, observed, len(rows))

    def iter_samples(self, root: Path, split: str, task: str) -> Iterator[UnifiedSample]:
        """Yield only schema-validated samples from the declared field mapping. / 仅产出来自声明字段映射的已校验样本。"""

        if task not in self.supported_tasks:
            raise DatasetProbeError(f"{self.name} does not support task={task!r}; supported={sorted(self.supported_tasks)}")
        probe = self.probe(root)
        manifest = json.loads((root / self.manifest_name).read_text(encoding="utf-8"))
        fields: Mapping[str, str] = manifest["fields"]
        for index, row in enumerate(_read_rows(probe.sample_file)):
            if str(row[fields["split"]]) != split or str(row[fields["task"]]) != task:
                continue
            images_value = row[fields["images"]]
            if not isinstance(images_value, list) or not images_value:
                raise DatasetProbeError(f"Row {index} has invalid images field {fields['images']!r}")
            roles = row.get(fields.get("image_roles", ""), [])
            if not isinstance(roles, list):
                roles = []
            images = [
                ImageRef(
                    image_id=f"{index}-{image_index}",
                    path=root / str(relative),
                    role=roles[image_index] if image_index < len(roles) else ("image" if len(images_value) == 1 else ("t1" if image_index == 0 else "t2")),
                )
                for image_index, relative in enumerate(images_value)
            ]
            missing_files = [str(image.path) for image in images if not image.path.is_file()]
            if missing_files:
                raise DatasetProbeError(f"Row {index} references missing images: {missing_files}")
            count = row.get(fields.get("count", ""))
            answers = row.get(fields.get("answers", ""), [])
            yield UnifiedSample(
                sample_id=stable_sample_id(str(row[fields["id"]]), images[0].path.relative_to(root), str(row[fields["question"]]), index),
                dataset=self.name,
                split=split,
                task=task,  # type: ignore[arg-type]
                images=images,
                question=str(row[fields["question"]]),
                ground_truth=GroundTruth(
                    count=int(count) if count not in (None, "") else None,
                    answers=[str(answer) for answer in answers] if isinstance(answers, list) else ([str(answers)] if answers not in (None, "") else []),
                    raw={"adapter_version": "1"},
                ),
            )


class VRSBenchVQAAdapter:
    """Read the official VRSBench evaluation VQA release without source edits.
    只读加载官方 VRSBench VQA 评测发布，不修改源数据。
    """

    name = "VRSBench"
    supported_tasks = {"general_vqa"}
    annotation_name = "VRSBench_EVAL_vqa.json"

    def probe(self, root: Path) -> AdapterProbe:
        """Verify the official annotation fields before yielding any sample.
        在产出样本前校验官方标注字段。
        """

        annotation = self._annotation_path(root)
        rows = _read_rows(annotation)
        required = {"image_id", "question", "ground_truth", "question_id", "type"}
        observed = tuple(sorted({key for row in rows[:20] for key in row}))
        for index, row in enumerate(rows):
            missing = sorted(required - set(row))
            if missing:
                raise DatasetProbeError(f"Official VRSBench VQA row {index} misses fields: {missing}")
        return AdapterProbe(self.name, "official-eval-v1", annotation, observed, len(rows))

    def iter_samples(self, root: Path, split: str, task: str) -> Iterator[UnifiedSample]:
        """Map official VQA rows to validated unified samples in source order.
        按源顺序将官方 VQA 行映射为经校验的统一样本。
        """

        if split not in {"validation", "val"}:
            raise DatasetProbeError("Official VRSBench_EVAL_vqa.json supports split='validation' only")
        if task != "general_vqa":
            raise DatasetProbeError("Official VRSBench VQA adapter supports task='general_vqa' only")
        probe = self.probe(root)
        for index, row in enumerate(_read_rows(probe.sample_file)):
            image_id = str(row["image_id"])
            image_path = self._image_path(root, probe.sample_file.parent, image_id)
            question = str(row["question"])
            answer = str(row["ground_truth"])
            question_id = str(row["question_id"])
            yield UnifiedSample(
                sample_id=stable_sample_id(question_id, image_path.relative_to(root), question, index),
                dataset=self.name,
                split="validation",
                task="general_vqa",
                images=[ImageRef(image_id=image_id, path=image_path, role="image")],
                question=question,
                ground_truth=GroundTruth(
                    answers=[answer],
                    raw={
                        "adapter_version": "official-eval-v1",
                        "image_id": image_id,
                        "question_id": row["question_id"],
                        "question_type": row["type"],
                        "source_dataset": row.get("dataset"),
                    },
                ),
                metadata={
                    "source": "VRSBench",
                    "source_index": index,
                    "question_type": str(row["type"]),
                    "source_dataset": row.get("dataset"),
                },
            )

    def _annotation_path(self, root: Path) -> Path:
        direct = root / self.annotation_name
        if direct.is_file():
            return direct
        matches = sorted(root.rglob(self.annotation_name)) if root.is_dir() else []
        if len(matches) != 1:
            raise DatasetProbeError(
                f"Expected exactly one {self.annotation_name}; observed {len(matches)} under {root}"
            )
        return matches[0]

    @staticmethod
    def _image_path(root: Path, annotation_root: Path, image_id: str) -> Path:
        candidates = (
            root / image_id,
            annotation_root / image_id,
            root / "Images_val" / image_id,
            root / "Images_val" / "Images_val" / image_id,
            annotation_root / "Images_val" / "Images_val" / image_id,
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise DatasetProbeError(f"Official VRSBench image is missing: {image_id}")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Read a small explicit JSON or JSONL manifest without network fallback. / 读取显式 JSON 或 JSONL 清单且不使用网络回退。"""

    if not path.is_file():
        raise DatasetProbeError(f"Declared samples_file does not exist: {path}")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("samples", []) if isinstance(payload, dict) else []
    else:
        raise DatasetProbeError("samples_file must be .json or .jsonl")
    if not all(isinstance(row, dict) for row in rows):
        raise DatasetProbeError("All sample rows must be JSON objects")
    return list(rows)


ADAPTERS: dict[str, DatasetAdapter] = {
    "LEVIR-CC": ManifestDatasetAdapter("LEVIR-CC", {"change_caption", "change_qa"}),
    "VRSBench": VRSBenchVQAAdapter(),
    "MME-RealWorld": ManifestDatasetAdapter("MME-RealWorld", {"general_vqa", "multiple_choice_vqa"}),
    "XLRS-Bench-lite": ManifestDatasetAdapter("XLRS-Bench-lite", {"counting", "general_vqa", "caption", "grounding"}),
}


def get_adapter(name: str) -> DatasetAdapter:
    """Return a registered adapter or expose allowed names. / 返回注册适配器或暴露允许的名称。"""

    try:
        return ADAPTERS[name]
    except KeyError as error:
        raise DatasetProbeError(f"Unsupported dataset {name!r}; supported={sorted(ADAPTERS)}") from error
