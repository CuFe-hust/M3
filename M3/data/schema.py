"""Canonical records shared by dataset adapters, inference, and evaluation.
数据集适配器、推理与评测共用的统一记录。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


VALID_TASK_TYPES = {"caption", "vqa", "grounding", "change_caption"}


@dataclass
class CanonicalSample:
    """One multimodal remote-sensing sample for baseline inference.
    一条用于基线推理的多模态遥感样本。
    """

    id: str
    task_type: str
    images: list[Any]
    prompt: str
    answers: list[str] = field(default_factory=list)
    choices: list[str] = field(default_factory=list)
    boxes: list[list[float]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.id:
            raise ValueError("Canonical sample id must not be empty.")
        if self.task_type not in VALID_TASK_TYPES:
            raise ValueError(f"Unsupported task_type: {self.task_type}")
        if not self.images:
            raise ValueError(f"Sample {self.id} has no image inputs.")
        if not self.prompt.strip():
            raise ValueError(f"Sample {self.id} has an empty prompt.")

    def serializable(self) -> dict[str, Any]:
        result = asdict(self)
        result["images"] = [str(image) for image in self.images]
        return result


@dataclass
class CanonicalPrediction:
    """One canonical prediction produced by a model.
    模型产生的一条统一预测。
    """

    id: str
    task_type: str
    text: str
    answer: str | None = None
    boxes: list[list[float]] = field(default_factory=list)
    masks: list[Any] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    count: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.id:
            raise ValueError("Canonical prediction id must not be empty.")
        if self.task_type not in VALID_TASK_TYPES:
            raise ValueError(f"Unsupported task_type: {self.task_type}")
        if not isinstance(self.text, str):
            raise ValueError("Canonical prediction text must be a string.")

    def serializable(self) -> dict[str, Any]:
        return asdict(self)
