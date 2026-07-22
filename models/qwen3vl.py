"""Qwen3-VL baseline wrapper with canonical prediction output.
输出统一预测格式的 Qwen3-VL 基线封装。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from data.schema import CanonicalPrediction, CanonicalSample


@dataclass
class Qwen3VLSettings:
    """Runtime settings for the untouched Qwen3-VL base checkpoint.
    原始 Qwen3-VL 基础权重的运行时设置。
    """

    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    dtype: str = "auto"
    device_map: str = "auto"
    max_new_tokens: int = 256
    min_pixels: int | None = None
    max_pixels: int | None = None
    local_files_only: bool = False


class Qwen3VLBaseline:
    """Run one unmodified Qwen3-VL checkpoint on canonical samples.
    在统一样本上运行一个未修改的 Qwen3-VL 权重。
    """

    def __init__(self, settings: Qwen3VLSettings) -> None:
        self.settings = settings
        self.model, self.processor = self._load()

    def _load(self) -> tuple[Any, Any]:
        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise RuntimeError("Install requirements.txt before loading Qwen3-VL.") from error

        dtype = _resolve_dtype(torch, self.settings.dtype)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.settings.model_id,
            dtype=dtype,
            device_map=self.settings.device_map,
            local_files_only=self.settings.local_files_only,
        )
        processor_kwargs = {}
        if self.settings.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.settings.min_pixels
        if self.settings.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.settings.max_pixels
        processor = AutoProcessor.from_pretrained(
            self.settings.model_id,
            local_files_only=self.settings.local_files_only,
            **processor_kwargs,
        )
        model.eval()
        return model, processor

    def predict(self, sample: CanonicalSample) -> CanonicalPrediction:
        """Generate one deterministic baseline prediction for a sample.
        为一个样本生成一条确定性的基线预测。
        """

        sample.validate()
        messages = [{"role": "user", "content": _message_content(sample)}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=sample.images, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        generated = self.model.generate(**inputs, max_new_tokens=self.settings.max_new_tokens, do_sample=False)
        trimmed = [output[len(input_ids) :] for input_ids, output in zip(inputs.input_ids, generated)]
        answer = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        boxes = _extract_boxes(answer) if sample.task_type == "grounding" else []
        return CanonicalPrediction(
            id=sample.id,
            task_type=sample.task_type,
            text=answer,
            answer=_choice_letter(answer) if sample.choices else answer,
            boxes=boxes,
            meta={"model_id": self.settings.model_id, "raw_text": answer},
        )


def _resolve_dtype(torch: Any, name: str) -> Any:
    if name == "auto":
        return "auto"
    available = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in available:
        raise ValueError(f"Unsupported dtype: {name}")
    return available[name]


def _message_content(sample: CanonicalSample) -> list[dict[str, Any]]:
    content = [{"type": "image", "image": image} for image in sample.images]
    content.append({"type": "text", "text": sample.prompt})
    return content


def _choice_letter(text: str) -> str:
    match = re.search(r"(?<![A-Z])([A-E])(?![A-Z])", text.upper())
    return match.group(1) if match else text.strip()


def _extract_boxes(text: str) -> list[list[float]]:
    groups = re.findall(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", text)
    return [[float(value) for value in group] for group in groups]
