"""Standalone Qwen3-VL-8B-Instruct wrapper with canonical prediction output.
独立输出统一预测格式的 Qwen3-VL-8B-Instruct 封装。

This wrapper is intentionally self-contained and does not reuse the 4B
``models.qwen3_vl.baseline`` implementation.
本封装刻意保持自包含，不复用 4B 的 ``models.qwen3_vl.baseline`` 实现。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from data.schema import CanonicalPrediction, CanonicalSample


TASK_MAX_NEW_TOKENS = {
    "caption": 512,
    "change_caption": 512,
    "vqa": 64,
    "grounding": 128,
}


# Official Hugging Face checkpoint id. This wrapper never downloads weights.
# Hugging Face 官方权重标识。本封装绝不下载权重。
QWEN3_VL_8B_INSTRUCT = "Qwen/Qwen3-VL-8B-Instruct"


@dataclass
class Qwen3VL8BSettings:
    """Runtime settings for the untouched Qwen3-VL-8B-Instruct checkpoint.
    原始 Qwen3-VL-8B-Instruct 权重的运行时设置。
    """

    model_id: str = QWEN3_VL_8B_INSTRUCT
    dtype: str = "auto"
    device_map: str = "auto"
    max_new_tokens: int = 256
    min_pixels: int | None = None
    max_pixels: int | None = None
    local_files_only: bool = False


class Qwen3VL8BInstruct:
    """Run the unmodified Qwen3-VL-8B-Instruct checkpoint on canonical samples.
    在统一样本上运行未修改的 Qwen3-VL-8B-Instruct 权重。
    """

    def __init__(self, settings: Qwen3VL8BSettings | None = None) -> None:
        self.settings = settings or Qwen3VL8BSettings()
        self.model, self.processor = self._load()

    def _load(self) -> tuple[Any, Any]:
        try:
            import torch
            import transformers
            from transformers import AutoConfig, AutoProcessor
        except ImportError as error:
            raise RuntimeError("Install requirements.txt before loading Qwen3-VL-8B.") from error

        dtype = _resolve_dtype(torch, self.settings.dtype)
        config = AutoConfig.from_pretrained(
            self.settings.model_id,
            local_files_only=self.settings.local_files_only,
        )
        model_factory = _qwen_model_factory(transformers, config.model_type)
        model = model_factory.from_pretrained(
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
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(text=[text], images=sample.images, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        max_new_tokens = min(self.settings.max_new_tokens, _task_max_new_tokens(sample))
        generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [output[len(input_ids) :] for input_ids, output in zip(inputs.input_ids, generated)]
        answer = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        raw_boxes = _extract_boxes(answer) if sample.task_type == "grounding" else []
        boxes = []
        coordinate_status = "not_applicable"
        source_coordinate_system = None
        coordinate_conversion = None
        if sample.task_type == "grounding":
            boxes, coordinate_status, source_coordinate_system, coordinate_conversion = _grounding_postprocess(raw_boxes, sample)
        return CanonicalPrediction(
            id=sample.id,
            task_type=sample.task_type,
            text=answer,
            answer=_choice_letter(answer) if sample.choices else answer,
            boxes=boxes,
            meta={
                "model_id": self.settings.model_id,
                "max_new_tokens": max_new_tokens,
                "raw_text": answer,
                "raw_prediction": answer,
                "raw_grounding_boxes": raw_boxes,
                "official_pixel_grounding_boxes": _official_pixel_boxes(raw_boxes, sample),
                "source_coordinate_system": source_coordinate_system,
                "canonical_coordinate_system": "normalized_0_100" if sample.task_type == "grounding" else None,
                "coordinate_conversion": coordinate_conversion,
                "coordinate_status": coordinate_status,
            },
        )


def _resolve_dtype(torch: Any, name: str) -> Any:
    if name == "auto":
        return "auto"
    available = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in available:
        raise ValueError(f"Unsupported dtype: {name}")
    return available[name]


def _qwen_model_factory(transformers: Any, model_type: str) -> Any:
    """Select the native class for the Qwen3-VL multimodal checkpoint.
    为 Qwen3-VL 多模态权重选择原生类。
    """

    class_name = {
        "qwen3_vl": "Qwen3VLForConditionalGeneration",
    }.get(model_type)
    if class_name is None or not hasattr(transformers, class_name):
        raise RuntimeError(
            f"Unsupported Qwen model_type {model_type!r}; install a Transformers version "
            "that provides the matching native model class."
        )
    return getattr(transformers, class_name)


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


def _task_max_new_tokens(sample: CanonicalSample) -> int:
    if sample.task_type == "caption" and sample.meta.get("source") == "XLRS-Bench full English caption release":
        return 768
    return TASK_MAX_NEW_TOKENS[sample.task_type]


def _grounding_postprocess(
    raw_boxes: list[list[float]], sample: CanonicalSample
) -> tuple[list[list[float]], str, str, str]:
    coordinate_system = "qwen3vl_normalized_0_1000"
    conversion = "value * 100 / 1000"
    if not raw_boxes:
        return [], "missing_box", coordinate_system, conversion
    canonical = [[value * 100 / 1000 for value in box] for box in raw_boxes]
    in_range = all(0 <= value <= 1000 for box in raw_boxes for value in box)
    return canonical, "converted_model_native" if in_range else "out_of_native_range", coordinate_system, conversion


def _official_pixel_boxes(raw_boxes: list[list[float]], sample: CanonicalSample) -> list[list[float]]:
    if sample.task_type != "grounding" or "XLRS-Bench" not in str(sample.meta.get("source", "")):
        return []
    width = float(sample.meta["image_width"])
    height = float(sample.meta["image_height"])
    return [[box[0] / 1000 * width, box[1] / 1000 * height, box[2] / 1000 * width, box[3] / 1000 * height] for box in raw_boxes]
