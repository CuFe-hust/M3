"""Shared canonical model helpers. / 统一模型共享辅助函数。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image

from data.schema import CanonicalPrediction, CanonicalSample

TASK_MAX_NEW_TOKENS = {"caption": 512, "change_caption": 512, "vqa": 64, "grounding": 128}


def resolve_dtype(torch_module: Any, name: str) -> Any:
    if name == "auto":
        return "auto"
    available = {"float16": torch_module.float16, "bfloat16": torch_module.bfloat16, "float32": torch_module.float32}
    if name not in available:
        raise ValueError(f"Unsupported dtype: {name}")
    return available[name]


def task_max_new_tokens(sample: CanonicalSample, configured_max: int) -> int:
    limit = 768 if sample.task_type == "caption" and sample.meta.get("source") == "XLRS-Bench full English caption release" else TASK_MAX_NEW_TOKENS[sample.task_type]
    return min(configured_max, limit)


def load_rgb_images(images: list[Any]) -> list[Image.Image]:
    loaded: list[Image.Image] = []
    for value in images:
        if isinstance(value, Image.Image):
            loaded.append(value.convert("RGB").copy())
        elif isinstance(value, (str, Path)):
            with Image.open(value) as image:
                loaded.append(image.convert("RGB").copy())
        else:
            raise TypeError(f"Unsupported image input: {type(value).__name__}")
    return loaded


def choice_letter(text: str) -> str:
    match = re.search(r"(?<![A-Z])([A-E])(?![A-Z])", text.upper())
    return match.group(1) if match else text.strip()


def extract_boxes(text: str) -> list[list[float]]:
    groups = re.findall(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", text)
    return [[float(value) for value in group] for group in groups]


def first_parameter_device(model: Any) -> Any:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def attach_peft_adapter(model: Any, *, adapter_id: str | None, adapter_revision: str | None, local_files_only: bool, merge_adapter: bool) -> tuple[Any, bool]:
    if adapter_id is None:
        return model, False
    try:
        from peft import PeftModel
    except ImportError as error:
        raise RuntimeError("Install peft to load a LoRA adapter.") from error
    model = PeftModel.from_pretrained(model, adapter_id, is_trainable=False, revision=adapter_revision, local_files_only=local_files_only)
    if merge_adapter:
        model = model.merge_and_unload()
    return model, merge_adapter


def build_text_prediction(*, sample: CanonicalSample, text: str, model_type: str, model_id: str, max_new_tokens: int, grounding_source_scale: float, grounding_source_name: str, extra_meta: dict[str, Any] | None = None) -> CanonicalPrediction:
    raw_boxes = extract_boxes(text) if sample.task_type == "grounding" else []
    boxes = [[value * 100.0 / grounding_source_scale for value in box] for box in raw_boxes]
    grounding = sample.task_type == "grounding"
    in_range = all(0 <= value <= grounding_source_scale for box in raw_boxes for value in box)
    status = "not_applicable" if not grounding else ("missing_box" if not raw_boxes else ("converted_model_native" if in_range else "out_of_native_range"))
    pixel_boxes: list[list[float]] = []
    if grounding and "XLRS-Bench" in str(sample.meta.get("source", "")):
        width, height = float(sample.meta["image_width"]), float(sample.meta["image_height"])
        pixel_boxes = [[b[0] / grounding_source_scale * width, b[1] / grounding_source_scale * height, b[2] / grounding_source_scale * width, b[3] / grounding_source_scale * height] for b in raw_boxes]
    meta = {"model_type": model_type, "model_id": model_id, "max_new_tokens": max_new_tokens, "raw_text": text, "raw_prediction": text, "raw_grounding_boxes": raw_boxes, "official_pixel_grounding_boxes": pixel_boxes, "source_coordinate_system": grounding_source_name if grounding else None, "canonical_coordinate_system": "normalized_0_100" if grounding else None, "coordinate_conversion": f"value * 100 / {grounding_source_scale:g}" if grounding else None, "coordinate_status": status}
    meta.update(extra_meta or {})
    return CanonicalPrediction(id=sample.id, task_type=sample.task_type, text=text, answer=choice_letter(text) if sample.choices else text, boxes=boxes, meta=meta)
