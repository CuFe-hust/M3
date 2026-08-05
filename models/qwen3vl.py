"""Qwen3-VL canonical wrapper. / Qwen3-VL 统一推理封装。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from data.schema import CanonicalPrediction, CanonicalSample
from models.common import attach_peft_adapter, build_text_prediction, choice_letter as _choice_letter, extract_boxes as _extract_boxes, first_parameter_device, load_rgb_images, resolve_dtype, task_max_new_tokens

@dataclass
class Qwen3VLSettings:
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    adapter_id: str | None = None
    adapter_revision: str | None = None
    merge_adapter: bool = False
    dtype: str = "auto"
    device_map: str = "auto"
    max_new_tokens: int = 256
    min_pixels: int | None = None
    max_pixels: int | None = None
    local_files_only: bool = False
    grounding_source_scale: float = 1000.0

class Qwen3VLBaseline:
    """Run Qwen3-VL on canonical samples. / 在统一样本上运行 Qwen3-VL。"""
    def __init__(self, settings: Qwen3VLSettings) -> None:
        self.settings = settings
        self.adapter_merged = False
        self.model, self.processor = self._load()

    def _load(self) -> tuple[Any, Any]:
        try:
            import torch
            from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise RuntimeError("Install requirements-models.txt before loading Qwen3-VL.") from error
        config = AutoConfig.from_pretrained(self.settings.model_id, local_files_only=self.settings.local_files_only)
        if config.model_type != "qwen3_vl":
            raise RuntimeError(f"Qwen3VLBaseline does not support model_type {config.model_type!r}; use model.type=qwen35_4b or model.type=qwen35_9b for Qwen3.5.")
        model = Qwen3VLForConditionalGeneration.from_pretrained(self.settings.model_id, dtype=resolve_dtype(torch, self.settings.dtype), device_map=self.settings.device_map, local_files_only=self.settings.local_files_only)
        model, self.adapter_merged = attach_peft_adapter(model, adapter_id=self.settings.adapter_id, adapter_revision=self.settings.adapter_revision, local_files_only=self.settings.local_files_only, merge_adapter=self.settings.merge_adapter)
        kwargs = {key: value for key, value in (("min_pixels", self.settings.min_pixels), ("max_pixels", self.settings.max_pixels)) if value is not None}
        processor = AutoProcessor.from_pretrained(self.settings.model_id, local_files_only=self.settings.local_files_only, **kwargs)
        model.eval()
        return model, processor

    def predict(self, sample: CanonicalSample) -> CanonicalPrediction:
        sample.validate()
        images = load_rgb_images(sample.images)
        messages = [{"role": "user", "content": [*[{"type": "image", "image": image} for image in images], {"type": "text", "text": sample.prompt}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=images, padding=True, return_tensors="pt").to(first_parameter_device(self.model))
        limit = task_max_new_tokens(sample, self.settings.max_new_tokens)
        generated = self.model.generate(**inputs, max_new_tokens=limit, do_sample=False)
        trimmed = [output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, generated)]
        answer = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return build_text_prediction(sample=sample, text=answer, model_type="qwen3vl", model_id=self.settings.model_id, max_new_tokens=limit, grounding_source_scale=self.settings.grounding_source_scale, grounding_source_name="qwen3vl_normalized_0_1000", extra_meta={"base_model_id": self.settings.model_id, "adapter_id": self.settings.adapter_id, "adapter_merged": self.adapter_merged, "model_variant": "qwen3_vl_4b_instruct"})

# Preserve helpers imported by existing callers. / 保留现有调用方导入的辅助函数。
def _message_content(sample: CanonicalSample) -> list[dict[str, Any]]:
    return [*[{"type": "image", "image": image} for image in sample.images], {"type": "text", "text": sample.prompt}]

def _task_max_new_tokens(sample: CanonicalSample) -> int:
    return task_max_new_tokens(sample, 768)

def _grounding_postprocess(raw_boxes: list[list[float]], sample: CanonicalSample) -> tuple[list[list[float]], str, str, str]:
    scale = 1000.0
    boxes = [[value * 100.0 / scale for value in box] for box in raw_boxes]
    status = "missing_box" if not raw_boxes else ("converted_model_native" if all(0 <= value <= scale for box in raw_boxes for value in box) else "out_of_native_range")
    return boxes, status, "qwen3vl_normalized_0_1000", "value * 100 / 1000"

def _official_pixel_boxes(raw_boxes: list[list[float]], sample: CanonicalSample) -> list[list[float]]:
    if sample.task_type != "grounding" or "XLRS-Bench" not in str(sample.meta.get("source", "")): return []
    width, height = float(sample.meta["image_width"]), float(sample.meta["image_height"])
    return [[box[0] / 1000 * width, box[1] / 1000 * height, box[2] / 1000 * width, box[3] / 1000 * height] for box in raw_boxes]
