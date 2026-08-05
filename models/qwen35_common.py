"""Shared Qwen3.5 runtime. / Qwen3.5 共享运行时。"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from data.schema import CanonicalPrediction, CanonicalSample
from models.common import build_text_prediction, first_parameter_device, load_rgb_images, resolve_dtype, task_max_new_tokens

@dataclass(frozen=True)
class Qwen35RuntimeConfig:
    model_id: str
    expected_variant: str
    dtype: str
    device_map: str
    max_new_tokens: int
    min_pixels: int | None
    max_pixels: int | None
    local_files_only: bool
    grounding_source_scale: float

def load_qwen35_runtime(runtime: Qwen35RuntimeConfig) -> tuple[Any, Any]:
    opposite = {"4b": "Qwen/Qwen3.5-9B", "9b": "Qwen/Qwen3.5-4B"}[runtime.expected_variant]
    if runtime.model_id == opposite:
        raise ValueError(f"The {runtime.expected_variant.upper()} wrapper cannot load {opposite}.")
    try:
        import torch
        from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration
    except ImportError as error:
        raise RuntimeError("Install requirements-models.txt before loading Qwen3.5.") from error
    config = AutoConfig.from_pretrained(runtime.model_id, local_files_only=runtime.local_files_only)
    if config.model_type != "qwen3_5":
        raise RuntimeError(f"Expected Qwen3.5 model_type 'qwen3_5', got {config.model_type!r}.")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(runtime.model_id, dtype=resolve_dtype(torch, runtime.dtype), device_map=runtime.device_map, local_files_only=runtime.local_files_only)
    kwargs = {key: value for key, value in (("min_pixels", runtime.min_pixels), ("max_pixels", runtime.max_pixels)) if value is not None}
    processor = AutoProcessor.from_pretrained(runtime.model_id, local_files_only=runtime.local_files_only, **kwargs)
    model.eval()
    return model, processor

def predict_qwen35(*, model: Any, processor: Any, sample: CanonicalSample, config: Qwen35RuntimeConfig, public_model_type: str) -> CanonicalPrediction:
    sample.validate()
    images = load_rgb_images(sample.images)
    messages = [{"role": "user", "content": [*[{"type": "image", "image": image} for image in images], {"type": "text", "text": sample.prompt}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=[prompt], images=images, padding=True, return_tensors="pt").to(first_parameter_device(model))
    limit = task_max_new_tokens(sample, config.max_new_tokens)
    generated = model.generate(**inputs, max_new_tokens=limit, do_sample=False)
    trimmed = [output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, generated)]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    variant = f"qwen3_5_{config.expected_variant}"
    return build_text_prediction(sample=sample, text=answer, model_type=public_model_type, model_id=config.model_id, max_new_tokens=limit, grounding_source_scale=config.grounding_source_scale, grounding_source_name="qwen35_normalized_0_1000", extra_meta={"model_variant": variant, "enable_thinking": False})
