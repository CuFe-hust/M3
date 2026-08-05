"""MiniCPM-V-4.6 canonical wrapper. / MiniCPM-V-4.6 统一封装。"""
from dataclasses import dataclass
from typing import Any
from data.schema import CanonicalPrediction, CanonicalSample
from models.common import build_text_prediction, first_parameter_device, load_rgb_images, resolve_dtype, task_max_new_tokens

@dataclass
class MiniCPMV46Settings:
    model_id: str = "openbmb/MiniCPM-V-4.6"
    dtype: str = "auto"
    device_map: str = "auto"
    max_new_tokens: int = 256
    downsample_mode: str = "16x"
    max_slice_nums: int = 36
    local_files_only: bool = False
    grounding_source_scale: float = 100.0

class MiniCPMV46Baseline:
    def __init__(self, settings: MiniCPMV46Settings) -> None:
        if settings.downsample_mode not in {"4x", "16x"} or settings.max_slice_nums < 1:
            raise ValueError("downsample_mode must be 4x/16x and max_slice_nums must be positive.")
        self.settings = settings
        self.model, self.processor = self._load()
    def _load(self) -> tuple[Any, Any]:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as error:
            raise RuntimeError("Install requirements-models.txt before loading MiniCPM-V.") from error
        processor = AutoProcessor.from_pretrained(self.settings.model_id, local_files_only=self.settings.local_files_only)
        model = AutoModelForImageTextToText.from_pretrained(self.settings.model_id, torch_dtype=resolve_dtype(torch, self.settings.dtype), device_map=self.settings.device_map, local_files_only=self.settings.local_files_only)
        model.eval(); return model, processor
    def predict(self, sample: CanonicalSample) -> CanonicalPrediction:
        sample.validate(); images = load_rgb_images(sample.images)
        messages = [{"role": "user", "content": [*[{"type": "image", "image": image} for image in images], {"type": "text", "text": sample.prompt}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", enable_thinking=False, downsample_mode=self.settings.downsample_mode, max_slice_nums=self.settings.max_slice_nums).to(first_parameter_device(self.model))
        limit = task_max_new_tokens(sample, self.settings.max_new_tokens)
        generated = self.model.generate(**inputs, downsample_mode=self.settings.downsample_mode, max_new_tokens=limit, do_sample=False)
        trimmed = [output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, generated)]
        answer = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return build_text_prediction(sample=sample, text=answer, model_type="minicpmv46", model_id=self.settings.model_id, max_new_tokens=limit, grounding_source_scale=self.settings.grounding_source_scale, grounding_source_name="minicpmv46_normalized_0_100", extra_meta={"downsample_mode": self.settings.downsample_mode, "max_slice_nums": self.settings.max_slice_nums, "enable_thinking": False})
