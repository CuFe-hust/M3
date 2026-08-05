"""Ovis2.5 canonical wrapper. / Ovis2.5 统一封装。"""
from dataclasses import dataclass
from typing import Any
from data.schema import CanonicalPrediction, CanonicalSample
from models.common import build_text_prediction, load_rgb_images, resolve_dtype, task_max_new_tokens

@dataclass
class Ovis25Settings:
    model_id: str = "ATH-MaaS/Ovis2.5-2B"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    max_new_tokens: int = 256
    min_pixels: int = 448 * 448
    max_pixels: int = 1792 * 1792
    local_files_only: bool = False
    grounding_source_scale: float = 100.0

class Ovis25Baseline:
    def __init__(self, settings: Ovis25Settings) -> None:
        if settings.min_pixels <= 0 or settings.max_pixels < settings.min_pixels:
            raise ValueError("Ovis pixel bounds are invalid.")
        self.settings = settings; self.model = self._load()
    def _load(self) -> Any:
        try:
            import torch
            from transformers import AutoModelForCausalLM
        except ImportError as error:
            raise RuntimeError("Install requirements-models.txt before loading Ovis.") from error
        model = AutoModelForCausalLM.from_pretrained(self.settings.model_id, torch_dtype=resolve_dtype(torch, self.settings.dtype), trust_remote_code=True, device_map=self.settings.device_map, local_files_only=self.settings.local_files_only)
        model.eval(); return model
    def predict(self, sample: CanonicalSample) -> CanonicalPrediction:
        sample.validate(); images = load_rgb_images(sample.images); limit = task_max_new_tokens(sample, self.settings.max_new_tokens)
        image_prefix = "\n".join("<image>" for _ in images)
        prompt = f"{image_prefix}\n{sample.prompt}"
        if hasattr(self.model, "chat"):
            result = self.model.chat(prompt=prompt, images=images, videos=None, min_pixels=self.settings.min_pixels, max_pixels=self.settings.max_pixels, enable_thinking=False, do_sample=False, max_new_tokens=limit)
            answer = result if isinstance(result, str) else result[0] if isinstance(result, tuple) and result and isinstance(result[0], str) else None
        elif hasattr(self.model, "preprocess_inputs"):
            messages = [{"role": "user", "content": [*[{"type": "image", "image": image} for image in images], {"type": "text", "text": sample.prompt}]}]
            input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(messages=messages, min_pixels=self.settings.min_pixels, max_pixels=self.settings.max_pixels, add_generation_prompt=True, enable_thinking=False)
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            pixel_values = pixel_values.to(device) if pixel_values is not None else None
            grid_thws = grid_thws.to(device) if grid_thws is not None else None
            outputs = self.model.generate(inputs=input_ids, pixel_values=pixel_values, grid_thws=grid_thws, enable_thinking=False, do_sample=False, max_new_tokens=limit)
            answer = self.model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            answer = None
        if answer is None: raise TypeError(f"Unsupported Ovis chat result: {type(result).__name__}")
        answer = answer.strip()
        return build_text_prediction(sample=sample, text=answer, model_type="ovis25", model_id=self.settings.model_id, max_new_tokens=limit, grounding_source_scale=self.settings.grounding_source_scale, grounding_source_name="ovis25_normalized_0_100", extra_meta={"min_pixels": self.settings.min_pixels, "max_pixels": self.settings.max_pixels, "enable_thinking": False})
