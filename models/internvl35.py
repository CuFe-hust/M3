"""InternVL3.5 canonical wrapper. / InternVL3.5 统一封装。"""
from dataclasses import dataclass
from typing import Any
from PIL import Image
from data.schema import CanonicalPrediction, CanonicalSample
from models.common import attach_peft_adapter, build_text_prediction, first_parameter_device, load_rgb_images, resolve_dtype, task_max_new_tokens

@dataclass
class InternVL35Settings:
    model_id: str = "OpenGVLab/InternVL3_5-8B"
    adapter_id: str | None = None
    adapter_revision: str | None = None
    merge_adapter: bool = False
    dtype: str = "bfloat16"
    device_map: str = "auto"
    max_new_tokens: int = 256
    input_size: int = 448
    max_num_tiles: int = 12
    use_thumbnail: bool = True
    low_cpu_mem_usage: bool = True
    use_flash_attn: bool = False
    local_files_only: bool = False
    grounding_source_scale: float = 1000.0

class InternVL35Baseline:
    def __init__(self, settings: InternVL35Settings) -> None:
        if settings.input_size < 1 or settings.max_num_tiles < 1: raise ValueError("InternVL image settings must be positive.")
        self.settings = settings; self.adapter_merged = False; self.model, self.tokenizer = self._load()
    def _load(self) -> tuple[Any, Any]:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error: raise RuntimeError("Install requirements-models.txt before loading InternVL.") from error
        model = AutoModel.from_pretrained(self.settings.model_id, torch_dtype=resolve_dtype(torch, self.settings.dtype), low_cpu_mem_usage=self.settings.low_cpu_mem_usage, use_flash_attn=self.settings.use_flash_attn, trust_remote_code=True, device_map=self.settings.device_map, local_files_only=self.settings.local_files_only)
        model, self.adapter_merged = attach_peft_adapter(model, adapter_id=self.settings.adapter_id, adapter_revision=self.settings.adapter_revision, local_files_only=self.settings.local_files_only, merge_adapter=self.settings.merge_adapter)
        tokenizer = AutoTokenizer.from_pretrained(self.settings.model_id, trust_remote_code=True, use_fast=False, local_files_only=self.settings.local_files_only)
        model.eval(); return model, tokenizer
    def _prepare_image(self, image: Image.Image) -> Any:
        try:
            import torch
            from torchvision import transforms
            from torchvision.transforms.functional import InterpolationMode
        except ImportError as error: raise RuntimeError("Install torchvision for InternVL image preprocessing.") from error
        tiles = dynamic_preprocess(image, self.settings.input_size, self.settings.max_num_tiles, self.settings.use_thumbnail)
        transform = transforms.Compose([transforms.Lambda(lambda item: item.convert("RGB")), transforms.Resize((self.settings.input_size, self.settings.input_size), interpolation=InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        return torch.stack([transform(tile) for tile in tiles])
    def predict(self, sample: CanonicalSample) -> CanonicalPrediction:
        sample.validate(); images = load_rgb_images(sample.images); tensors = [self._prepare_image(image) for image in images]
        import torch
        counts = [tensor.size(0) for tensor in tensors]
        parameter = next(self.model.parameters())
        pixels = torch.cat(tensors, dim=0).to(device=first_parameter_device(self.model), dtype=parameter.dtype)
        prefix = "<image>" if len(images) == 1 else "\n".join(f"Image-{index}: <image>" for index in range(1, len(images) + 1))
        limit = task_max_new_tokens(sample, self.settings.max_new_tokens)
        answer = self.model.chat(self.tokenizer, pixels, f"{prefix}\n{sample.prompt}", {"max_new_tokens": limit, "do_sample": False}, num_patches_list=counts).strip()
        return build_text_prediction(sample=sample, text=answer, model_type="internvl35", model_id=self.settings.model_id, max_new_tokens=limit, grounding_source_scale=self.settings.grounding_source_scale, grounding_source_name="internvl35_normalized_0_1000", extra_meta={"base_model_id": self.settings.model_id, "adapter_id": self.settings.adapter_id, "adapter_merged": self.adapter_merged, "input_size": self.settings.input_size, "max_num_tiles": self.settings.max_num_tiles, "num_patches_list": counts})

def dynamic_preprocess(image: Image.Image, image_size: int, max_num: int, use_thumbnail: bool) -> list[Image.Image]:
    width, height = image.size; ratio = width / height
    candidates = sorted({(cols, rows) for tiles in range(1, max_num + 1) for cols in range(1, tiles + 1) for rows in range(1, tiles + 1) if 1 <= cols * rows <= max_num}, key=lambda item: item[0] * item[1])
    cols, rows = min(candidates, key=lambda item: abs(ratio - item[0] / item[1]))
    resized = image.resize((cols * image_size, rows * image_size))
    tiles = [resized.crop(((index % cols) * image_size, (index // cols) * image_size, (index % cols + 1) * image_size, (index // cols + 1) * image_size)) for index in range(cols * rows)]
    if use_thumbnail and len(tiles) > 1: tiles.append(image.resize((image_size, image_size)))
    return tiles
