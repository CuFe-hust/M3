"""Benchmark VLM wrapper for Qwen3-VL-4B and InternVL3.5-8B with optional LoRA.
支持 Qwen3-VL-4B / InternVL3.5-8B 及可选 LoRA 的评测模型封装。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PIL import Image

from data.schema import CanonicalPrediction, CanonicalSample
from models.qwen3vl import (
    TASK_MAX_NEW_TOKENS,
    _choice_letter,
    _extract_boxes,
    _grounding_postprocess,
    _message_content,
)


MODEL_TYPES = ("qwen3-vl-4b", "internvl3.5-8b")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resolve_load_paths(model_path: Path, lora_path: Path | None) -> tuple[Path, Path | None]:
    """Resolve the effective model path and an optional PEFT adapter path.
    解析有效模型路径与可选的 PEFT 适配器路径。

    A LoRA directory may contain either ``adapter_config.json`` (PEFT adapter,
    loaded on top of ``model_path``) or a full merged ``config.json`` + weights
    directory (used directly as the effective model).
    LoRA 目录可以是含 ``adapter_config.json`` 的 PEFT 适配器（叠加在基座上），
    也可以是含 ``config.json`` 与权重的完整合并目录（直接作为有效模型）。
    """

    if lora_path is None:
        return model_path, None
    lora_path = lora_path.expanduser().resolve()
    if (lora_path / "adapter_config.json").is_file():
        return model_path, lora_path
    if (lora_path / "config.json").is_file() and any(lora_path.rglob("*.safetensors")):
        return lora_path, None
    raise ValueError(
        f"lora_path {lora_path} is neither a PEFT adapter dir nor a merged full-model dir. "
        f"lora_path 既不是 PEFT 适配器目录，也不是合并后的完整模型目录。"
    )


def resolve_device(requested: str | None) -> str:
    """Pick a concrete torch device string for inference.
    为推理选择一个具体的 torch 设备字符串。
    """

    if requested not in (None, "", "auto"):
        return requested
    torch = _torch()
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BenchmarkVLM:
    """Run Qwen3-VL-4B or InternVL3.5-8B on canonical benchmark samples.
    在统一评测样本上运行 Qwen3-VL-4B 或 InternVL3.5-8B。
    """

    def __init__(
        self,
        model_type: str,
        model_path: Path,
        lora_path: Path | None = None,
        dtype: str = "bfloat16",
        device: str | None = None,
        local_files_only: bool = True,
        max_new_tokens: int = 256,
        max_tiles: int = 12,
    ) -> None:
        if model_type not in MODEL_TYPES:
            raise ValueError(f"Unsupported model_type {model_type!r}; choose from {MODEL_TYPES}")
        self.model_type = model_type
        self.model_path = Path(model_path).expanduser().resolve()
        self.lora_path = Path(lora_path).expanduser().resolve() if lora_path else None
        self.effective_model_path, self.adapter_path = resolve_load_paths(
            self.model_path, self.lora_path
        )
        self.dtype = dtype
        self.device = resolve_device(device)
        self.local_files_only = local_files_only
        self.max_new_tokens = int(max_new_tokens)
        self.max_tiles = int(max_tiles)
        self.model: Any = None
        self.processor: Any = None
        self.tokenizer: Any = None
        self._load()

    def _load(self) -> None:
        if self.model_type == "qwen3-vl-4b":
            self._load_qwen3()
        else:
            self._load_internvl()

    def _load_qwen3(self) -> None:
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise RuntimeError("Install transformers >= 4.57 before loading Qwen3-VL.") from error
        dtype = _torch_dtype(self.dtype)
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "local_files_only": self.local_files_only,
            "low_cpu_mem_usage": True,
        }
        if self.device.startswith("cuda") or self.device == "auto":
            load_kwargs["device_map"] = "auto" if self.device == "auto" else {"": self.device}
        model = Qwen3VLForConditionalGeneration.from_pretrained(self.effective_model_path, **load_kwargs)
        if self.adapter_path is not None:
            model = _load_peft_adapter(model, self.adapter_path)
        if "device_map" not in load_kwargs:
            model = model.to(self.device)
        model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.effective_model_path,
            local_files_only=self.local_files_only,
        )
        self.model = model

    def _load_internvl(self) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Install transformers before loading InternVL3.5.") from error
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.effective_model_path,
                trust_remote_code=True,
                use_fast=False,
                fix_mistral_regex=True,
                local_files_only=self.local_files_only,
            )
        except TypeError:
            tokenizer = AutoTokenizer.from_pretrained(
                self.effective_model_path,
                trust_remote_code=True,
                use_fast=False,
                local_files_only=self.local_files_only,
            )
        model = AutoModel.from_pretrained(
            self.effective_model_path,
            torch_dtype=_torch_dtype(self.dtype),
            low_cpu_mem_usage=True,
            use_flash_attn=False,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        if self.adapter_path is not None:
            model = _load_peft_adapter(model, self.adapter_path)
        model = model.eval().to(self.device)
        self.tokenizer = tokenizer
        self.model = model

    def predict(self, sample: CanonicalSample) -> CanonicalPrediction:
        """Produce one canonical prediction with latency/VRAM metadata.
        生成一条带时延与显存元数据的统一预测。
        """

        sample.validate()
        started = time.perf_counter()
        if self.model_type == "qwen3-vl-4b":
            prediction = self._predict_qwen3(sample)
        else:
            prediction = self._predict_internvl(sample)
        elapsed = time.perf_counter() - started
        prediction.meta["latency_seconds"] = round(elapsed, 6)
        prediction.meta["peak_vram_gb"] = _peak_vram_gb(self.device)
        return prediction

    def _predict_qwen3(self, sample: CanonicalSample) -> CanonicalPrediction:
        messages = [{"role": "user", "content": _message_content(sample)}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=sample.images, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        max_new_tokens = min(self.max_new_tokens, _task_max_new_tokens(sample))
        generated = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [output[len(input_ids) :] for input_ids, output in zip(inputs.input_ids, generated)]
        answer = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        raw_boxes = _extract_boxes(answer) if sample.task_type == "grounding" else []
        boxes: list[list[float]] = []
        coordinate_status = "not_applicable"
        source_coordinate_system: str | None = None
        coordinate_conversion: str | None = None
        if sample.task_type == "grounding":
            boxes, coordinate_status, source_coordinate_system, coordinate_conversion = _grounding_postprocess(
                raw_boxes, sample
            )
        return CanonicalPrediction(
            id=sample.id,
            task_type=sample.task_type,
            text=answer,
            answer=_choice_letter(answer) if sample.choices else answer,
            boxes=boxes,
            meta={
                "model_type": self.model_type,
                "model_path": str(self.effective_model_path),
                "adapter_path": str(self.adapter_path) if self.adapter_path else None,
                "raw_text": answer,
                "raw_grounding_boxes": raw_boxes,
                "source_coordinate_system": source_coordinate_system,
                "canonical_coordinate_system": "normalized_0_100" if sample.task_type == "grounding" else None,
                "coordinate_conversion": coordinate_conversion,
                "coordinate_status": coordinate_status,
            },
        )

    def _predict_internvl(self, sample: CanonicalSample) -> CanonicalPrediction:
        torch = _torch()
        tensors = [_internvl_image_tensor(image, self.max_tiles) for image in sample.images]
        pixel_values = torch.cat(tensors, dim=0).to(device=self.device, dtype=_torch_dtype(self.dtype))
        num_patches_list = [tensor.size(0) for tensor in tensors]
        prompt = _internvl_prompt(sample)
        max_new_tokens = min(self.max_new_tokens, _task_max_new_tokens(sample))
        generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}
        chat_kwargs: dict[str, Any] = {}
        if len(tensors) > 1:
            chat_kwargs["num_patches_list"] = num_patches_list
        chat_model = getattr(self.model, "base_model", self.model)
        with torch.inference_mode():
            raw = chat_model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config,
                history=None,
                return_history=False,
                **chat_kwargs,
            )
        answer = str(raw).strip()
        raw_boxes = _extract_boxes(answer) if sample.task_type == "grounding" else []
        boxes: list[list[float]] = []
        coordinate_status = "not_applicable"
        if sample.task_type == "grounding":
            boxes, coordinate_status = _internvl_grounding_boxes(raw_boxes)
        return CanonicalPrediction(
            id=sample.id,
            task_type=sample.task_type,
            text=answer,
            answer=_choice_letter(answer) if sample.choices else answer,
            boxes=boxes,
            meta={
                "model_type": self.model_type,
                "model_path": str(self.effective_model_path),
                "adapter_path": str(self.adapter_path) if self.adapter_path else None,
                "raw_text": answer,
                "raw_grounding_boxes": raw_boxes,
                "source_coordinate_system": "internvl_prompt_normalized_0_100",
                "canonical_coordinate_system": "normalized_0_100" if sample.task_type == "grounding" else None,
                "coordinate_conversion": "value as-is" if sample.task_type == "grounding" else None,
                "coordinate_status": coordinate_status,
            },
        )


def _load_peft_adapter(model: Any, adapter_path: Path) -> Any:
    try:
        from peft import PeftModel
    except ImportError as error:
        raise RuntimeError("Install peft to load LoRA adapters. 加载 LoRA 需要安装 peft。") from error
    return PeftModel.from_pretrained(model, str(adapter_path))


def _torch_dtype(name: str) -> Any:
    torch = _torch()
    if name == "auto":
        return "auto"
    available = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in available:
        raise ValueError(f"Unsupported dtype: {name}")
    return available[name]


def _peak_vram_gb(device: str) -> float:
    torch = _torch()
    if not (device.startswith("cuda") and torch.cuda.is_available()):
        return 0.0
    try:
        index = int(device.split(":")[1]) if ":" in device else 0
    except (IndexError, ValueError):
        index = 0
    return round(torch.cuda.max_memory_allocated(index) / 1024**3, 6)


def _task_max_new_tokens(sample: CanonicalSample) -> int:
    if sample.task_type == "caption" and sample.meta.get("source", "").startswith("XLRS-Bench"):
        return 768
    return TASK_MAX_NEW_TOKENS[sample.task_type]


def _internvl_grounding_boxes(raw_boxes: list[list[float]]) -> tuple[list[list[float]], str]:
    """Convert InternVL prompt-space boxes (0..100 or 0..1) to canonical 0..100.
    将 InternVL 提示空间框（0..100 或 0..1）转换为统一 0..100。
    """

    if not raw_boxes:
        return [], "missing_box"
    canonical = []
    for box in raw_boxes:
        scale = 100.0 if max(abs(value) for value in box) <= 1 else 1.0
        canonical.append([value * scale for value in box])
    return canonical, "converted_prompt_native"


def _internvl_prompt(sample: CanonicalSample) -> str:
    """Build the InternVL3.5 chat prompt with image placeholders.
    构建带图像占位符的 InternVL3.5 对话提示。
    """

    if len(sample.images) > 1:
        return (
            "Image-1 (before): <image>\n"
            "Image-2 (after): <image>\n"
            f"{sample.prompt}"
        )
    return f"<image>\n{sample.prompt}"


def _internvl_image_tensor(image: Any, max_num: int = 12) -> torch.Tensor:
    """Preprocess one image into InternVL dynamic tiles (official 448px path).
    按官方 448px 动态切图方式预处理一张图像。
    """

    if isinstance(image, Image.Image):
        opened = image.convert("RGB")
    else:
        with Image.open(str(image)) as loaded:
            opened = loaded.convert("RGB")
    tiles = dynamic_preprocess(opened, image_size=448, use_thumbnail=True, max_num=max_num)
    transform = _internvl_transform(448)
    return torch.stack([transform(tile) for tile in tiles])


def _internvl_transform(input_size: int) -> Any:
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    return T.Compose(
        [
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    """InternVL repository official dynamic preprocessing.
    复用 InternVL 仓库官方动态切图预处理。
    """

    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    target_ratio = _closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_ratio[0]
    target_height = image_size * target_ratio[1]
    blocks = target_ratio[0] * target_ratio[1]
    resized = image.resize((target_width, target_height))
    processed = []
    for block in range(blocks):
        box = (
            (block % (target_width // image_size)) * image_size,
            (block // (target_width // image_size)) * image_size,
            ((block % (target_width // image_size)) + 1) * image_size,
            ((block // (target_width // image_size)) + 1) * image_size,
        )
        processed.append(resized.crop(box))
    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))
    return processed


def _closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        ratio_diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def _torch() -> Any:
    """Import torch lazily so non-GPU import surfaces fail only on use.
    惰性导入 torch，避免在非 GPU 环境导入即失败。
    """

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Install torch before running the benchmark VLM. 运行评测模型需要安装 torch。") from error
    return torch


__all__ = [
    "BenchmarkVLM",
    "MODEL_TYPES",
    "dynamic_preprocess",
    "resolve_device",
    "resolve_load_paths",
]
