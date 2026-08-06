"""Qwen3-VL baseline wrapper with lazy loading and plain text generation.

Qwen3-VL 基线封装：惰性加载处理器与模型，提供基于 UnifiedSample 的文本
生成。评测后处理（选项字母、框提取等）不属于本模块，由 evaluation 层负责；
本模块不包含任何 Agent 逻辑。
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from models.qwen_transformers import QwenTransformersError


class Qwen3VLSettings(BaseModel):
    """Declaration-only settings for the Qwen3-VL baseline wrapper.
    Qwen3-VL 基线封装的纯声明配置。"""

    model_config = ConfigDict(extra="forbid")

    model: str = "qwen3-vl-4b-instruct"
    max_new_tokens: int = Field(default=128, gt=0)
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    device_map: str = "auto"
    allow_download: bool = False
    local_files_only: bool | None = None
    revision: str | None = None

    @model_validator(mode="after")
    def validate_offline_flags(self) -> "Qwen3VLSettings":
        """local_files_only and allow_download must not conflict.
        local_files_only 与 allow_download 不得冲突。"""
        if self.local_files_only is not None and self.allow_download and self.local_files_only:
            raise ValueError(
                "allow_download and local_files_only=True cannot both be set"
            )
        return self

    def effective_local_files_only(self) -> bool:
        """Resolved offline flag: local-first by default. / 解析后的离线开关。"""
        if self.local_files_only is not None:
            return self.local_files_only
        return not self.allow_download


class Qwen3VLBaseline:
    """Zero-shot text baseline over a local Qwen3-VL checkpoint.
    本地 Qwen3-VL 权重的零样本文本基线。"""

    name = "qwen3_vl_baseline"

    def __init__(
        self,
        settings: Qwen3VLSettings | Any,
        *,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        self.settings = settings
        started = time.perf_counter()
        if (model is None) != (processor is None):
            raise ValueError("model and processor must be supplied together")
        self.model, self.processor = (
            (model, processor) if model is not None else self._load()
        )
        self.load_seconds = round(time.perf_counter() - started, 6)

    def _load(self) -> tuple[Any, Any]:
        """Load the declared checkpoint lazily without network fallback.
        惰性加载声明的权重，不做网络回退。"""
        try:
            import torch
            import transformers
            from transformers import AutoConfig, AutoProcessor
        except ImportError as error:
            raise QwenTransformersError(
                "Install requirements.txt before loading local Qwen."
            ) from error
        local_files_only = self.settings.effective_local_files_only()
        revision = self.settings.revision
        dtype: Any = "auto"
        if self.settings.dtype != "auto":
            dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[self.settings.dtype]
        config = AutoConfig.from_pretrained(
            self.settings.model,
            local_files_only=local_files_only,
            trust_remote_code=True,
            revision=revision,
        )
        class_name = "Qwen3VLForConditionalGeneration"
        model_factory = getattr(transformers, class_name, None)
        if model_factory is None:
            raise QwenTransformersError(
                "Qwen3-VL baseline requires a Transformers version providing "
                f"{class_name}."
            )
        model = model_factory.from_pretrained(
            self.settings.model,
            dtype=dtype,
            device_map=self.settings.device_map,
            local_files_only=local_files_only,
            trust_remote_code=True,
            revision=revision,
        )
        processor = AutoProcessor.from_pretrained(
            self.settings.model,
            local_files_only=local_files_only,
            trust_remote_code=True,
            revision=revision,
        )
        model.eval()
        return model, processor

    def generate_text(self, *, text: str, images: list[Any] | None = None) -> str:
        """Run deterministic greedy text generation for one prompt using an
        explicit multimodal content list.
        使用显式多模态 content 列表对一条提示执行确定性贪心文本生成。"""
        content: list[dict[str, Any]] = []
        for image in images or []:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt], images=images or None, padding=True, return_tensors="pt"
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.settings.max_new_tokens,
            do_sample=False,
        )
        input_tokens = int(inputs["input_ids"].shape[-1])
        trimmed = [output[input_tokens:] for output in generated]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
