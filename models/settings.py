"""Model settings declarations; no environment reading, no secret values.

模型配置声明；不读取环境变量、不保存密钥值。环境变量解析由
application/settings.py 负责。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class QwenSettings(BaseModel):
    """Settings for the local Transformers Qwen backend.
    本地 Transformers Qwen 后端的配置。"""

    model_config = ConfigDict(extra="forbid")

    model: str = "qwen3-vl-4b-instruct"
    max_tokens: int = Field(default=4096, gt=0)
    spatial_review_max_tokens: int = Field(default=128, gt=0)
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    device_map: str = "auto"
    use_kernels: bool = False
    local_files_only: bool = False
    min_pixels: int | None = Field(default=None, gt=0)
    max_pixels: int | None = Field(default=None, gt=0)


class DeepSeekSettings(BaseModel):
    """Settings for a DeepSeek structured judge client (declaration only).
    未来 DeepSeek 结构化评估客户端的配置（仅声明）。"""

    model_config = ConfigDict(extra="forbid")

    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    api_key_env: str = "DEEPSEEK_API_KEY"
    timeout_seconds: int = Field(default=120, gt=0)
    max_retries: int = Field(default=3, ge=0)


class ModelSettings(BaseModel):
    """Group model settings without storing secret values.
    聚合模型配置且不保存密钥值。"""

    model_config = ConfigDict(extra="forbid")

    qwen: QwenSettings = Field(default_factory=QwenSettings)
    deepseek: DeepSeekSettings = Field(default_factory=DeepSeekSettings)
