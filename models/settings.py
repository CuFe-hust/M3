"""Model settings declarations; no environment reading, no secret values.

模型配置声明；不读取环境变量、不保存密钥值。环境变量解析由
application/settings.py 负责。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from models.base import is_local_model_path, validate_logical_model_id


class QwenSettings(BaseModel):
    """Settings for the local Transformers Qwen backend.
    本地 Transformers Qwen 后端的配置。"""

    model_config = ConfigDict(extra="forbid")

    # Physical checkpoint path / name passed to from_pretrained.
    # 传给 from_pretrained 的物理 checkpoint 路径/名称。
    model: str = "qwen3-vl-4b-instruct"
    # Logical, machine-independent model identity used for hashes and traces.
    # 用于哈希与 trace 的逻辑、与机器无关的模型身份。
    cache_model_id: str | None = None
    max_tokens: int = Field(default=4096, gt=0)
    spatial_review_max_tokens: int = Field(default=128, gt=0)
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    device_map: str = "auto"
    use_kernels: bool = False
    # Single source of truth for online behaviour: offline by default.
    # 联网行为的唯一来源：默认离线。
    allow_download: bool = False
    min_pixels: int | None = Field(default=None, gt=0)
    max_pixels: int | None = Field(default=None, gt=0)
    revision: str | None = None

    @model_validator(mode="after")
    def validate_model_identity(self) -> "QwenSettings":
        """A local checkpoint path (POSIX, Windows drive, UNC, or file URI)
        must carry an explicit logical cache model id, which itself must be a
        logical identifier — never a local path — so hashes and traces never
        leak machine paths. 本地 checkpoint 路径（POSIX、Windows drive、UNC
        或 file URI）必须携带显式逻辑缓存模型 ID；该 ID 本身必须是逻辑标识
        符而非本地路径，使哈希与 trace 永不泄漏机器路径。"""
        if is_local_model_path(self.model) and not self.cache_model_id:
            raise ValueError("cache_model_id is required when model is a local path")
        if self.cache_model_id is not None:
            self.cache_model_id = validate_logical_model_id(
                self.cache_model_id,
                where="cache_model_id",
            )
        return self

    @property
    def effective_cache_model_id(self) -> str:
        """Logical model identity for hashes and traces; the returned value is
        always validated — absolute local checkpoint paths are rejected without
        an explicit cache_model_id, and the declared model name is safe.
        用于哈希与 trace 的逻辑模型身份；返回值始终经过校验——无显式
        cache_model_id 的绝对本地 checkpoint 路径已被拒绝，声明模型名安全。"""
        if self.cache_model_id is not None:
            return self.cache_model_id
        return validate_logical_model_id(self.model, where="model")


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
