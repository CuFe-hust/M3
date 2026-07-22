"""Validated local settings with environment overrides.
带环境变量覆盖的本地校验配置。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class QwenSettings(BaseModel):
    """Settings for a future OpenAI-compatible Qwen endpoint.
    未来 OpenAI 兼容 Qwen 端点的配置。
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["vllm", "transformers"] = "vllm"
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "qwen3-vl-4b-instruct"
    api_key_env: str = "QWEN_API_KEY"
    timeout_seconds: int = Field(default=120, gt=0)
    max_retries: int = Field(default=2, ge=0)
    temperature: float = Field(default=0.0, ge=0.0)
    max_tokens: int = Field(default=4096, gt=0)
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    device_map: str = "auto"
    local_files_only: bool = False
    min_pixels: int | None = Field(default=None, gt=0)
    max_pixels: int | None = Field(default=None, gt=0)


class DeepSeekSettings(BaseModel):
    """Settings for a future DeepSeek structured judge client.
    未来 DeepSeek 结构化评估客户端的配置。
    """

    model_config = ConfigDict(extra="forbid")

    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    api_key_env: str = "DEEPSEEK_API_KEY"
    timeout_seconds: int = Field(default=120, gt=0)
    max_retries: int = Field(default=3, ge=0)


class ModelSettings(BaseModel):
    """Group model settings without storing secret values.
    聚合模型配置且不保存密钥值。
    """

    model_config = ConfigDict(extra="forbid")

    qwen: QwenSettings = Field(default_factory=QwenSettings)
    deepseek: DeepSeekSettings = Field(default_factory=DeepSeekSettings)


class CountingSettings(BaseModel):
    """Deterministic defaults shared by future point-counting components.
    未来点式计数组件共用的确定性默认配置。
    """

    model_config = ConfigDict(extra="forbid")

    tile_core_size: int = Field(default=896, gt=0)
    halo_size: int = Field(default=128, ge=0)
    model_max_side: int = Field(default=1280, gt=0)
    max_pixels_without_tiling: int = Field(default=1_600_000, gt=0)
    boundary_band_px: int = Field(default=32, ge=0)
    min_confidence: float = Field(default=0.2, ge=0.0, le=1.0)
    max_points_per_tile: int = Field(default=200, gt=0)
    sequential: bool = True
    concurrency: int = Field(default=1, ge=1)
    seam_verify: bool = True
    recursive_split_enabled: bool = True
    max_recursive_depth: int = Field(default=2, ge=0)
    min_core_size: int = Field(default=224, gt=0)
    seam_crop_margin_px: int = Field(default=128, ge=0)
    unresolved_conflict_policy: Literal["flag_for_review"] = "flag_for_review"
    prompt_version: str = "count-point-v2"

    def model_post_init(self, __context: Any) -> None:
        if self.sequential and self.concurrency != 1:
            raise ValueError("sequential counting requires concurrency=1")


class RunSettings(BaseModel):
    """Settings that determine durable local run artifacts.
    决定本地可持久化运行产物的配置。
    """

    model_config = ConfigDict(extra="forbid")

    root: Path = Path("outputs/runs")
    save_tiles: bool = False
    save_annotated_images: bool = True
    save_raw_responses: bool = True

class RouterSettings(BaseModel):
    """Sparse router thresholds and safety limits. / 稀疏路由阈值和安全限制。"""
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    trust_dataset_task_type: bool = True
    use_router_agent_when_task_missing: bool = True
    high_confidence_threshold: float = Field(default=0.8, ge=0, le=1)
    medium_confidence_threshold: float = Field(default=0.6, ge=0, le=1)
    max_total_experts: int = Field(default=3, ge=1)
    repair_attempts: int = Field(default=1, ge=0, le=1)
    enable_rule_fallback: bool = True

    def model_post_init(self, __context: Any) -> None:
        if self.medium_confidence_threshold >= self.high_confidence_threshold:
            raise ValueError("router medium threshold must be below high threshold")


class PathSettings(BaseModel):
    """Project-relative input locations.
    项目相对输入位置。
    """

    model_config = ConfigDict(extra="forbid")

    dataset_root: Path = Path("dataset")


class AppSettings(BaseModel):
    """Top-level validated settings for the new local runtime.
    新本地运行时的顶层校验配置。
    """

    model_config = ConfigDict(extra="forbid")

    models: ModelSettings = Field(default_factory=ModelSettings)
    counting: CountingSettings = Field(default_factory=CountingSettings)
    runs: RunSettings = Field(default_factory=RunSettings)
    router: RouterSettings = Field(default_factory=RouterSettings)
    paths: PathSettings = Field(default_factory=PathSettings)


class EnvironmentOverrides(BaseSettings):
    """Read non-secret endpoint and path overrides from dotenv or process env.
    从 dotenv 或进程环境读取非密钥端点与路径覆盖。
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    qwen_backend: Literal["vllm", "transformers"] | None = Field(default=None, validation_alias="QWEN_BACKEND")
    qwen_base_url: str | None = Field(default=None, validation_alias="QWEN_BASE_URL")
    qwen_model: str | None = Field(default=None, validation_alias="QWEN_MODEL")
    deepseek_base_url: str | None = Field(default=None, validation_alias="DEEPSEEK_BASE_URL")
    deepseek_model: str | None = Field(default=None, validation_alias="DEEPSEEK_MODEL")
    dataset_root: str | None = Field(default=None, validation_alias="DATASET_ROOT")
    output_root: str | None = Field(default=None, validation_alias="OUTPUT_ROOT")


def load_settings(path: Path | None = None, environ: Mapping[str, str] | None = None) -> AppSettings:
    """Load YAML settings and apply documented environment overrides.
    加载 YAML 配置并应用已文档化的环境变量覆盖。
    """

    payload: dict[str, Any] = {}
    if path is not None:
        with path.open(encoding="utf-8") as file:
            loaded = yaml.safe_load(file) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Configuration root must be a mapping: {path}")
        payload = loaded
    environment = _environment_values(environ)
    _apply_environment_overrides(payload, environment)
    return AppSettings.model_validate(payload)


def _environment_values(environ: Mapping[str, str] | None) -> dict[str, str]:
    """Produce recognized environment overrides without loading API-key values.
    生成已识别的环境变量覆盖且不加载 API 密钥值。
    """

    if environ is not None:
        return dict(environ)
    overrides = EnvironmentOverrides()
    values = overrides.model_dump(exclude_none=True)
    return {
        "QWEN_BACKEND": values.get("qwen_backend", ""),
        "QWEN_BASE_URL": values.get("qwen_base_url", ""),
        "QWEN_MODEL": values.get("qwen_model", ""),
        "DEEPSEEK_BASE_URL": values.get("deepseek_base_url", ""),
        "DEEPSEEK_MODEL": values.get("deepseek_model", ""),
        "DATASET_ROOT": values.get("dataset_root", ""),
        "OUTPUT_ROOT": values.get("output_root", ""),
    }


def _apply_environment_overrides(payload: dict[str, Any], environ: Mapping[str, str]) -> None:
    """Apply non-secret endpoint and path overrides without persisting secrets.
    应用非密钥端点与路径覆盖且不持久化密钥。
    """

    models = payload.setdefault("models", {})
    qwen = models.setdefault("qwen", {})
    deepseek = models.setdefault("deepseek", {})
    paths = payload.setdefault("paths", {})
    runs = payload.setdefault("runs", {})
    for key, destination in (
        ("QWEN_BACKEND", (qwen, "backend")),
        ("QWEN_BASE_URL", (qwen, "base_url")),
        ("QWEN_MODEL", (qwen, "model")),
        ("DEEPSEEK_BASE_URL", (deepseek, "base_url")),
        ("DEEPSEEK_MODEL", (deepseek, "model")),
        ("DATASET_ROOT", (paths, "dataset_root")),
        ("OUTPUT_ROOT", (runs, "root")),
    ):
        value = environ.get(key)
        # An explicit YAML run root scopes reproducible artifacts; dotenv must not redirect it.
        # 显式 YAML 运行根目录限定可复现产物；dotenv 不得重定向它。
        if value and not (key == "OUTPUT_ROOT" and destination[1] in destination[0]):
            destination[0][destination[1]] = value
