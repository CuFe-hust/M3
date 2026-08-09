"""Application settings: YAML loading, environment references, and safe
snapshots. 应用配置：YAML 加载、环境变量引用与安全快照。

Secret values never enter settings, snapshots, repr, or artifacts: only the
environment variable NAME (api_key_env) is declared, the value is read once by
the composition root and injected directly into the judge client.
密钥值绝不进入配置、快照、repr 或产物：只声明环境变量名（api_key_env），
值由组合根读取一次并直接注入 judge 客户端。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agents.change.settings import AgentChangeSettings
from agents.counting.settings import (
    AgentCountingSettings,
    CountingSettings,
    YoloCountingSettings,
)
from models.settings import ModelSettings


class RunSettings(BaseModel):
    """Run output directory and artifact-saving switches.
    运行输出目录与产物保存开关。"""

    model_config = ConfigDict(extra="forbid")

    root: Path = Path("outputs/runs")
    save_tiles: bool = False
    save_annotated_images: bool = True
    save_raw_responses: bool = True


class RouterSettings(BaseModel):
    """Task resolution and per-sample budget defaults.
    任务解析与逐样本预算默认值。"""

    model_config = ConfigDict(extra="forbid")

    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    default_qwen_calls: int = Field(default=50, ge=1)
    default_deepseek_calls: int = Field(default=10, ge=0)
    fallback_on_partial: bool = False


class PathSettings(BaseModel):
    """Dataset resolution root. / 数据集解析根。"""

    model_config = ConfigDict(extra="forbid")

    dataset_root: Path = Path("dataset")


class BackendSettings(BaseModel):
    """Counting backend configuration group. / 计数后端配置组。"""

    model_config = ConfigDict(extra="forbid")

    yolo: YoloCountingSettings = Field(default_factory=YoloCountingSettings)


class AgentsSettings(BaseModel):
    """Agent-level configuration group. / Agent 级配置组。"""

    model_config = ConfigDict(extra="forbid")

    counting: AgentCountingSettings = Field(default_factory=AgentCountingSettings)
    change: AgentChangeSettings = Field(default_factory=AgentChangeSettings)


class AppSettings(BaseModel):
    """Full application settings; no secret values by construction.
    完整应用配置；构造上不含任何密钥值。"""

    model_config = ConfigDict(extra="forbid")

    models: ModelSettings = Field(default_factory=ModelSettings)
    counting: CountingSettings = Field(default_factory=CountingSettings)
    runs: RunSettings = Field(default_factory=RunSettings)
    router: RouterSettings = Field(default_factory=RouterSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    backend: BackendSettings = Field(default_factory=BackendSettings)
    agents: AgentsSettings = Field(default_factory=AgentsSettings)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_counting_execution_policy(cls, data: Any) -> Any:
        """Move legacy YOLO-scoped execution policy at the settings boundary."""

        if not isinstance(data, Mapping):
            return data
        payload = dict(data)
        counting = dict(payload.get("counting", {}))
        backend = dict(payload.get("backend", {}))
        yolo = dict(backend.get("yolo", {}))
        aliases = {
            "fallback_to_qwen_on_unavailable": "fallback_on_backend_unavailable",
            "fallback_to_qwen_on_error": "fallback_on_backend_error",
            "verify_empty_with_qwen": "verify_empty_detection",
        }
        for legacy, current in aliases.items():
            if legacy not in yolo:
                continue
            if current in counting:
                raise ValueError(
                    f"cannot configure both legacy key {legacy!r} and {current!r}"
                )
            counting[current] = yolo.pop(legacy)
        if "trust_empty_detection" in backend:
            if "trust_empty_detection" in counting:
                raise ValueError(
                    "cannot configure both legacy backend trust_empty_detection "
                    "and counting trust_empty_detection"
                )
            counting["trust_empty_detection"] = backend.pop("trust_empty_detection")
        backend["yolo"] = yolo
        payload["backend"] = backend
        payload["counting"] = counting
        return payload

    def to_config_payload(self) -> dict[str, Any]:
        """JSON-safe reproduction snapshot for run manifests: no secret
        values, no Path objects, paths serialized as strings with forward
        slashes on every platform. Configured host paths are preserved
        verbatim — this snapshot is reproduction/debug oriented and carries
        the host's own path semantics. 供 run manifest 使用的 JSON 安全复现
        快照：无密钥值、无 Path 对象、路径在所有平台统一正斜杠字符串。
        配置的主机路径原样保留——本快照面向复现/调试，承载主机自身的路径
        语义。"""

        payload = self.model_dump(mode="json")
        payload["runs"]["root"] = _posix(self.runs.root)
        payload["paths"]["dataset_root"] = _posix(self.paths.dataset_root)
        for profile_name, profile in (
            ("segformer_isaid", self.models.segformer_isaid),
            ("segformer_oem", self.models.segformer_oem),
        ):
            payload["models"][profile_name]["model_path"] = _posix(
                profile.model_path
            )
            if profile.processor_path is not None:
                payload["models"][profile_name]["processor_path"] = _posix(
                    profile.processor_path
                )
        for backend_name, profile in self.models.segformer_experts.items():
            serialized = payload["models"]["segformer_experts"][backend_name]
            serialized["model_path"] = _posix(profile.model_path)
            if profile.processor_path is not None:
                serialized["processor_path"] = _posix(profile.processor_path)
        return payload

    def safe_snapshot(self) -> dict[str, Any]:
        """Alias of to_config_payload: the only allowed serialization of
        settings. 与 to_config_payload 相同的安全快照：配置唯一允许的序列化
        形式。"""

        return self.to_config_payload()


_ENV_OVERRIDES = {
    "QWEN_MODEL": ("models", "qwen", "model"),
    "SEGFORMER_ISAID_MODEL": ("models", "segformer_isaid", "model_path"),
    "SEGFORMER_OEM_MODEL": ("models", "segformer_oem", "model_path"),
    "DEEPSEEK_BASE_URL": ("models", "deepseek", "base_url"),
    "DEEPSEEK_MODEL": ("models", "deepseek", "model"),
    "DATASET_ROOT": ("paths", "dataset_root"),
    "OUTPUT_ROOT": ("runs", "root"),
}


def load_settings(
    path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> AppSettings:
    """Load settings from defaults, an optional YAML file, and environment
    overrides (environment wins). Secret VALUES are never read — only declared
    env var names travel through settings. 从默认值、可选 YAML 文件与环境变量
    覆盖加载配置（环境变量优先）。密钥值绝不读取——只有声明的环境变量名经过
    配置。"""

    settings = AppSettings()
    if path is not None and path.is_file():
        merged = settings.model_dump()
        raw = AppSettings.migrate_legacy_counting_execution_policy(_load_yaml(path))
        _deep_merge(merged, raw)
        settings = AppSettings.model_validate(merged)
    environ = environ if environ is not None else os.environ
    overrides: dict[str, Any] = {}
    for env_name, target in _ENV_OVERRIDES.items():
        value = environ.get(env_name)
        if value is None or value == "":
            continue
        node = overrides
        for key in target[:-1]:
            node = node.setdefault(key, {})
        node[target[-1]] = value
    if overrides:
        merged = settings.model_dump()
        _deep_merge(merged, overrides)
        settings = AppSettings.model_validate(merged)
    return settings


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid settings YAML: {type(exc).__name__}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("settings YAML must map to an object")
    return raw


def _posix(path: Path) -> str:
    """POSIX serialization with forward-slash separators on every platform
    (as_posix alone keeps backslashes on POSIX hosts). 所有平台统一正斜杠
    的 POSIX 序列化（仅 as_posix 在 POSIX 主机上会保留反斜杠）。"""

    return path.as_posix().replace("\\", "/")


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
