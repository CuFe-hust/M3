"""Shared Qwen3.5 base with named PEFT LoRA adapter bindings.

共享 Qwen3.5 基座与命名 PEFT LoRA adapter 绑定。可选重依赖仅在 engine
实际加载模型时导入；导入本模块不会加载 torch、Transformers、PEFT 或权重。
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from models.base import (
    ModelCacheIdentity,
    ModelT,
    RequestMeta,
    validate_local_model_asset,
)
from models.cache import JsonResponseCache
from models.qwen_transformers import QWEN_CLIENT_VERSION, QwenTransformersClient
from models.settings import QwenAdapterSettings, QwenSettings


MULTI_ADAPTER_CLIENT_VERSION = f"{QWEN_CLIENT_VERSION}:multi-adapter-v1"
_ADAPTER_CONFIG_FILENAME = "adapter_config.json"
_ADAPTER_WEIGHTS_FILENAME = "adapter_model.safetensors"


class QwenAdapterError(RuntimeError):
    """Stable public failure for invalid or incompatible adapter assets.
    adapter 资产无效或不兼容时的稳定公共失败。"""


@dataclass(frozen=True)
class QwenAdapterSpec:
    """Validated deployment asset plus its path-free logical identity.
    已验证部署资产及其不含路径的逻辑身份。"""

    name: str
    path: Path
    logical_id: str
    revision: str
    peft_version: str | None
    declared_base_model: str
    target_modules: str | tuple[str, ...]
    weight_keys: tuple[str, ...]

    def audit_payload(self) -> dict[str, str | None]:
        return {
            "logical_id": self.logical_id,
            "revision": self.revision,
            "peft_version": self.peft_version,
        }


class BoundQwenAdapterClient:
    """Lightweight protocol client permanently bound to one logical adapter.
    永久绑定一个逻辑 adapter 的轻量协议 client。"""

    def __init__(
        self,
        client: QwenTransformersClient,
        *,
        identity: ModelCacheIdentity,
        binding_name: str,
        adapter_logical_id: str,
    ) -> None:
        self._client = client
        self._cache_identity = identity
        self.binding_name = binding_name
        self.adapter_logical_id = adapter_logical_id

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return self._cache_identity

    async def complete_json(
        self,
        *,
        messages: list[dict[str, Any]],
        response_model: type[ModelT],
        request_meta: RequestMeta,
        max_tokens: int | None = None,
    ) -> ModelT:
        return await self._client.complete_json(
            messages=messages,
            response_model=response_model,
            request_meta=request_meta,
            max_tokens=max_tokens,
        )


class MultiAdapterQwenEngine:
    """Own one base/processor, all named adapters, and one generation lock.
    独占一份基座/processor、全部命名 adapter 与一把生成锁。"""

    def __init__(
        self,
        settings: QwenSettings,
        *,
        adapters: Mapping[str, QwenAdapterSettings],
        project_root: Path,
        repair_prompt: str | None = None,
        cache: JsonResponseCache | None = None,
        model: Any | None = None,
        processor: Any | None = None,
        adapter_loader: Callable[[Any, tuple[QwenAdapterSpec, ...]], Any] | None = None,
    ) -> None:
        self.settings = settings
        self._repair_prompt = repair_prompt
        self._cache = cache
        enabled = {
            name: adapter for name, adapter in adapters.items() if adapter.enabled
        }
        specs = tuple(
            _validate_adapter_asset(name, adapter, project_root=project_root)
            for name, adapter in sorted(enabled.items())
        )
        if (model is None) != (processor is None):
            raise ValueError("model and processor must be supplied together")
        if model is None:
            base_client = QwenTransformersClient(
                settings,
                repair_prompt=repair_prompt,
                cache=cache,
            )
            model, processor = base_client.model, base_client.processor
            self.base_load_seconds = base_client.load_seconds
        else:
            self.base_load_seconds = 0.0
        assert processor is not None
        _validate_base_and_targets(model, settings, specs)
        if specs:
            loader = adapter_loader or _load_peft_adapters
            try:
                model = loader(model, specs)
            except QwenAdapterError:
                raise
            except Exception as error:
                raise QwenAdapterError("QWEN_ADAPTER_LOAD_FAILED") from error
            _validate_loaded_weights(model, specs)
        if hasattr(model, "requires_grad_"):
            model.requires_grad_(False)
        if hasattr(model, "eval"):
            model.eval()
        self.model = model
        self.processor = processor
        self._specs = MappingProxyType({spec.name: spec for spec in specs})
        self._generation_lock = asyncio.Lock()

    @property
    def adapter_inventory(self) -> Mapping[str, dict[str, str | None]]:
        """Path-free immutable adapter audit inventory.
        不含路径的不可变 adapter 审计清单。"""

        return MappingProxyType(
            {name: spec.audit_payload() for name, spec in self._specs.items()}
        )

    @property
    def runtime_identity(self) -> dict[str, Any]:
        """Portable engine identity suitable for run artifacts.
        适合运行产物的可移植 engine 身份。"""

        return {
            "base_model_id": self.settings.effective_cache_model_id,
            "base_revision": self.settings.revision,
            "client_version": MULTI_ADAPTER_CLIENT_VERSION,
            "adapters": {
                name: dict(payload)
                for name, payload in self.adapter_inventory.items()
            },
        }

    def bind(self, adapter_name: str) -> BoundQwenAdapterClient:
        """Create one immutable client view; unknown names never fall back.
        创建不可变 client 视图；未知名称绝不回退。"""

        if adapter_name == "base":
            adapter_logical_id = "base"
            adapter_revision = None
            peft_version = None
        else:
            try:
                spec = self._specs[adapter_name]
            except KeyError:
                raise QwenAdapterError("QWEN_ADAPTER_BINDING_UNKNOWN") from None
            adapter_logical_id = spec.logical_id
            adapter_revision = spec.revision
            peft_version = spec.peft_version
        identity = ModelCacheIdentity(
            model=self.settings.effective_cache_model_id,
            generation={
                "temperature": 0.0,
                "do_sample": False,
                "max_tokens": self.settings.max_tokens,
                "min_pixels": self.settings.min_pixels,
                "max_pixels": self.settings.max_pixels,
                "adapter": {
                    "logical_id": adapter_logical_id,
                    "revision": adapter_revision,
                    "peft_version": peft_version,
                },
            },
            client_version=MULTI_ADAPTER_CLIENT_VERSION,
            revision=self.settings.revision,
        )
        client = QwenTransformersClient(
            self.settings,
            repair_prompt=self._repair_prompt,
            cache=self._cache,
            model=self.model,
            processor=self.processor,
            generation_lock=self._generation_lock,
            generation_activation=lambda: self._activate(adapter_name),
        )
        return BoundQwenAdapterClient(
            client,
            identity=identity,
            binding_name=adapter_name,
            adapter_logical_id=adapter_logical_id,
        )

    def _activate(self, adapter_name: str) -> None:
        """Switch adapter only while the shared generation lock is held.
        仅在共享生成锁持有期间切换 adapter。"""

        if adapter_name == "base":
            disable = getattr(self.model, "disable_adapter_layers", None)
            if self._specs and not callable(disable):
                raise QwenAdapterError("QWEN_BASE_BINDING_UNSUPPORTED")
            if callable(disable):
                disable()
            return
        enable = getattr(self.model, "enable_adapter_layers", None)
        if callable(enable):
            enable()
        setter = getattr(self.model, "set_adapter", None)
        if not callable(setter):
            raise QwenAdapterError("QWEN_ADAPTER_SWITCH_UNSUPPORTED")
        setter(adapter_name)


def _validate_adapter_asset(
    name: str,
    settings: QwenAdapterSettings,
    *,
    project_root: Path,
) -> QwenAdapterSpec:
    path = settings.path.expanduser()
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    config_path = path / _ADAPTER_CONFIG_FILENAME
    weights_path = path / _ADAPTER_WEIGHTS_FILENAME
    if not config_path.is_file():
        raise QwenAdapterError("QWEN_ADAPTER_CONFIG_MISSING")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QwenAdapterError("QWEN_ADAPTER_CONFIG_INVALID") from error
    if not isinstance(payload, dict):
        raise QwenAdapterError("QWEN_ADAPTER_CONFIG_INVALID")
    if str(payload.get("peft_type", "")).upper() != "LORA":
        raise QwenAdapterError("QWEN_ADAPTER_NOT_LORA")
    if payload.get("modules_to_save") not in (None, [], ()):
        raise QwenAdapterError("QWEN_ADAPTER_MODULES_TO_SAVE_UNSUPPORTED")
    if payload.get("auxiliary_heads") not in (None, [], {}, ()):
        raise QwenAdapterError("QWEN_ADAPTER_AUXILIARY_HEAD_UNSUPPORTED")
    try:
        has_legacy_head_asset = any(
            "visual_planner_roi_head" in candidate.name
            for candidate in path.iterdir()
        )
    except OSError as error:
        raise QwenAdapterError("QWEN_ADAPTER_DIRECTORY_UNREADABLE") from error
    if has_legacy_head_asset:
        raise QwenAdapterError("QWEN_ADAPTER_AUXILIARY_HEAD_UNSUPPORTED")
    target_modules = payload.get("target_modules")
    if isinstance(target_modules, str) and target_modules.strip():
        normalized_targets: str | tuple[str, ...] = target_modules
    elif isinstance(target_modules, list) and target_modules and all(
        isinstance(value, str) and value.strip() for value in target_modules
    ):
        normalized_targets = tuple(target_modules)
    else:
        raise QwenAdapterError("QWEN_ADAPTER_TARGET_MODULES_INVALID")
    try:
        validate_local_model_asset(
            weights_path,
            expected_sha256=settings.revision,
        )
        weight_keys = _safetensor_keys(weights_path)
    except QwenAdapterError:
        raise
    except Exception as error:
        raise QwenAdapterError(type(error).__name__) from error
    if not weight_keys or any("visual_planner_roi_head" in key for key in weight_keys):
        raise QwenAdapterError("QWEN_ADAPTER_WEIGHT_KEYS_INVALID")
    base_name = payload.get("base_model_name_or_path")
    if not isinstance(base_name, str) or not base_name.strip():
        raise QwenAdapterError("QWEN_ADAPTER_BASE_IDENTITY_MISSING")
    return QwenAdapterSpec(
        name=name,
        path=path,
        logical_id=settings.logical_id,
        revision=settings.revision,
        peft_version=(
            str(payload["peft_version"]).strip()
            if payload.get("peft_version") is not None
            else None
        ),
        declared_base_model=base_name.strip(),
        target_modules=normalized_targets,
        weight_keys=weight_keys,
    )


def _safetensor_keys(path: Path) -> tuple[str, ...]:
    """Read only the bounded safetensors JSON header using the standard library.
    仅使用标准库读取有界的 safetensors JSON header。"""

    try:
        with path.open("rb") as file:
            raw_length = file.read(8)
            if len(raw_length) != 8:
                raise QwenAdapterError("QWEN_ADAPTER_WEIGHTS_INVALID")
            header_length = int.from_bytes(raw_length, "little", signed=False)
            if header_length <= 0 or header_length > 64 * 1024 * 1024:
                raise QwenAdapterError("QWEN_ADAPTER_WEIGHTS_INVALID")
            header = file.read(header_length)
        payload = json.loads(header.decode("utf-8"))
    except QwenAdapterError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QwenAdapterError("QWEN_ADAPTER_WEIGHTS_INVALID") from error
    if not isinstance(payload, dict):
        raise QwenAdapterError("QWEN_ADAPTER_WEIGHTS_INVALID")
    return tuple(sorted(str(key) for key in payload if key != "__metadata__"))


def _validate_base_and_targets(
    model: Any,
    settings: QwenSettings,
    specs: tuple[QwenAdapterSpec, ...],
) -> None:
    if not specs:
        return
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type != "qwen3_5":
        raise QwenAdapterError("QWEN_ADAPTER_BASE_MODEL_TYPE_INCOMPATIBLE")
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules):
        raise QwenAdapterError("QWEN_ADAPTER_TARGET_MODULES_UNAVAILABLE")
    module_names = tuple(name for name, _ in named_modules())
    for spec in specs:
        if _model_leaf(spec.declared_base_model) != _model_leaf(settings.model):
            raise QwenAdapterError("QWEN_ADAPTER_BASE_IDENTITY_INCOMPATIBLE")
        targets = spec.target_modules
        if isinstance(targets, str):
            try:
                pattern = re.compile(targets)
            except re.error as error:
                raise QwenAdapterError("QWEN_ADAPTER_TARGET_MODULES_INVALID") from error
            if not any(pattern.fullmatch(name) for name in module_names):
                raise QwenAdapterError("QWEN_ADAPTER_TARGET_MODULES_INCOMPATIBLE")
        else:
            if any(
                not any(name == target or name.endswith(f".{target}") for name in module_names)
                for target in targets
            ):
                raise QwenAdapterError("QWEN_ADAPTER_TARGET_MODULES_INCOMPATIBLE")


def _model_leaf(value: str) -> str:
    """Compare model declarations by their portable terminal name only.
    仅用可移植末端名称比较模型声明。"""

    normalized = value.strip().replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"[^a-z0-9]", "", normalized.casefold())


def _load_peft_adapters(model: Any, specs: tuple[QwenAdapterSpec, ...]) -> Any:
    try:
        from peft import PeftModel
    except ImportError as error:
        raise QwenAdapterError("QWEN_ADAPTER_PEFT_UNAVAILABLE") from error
    first, *remaining = specs
    wrapped = PeftModel.from_pretrained(
        model,
        str(first.path),
        adapter_name=first.name,
        is_trainable=False,
        local_files_only=True,
    )
    for spec in remaining:
        wrapped.load_adapter(
            str(spec.path),
            adapter_name=spec.name,
            is_trainable=False,
            local_files_only=True,
        )
    return wrapped


def _validate_loaded_weights(model: Any, specs: tuple[QwenAdapterSpec, ...]) -> None:
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise QwenAdapterError("QWEN_ADAPTER_WEIGHT_CONSUMPTION_UNVERIFIED")
    actual_keys = tuple(str(key) for key in state_dict())
    for spec in specs:
        normalized_actual = {
            _normalize_loaded_lora_key(key, spec.name) for key in actual_keys
        }
        missing = [
            key
            for key in spec.weight_keys
            if not any(
                actual == key
                or actual.endswith(f".{key}")
                or key.endswith(f".{actual}")
                for actual in normalized_actual
            )
        ]
        if missing:
            raise QwenAdapterError("QWEN_ADAPTER_WEIGHT_CONSUMPTION_INCOMPLETE")


def _normalize_loaded_lora_key(key: str, adapter_name: str) -> str:
    return key.replace(
        f".lora_A.{adapter_name}.weight", ".lora_A.weight"
    ).replace(
        f".lora_B.{adapter_name}.weight", ".lora_B.weight"
    )


__all__ = [
    "BoundQwenAdapterClient",
    "MultiAdapterQwenEngine",
    "QwenAdapterError",
    "QwenAdapterSpec",
]
