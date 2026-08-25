"""Unified model entry point for main-flow models.
主流程模型的统一模型入口。

Calling ``create_model(name, **kwargs)`` is the only supported way to
construct a main-flow model wrapper. Adding a new model means adding one
``@register(name)`` builder function in this file. Builders import concrete
models lazily so ``import models.entry`` never loads transformers or torch.
调用 ``create_model(name, **kwargs)`` 是构建主流程模型封装的唯一入口；
新增模型只需在本文件增加一个 ``@register(name)`` 构建函数。builder 惰性
导入具体模型，使 ``import models.entry`` 不加载 transformers 或 torch。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

ModelName = Literal[
    "qwen_transformers",
    "qwen3_vl_baseline",
    "qwen3_5_transformers",
    "qwen3_5_multi_adapter",
    "segformer_transformers",
]

ModelBuilder = Callable[..., object]

_REGISTRY: dict[str, ModelBuilder] = {}


def register(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    """Register one model builder under a stable name.
    以稳定名称注册一个模型构建函数。"""

    def decorator(builder: ModelBuilder) -> ModelBuilder:
        if name in _REGISTRY:
            raise ValueError(f"Model entry already registered: {name}")
        _REGISTRY[name] = builder
        return builder

    return decorator


@register("qwen_transformers")
def build_qwen_transformers_client(
    *,
    settings: Any,
    repair_prompt: str | None = None,
    cache: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Build the shared local Transformers Qwen client.
    构建共享的本地 Transformers Qwen 客户端。"""
    from models.qwen_transformers import QwenTransformersClient

    return QwenTransformersClient(
        settings,
        repair_prompt=repair_prompt,
        cache=cache,
        **kwargs,
    )


@register("qwen3_vl_baseline")
def build_qwen3_vl_baseline(*, settings: Any, **kwargs: Any) -> Any:
    """Build the Qwen3-VL baseline wrapper.
    构建 Qwen3-VL 基线封装。"""
    from models.qwen3_vl.baseline import Qwen3VLBaseline

    return Qwen3VLBaseline(settings, **kwargs)


@register("qwen3_5_transformers")
def build_qwen3_5_transformers_client(
    *,
    settings: Any,
    repair_prompt: str | None = None,
    cache: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Build the Qwen3.5 Transformers client via its model folder.
    经 Qwen3.5 模型目录构建其 Transformers 客户端。"""
    from models.qwen3_5.model import QwenTransformersClient

    return QwenTransformersClient(
        settings,
        repair_prompt=repair_prompt,
        cache=cache,
        **kwargs,
    )


@register("qwen3_5_multi_adapter")
def build_qwen3_5_multi_adapter_engine(
    *,
    settings: Any,
    adapters: Any,
    project_root: Any,
    repair_prompt: str | None = None,
    cache: Any | None = None,
    **kwargs: Any,
) -> Any:
    """Build one shared Qwen3.5 base with named PEFT adapters lazily.
    惰性构建一份共享 Qwen3.5 基座及命名 PEFT adapter。"""

    from models.qwen3_5.multi_adapter import MultiAdapterQwenEngine

    return MultiAdapterQwenEngine(
        settings,
        adapters=adapters,
        project_root=project_root,
        repair_prompt=repair_prompt,
        cache=cache,
        **kwargs,
    )


@register("segformer_transformers")
def build_segformer_transformers_client(
    *,
    settings: Any,
    **kwargs: Any,
) -> Any:
    """Build the local dense SegFormer client without eager concrete imports."""

    from models.segformer_transformers import SegFormerTransformersClient

    return SegFormerTransformersClient(settings, **kwargs)


def create_model(name: str, **kwargs: Any) -> Any:
    """Create one model wrapper through the unified entry.
    通过统一入口创建一个模型封装。"""
    builder = _REGISTRY.get(name)
    if builder is None:
        raise KeyError(f"Unknown model entry {name!r}; registered: {sorted(_REGISTRY)}")
    return builder(**kwargs)


def list_models() -> tuple[str, ...]:
    """Return registered model names in registration order as an immutable
    tuple. 按注册顺序返回不可变的已注册模型名元组。"""
    return tuple(_REGISTRY)
