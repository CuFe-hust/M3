"""Unified model entry point for main-flow models.
主流程模型的统一模型入口。

Calling ``create_model(name, **kwargs)`` is the only supported way to
construct a main-flow model wrapper. Adding a new model means adding one
``@register(name)`` builder function in this file.
调用 ``create_model(name, **kwargs)`` 是构建主流程模型封装的唯一入口；
新增模型只需在本文件增加一个 ``@register(name)`` 构建函数。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


_REGISTRY: dict[str, Callable[..., Any]] = {}


def register(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register one model builder under a stable name.
    以稳定名称注册一个模型构建函数。
    """

    def decorator(builder: Callable[..., Any]) -> Callable[..., Any]:
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
    构建共享的本地 Transformers Qwen 客户端。
    """

    # Lazy import keeps `import models` free of HF offline side effects.
    # 惰性导入使 `import models` 不触发 HF 离线环境副作用。
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
    构建 Qwen3-VL 基线封装。
    """

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
    经 Qwen3.5 模型目录构建其 Transformers 客户端。
    """

    from models.qwen3_5.model import QwenTransformersClient

    return QwenTransformersClient(
        settings,
        repair_prompt=repair_prompt,
        cache=cache,
        **kwargs,
    )


def create_model(name: str, **kwargs: Any) -> Any:
    """Create one model wrapper through the unified entry.
    通过统一入口创建一个模型封装。
    """

    builder = _REGISTRY.get(name)
    if builder is None:
        raise KeyError(f"Unknown model entry {name!r}; registered: {sorted(_REGISTRY)}")
    return builder(**kwargs)


def list_models() -> list[str]:
    """Return registered model names in registration order.
    按注册顺序返回已注册的模型名称。
    """

    return list(_REGISTRY)
