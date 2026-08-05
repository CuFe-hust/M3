"""Qwen3.5 model wrapper reusing the shared Transformers client.
复用共享 Transformers 客户端的 Qwen3.5 模型封装。
"""

from __future__ import annotations

from models.qwen_transformers import QwenTransformersClient, QwenTransformersError

__all__ = ["QwenTransformersClient", "QwenTransformersError"]
