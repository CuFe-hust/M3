"""Standalone Qwen3-VL-8B-Instruct model folder.
独立的 Qwen3-VL-8B-Instruct 模型目录。

The wrapper is self-contained in ``models.qwen3_vl_8b.model`` and defaults to
the 8B Instruct checkpoint. This folder provides a stable per-model import
path and keeps local 8B weights outside the repository; the repository copy
never downloads weight files.
封装独立实现于 ``models.qwen3_vl_8b.model``，默认指向 8B Instruct 权重。
本目录提供稳定的按模型导入路径，并将本地 8B 权重放在仓库之外；
仓库副本不下载权重文件。
"""

from models.qwen3_vl_8b.model import (
    QWEN3_VL_8B_INSTRUCT,
    Qwen3VL8BInstruct,
    Qwen3VL8BSettings,
)

__all__ = [
    "QWEN3_VL_8B_INSTRUCT",
    "Qwen3VL8BInstruct",
    "Qwen3VL8BSettings",
]
