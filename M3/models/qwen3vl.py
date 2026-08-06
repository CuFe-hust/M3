"""Backward-compatible alias for the Qwen3-VL baseline wrapper.
Qwen3-VL 基线封装的向后兼容别名。

The real implementation now lives in ``models.qwen3_vl.baseline``. This module
keeps the historical ``models.qwen3vl`` import path and the report metadata
string ``models.qwen3vl.Qwen3VLBaseline`` unchanged.
真实实现已迁移至 ``models.qwen3_vl.baseline``。本模块保留历史
``models.qwen3vl`` 导入路径及报告元数据字符串 ``models.qwen3vl.Qwen3VLBaseline``
不变。
"""

from models.qwen3_vl.baseline import (
    Qwen3VLBaseline,
    Qwen3VLSettings,
    TASK_MAX_NEW_TOKENS,
    _choice_letter,
    _extract_boxes,
    _grounding_postprocess,
    _message_content,
    _official_pixel_boxes,
    _qwen_model_factory,
    _resolve_dtype,
    _task_max_new_tokens,
    _uses_qwen35_chat_template,
)

__all__ = [
    "Qwen3VLBaseline",
    "Qwen3VLSettings",
    "TASK_MAX_NEW_TOKENS",
    "_choice_letter",
    "_extract_boxes",
    "_grounding_postprocess",
    "_message_content",
    "_official_pixel_boxes",
    "_qwen_model_factory",
    "_resolve_dtype",
    "_task_max_new_tokens",
    "_uses_qwen35_chat_template",
]
