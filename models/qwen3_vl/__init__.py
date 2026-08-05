"""Qwen3-VL model folder.
Qwen3-VL 模型目录。

The folder keeps the model wrapper and its weights together. Weights are
stored under ``weights/`` on the machine that runs inference; the local
development copy does not download them.
该目录将模型封装与权重放在一起；权重存放在运行推理机器的 ``weights/``
目录，本地开发副本不下载。
"""

from models.qwen3_vl.baseline import Qwen3VLBaseline, Qwen3VLSettings

__all__ = ["Qwen3VLBaseline", "Qwen3VLSettings"]
