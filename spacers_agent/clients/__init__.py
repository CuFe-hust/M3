"""Test and training model clients outside the main-flow models/ entry.
主流程 models/ 入口之外的测试与训练用模型客户端。

Main-flow models are constructed only through ``models.entry.create_model``;
these clients are used by tests, offline doubles, and the text-only judge.
主流程模型只通过 ``models.entry.create_model`` 构建；本包客户端用于测试、
离线替身和纯文本评审。
"""

from models.base import RequestMeta, VisionLanguageClient, image_to_data_url
from spacers_agent.clients.deepseek import DeepSeekJudgeClient
from spacers_agent.clients.mock import MockVisionClient

__all__ = [
    "DeepSeekJudgeClient",
    "MockVisionClient",
    "RequestMeta",
    "VisionLanguageClient",
    "image_to_data_url",
]
