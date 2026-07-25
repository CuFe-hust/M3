"""Counting backend protocol, registry, and selector.
计数后端协议、注册表与选择器。
"""

from spacers_agent.agents.counting.backends.base import BackendSelection, CountingBackend, CountingRequest
from spacers_agent.agents.counting.backends.qwen_point import QwenPointCountingBackend
from spacers_agent.agents.counting.backends.registry import BackendRegistry
from spacers_agent.agents.counting.backends.selector import BackendSelector
from spacers_agent.agents.counting.backends.vrsbench_qwen_count import VRSBenchQwenCountBackend

__all__ = [
    "BackendRegistry",
    "BackendSelection",
    "BackendSelector",
    "CountingBackend",
    "CountingRequest",
    "QwenPointCountingBackend",
    "VRSBenchQwenCountBackend",
]
