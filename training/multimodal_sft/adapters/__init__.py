"""Concrete model-family adapters.

These modules are the only place where model-specific structure knowledge is
allowed to live.
"""

from .qwen3_5 import Qwen3_5Adapter, Qwen35Adapter
from .qwen3_vl import Qwen3VLAdapter

__all__ = ["Qwen3_5Adapter", "Qwen35Adapter", "Qwen3VLAdapter"]

