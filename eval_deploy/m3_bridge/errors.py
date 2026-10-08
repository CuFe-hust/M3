"""Stable bridge error type mapped by the HTTP shell to non-2xx JSON.

桥接稳定错误类型；由 HTTP 外壳（server.py 的补丁）映射为非 2xx JSON。

This module must stay dependency-free: it is imported by model_adapter.py in
every MODEL_MODE, including demo/test runs that never load the bridge.
本模块必须保持零依赖：model_adapter.py 在任意 MODEL_MODE（含 demo/测试）
下都会导入它。
"""

from __future__ import annotations


class AdapterError(Exception):
    """One stable error code for the HTTP layer; never carries raw text.

    供 HTTP 层使用的稳定错误码；绝不携带原始异常文本。
    """

    def __init__(self, status: int, code: str) -> None:
        super().__init__(code)
        self.status = status
        self.code = code
