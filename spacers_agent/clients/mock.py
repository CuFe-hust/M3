"""Offline structured-response client for deterministic integration tests.
用于确定性集成测试的离线结构化响应客户端。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from spacers_agent.clients.base import ModelT, RequestMeta, VisionLanguageClient


class MockVisionClient(VisionLanguageClient):
    """Return preconfigured JSON payloads keyed by request ID or hash.
    按请求 ID 或哈希返回预配置 JSON 载荷。
    """

    def __init__(self, responses: Mapping[str, BaseModel | dict[str, Any] | str]) -> None:
        self.responses = dict(responses)
        self.calls: list[RequestMeta] = []

    async def complete_json(
        self,
        *,
        messages: list[dict[str, Any]],
        response_model: type[ModelT],
        request_meta: RequestMeta,
    ) -> ModelT:
        """Validate a configured response without making a network request.
        不发起网络请求并校验预配置响应。
        """

        del messages
        self.calls.append(request_meta)
        value = self.responses.get(request_meta.request_id, self.responses.get(request_meta.request_hash))
        if value is None:
            raise KeyError(f"No mock response for {request_meta.request_id}")
        if isinstance(value, BaseModel):
            return response_model.model_validate(value.model_dump())
        if isinstance(value, str):
            return response_model.model_validate(json.loads(value))
        return response_model.model_validate(value)
