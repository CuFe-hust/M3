"""OpenAI-compatible Qwen client with safe local persistence and retries.
具备安全本地持久化与重试能力的 OpenAI 兼容 Qwen 客户端。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, InternalServerError, RateLimitError
from pydantic import BaseModel, ValidationError

from spacers_agent.clients.base import CacheEntry, JsonResponseCache, ModelT, RequestMeta, VisionLanguageClient, sanitize_messages
from spacers_agent.settings import QwenSettings


CompletionCreate = Callable[..., Awaitable[Any]]


class QwenClientError(RuntimeError):
    """Raised when a Qwen request exhausts its permitted recovery path.
    当 Qwen 请求耗尽允许的恢复路径时抛出。
    """


class QwenVLLMClient(VisionLanguageClient):
    """Issue one structured request at a time to a Qwen vLLM endpoint.
    一次向 Qwen vLLM 端点发送一条结构化请求。
    """

    def __init__(
        self,
        settings: QwenSettings,
        *,
        repair_prompt: str,
        cache: JsonResponseCache | None = None,
        completion_create: CompletionCreate | None = None,
        retry_base_seconds: float = 1.0,
    ) -> None:
        self.settings = settings
        self.repair_prompt = repair_prompt
        self.cache = cache
        self.retry_base_seconds = retry_base_seconds
        self._completion_create = completion_create or self._create_live_completion()

    async def complete_json(
        self,
        *,
        messages: list[dict[str, Any]],
        response_model: type[ModelT],
        request_meta: RequestMeta,
    ) -> ModelT:
        """Return cached or live schema-validated JSON with one repair attempt.
        返回缓存或线上经 Schema 校验的 JSON，并允许一次修复。
        """

        cached = self.cache.load(request_meta.request_hash) if self.cache else None
        if cached is not None:
            result = response_model.model_validate(cached.parsed)
            self._write_artifacts(
                request_meta,
                [cached.raw_response],
                result,
                [],
                cache_hit=True,
                response_metadata={"latency_seconds": 0.0, "token_usage": None},
            )
            return result

        attempt_errors: list[dict[str, Any]] = []
        raw_responses: list[str] = []
        active_messages = messages
        repair_used = False
        transport_attempt = 0
        while True:
            started = time.perf_counter()
            raw_response = ""
            try:
                response = await self._completion_create(
                    model=self.settings.model,
                    messages=active_messages,
                    temperature=self.settings.temperature,
                    max_tokens=self.settings.max_tokens,
                    response_format={"type": "json_object"},
                )
                raw_response = _response_content(response)
                raw_responses.append(raw_response)
                result = _validate_response(raw_response, response_model)
                if self.cache:
                    self.cache.save(
                        request_meta.request_hash,
                        CacheEntry(raw_response=raw_response, parsed=result.model_dump(mode="json")),
                    )
                self._write_artifacts(
                    request_meta,
                    raw_responses,
                    result,
                    attempt_errors,
                    cache_hit=False,
                    response_metadata={
                        "latency_seconds": round(time.perf_counter() - started, 6),
                        "token_usage": _token_usage(response),
                    },
                )
                return result
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                attempt_errors.append(_attempt_error(transport_attempt, error, started, retryable=False))
                if not repair_used:
                    repair_used = True
                    active_messages = _repair_messages(self.repair_prompt, raw_response, str(error))
                    continue
                self._write_artifacts(
                    request_meta,
                    raw_responses,
                    None,
                    attempt_errors,
                    cache_hit=False,
                    response_metadata=None,
                )
                raise QwenClientError(f"Qwen JSON validation failed after repair: {error}") from error
            except Exception as error:
                retryable = _is_retryable(error)
                attempt_errors.append(_attempt_error(transport_attempt, error, started, retryable=retryable))
                if not retryable or transport_attempt >= self.settings.max_retries:
                    self._write_artifacts(
                        request_meta,
                        raw_responses,
                        None,
                        attempt_errors,
                        cache_hit=False,
                        response_metadata=None,
                    )
                    raise QwenClientError(f"Qwen request failed: {error}") from error
                await asyncio.sleep(self.retry_base_seconds * (2**transport_attempt))
                transport_attempt += 1

    def _create_live_completion(self) -> CompletionCreate:
        """Create an async OpenAI-compatible call using an environment-only key.
        使用仅来自环境变量的密钥创建异步 OpenAI 兼容调用。
        """

        api_key = os.environ.get(self.settings.api_key_env)
        if not api_key:
            raise QwenClientError(f"Missing required environment variable: {self.settings.api_key_env}")
        client = AsyncOpenAI(api_key=api_key, base_url=self.settings.base_url, timeout=self.settings.timeout_seconds)
        return client.chat.completions.create

    def _write_artifacts(
        self,
        request_meta: RequestMeta,
        raw_responses: list[str],
        result: BaseModel | None,
        attempt_errors: list[dict[str, Any]],
        *,
        cache_hit: bool,
        response_metadata: dict[str, Any] | None,
    ) -> None:
        """Persist raw, parsed, validation, and safe request metadata for recovery.
        为恢复持久化原始响应、解析结果、校验信息与安全请求元数据。
        """

        if request_meta.artifact_dir is None:
            return
        artifact_dir = request_meta.artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "request_meta.json").write_text(
            json.dumps(request_meta.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rendered_raw = "\n\n".join(
            f"[response_attempt={index}]\n{value}" for index, value in enumerate(raw_responses, start=1)
        )
        (artifact_dir / "raw_response.txt").write_text(rendered_raw, encoding="utf-8")
        validation = {
            "cache_hit": cache_hit,
            "attempt_errors": attempt_errors,
            "response_metadata": response_metadata,
            "valid": result is not None,
        }
        (artifact_dir / "validation.json").write_text(
            json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if result is not None:
            (artifact_dir / "parsed.json").write_text(
                json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )


def _response_content(response: Any) -> str:
    """Extract non-empty assistant content from an OpenAI-compatible response.
    从 OpenAI 兼容响应提取非空助手内容。
    """

    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError("Qwen response has no assistant content") from error
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Qwen response content is empty")
    return content


def _token_usage(response: Any) -> dict[str, int | None] | None:
    """Extract token counters when an OpenAI-compatible server provides them.
    当 OpenAI 兼容服务提供 token 计数时提取该计数。
    """

    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _validate_response(raw_response: str, response_model: type[ModelT]) -> ModelT:
    """Remove an optional Markdown fence and validate JSON with Pydantic.
    移除可选 Markdown 围栏并使用 Pydantic 校验 JSON。
    """

    normalized = _strip_json_fence(raw_response)
    return response_model.model_validate(json.loads(normalized))


def _strip_json_fence(value: str) -> str:
    """Normalize a single fenced JSON response without accepting prose.
    规范化单个带围栏 JSON 响应且不接受额外散文。
    """

    stripped = value.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or not lines[-1].strip().startswith("```"):
        raise ValueError("Unterminated JSON fence")
    return "\n".join(lines[1:-1]).strip()


def _repair_messages(repair_prompt: str, raw_response: str, validation_error: str) -> list[dict[str, Any]]:
    """Build a no-image repair request that cannot add visual evidence.
    构建不含图像的修复请求，避免新增视觉依据。
    """

    return [
        {"role": "system", "content": repair_prompt},
        {
            "role": "user",
            "content": json.dumps(
                {"validation_error": validation_error, "raw_output": raw_response},
                ensure_ascii=False,
            ),
        },
    ]


def _is_retryable(error: Exception) -> bool:
    """Classify only transient transport and service failures as retryable.
    仅将瞬时传输和服务失败归类为可重试。
    """

    return isinstance(
        error,
        (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError, httpx.TimeoutException),
    ) or getattr(error, "status_code", None) in {429, 500, 502, 503, 504}


def _attempt_error(attempt: int, error: Exception, started: float, *, retryable: bool) -> dict[str, Any]:
    """Create safe retry metadata without credentials or request payloads.
    创建不含凭据或请求载荷的安全重试元数据。
    """

    return {
        "attempt": attempt,
        "error_type": type(error).__name__,
        "error": str(error),
        "latency_seconds": round(time.perf_counter() - started, 6),
        "retryable": retryable,
    }
