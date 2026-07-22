"""Text-only DeepSeek structured judge client with bounded recovery and artifacts.
仅文本 DeepSeek 结构化评估器客户端，具有限定恢复和产物记录。
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
from pydantic import ValidationError

from spacers_agent.clients.base import CacheEntry, JsonResponseCache, RequestMeta
from spacers_agent.evaluation import DeepSeekJudgeResult
from spacers_agent.settings import DeepSeekSettings


CompletionCreate = Callable[..., Awaitable[Any]]


class EmptyJudgeResponseError(ValueError):
    """Raised when DeepSeek returns a response without usable assistant content.
    当 DeepSeek 返回没有可用助手内容的响应时抛出。
    """


class DeepSeekJudgeError(RuntimeError):
    """Raised when the structured judge cannot complete its bounded recovery path.
    当结构化评估器无法完成其限定恢复路径时抛出。
    """


class DeepSeekJudgeClient:
    """Issue text-only JSON judge requests to an OpenAI-compatible DeepSeek API.
    向 OpenAI 兼容 DeepSeek API 发起仅文本 JSON 评估请求。
    """

    def __init__(
        self,
        settings: DeepSeekSettings,
        *,
        judge_prompt: str,
        repair_prompt: str,
        cache: JsonResponseCache | None = None,
        completion_create: CompletionCreate | None = None,
        retry_base_seconds: float = 1.0,
    ) -> None:
        self.settings = settings
        self.judge_prompt = judge_prompt
        self.repair_prompt = repair_prompt
        self.cache = cache
        self.retry_base_seconds = retry_base_seconds
        self._completion_create = completion_create or self._create_live_completion()

    async def judge(self, payload: dict[str, Any], *, request_meta: RequestMeta) -> DeepSeekJudgeResult:
        """Return a cached or live schema-validated text-only judge result.
        返回缓存或在线的经 Schema 校验仅文本评估结果。
        """

        _assert_text_only_payload(payload)
        cached = self.cache.load(request_meta.request_hash) if self.cache else None
        if cached is not None:
            result = DeepSeekJudgeResult.model_validate(cached.parsed)
            self._write_artifacts(request_meta, [cached.raw_response], result, [], cache_hit=True, metadata={"latency_seconds": 0.0, "token_usage": None})
            return result

        messages = _judge_messages(self.judge_prompt, payload)
        raw_responses: list[str] = []
        errors: list[dict[str, Any]] = []
        repair_used = False
        transport_attempt = 0
        while True:
            started = time.perf_counter()
            raw_response = ""
            try:
                response = await self._completion_create(
                    model=self.settings.model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=2048,
                    response_format={"type": "json_object"},
                )
                raw_response = _response_content(response)
                raw_responses.append(raw_response)
                result = DeepSeekJudgeResult.model_validate(json.loads(_strip_json_fence(raw_response)))
                if self.cache:
                    self.cache.save(
                        request_meta.request_hash,
                        CacheEntry(raw_response=raw_response, parsed=result.model_dump(mode="json")),
                    )
                self._write_artifacts(
                    request_meta,
                    raw_responses,
                    result,
                    errors,
                    cache_hit=False,
                    metadata={"latency_seconds": round(time.perf_counter() - started, 6), "token_usage": _token_usage(response)},
                )
                return result
            except EmptyJudgeResponseError as error:
                errors.append(_error_record(transport_attempt, error, started, retryable=True))
                if transport_attempt >= self.settings.max_retries:
                    self._write_artifacts(request_meta, raw_responses, None, errors, cache_hit=False, metadata=None)
                    raise DeepSeekJudgeError(f"DeepSeek judge request failed: {error}") from error
                await asyncio.sleep(self.retry_base_seconds * (2**transport_attempt))
                transport_attempt += 1
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                errors.append(_error_record(transport_attempt, error, started, retryable=False))
                if not repair_used:
                    repair_used = True
                    messages = _repair_messages(self.repair_prompt, raw_response, str(error))
                    continue
                self._write_artifacts(request_meta, raw_responses, None, errors, cache_hit=False, metadata=None)
                raise DeepSeekJudgeError(f"DeepSeek JSON validation failed after repair: {error}") from error
            except Exception as error:
                retryable = _is_retryable(error)
                errors.append(_error_record(transport_attempt, error, started, retryable=retryable))
                if not retryable or transport_attempt >= self.settings.max_retries:
                    self._write_artifacts(request_meta, raw_responses, None, errors, cache_hit=False, metadata=None)
                    raise DeepSeekJudgeError(f"DeepSeek judge request failed: {error}") from error
                await asyncio.sleep(self.retry_base_seconds * (2**transport_attempt))
                transport_attempt += 1

    def _create_live_completion(self) -> CompletionCreate:
        """Create a client with a key read only from the configured environment variable.
        使用仅从配置环境变量读取的密钥创建客户端。
        """

        api_key = os.environ.get(self.settings.api_key_env)
        if not api_key:
            raise DeepSeekJudgeError(f"Missing required environment variable: {self.settings.api_key_env}")
        client = AsyncOpenAI(api_key=api_key, base_url=self.settings.base_url, timeout=self.settings.timeout_seconds)
        return client.chat.completions.create

    def _write_artifacts(
        self,
        request_meta: RequestMeta,
        raw_responses: list[str],
        result: DeepSeekJudgeResult | None,
        errors: list[dict[str, Any]],
        *,
        cache_hit: bool,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Persist raw, parsed, validation, latency, and token evidence without images.
        持久化不含图像的原始、解析、校验、延迟和 token 证据。
        """

        if request_meta.artifact_dir is None:
            return
        directory = request_meta.artifact_dir
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "request_meta.json").write_text(
            json.dumps(request_meta.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raw = "\n\n".join(f"[response_attempt={index}]\n{value}" for index, value in enumerate(raw_responses, start=1))
        (directory / "raw_response.txt").write_text(raw, encoding="utf-8")
        (directory / "validation.json").write_text(
            json.dumps({"cache_hit": cache_hit, "attempt_errors": errors, "response_metadata": metadata, "valid": result is not None}, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if result is not None:
            (directory / "parsed.json").write_text(
                result.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )


def _judge_messages(prompt: str, payload: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Evaluate the following structured evidence and return JSON only:\n" + json.dumps(payload, ensure_ascii=False)},
    ]


def _repair_messages(prompt: str, raw_response: str, validation_error: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps({"validation_error": validation_error, "raw_output": raw_response}, ensure_ascii=False)},
    ]


def _assert_text_only_payload(payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False).casefold()
    forbidden = ("data:image/", "base64", "image_bytes", "image_path", "pixel_array")
    if any(marker in encoded for marker in forbidden):
        raise ValueError("DeepSeek judge payload must contain text and structured evidence only")


def _response_content(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as error:
        raise EmptyJudgeResponseError("DeepSeek response has no assistant content") from error
    if not isinstance(content, str) or not content.strip():
        raise EmptyJudgeResponseError("DeepSeek response content is empty")
    return content


def _strip_json_fence(value: str) -> str:
    stripped = value.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or not lines[-1].strip().startswith("```"):
        raise ValueError("Unterminated JSON fence")
    return "\n".join(lines[1:-1]).strip()


def _token_usage(response: Any) -> dict[str, int | None] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _is_retryable(error: Exception) -> bool:
    return isinstance(
        error,
        (EmptyJudgeResponseError, APIConnectionError, APITimeoutError, InternalServerError, RateLimitError, httpx.TimeoutException),
    ) or getattr(error, "status_code", None) in {429, 500, 502, 503, 504}


def _error_record(attempt: int, error: Exception, started: float, *, retryable: bool) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "error_type": type(error).__name__,
        "error": str(error),
        "latency_seconds": round(time.perf_counter() - started, 6),
        "retryable": retryable,
    }
