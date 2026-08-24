"""Concrete text-only DeepSeek judge client with bounded recovery, caching,
and artifacts. HTTP uses only the Python standard library so the base wheel
keeps its minimal dependency surface.

具体仅文本 DeepSeek judge 客户端：受限恢复、缓存与产物。HTTP 仅用 Python
标准库实现，保持基础 wheel 最小依赖面。客户端不读取环境变量（api_key 由
composition root 注入）、不加载视觉模型；公共错误消息只携带稳定 code，
产物中的错误记录只含稳定类型名，绝不携带原始响应或异常文本。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from evaluation.judges.base import (
    DeepSeekJudgeResult,
    ModelT,
    stable_error_label,
)
from models.base import RequestMeta
from models.cache import CacheEntry, JsonResponseCache
from models.settings import DeepSeekSettings


class DeepSeekJudgeError(RuntimeError):
    """Stable public judge error; the message carries only a fixed code and
    never raw response or exception text. 稳定的公共 judge 错误；消息只携带
    固定 code，绝不携带原始响应或异常文本。"""


class EmptyJudgeResponseError(ValueError):
    """Internal signal for a response without usable assistant content.
    响应无可用的助手内容时使用的内部信号。"""


class JudgeTransportError(RuntimeError):
    """Stable transport error; status_code drives retry decisions and is None
    for connection-level failures. 稳定传输错误；status_code 驱动重试决策，
    连接级失败为 None。"""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


JudgeTransport = Callable[..., str]

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def urllib_judge_transport(
    *,
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    base_url: str,
    timeout_seconds: int,
) -> str:
    """Post one chat-completion request using only the Python standard library.
    Errors are normalized to JudgeTransportError with a stable status code and
    no raw response text. 仅用 Python 标准库发起一次 chat-completion 请求；
    错误归一化为带稳定 status code 且无原始响应文本的 JudgeTransportError。"""

    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise JudgeTransportError(
            "judge transport http error", status_code=error.code
        ) from error
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        raise JudgeTransportError(
            "judge transport connection error", status_code=None
        ) from error
    except json.JSONDecodeError as error:
        raise JudgeTransportError("judge transport invalid response body") from error
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise JudgeTransportError("judge transport missing assistant content") from error
    if not isinstance(content, str) or not content.strip():
        raise JudgeTransportError("judge transport empty content")
    return content


class DeepSeekJudgeClient:
    """Issue text-only JSON judge requests to an OpenAI-compatible DeepSeek
    API. The api_key is injected by the composition root; the client never
    reads the environment. 向 OpenAI 兼容 DeepSeek API 发起仅文本 JSON judge
    请求。api_key 由 composition root 注入；客户端绝不读取环境变量。"""

    def __init__(
        self,
        settings: DeepSeekSettings,
        *,
        api_key: str,
        judge_prompt: str,
        repair_prompt: str,
        cache: JsonResponseCache | None = None,
        transport: JudgeTransport | None = None,
        retry_base_seconds: float = 1.0,
    ) -> None:
        if not api_key:
            raise DeepSeekJudgeError("deepseek judge client requires an api key")
        self.settings = settings
        self.api_key = api_key
        self.judge_prompt = judge_prompt
        self.repair_prompt = repair_prompt
        self.cache = cache
        self.retry_base_seconds = retry_base_seconds
        self._transport = transport or urllib_judge_transport

    def judge(
        self,
        payload: Mapping[str, Any],
        *,
        request_meta: RequestMeta,
    ) -> DeepSeekJudgeResult:
        """Return a cached or live schema-validated counting judge result.
        返回缓存或在线、经 Schema 校验的计数 judge 结果。"""

        return self.judge_json(
            payload, response_model=DeepSeekJudgeResult, request_meta=request_meta
        )

    def judge_json(
        self,
        payload: Mapping[str, Any],
        *,
        response_model: type[ModelT],
        request_meta: RequestMeta,
        system_prompt: str | None = None,
        repair_with_original_payload: bool = False,
    ) -> ModelT:
        """Return a schema-validated text-only result for a declared Judge
        contract. Cache hits skip the transport; failed recoveries raise
        DeepSeekJudgeError with a fixed code. 按声明的 Judge 契约返回经
        Schema 校验的纯文本结果。缓存命中跳过传输；恢复失败抛固定 code 的
        DeepSeekJudgeError。"""

        _assert_text_only_payload(payload)
        cached = self.cache.load(request_meta.request_hash) if self.cache else None
        if cached is not None:
            result = response_model.model_validate(cached.parsed)
            self._write_artifacts(
                request_meta, [cached.raw_response], result, [], cache_hit=True
            )
            return result
        messages = _judge_messages(system_prompt or self.judge_prompt, payload)
        raw_responses: list[str] = []
        errors: list[dict[str, Any]] = []
        repair_used = False
        attempt = 0
        while True:
            started = time.perf_counter()
            try:
                raw = self._transport(
                    model=self.settings.model,
                    messages=messages,
                    api_key=self.api_key,
                    base_url=self.settings.base_url,
                    timeout_seconds=self.settings.timeout_seconds,
                )
                if not isinstance(raw, str) or not raw.strip():
                    raise EmptyJudgeResponseError()
            except EmptyJudgeResponseError as error:
                errors.append(_error_record(attempt, error, started, retryable=True))
                if attempt >= self.settings.max_retries:
                    self._write_artifacts(
                        request_meta, raw_responses, None, errors, cache_hit=False
                    )
                    raise DeepSeekJudgeError("DEEPSEEK_JUDGE_EMPTY_RESPONSE") from error
                time.sleep(self.retry_base_seconds * (2**attempt))
                attempt += 1
                continue
            except Exception as error:
                retryable = _is_retryable(error)
                errors.append(_error_record(attempt, error, started, retryable=retryable))
                if not retryable or attempt >= self.settings.max_retries:
                    self._write_artifacts(
                        request_meta, raw_responses, None, errors, cache_hit=False
                    )
                    raise DeepSeekJudgeError("DEEPSEEK_JUDGE_TRANSPORT_FAILED") from error
                time.sleep(self.retry_base_seconds * (2**attempt))
                attempt += 1
                continue
            raw_responses.append(raw)
            try:
                result = response_model.model_validate(json.loads(_strip_json_fence(raw)))
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                errors.append(_error_record(attempt, error, started, retryable=False))
                if not repair_used:
                    repair_used = True
                    messages = _repair_messages(
                        self.repair_prompt,
                        raw,
                        stable_error_label(error),
                        original_payload=(
                            payload if repair_with_original_payload else None
                        ),
                    )
                    continue
                self._write_artifacts(
                    request_meta, raw_responses, None, errors, cache_hit=False
                )
                raise DeepSeekJudgeError("DEEPSEEK_JUDGE_INVALID_JSON") from error
            if self.cache is not None:
                self.cache.save(
                    request_meta.request_hash,
                    CacheEntry(raw_response=raw, parsed=result.model_dump(mode="json")),
                )
            self._write_artifacts(
                request_meta,
                raw_responses,
                result,
                errors,
                cache_hit=False,
                metadata={"latency_seconds": round(time.perf_counter() - started, 6)},
            )
            return result

    def _write_artifacts(
        self,
        request_meta: RequestMeta,
        raw_responses: list[str],
        result: BaseModel | None,
        errors: list[dict[str, Any]],
        *,
        cache_hit: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist raw, parsed, and validation evidence without images; error
        records carry stable type names only, never raw exception text.
        持久化不含图像的原始、解析与校验证据；错误记录只携带稳定类型名，
        绝不携带原始异常文本。"""

        if request_meta.artifact_dir is None:
            return
        directory = request_meta.artifact_dir
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "request_meta.json").write_text(
            json.dumps(
                request_meta.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raw = "\n\n".join(
            f"[response_attempt={index}]\n{value}"
            for index, value in enumerate(raw_responses, start=1)
        )
        (directory / "raw_response.txt").write_text(raw, encoding="utf-8")
        (directory / "validation.json").write_text(
            json.dumps(
                {
                    "cache_hit": cache_hit,
                    "attempt_errors": errors,
                    "response_metadata": metadata,
                    "valid": result is not None,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if result is not None:
            (directory / "parsed.json").write_text(
                result.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )


def _judge_messages(
    prompt: str, payload: Mapping[str, Any]
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": "Evaluate the following structured evidence and return JSON only:\n"
            + json.dumps(payload, ensure_ascii=False),
        },
    ]


def _repair_messages(
    prompt: str,
    raw_response: str,
    validation_error: str,
    *,
    original_payload: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    repair_payload: dict[str, Any] = {
        "validation_error": validation_error,
        "raw_output": raw_response,
    }
    if original_payload is not None:
        repair_payload["original_payload"] = dict(original_payload)
    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": json.dumps(repair_payload, ensure_ascii=False),
        },
    ]


def _assert_text_only_payload(payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False).casefold()
    forbidden = ("data:image/", "base64", "image_bytes", "image_path", "pixel_array")
    if any(marker in encoded for marker in forbidden):
        raise ValueError(
            "DeepSeek judge payload must contain text and structured evidence only"
        )


def _strip_json_fence(value: str) -> str:
    stripped = value.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or not lines[-1].strip().startswith("```"):
        raise ValueError("Unterminated JSON fence")
    return "\n".join(lines[1:-1]).strip()


def _is_retryable(error: Exception) -> bool:
    """JudgeTransportError retries on server/rate-limit codes and on
    connection-level failures; every other exception type fails immediately.
    JudgeTransportError 在服务端/限流 code 与连接级失败时重试；其他异常类型
    立即失败。"""

    if not isinstance(error, JudgeTransportError):
        return False
    return error.status_code is None or error.status_code in _RETRYABLE_STATUS_CODES


def _error_record(
    attempt: int, error: Exception, started: float, *, retryable: bool
) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "error_type": type(error).__name__,
        "latency_seconds": round(time.perf_counter() - started, 6),
        "retryable": retryable,
    }
