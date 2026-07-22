"""In-process Qwen3-VL Transformers client with structured local artifacts.
进程内 Qwen3-VL Transformers 客户端及结构化本地产物。
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, ValidationError

from spacers_agent.clients.base import (
    CacheEntry,
    JsonResponseCache,
    ModelT,
    RequestMeta,
    VisionLanguageClient,
    sanitize_messages,
)
from spacers_agent.settings import QwenSettings


class QwenTransformersError(RuntimeError):
    """Report a visible local model loading, generation, or validation failure.
    报告可见的本地模型加载、生成或校验失败。
    """


class QwenTransformersClient(VisionLanguageClient):
    """Run the configured local checkpoint directly through Transformers.
    通过 Transformers 直接运行配置的本地权重。
    """

    def __init__(
        self,
        settings: QwenSettings,
        *,
        repair_prompt: str | None = None,
        cache: JsonResponseCache | None = None,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        self.settings = settings
        self.repair_prompt = repair_prompt
        self.cache = cache
        started = time.perf_counter()
        if (model is None) != (processor is None):
            raise ValueError("model and processor must be supplied together")
        self.model, self.processor = (model, processor) if model is not None else self._load()
        self.load_seconds = round(time.perf_counter() - started, 6)

    def _load(self) -> tuple[Any, Any]:
        """Load only the declared checkpoint without an endpoint or download fallback.
        仅加载声明的权重，不使用端点或下载回退。
        """

        try:
            import torch
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as error:
            raise QwenTransformersError("Install requirements.txt before loading local Qwen3-VL.") from error
        dtype: Any = "auto"
        if self.settings.dtype != "auto":
            dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[self.settings.dtype]
        processor_kwargs: dict[str, Any] = {}
        if self.settings.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.settings.min_pixels
        if self.settings.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.settings.max_pixels
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.settings.model,
            dtype=dtype,
            device_map=self.settings.device_map,
            local_files_only=self.settings.local_files_only,
        )
        processor = AutoProcessor.from_pretrained(
            self.settings.model,
            local_files_only=self.settings.local_files_only,
            **processor_kwargs,
        )
        model.eval()
        return model, processor

    async def complete_json(
        self,
        *,
        messages: list[dict[str, Any]],
        response_model: type[ModelT],
        request_meta: RequestMeta,
    ) -> ModelT:
        """Generate once, validate JSON, and persist auditable local-call metadata.
        生成一次、校验 JSON，并保存可审计的本地调用元数据。
        """

        cached = self.cache.load(request_meta.request_hash) if self.cache else None
        if cached is not None:
            result = response_model.model_validate(cached.parsed)
            self._write_artifacts(
                request_meta,
                messages,
                cached.raw_response,
                result,
                cache_hit=True,
                validation_error=None,
                metadata={"latency_seconds": 0.0, "token_usage": None},
            )
            return result
        started = time.perf_counter()
        raw_responses: list[str] = []
        attempt_errors: list[dict[str, Any]] = []
        token_usage: dict[str, int] | None = None
        try:
            raw_response, token_usage = await asyncio.to_thread(self._generate, messages, response_model)
            raw_responses.append(raw_response)
            try:
                result = _validate_response(raw_response, response_model)
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                attempt_errors.append(_validation_attempt_error(1, error))
                if self.repair_prompt is None:
                    raise
                repair_messages = _repair_messages(self.repair_prompt, raw_response, str(error))
                repaired, repair_usage = await asyncio.to_thread(self._generate, repair_messages, response_model)
                raw_responses.append(repaired)
                token_usage = _sum_token_usage(token_usage, repair_usage)
                result = _validate_response(repaired, response_model)
        except (json.JSONDecodeError, ValidationError, ValueError) as error:
            if not attempt_errors or attempt_errors[-1]["error"] != str(error):
                attempt_errors.append(_validation_attempt_error(len(raw_responses), error))
            self._write_artifacts(
                request_meta,
                messages,
                _render_raw_responses(raw_responses),
                None,
                cache_hit=False,
                validation_error=f"{type(error).__name__}: {error}",
                metadata={
                    "latency_seconds": round(time.perf_counter() - started, 6),
                    "token_usage": token_usage,
                    "attempt_errors": attempt_errors,
                    "repair_used": len(raw_responses) > 1,
                },
            )
            raise QwenTransformersError(f"Local Qwen JSON validation failed after repair: {error}") from error
        except Exception as error:
            self._write_artifacts(
                request_meta,
                messages,
                _render_raw_responses(raw_responses),
                None,
                cache_hit=False,
                validation_error=f"{type(error).__name__}: {error}",
                metadata={
                    "latency_seconds": round(time.perf_counter() - started, 6),
                    "token_usage": token_usage,
                    "attempt_errors": attempt_errors,
                    "repair_used": len(raw_responses) > 1,
                },
            )
            raise QwenTransformersError(f"Local Qwen generation failed: {error}") from error
        rendered_raw = _render_raw_responses(raw_responses)
        if self.cache:
            self.cache.save(
                request_meta.request_hash,
                CacheEntry(raw_response=rendered_raw, parsed=result.model_dump(mode="json")),
            )
        self._write_artifacts(
            request_meta,
            messages,
            rendered_raw,
            result,
            cache_hit=False,
            validation_error=None,
            metadata={
                "latency_seconds": round(time.perf_counter() - started, 6),
                "token_usage": token_usage,
                "attempt_errors": attempt_errors,
                "repair_used": len(raw_responses) > 1,
                "local_recoveries": _result_local_recoveries(result),
            },
        )
        return result

    def _generate(
        self,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
    ) -> tuple[str, dict[str, int]]:
        """Convert data URLs to PIL images and run deterministic generation.
        将数据 URL 转为 PIL 图像并执行确定性生成。
        """

        model_messages, images = _transformer_messages(messages, response_model)
        text = self.processor.apply_chat_template(model_messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images or None, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.settings.max_tokens,
            do_sample=False,
        )
        input_tokens = int(inputs.input_ids.shape[-1])
        trimmed = [output[input_tokens:] for output in generated]
        raw = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        output_tokens = int(trimmed[0].shape[-1])
        return raw, {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    def _write_artifacts(
        self,
        request_meta: RequestMeta,
        messages: list[dict[str, Any]],
        raw_response: str,
        result: BaseModel | None,
        *,
        cache_hit: bool,
        validation_error: str | None,
        metadata: dict[str, Any],
    ) -> None:
        """Persist sanitized inputs, raw output, parsed output, timing, and tokens.
        保存脱敏输入、原始输出、解析输出、耗时和 token 信息。
        """

        if request_meta.artifact_dir is None:
            return
        directory = request_meta.artifact_dir
        directory.mkdir(parents=True, exist_ok=True)
        _write_json(directory / "request_meta.json", request_meta.model_dump(mode="json"))
        _write_json(directory / "request.json", {"messages": sanitize_messages(messages)})
        (directory / "raw_response.txt").write_text(raw_response, encoding="utf-8")
        _write_json(
            directory / "validation.json",
            {
                "backend": "transformers",
                "cache_hit": cache_hit,
                "valid": result is not None,
                "validation_error": validation_error,
                "response_metadata": metadata,
            },
        )
        if result is not None:
            _write_json(directory / "parsed.json", result.model_dump(mode="json"))


def _transformer_messages(
    messages: list[dict[str, Any]],
    response_model: type[BaseModel],
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    """Translate OpenAI-style local messages and append the exact response schema.
    转换 OpenAI 风格的本地消息，并附加精确响应 Schema。
    """

    converted: list[dict[str, Any]] = []
    images: list[Image.Image] = []
    schema_instruction = (
        "Return valid JSON only. The JSON must match this schema exactly: "
        + json.dumps(response_model.model_json_schema(), ensure_ascii=False, separators=(",", ":"))
    )
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            text = content + ("\n\n" + schema_instruction if message.get("role") == "system" else "")
            converted.append({"role": message.get("role", "user"), "content": text})
            continue
        converted_content: list[dict[str, Any]] = []
        for item in content or []:
            if item.get("type") == "image_url":
                image = _decode_data_url(str(item.get("image_url", {}).get("url", "")))
                images.append(image)
                converted_content.append({"type": "image", "image": image})
            elif item.get("type") == "text":
                converted_content.append({"type": "text", "text": str(item.get("text", ""))})
            else:
                raise ValueError(f"Unsupported local message item: {item.get('type')!r}")
        converted.append({"role": message.get("role", "user"), "content": converted_content})
    if not any(message.get("role") == "system" for message in converted):
        converted.insert(0, {"role": "system", "content": schema_instruction})
    return converted, images


def _decode_data_url(value: str) -> Image.Image:
    """Decode one in-memory image URL without accepting remote URLs.
    解码一条内存图像 URL，且不接受远程 URL。
    """

    if not value.startswith("data:image/") or ";base64," not in value:
        raise ValueError("Transformers backend accepts only data:image Base64 URLs")
    _, encoded = value.split(",", 1)
    with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as opened:
        return opened.convert("RGB")


def _validate_response(raw_response: str, response_model: type[ModelT]) -> ModelT:
    """Validate an optional fenced JSON object without accepting surrounding prose.
    校验可选围栏 JSON 对象，不接受前后散文。
    """

    stripped = raw_response.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) < 2:
            raise ValueError("Unterminated JSON fence")
        stripped = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]).strip()
    recovery: str | None = None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as error:
        recovered = _recover_truncated_json(stripped, error)
        if recovered is None:
            raise
        payload, recovery = recovered
    if recovery is not None and isinstance(payload, dict) and "geometry" in response_model.model_fields:
        geometry = dict(payload.get("geometry") or {})
        normalizations = list(geometry.get("input_normalizations") or [])
        normalizations.append(recovery)
        geometry["input_normalizations"] = list(dict.fromkeys(normalizations))
        payload["geometry"] = geometry
    return response_model.model_validate(payload)


def _recover_truncated_json(
    value: str,
    error: json.JSONDecodeError,
) -> tuple[Any, str] | None:
    """Close or prune only an incomplete JSON tail at end-of-output.
    仅闭合或裁剪输出末尾不完整的 JSON 尾部。
    """

    stripped = value.rstrip()
    if not stripped.startswith(("{", "[")):
        return None
    near_end = error.pos >= max(0, len(stripped) - 2)
    if not near_end and not error.msg.startswith("Unterminated string"):
        return None
    frames, in_string, dangling_escape = _scan_json_frames(stripped)
    if not frames:
        return None

    closed = stripped
    if in_string:
        if dangling_escape and closed.endswith("\\"):
            closed = closed[:-1]
        closed += '"'
    closed = closed.rstrip()
    if closed.endswith(","):
        closed = closed[:-1].rstrip()
    direct = _load_with_closed_frames(closed, frames)
    if direct is not None:
        return direct, "truncated_json_closed_locally"

    deepest = frames[-1]
    parent = frames[-2] if len(frames) > 1 else None
    if deepest["kind"] == "{" and parent is not None and parent["kind"] == "[":
        cut = int(deepest["open"])
    else:
        comma = deepest.get("last_comma")
        if comma is None:
            return None
        cut = int(comma)
    pruned = stripped[:cut].rstrip()
    if pruned.endswith(","):
        pruned = pruned[:-1].rstrip()
    remaining, remaining_in_string, _ = _scan_json_frames(pruned)
    if remaining_in_string or not remaining:
        return None
    recovered = _load_with_closed_frames(pruned, remaining)
    if recovered is None:
        return None
    return recovered, "truncated_json_incomplete_tail_pruned"


def _scan_json_frames(value: str) -> tuple[list[dict[str, int | str | None]], bool, bool]:
    """Track open JSON containers and their last top-level commas.
    跟踪未闭合 JSON 容器及各自最后一个顶层逗号。
    """

    frames: list[dict[str, int | str | None]] = []
    in_string = False
    escaped = False
    for index, character in enumerate(value):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "{[":
            frames.append({"kind": character, "open": index, "last_comma": None})
        elif character in "}]":
            if not frames:
                return [], in_string, escaped
            expected = "{" if character == "}" else "["
            if frames[-1]["kind"] != expected:
                return [], in_string, escaped
            frames.pop()
        elif character == "," and frames:
            frames[-1]["last_comma"] = index
    return frames, in_string, escaped


def _load_with_closed_frames(value: str, frames: list[dict[str, int | str | None]]) -> Any | None:
    """Close known containers and return decoded JSON when it is valid.
    闭合已知容器，并在结果有效时返回解码后的 JSON。
    """

    suffix = "".join("}" if frame["kind"] == "{" else "]" for frame in reversed(frames))
    try:
        return json.loads(value + suffix)
    except json.JSONDecodeError:
        return None


def _repair_messages(repair_prompt: str, raw_response: str, validation_error: str) -> list[dict[str, Any]]:
    """Build a text-only format repair request that cannot inspect the image.
    构建无法查看图像的纯文本格式修复请求。
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


def _render_raw_responses(values: list[str]) -> str:
    return "\n\n".join(
        f"[response_attempt={index}]\n{value}" for index, value in enumerate(values, start=1)
    )


def _result_local_recoveries(result: BaseModel) -> list[str]:
    """Expose local truncation recovery markers in call metadata.
    在调用元数据中公开本地截断恢复标记。
    """

    geometry = getattr(result, "geometry", None)
    if not isinstance(geometry, dict):
        return []
    return [
        str(value)
        for value in geometry.get("input_normalizations", [])
        if str(value).startswith("truncated_json_")
    ]


def _validation_attempt_error(attempt: int, error: Exception) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "error_type": type(error).__name__,
        "error": str(error),
        "retryable": False,
    }


def _sum_token_usage(
    first: dict[str, int] | None,
    second: dict[str, int] | None,
) -> dict[str, int] | None:
    if first is None:
        return second
    if second is None:
        return first
    return {key: int(first.get(key, 0)) + int(second.get(key, 0)) for key in set(first) | set(second)}


def _write_json(path: Path, value: Any) -> None:
    """Atomically write one UTF-8 JSON artifact.
    原子写入一份 UTF-8 JSON 产物。
    """

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
