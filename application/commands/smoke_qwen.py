"""Public `smoke-qwen` CLI command: one direct VisionLanguageClient request.

公开 `smoke-qwen` CLI 命令：一次直接 VisionLanguageClient 冒烟请求。
不经过 Agent 路由；测试注入 fake 客户端；恰好一次模型调用。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from application.prompts import PromptCatalog
from application.settings import load_settings
from models.base import (
    RequestMeta,
    build_request_hash,
    require_model_cache_identity,
)
from models.cache import JsonResponseCache
from models.entry import create_model
from models.images import detect_image_mime, image_to_data_url

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_INTERRUPTED = 130


class _SmokeResponse(BaseModel):
    """Minimal smoke response schema. / 最小冒烟响应 schema。"""

    model_config = ConfigDict(extra="forbid")

    message: str


def run_smoke_qwen(
    args: argparse.Namespace,
    *,
    qwen_client=None,
) -> int:
    """Send exactly one direct image+question request and print the answer.
    发送恰好一次直接的 图片+问题 请求并输出回答。"""

    try:
        settings = load_settings(
            Path(args.config) if getattr(args, "config", None) else None,
        )
        project_root = Path(__file__).resolve().parents[2]
        image_path = Path(args.image)
        data = image_path.read_bytes()
        mime = detect_image_mime(image_path)
        content = [
            {
                "type": "image_url",
                "image_url": {"url": image_to_data_url(data, mime)},
            },
            {
                "type": "text",
                "text": json.dumps({"question": args.question}, ensure_ascii=False),
            },
        ]
        messages = [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": content},
        ]
        catalog = PromptCatalog(project_root / "prompts")
        client = qwen_client or create_model(
            "qwen_transformers",
            settings=settings.models.qwen,
            repair_prompt=catalog["json_repair"],
            cache=JsonResponseCache(settings.runs.root / "service" / "cache"),
        )
        identity = require_model_cache_identity(client, component="smoke_qwen")
        image_sha256 = hashlib.sha256(data).hexdigest()
        request_hash = build_request_hash(
            model=identity.model,
            generation=identity.generation_payload(),
            prompt_version="smoke-qwen-v1",
            messages=messages,
            image_sha256=image_sha256,
            response_schema=_SmokeResponse.model_json_schema(),
            client_version=identity.client_version,
            model_revision=identity.revision,
        )
        result = asyncio.run(
            client.complete_json(
                messages=messages,
                response_model=_SmokeResponse,
                request_meta=RequestMeta(
                    request_id="smoke-qwen",
                    request_hash=request_hash,
                    prompt_version="smoke-qwen-v1",
                    sample_id="smoke-qwen",
                    image_sha256=image_sha256,
                    artifact_dir=settings.runs.root / "service" / "smoke-qwen",
                ),
            )
        )
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    except Exception as error:
        print(
            json.dumps({"status": "failed", "error": f"{type(error).__name__}"}),
            file=sys.stderr,
        )
        return EXIT_RUNTIME
    print(
        json.dumps(
            {
                "status": "ok",
                "model": settings.models.qwen.model,
                "message": result.message,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return EXIT_OK
