"""Public `health` CLI command: model/service readiness metadata.

公开 `health` CLI 命令：模型/服务就绪元数据。正常模式只输出元数据与
环境变量名，绝不输出密钥值；`--live` 路径注入 fake 客户端（测试）或
构造真实客户端（composition root）后恰好调用一次。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from application.prompts import PromptCatalog
from application.settings import load_settings
from evaluation.judges.deepseek import DeepSeekJudgeClient
from models.base import (
    RequestMeta,
    build_request_hash,
    require_model_cache_identity,
)
from models.cache import JsonResponseCache
from models.entry import create_model

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_INTERRUPTED = 130


class _HealthProbe(BaseModel):
    """Minimal schema for one live health probe. / 一次 live 探测的最小 schema。"""

    model_config = ConfigDict(extra="forbid")

    status: str


def run_health(
    args: argparse.Namespace,
    *,
    qwen_client=None,
    judge_client=None,
) -> int:
    """Print readiness metadata; --live performs exactly one probe call.
    输出就绪元数据；--live 恰好执行一次探测调用。"""

    try:
        settings = load_settings(
            Path(args.config) if getattr(args, "config", None) else None,
        )
        project_root = Path(__file__).resolve().parents[2]
        if args.component == "qwen":
            if args.live:
                catalog = PromptCatalog(project_root / "prompts")
                if qwen_client is None:
                    engine = create_model(
                        "qwen3_5_multi_adapter",
                        settings=settings.models.qwen,
                        adapters=settings.models.qwen_adapters,
                        project_root=project_root,
                        repair_prompt=catalog["json_repair"],
                        cache=JsonResponseCache(
                            settings.runs.root / "service" / "cache"
                        ),
                    )
                    client = engine.bind(
                        settings.models.qwen_adapter_bindings.planner
                    )
                else:
                    client = qwen_client
                identity = require_model_cache_identity(client, component="health")
                messages = [
                    {
                        "role": "user",
                        "content": 'Return valid JSON only: {"status": "ok"}',
                    }
                ]
                request_hash = build_request_hash(
                    model=identity.model,
                    generation=identity.generation_payload(),
                    prompt_version="health-v1",
                    messages=messages,
                    image_sha256=None,
                    response_schema=_HealthProbe.model_json_schema(),
                    client_version=identity.client_version,
                    model_revision=identity.revision,
                )
                probe = asyncio.run(
                    client.complete_json(
                        messages=messages,
                        response_model=_HealthProbe,
                        request_meta=RequestMeta(
                            request_id="health:qwen",
                            request_hash=request_hash,
                            prompt_version="health-v1",
                            sample_id="health",
                            artifact_dir=settings.runs.root / "service" / "health",
                        ),
                    )
                )
                payload = {
                    "status": "ok",
                    "component": "qwen",
                    "model": settings.models.qwen.model,
                    "probe_status": probe.status,
                }
            else:
                payload = {
                    "status": "ready",
                    "component": "qwen",
                    "model": settings.models.qwen.model,
                    "cache_model_id": settings.models.qwen.effective_cache_model_id,
                    "allow_download": settings.models.qwen.allow_download,
                }
        else:  # deepseek
            if args.live:
                catalog = PromptCatalog(project_root / "prompts")
                client = judge_client or DeepSeekJudgeClient(
                    settings.models.deepseek,
                    api_key=os.environ.get(settings.models.deepseek.api_key_env) or "",
                    judge_prompt=catalog["count_judge"],
                    repair_prompt=catalog["json_repair"],
                    cache=JsonResponseCache(
                        settings.runs.root / "service" / "deepseek_cache"
                    ),
                )
                request_hash = build_request_hash(
                    model=settings.models.deepseek.model,
                    generation={},
                    prompt_version="health-v1",
                    messages=[
                        {"role": "user", "content": "health probe"},
                    ],
                    image_sha256=None,
                    response_schema=_HealthProbe.model_json_schema(),
                )
                probe = client.judge_json(
                    payload={"question": "health probe"},
                    response_model=_HealthProbe,
                    request_meta=RequestMeta(
                        request_id="health:deepseek",
                        request_hash=request_hash,
                        prompt_version="health-v1",
                        sample_id="health",
                        artifact_dir=settings.runs.root / "service" / "health",
                    ),
                )
                payload = {
                    "status": "ok",
                    "component": "deepseek",
                    "model": settings.models.deepseek.model,
                    "probe_status": probe.status,
                }
            else:
                # Only the environment variable NAME is ever printed; the
                # secret value never leaves the environment.
                # 只输出环境变量名；密钥值绝不离开环境。
                payload = {
                    "status": "ready",
                    "component": "deepseek",
                    "model": settings.models.deepseek.model,
                    "api_key_env": settings.models.deepseek.api_key_env,
                }
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    except Exception as error:
        print(
            json.dumps({"status": "failed", "error": f"{type(error).__name__}"}),
            file=sys.stderr,
        )
        return EXIT_RUNTIME
    print(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    )
    return EXIT_OK
