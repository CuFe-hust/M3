"""Shared CLI command helpers and stable exit codes. / 共享 CLI 辅助函数和稳定退出码。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spacers_agent.clients.base import JsonResponseCache
from spacers_agent.clients.qwen_vllm import QwenVLLMClient
from spacers_agent.settings import AppSettings

EXIT_OK = 0
EXIT_ARGUMENT = 2
EXIT_DATA = 3
EXIT_QWEN_UNAVAILABLE = 4
EXIT_QWEN_FAILED = 5
EXIT_PARTIAL = 6
EXIT_DEEPSEEK_FAILED = 7
EXIT_INVARIANT = 8


def project_root() -> Path:
    """Return the repository root without a user-specific path. / 返回不含用户特定路径的仓库根目录。"""

    return Path(__file__).resolve().parents[2]


def prompts() -> dict[str, str]:
    """Load the versioned prompt files needed by commands. / 加载命令所需的版本化 Prompt 文件。"""

    root = project_root() / "prompts"
    return {
        "count": (root / "count_tile_v2.md").read_text(encoding="utf-8"),
        "target": (root / "target_parse_v1.md").read_text(encoding="utf-8"),
        "change": (root / "change_v1.md").read_text(encoding="utf-8"),
        "spatial": (root / "spatial_v2.md").read_text(encoding="utf-8"),
        "general": (root / "general_vqa_v2.md").read_text(encoding="utf-8"),
        "seam": (root / "seam_verify_v1.md").read_text(encoding="utf-8"),
    }


def qwen_client(settings: AppSettings, run_dir: Path) -> QwenVLLMClient:
    """Create the run-scoped Qwen client and cache. / 创建运行范围的 Qwen 客户端和缓存。"""

    repair = (project_root() / "prompts" / "json_repair_v1.md").read_text(encoding="utf-8")
    return QwenVLLMClient(settings.models.qwen, repair_prompt=repair, cache=JsonResponseCache(run_dir / "cache"))


def emit_summary(payload: dict[str, Any]) -> None:
    """Print a machine-readable final stdout line. / 输出机器可读的最终标准输出行。"""

    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
