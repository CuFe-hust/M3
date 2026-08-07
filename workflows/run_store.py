"""Durable local run manifests, snapshots, and event storage.

可恢复的本地运行清单、快照与事件存储。RunStore 只接受可序列化的
config payload、model IDs 与 prompt 路径——不依赖任何完整配置对象、不
构造 Agent、不调用任何模型；创建 run 本身绝不触发模型调用。所有写入
（manifest、config 快照、Prompt 副本、事件日志）均为原子替换；快照在写
入前做密钥校验，绝不记录 API key。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from workflows.events import EventWriter

# Sensitive key names rejected in snapshots (normalized before matching).
# 快照中拒绝的敏感键名（匹配前先归一化）。
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "secret",
        "token",
        "password",
        "base64",
        "credential",
        "private_key",
    }
)
# High-risk sensitive value prefixes, checked after lstrip().lower().
# 高风险敏感值前缀；在 lstrip().lower() 后检查。
_SENSITIVE_VALUE_PREFIXES = (
    "sk-",
    "bearer ",
    "data:image/",
    "-----begin private key-----",
)


class RunManifest(BaseModel):
    """Reproducibility metadata stored before a run can call a model.
    在运行调用模型前保存的可复现元数据。"""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    created_at: datetime
    git_commit: str | None
    git_dirty: bool | None
    config_hash: str
    prompt_hashes: dict[str, str]
    model_ids: dict[str, str]
    dataset: str | None = None
    split: str | None = None
    sample_filter: str | None = None


class RunStore:
    """Create a run directory without recording API keys or image payloads.
    创建运行目录且不记录 API 密钥或图像载荷。"""

    def __init__(self, root: Path, project_root: Path) -> None:
        self.root = root
        self.project_root = project_root

    def create_run(
        self,
        *,
        config_payload: Mapping[str, Any],
        model_ids: Mapping[str, str],
        prompt_paths: list[Path],
        run_id: str | None = None,
        dataset: str | None = None,
        split: str | None = None,
        sample_filter: str | None = None,
    ) -> RunManifest:
        """Create manifest, config snapshot, Prompt copies, and event log.
        No model client exists in this path, so creating a run can never call
        a model. 创建清单、配置快照、Prompt 副本和事件日志。该路径不含任何
        模型客户端，创建 run 绝不会调用模型。"""

        # Snapshots must never record credentials; reject before any I/O.
        # 快照绝不能记录凭据；在任何 I/O 之前拒绝。
        _reject_secrets(config_payload, "config payload")
        _reject_secrets(model_ids, "model ids")

        resolved_run_id = run_id or _new_run_id()
        run_dir = self.root / resolved_run_id
        if run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        prompts_dir = run_dir / "prompts.snapshot"
        prompts_dir.mkdir(parents=True)
        prompt_hashes = _snapshot_prompts(prompt_paths, prompts_dir)
        config_payload_plain = json.loads(json.dumps(config_payload, ensure_ascii=False))
        manifest = RunManifest(
            run_id=resolved_run_id,
            created_at=datetime.now(timezone.utc),
            git_commit=_git_value(self.project_root, "rev-parse", "HEAD"),
            git_dirty=_git_dirty(self.project_root),
            config_hash=_stable_hash(config_payload_plain),
            prompt_hashes=prompt_hashes,
            model_ids=dict(model_ids),
            dataset=dataset,
            split=split,
            sample_filter=sample_filter,
        )
        _write_json(run_dir / "manifest.json", manifest.model_dump(mode="json"))
        _write_json(run_dir / "config.snapshot.json", config_payload_plain)
        EventWriter(run_dir / "events.jsonl").write(
            "RUN_CREATED", details={"run_id": resolved_run_id}
        )
        return manifest


def _new_run_id() -> str:
    """Create a sortable local run identifier.
    创建可排序的本地运行标识。"""

    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"


def _snapshot_prompts(prompt_paths: list[Path], destination: Path) -> dict[str, str]:
    """Copy Prompt files and return content hashes keyed by filename. Copies
    are atomic: each file lands via a temporary name then replace.
    复制 Prompt 文件并返回按文件名索引的内容哈希。副本原子化：先写临时名
    再 replace。"""

    hashes: dict[str, str] = {}
    for prompt_path in prompt_paths:
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Prompt file does not exist: {prompt_path}")
        target = destination / prompt_path.name
        if target.exists():
            raise ValueError(f"Duplicate Prompt filename: {prompt_path.name}")
        temporary = destination / (target.name + ".tmp")
        shutil.copy2(prompt_path, temporary)
        temporary.replace(target)
        hashes[prompt_path.name] = _sha256_file(prompt_path)
    return hashes


def _sha256_file(path: Path) -> str:
    """Return a SHA256 digest for a small versioned asset.
    返回小型版本化资源的 SHA256 摘要。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(value: object) -> str:
    """Hash JSON-compatible deterministic metadata.
    对 JSON 兼容的确定性元数据计算哈希。"""

    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    """Write one UTF-8 JSON artifact with a stable layout, atomically.
    使用稳定布局原子写入一份 UTF-8 JSON 产物。"""

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _reject_secrets(value: Any, where: str) -> None:
    """Recursively reject sensitive keys and high-risk value prefixes in a
    snapshot payload; error messages never echo the offending value.
    递归拒绝快照载荷中的敏感键与高风险值前缀；错误消息绝不回显违规值。"""

    if isinstance(value, str):
        normalized = value.lstrip().lower()
        for prefix in _SENSITIVE_VALUE_PREFIXES:
            if normalized.startswith(prefix):
                raise ValueError(f"{where} contains a sensitive value prefix")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = str(key).lower().replace("-", "_").replace(" ", "_")
            if normalized_key in _SENSITIVE_KEYS:
                raise ValueError(f"{where} contains a sensitive key")
            _reject_secrets(item, f"{where}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secrets(item, f"{where}[{index}]")
        return


def _git_value(project_root: Path, *arguments: str) -> str | None:
    """Read a Git value without failing a local run outside a repository.
    读取 Git 值且在仓库外不使本地运行失败。"""

    completed = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value or None


def _git_dirty(project_root: Path) -> bool | None:
    """Return whether tracked files differ from Git HEAD.
    返回已跟踪文件是否与 Git HEAD 不同。"""

    completed = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return bool(completed.stdout.strip())
