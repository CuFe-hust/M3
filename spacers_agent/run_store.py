"""Durable local run manifests, snapshots, and event storage.
可恢复的本地运行清单、快照与事件存储。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict

from spacers_agent.events import EventWriter
from spacers_agent.settings import AppSettings


class RunManifest(BaseModel):
    """Reproducibility metadata stored before a run can call a model.
    在运行调用模型前保存的可复现元数据。
    """

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
    创建运行目录且不记录 API 密钥或图像载荷。
    """

    def __init__(self, root: Path, project_root: Path) -> None:
        self.root = root
        self.project_root = project_root

    def create_run(
        self,
        settings: AppSettings,
        *,
        prompt_paths: list[Path],
        run_id: str | None = None,
        dataset: str | None = None,
        split: str | None = None,
        sample_filter: str | None = None,
    ) -> RunManifest:
        """Create manifest, config snapshot, Prompt copies, and event log.
        创建清单、配置快照、Prompt 副本和事件日志。
        """

        resolved_run_id = run_id or _new_run_id()
        run_dir = self.root / resolved_run_id
        if run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {run_dir}")
        prompts_dir = run_dir / "prompts.snapshot"
        prompts_dir.mkdir(parents=True)
        config_payload = settings.model_dump(mode="json")
        prompt_hashes = _snapshot_prompts(prompt_paths, prompts_dir)
        manifest = RunManifest(
            run_id=resolved_run_id,
            created_at=datetime.now(timezone.utc),
            git_commit=_git_value(self.project_root, "rev-parse", "HEAD"),
            git_dirty=_git_dirty(self.project_root),
            config_hash=_stable_hash(config_payload),
            prompt_hashes=prompt_hashes,
            model_ids={
                "qwen": settings.models.qwen.model,
                "deepseek": settings.models.deepseek.model,
            },
            dataset=dataset,
            split=split,
            sample_filter=sample_filter,
        )
        _write_json(run_dir / "manifest.json", manifest.model_dump(mode="json"))
        with (run_dir / "config.snapshot.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(config_payload, file, allow_unicode=True, sort_keys=True)
        EventWriter(run_dir / "events.jsonl").write("RUN_CREATED", details={"run_id": resolved_run_id})
        return manifest


def _new_run_id() -> str:
    """Create a sortable local run identifier.
    创建可排序的本地运行标识。
    """

    return f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"


def _snapshot_prompts(prompt_paths: list[Path], destination: Path) -> dict[str, str]:
    """Copy Prompt files and return content hashes keyed by filename.
    复制 Prompt 文件并返回按文件名索引的内容哈希。
    """

    hashes: dict[str, str] = {}
    for prompt_path in prompt_paths:
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Prompt file does not exist: {prompt_path}")
        target = destination / prompt_path.name
        if target.exists():
            raise ValueError(f"Duplicate Prompt filename: {prompt_path.name}")
        shutil.copy2(prompt_path, target)
        hashes[prompt_path.name] = _sha256_file(prompt_path)
    return hashes


def _sha256_file(path: Path) -> str:
    """Return a SHA256 digest for a small versioned asset.
    返回小型版本化资源的 SHA256 摘要。
    """

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(value: object) -> str:
    """Hash JSON-compatible deterministic metadata.
    对 JSON 兼容的确定性元数据计算哈希。
    """

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> None:
    """Write one UTF-8 JSON artifact with a stable layout.
    使用稳定布局写入一份 UTF-8 JSON 产物。
    """

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git_value(project_root: Path, *arguments: str) -> str | None:
    """Read a Git value without failing a local run outside a repository.
    读取 Git 值且在仓库外不使本地运行失败。
    """

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
    返回已跟踪文件是否与 Git HEAD 不同。
    """

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
