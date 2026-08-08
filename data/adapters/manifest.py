"""Self-contained dataset-probe write-back adapter for run manifests.

数据层自包含的 run manifest dataset_probe 写回适配。本模块位于 data 包，
不导入 workflows（data 层零业务依赖）：manifest.json 的读取校验与原子写回
全部自包含。缺失/损坏/违反最小 schema（run_id 非空字符串）的 manifest 以
稳定错误码失败；probe 载荷为 JSON-safe 且经过敏感扫描；写回原子
（tmp + replace），成功无临时文件残留；已存在的 probe 幂等覆盖。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data.adapters.base import AdapterProbe


class ManifestAdapterError(ValueError):
    """Stable error for manifest-adapter failures; the public message carries
    only the stable code, never raw manifest content.
    manifest 适配失败的稳定错误；公共消息只携带稳定 code，绝不携带原始
    manifest 内容。"""

    def __init__(self, code: str) -> None:
        super().__init__(f"MANIFEST_ADAPTER_FAILED:{code}")
        self.code = code


_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "access_token",
        "refresh_token",
        "private_key",
        "password",
        "credential",
        "token",
        "secret",
    }
)
_SENSITIVE_VALUE_PREFIXES = (
    "sk-",
    "bearer ",
    "data:image/",
    "-----begin private key-----",
)


def _probe_payload(probe: AdapterProbe) -> dict[str, object]:
    """Serialize an AdapterProbe to a JSON-safe mapping. Paths use the repo's
    POSIX serialization convention and tuples become lists; optional fields
    are omitted when empty. 将 AdapterProbe 序列化为 JSON-safe 映射：路径按
    仓库 POSIX 序列化约定转字符串、tuple 转 list；可选字段为空时省略。"""

    payload: dict[str, object] = {
        "dataset": probe.dataset,
        "version": probe.version,
        "sample_file": probe.sample_file.as_posix(),
        "observed_fields": list(probe.observed_fields),
        "sample_count": probe.sample_count,
    }
    if probe.task is not None:
        payload["task"] = probe.task
    if probe.available_tasks:
        payload["available_tasks"] = list(probe.available_tasks)
    return payload


def _reject_sensitive(value: Any) -> None:
    """Reject high-risk sensitive keys and value prefixes recursively; the
    probe payload is adapter-controlled, this is a uniform safety net.
    递归拒绝高风险敏感键与值前缀；probe 载荷由适配器受控，本检查是统一
    安全网。"""

    if isinstance(value, str):
        normalized = value.lstrip().lower()
        if any(normalized.startswith(prefix) for prefix in _SENSITIVE_VALUE_PREFIXES):
            raise ValueError("dataset probe contains a sensitive value prefix")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            normalized_key = str(key).lower().replace("-", "_").replace(" ", "_")
            if normalized_key in _SENSITIVE_KEYS:
                raise ValueError("dataset probe contains a sensitive key")
            _reject_sensitive(item)
    elif isinstance(value, list):
        for item in value:
            _reject_sensitive(item)


def update_manifest_probe(run_dir: Path, probe: AdapterProbe) -> dict[str, Any]:
    """Read manifest.json, attach the dataset probe, and rewrite it atomically.
    Missing, unparseable, or schema-violating manifests fail with stable codes;
    an existing probe is overwritten (idempotent). Returns the updated manifest
    mapping. 读取 manifest.json，写入 dataset_probe 并原子重写。缺失/无法
    解析/违反最小 schema 的 manifest 以稳定错误码失败；已存在的 probe 幂等
    覆盖。返回更新后的 manifest 映射。"""

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ManifestAdapterError("MANIFEST_MISSING")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ManifestAdapterError("MANIFEST_INVALID") from exc
    if not isinstance(raw, dict):
        raise ManifestAdapterError("MANIFEST_INVALID")
    run_id = raw.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ManifestAdapterError("MANIFEST_SCHEMA")
    payload = _probe_payload(probe)
    _reject_sensitive(payload)
    updated = dict(raw)
    updated["dataset_probe"] = payload
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return updated
