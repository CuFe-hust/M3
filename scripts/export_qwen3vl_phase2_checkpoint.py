#!/usr/bin/env python3
"""Phase 2 composite checkpoint exporter (task docs/train/04).

Exports a resumable Phase 2 composite training checkpoint

    base Qwen3-VL checkpoint
    + LLM LoRA adapter
    + fully trained main Merger / DeepStack Merger state

into a complete deployable checkpoint that can be loaded directly by
``AutoModelForImageTextToText.from_pretrained()`` and the project's
Qwen3-VL main flow.

The exporter only restores, merges, saves and validates the model: it never
reads training data, never trains, and never re-interprets data configs.
导出器只恢复、合并、保存和验证模型；不读取训练集、不执行训练、不重新
解释数据配置。

Fixed export order (doc 04 section 5):

    validate training manifest -> load base -> enumerate expected merger
    keys -> strict merger load -> attach LLM LoRA -> verify PEFT adapter
    keys/targets -> merge_and_unload -> audit final model/config -> save
    model + processor -> copy auxiliary files -> offline reload validation
    -> optional forward check -> write export manifest -> atomic publish.

All saves and reload validations happen in an explicit temp directory next
to the final output; the final directory is published with a same-filesystem
atomic rename only after every gate passes. 所有保存与 reload 验证都在最终
输出旁的显式临时目录完成；全部门禁通过后才用同文件系统原子 rename 发布。

Importing this module loads no weights: torch/transformers/peft are imported
lazily inside the execution path, so ``--help`` works without them.
本模块 import 不加载权重；torch/transformers/peft 均在执行路径内惰性导入。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# Stable constants (mirrors the composite checkpoint contract produced by
# scripts/finetune_qwen3vl_phase2.py).
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1
TRAINING_MANIFEST_FILENAME = "phase2_training_manifest.json"
EXPORT_MANIFEST_FILENAME = "phase2_export_manifest.json"
MERGER_STATE_FILENAME = "merger_model.safetensors"
TEMP_DIR_SUFFIX = ".export-tmp"

# Small base-side config files that save_pretrained does not emit; copied
# from the validated base checkpoint only when missing in the output dir,
# never overwriting anything already produced.
# save_pretrained 不会产出的辅助配置；仅当输出目录缺失时从已验证 base 复制。
_AUXILIARY_FILES = (
    "generation_config.json",
    "chat_template.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
)

# PEFT name prefixes used by peft 0.20 wrappers.
_PEFT_NAME_PREFIXES = ("base_model.model.", "base_model.")

# Secret-like manifest key patterns (fail-closed safety scan).
_SECRET_KEY_PATTERNS = (
    "api_key", "apikey", "authorization", "password", "credential",
    "private_key", "secret",
)

logger_name = "export_qwen3vl_phase2_checkpoint"


# ---------------------------------------------------------------------------
# Stable error types (public messages never carry credentials, raw exception
# dumps or machine absolute paths).
# 稳定错误类型（公共消息不携带凭据、原始异常全文或机器绝对路径）。
# ---------------------------------------------------------------------------


class ExportError(Exception):
    """Base class for exporter errors. / 导出器错误基类。"""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"export error {code}: {detail}")
        self.code = code
        self.detail = detail


class CheckpointValidationError(ExportError):
    """Composite checkpoint layout/checksum/manifest validation failed.
    复合 checkpoint 布局/checksum/manifest 校验失败。"""


class BaseIdentityError(ExportError):
    """CLI base checkpoint does not match the training manifest identity.
    CLI base checkpoint 与训练 manifest 身份不一致。"""


class MergerLoadError(ExportError):
    """Strict merger state load failed (missing/unexpected/shape/dtype/count).
    merger state 严格加载失败（缺失/意外/shape/dtype/数量）。"""


class LoRAValidationError(ExportError):
    """PEFT adapter attach/verify/merge failed.
    PEFT adapter 挂载/校验/合并失败。"""


class SaveError(ExportError):
    """Model/processor saving failed inside the temp directory.
    临时目录内模型/processor 保存失败。"""


class ReloadValidationError(ExportError):
    """Offline reload validation of the exported directory failed.
    导出目录的离线 reload 验证失败。"""


class PublishError(ExportError):
    """Atomic publish of the final directory failed.
    最终目录原子发布失败。"""


# ---------------------------------------------------------------------------
# Lazy heavy-dependency loaders (module import must stay weight-free).
# ---------------------------------------------------------------------------


def _torch() -> Any:
    import torch  # noqa: PLC0415

    return torch


def _transformers() -> Any:
    import transformers  # noqa: PLC0415

    return transformers


def _peft() -> Any:
    import peft  # noqa: PLC0415

    return peft


# ---------------------------------------------------------------------------
# Seams (thin wrappers around the transformers/peft API so unit tests can
# substitute the tiny fake model tree without loading any 8B weights).
# 测试 seam：围绕 transformers/peft 的薄封装，单测可注入 fake 模型树。
# ---------------------------------------------------------------------------


def load_base_components(
    model_id: str | Path,
    torch_dtype: Any,
    device: str,
    local_files_only: bool,
) -> tuple[Any, Any, Any]:
    """Load config + model + processor from the base checkpoint; validates
    model_type == 'qwen3_vl'. 从 base checkpoint 加载 config/model/processor。"""
    transformers_mod = _transformers()
    config = transformers_mod.AutoConfig.from_pretrained(
        model_id, local_files_only=local_files_only
    )
    if config.model_type != "qwen3_vl":
        raise BaseIdentityError(
            "unsupported_model_type",
            f"expected qwen3_vl, got {config.model_type!r}",
        )
    model = transformers_mod.AutoModelForImageTextToText.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch_dtype,
        device_map=device,
        local_files_only=local_files_only,
    )
    processor = transformers_mod.AutoProcessor.from_pretrained(
        model_id, local_files_only=local_files_only
    )
    return config, model, processor


def load_processor_from_dir(
    path: str | Path, local_files_only: bool
) -> Any:
    """Load the processor saved inside the composite checkpoint.
    加载复合 checkpoint 中保存的 processor。"""
    transformers_mod = _transformers()
    return transformers_mod.AutoProcessor.from_pretrained(
        path, local_files_only=local_files_only
    )


def _peft_from_pretrained(base_model: Any, adapter_dir: str | Path) -> Any:
    """Attach the saved LoRA adapter via the official PEFT loading API.
    通过 PEFT 官方加载接口挂载 LoRA adapter。"""
    return _peft().PeftModel.from_pretrained(base_model, str(adapter_dir))


def _auto_config_from_pretrained(path: str | Path, local_files_only: bool) -> Any:
    transformers_mod = _transformers()
    return transformers_mod.AutoConfig.from_pretrained(
        path, local_files_only=local_files_only
    )


def _auto_model_from_pretrained(
    path: str | Path,
    torch_dtype: Any,
    device: str,
    local_files_only: bool,
) -> Any:
    transformers_mod = _transformers()
    return transformers_mod.AutoModelForImageTextToText.from_pretrained(
        path,
        torch_dtype=torch_dtype,
        device_map=device,
        local_files_only=local_files_only,
    )


def _auto_processor_from_pretrained(path: str | Path, local_files_only: bool) -> Any:
    transformers_mod = _transformers()
    return transformers_mod.AutoProcessor.from_pretrained(
        path, local_files_only=local_files_only
    )


# ---------------------------------------------------------------------------
# Files, checksums, runtime helpers
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    """Streaming sha256 of a file (stable across machines).
    文件的流式 sha256（跨机器稳定）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: str | Path, payload: Any) -> None:
    """Write JSON via temp file + atomic replace; no half-written artifacts.
    临时文件 + 原子替换写 JSON；不留下半个文件。"""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=False)
        handle.write("\n")
    os.replace(tmp, path)


def git_head(repo_root: str | Path) -> str:
    """Current git HEAD hash; 'unknown' when not available.
    当前 git HEAD；不可用时为 'unknown'。"""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:  # noqa: BLE001 - best effort only
        pass
    return "unknown"


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


def resolve_dtype(torch_module: Any, name: str) -> Any:
    """Resolve the CLI dtype name to a torch dtype or 'auto'.
    将 CLI dtype 名称解析为 torch dtype 或 'auto'。"""
    if name == "auto":
        return "auto"
    mapping = {
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }
    if name not in mapping:
        raise ExportError("unsupported_dtype", name)
    return mapping[name]


# ---------------------------------------------------------------------------
# Identities (fingerprint logic MUST stay in sync with
# scripts/finetune_qwen3vl_phase2.py model_logical_identity / processor_identity;
# tests assert the two modules agree on the same config/processor).
# 指纹逻辑必须与训练脚本保持一致；测试会断言两模块对同一 config 计算一致。
# ---------------------------------------------------------------------------


def model_logical_identity(config: Any) -> dict:
    """Path-independent logical identity of the base model.
    与机器路径无关的 base model 逻辑身份。"""
    vision_config = getattr(config, "vision_config", None)
    vision_fields = {}
    if vision_config is not None:
        for name in (
            "hidden_size", "out_hidden_size", "depth", "spatial_merge_size",
            "patch_size", "deepstack_visual_indexes",
        ):
            if hasattr(vision_config, name):
                vision_fields[name] = getattr(vision_config, name)
    fields = {
        "model_type": config.model_type,
        "architectures": list(config.architectures or []),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "intermediate_size": getattr(config, "intermediate_size", None),
        "vocab_size": getattr(config, "vocab_size", None),
        "vision_config": vision_fields,
    }
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return {
        "fields": fields,
        "fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "revision": getattr(config, "_commit_hash", None),
    }


def processor_identity(processor: Any) -> dict:
    """Path-independent processor identity: class + tokenizer fingerprint.
    与路径无关的 processor 身份：类名 + tokenizer 指纹。"""
    tokenizer = getattr(processor, "tokenizer", None)
    tokenizer_fields = {}
    if tokenizer is not None:
        tokenizer_fields = {
            "class": tokenizer.__class__.__name__,
            "vocab_size": getattr(tokenizer, "vocab_size", None),
            "has_chat_template": bool(getattr(tokenizer, "chat_template", None)),
        }
    fields = {
        "processor_class": processor.__class__.__name__,
        "tokenizer": tokenizer_fields,
    }
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return {
        "fields": fields,
        "fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


# ---------------------------------------------------------------------------
# Model structure location (structure-based, mirrors finetune script).
# ---------------------------------------------------------------------------


def locate_roots(model: Any) -> tuple[str, str]:
    """Locate the vision root (owner of merger + deepstack_merger_list) and
    the text root (owner of layers + embed_tokens); returns their full names.
    定位视觉根与文本根，返回全路径名。"""
    torch = _torch()
    vision_name, text_name = None, None
    for name, module in model.named_modules():
        if vision_name is None and isinstance(
            getattr(module, "deepstack_merger_list", None), torch.nn.ModuleList
        ) and isinstance(getattr(module, "merger", None), torch.nn.Module):
            vision_name = name
        if text_name is None and isinstance(
            getattr(module, "layers", None), torch.nn.ModuleList
        ) and isinstance(getattr(module, "embed_tokens", None), torch.nn.Module):
            text_name = name
    if not vision_name:
        raise MergerLoadError("vision_root_not_found", "expected module with merger + deepstack_merger_list")
    if not text_name:
        raise MergerLoadError("text_root_not_found", "expected module with layers + embed_tokens")
    return vision_name, text_name


def enumerate_merger_names(model: Any, vision_root_name: str) -> list[str]:
    """Full module paths of the main merger and every deepstack merger.
    主 merger 与全部 deepstack merger 的全路径。"""
    vision_root = model.get_submodule(vision_root_name)
    names = [f"{vision_root_name}.merger"]
    for index, _merger in enumerate(vision_root.deepstack_merger_list):
        names.append(f"{vision_root_name}.deepstack_merger_list.{index}")
    return names


def expected_deepstack_count(config: Any) -> int | None:
    """Number of deepstack mergers declared by the config, when available.
    配置声明的 deepstack merger 数量（可用时）。"""
    vision_config = getattr(config, "vision_config", None)
    if vision_config is None:
        return None
    indexes = getattr(vision_config, "deepstack_visual_indexes", None)
    if indexes is None:
        return None
    try:
        return len(list(indexes))
    except TypeError:
        return None


def _under_any(name: str, roots: Sequence[str]) -> bool:
    return any(name == root or name.startswith(root + ".") for root in roots)


def expected_merger_table(model: Any, merger_names: Sequence[str]) -> dict[str, dict]:
    """Name -> {shape, dtype, numel} of every parameter and persistent
    buffer inside the merger subtrees. 枚举 merger 子树内全部参数与
    persistent buffer 的 name/shape/dtype/numel。"""
    torch = _torch()
    table: dict[str, dict] = {}
    for name, parameter in model.named_parameters():
        if _under_any(name, merger_names):
            table[name] = {
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "numel": int(parameter.numel()),
            }
    for name, buffer in model.named_buffers():
        if _under_any(name, merger_names) and buffer is not None and isinstance(buffer, torch.Tensor):
            table[name] = {
                "shape": list(buffer.shape),
                "dtype": str(buffer.dtype),
                "numel": int(buffer.numel()),
            }
    return table


def _strip_peft_prefix(name: str) -> str:
    for prefix in _PEFT_NAME_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


# peft saves on-disk adapter keys as ``lora_A.weight`` but names the
# in-memory parameter ``lora_A.default.weight`` (active adapter "default");
# the consumption check must treat both spellings as the same tensor.
# peft 磁盘上的 adapter key 是 lora_A.weight，内存参数名为
# lora_A.default.weight；消费检查必须把两种拼写视为同一张量。
_LORA_DEFAULT_RE = re.compile(r"(\.lora_[^.]+)\.default(\..*)?$")


def _normalize_lora_key(name: str) -> str:
    return _LORA_DEFAULT_RE.sub(r"\1\2", name)


# ---------------------------------------------------------------------------
# 1. Composite checkpoint validation (before any big weight load).
# ---------------------------------------------------------------------------


def validate_checkpoint_layout(checkpoint_path: str | Path) -> dict:
    """Verify the composite checkpoint layout and manifest schema; returns
    the parsed training manifest. No weights are loaded here.
    校验复合 checkpoint 布局与 manifest schema；不加载权重。"""
    checkpoint_path = Path(checkpoint_path)
    required = {
        "adapter/adapter_config.json": checkpoint_path / "adapter" / "adapter_config.json",
        "adapter/adapter_model.safetensors": checkpoint_path / "adapter" / "adapter_model.safetensors",
        MERGER_STATE_FILENAME: checkpoint_path / MERGER_STATE_FILENAME,
        "processor": checkpoint_path / "processor",
        TRAINING_MANIFEST_FILENAME: checkpoint_path / TRAINING_MANIFEST_FILENAME,
    }
    missing = [name for name, path in required.items() if not path.is_file() and name != "processor"]
    if not (checkpoint_path / "processor").is_dir():
        missing.append("processor")
    if missing:
        raise CheckpointValidationError(
            "checkpoint_incomplete", "; ".join(missing)
        )
    manifest_path = checkpoint_path / TRAINING_MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointValidationError("manifest_unreadable", "phase2_training_manifest.json") from error
    if not isinstance(manifest, dict):
        raise CheckpointValidationError("manifest_invalid", "not a JSON object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise CheckpointValidationError(
            "manifest_schema_unsupported", str(manifest.get("schema_version"))
        )
    if manifest.get("checkpoint_type") != "phase2_composite":
        raise CheckpointValidationError(
            "checkpoint_type_mismatch", str(manifest.get("checkpoint_type"))
        )
    return manifest


def _manifest_safety_scan(value: Any, key_path: str, violations: list[str]) -> None:
    """Recursive scan: secret-like keys and absolute paths under path-like
    keys are hard violations. 递归扫描：疑似密钥键名与 path 键的绝对路径
    都是硬性违规。"""
    if isinstance(value, dict):
        for key, item in value.items():
            low = str(key).lower()
            next_path = f"{key_path}.{key}" if key_path else str(key)
            # exact secret-name matches plus *_token suffixes; the producer's
            # legitimate manifest uses e.g. data_sampling.group_key, which is
            # a repeat-group key, not a credential.
            if low in _SECRET_KEY_PATTERNS or low.endswith("_token"):
                violations.append(f"secret-like key: {next_path}")
            if "path" in low and isinstance(item, str):
                if _is_absolute_path(item):
                    violations.append(f"absolute path under {next_path}")
            _manifest_safety_scan(item, next_path, violations)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _manifest_safety_scan(item, f"{key_path}[{index}]", violations)


def _is_absolute_path(value: str) -> bool:
    if not value:
        return False
    if value.startswith(("/", "\\", "~")):
        return True
    if len(value) > 1 and value[1] == ":" and value[0].isalpha():
        return True  # windows drive
    return False


def validate_manifest_safety(manifest: dict) -> None:
    """The training manifest must not contain dangerous absolute result
    paths or secrets. 训练 manifest 不得包含危险绝对 result path 或 secret。"""
    violations: list[str] = []
    _manifest_safety_scan(manifest, "", violations)
    if violations:
        raise CheckpointValidationError(
            "manifest_unsafe", "; ".join(violations[:10])
        )


def verify_checkpoint_checksums(checkpoint_path: str | Path, manifest: dict) -> dict:
    """Verify adapter + merger file sha256 against the training manifest
    BEFORE loading any big weights. 在加载大权重前校验 adapter/merger 的
    sha256 与训练 manifest 一致。"""
    checkpoint_path = Path(checkpoint_path)
    adapter_meta = manifest.get("adapter") or {}
    merger_meta = manifest.get("merger") or {}
    results = {}

    adapter_file = checkpoint_path / "adapter" / "adapter_model.safetensors"
    actual_adapter = sha256_file(adapter_file)
    recorded_adapter = adapter_meta.get("file_sha256")
    if not recorded_adapter:
        raise CheckpointValidationError("manifest_adapter_sha256_missing", "adapter.file_sha256")
    if actual_adapter != recorded_adapter:
        raise CheckpointValidationError("adapter_checksum_mismatch", "adapter/adapter_model.safetensors")
    results["adapter"] = {"file": "adapter/adapter_model.safetensors", "file_sha256": actual_adapter}

    merger_file = checkpoint_path / MERGER_STATE_FILENAME
    actual_merger = sha256_file(merger_file)
    recorded_merger = merger_meta.get("file_sha256")
    if not recorded_merger:
        raise CheckpointValidationError("manifest_merger_sha256_missing", "merger.file_sha256")
    if actual_merger != recorded_merger:
        raise CheckpointValidationError("merger_checksum_mismatch", MERGER_STATE_FILENAME)
    results["merger"] = {"file": MERGER_STATE_FILENAME, "file_sha256": actual_merger}
    return results


def verify_base_identity(config: Any, manifest: dict) -> dict:
    """CLI base checkpoint logical identity/revision must match the training
    manifest (the local path is only a loading hint, never an identity).
    CLI base 的逻辑身份/revision 必须与训练 manifest 一致。"""
    identity = model_logical_identity(config)
    recorded = manifest.get("base_model") or {}
    if not recorded.get("fingerprint"):
        raise CheckpointValidationError("manifest_base_identity_missing", "base_model.fingerprint")
    if identity["fingerprint"] != recorded["fingerprint"]:
        raise BaseIdentityError("base_fingerprint_mismatch", "CLI base checkpoint differs from training manifest")
    recorded_revision = recorded.get("revision")
    if recorded_revision and identity.get("revision") and recorded_revision != identity["revision"]:
        raise BaseIdentityError("base_revision_mismatch", "CLI base checkpoint revision differs from training manifest")
    return identity


# ---------------------------------------------------------------------------
# 3/4/6. Merger strict load (three-way: model table vs manifest vs file).
# ---------------------------------------------------------------------------


def verify_manifest_merger_layout(manifest: dict, merger_names: Sequence[str], config: Any) -> None:
    """The manifest must declare exactly the main merger + every deepstack
    merger, matching both the model enumeration and the config declaration.
    manifest 必须声明与模型枚举、配置声明一致的主 merger 与全部 deepstack。"""
    declared = manifest.get("merger", {}).get("modules")
    if not isinstance(declared, list) or sorted(declared) != sorted(merger_names):
        raise MergerLoadError(
            "merger_module_declaration_mismatch",
            f"manifest={sorted(declared or [])} model={sorted(merger_names)}",
        )
    expected = expected_deepstack_count(config)
    if expected is not None:
        deepstack_names = [name for name in merger_names if "deepstack_merger_list" in name]
        if len(deepstack_names) != expected:
            raise MergerLoadError(
                "deepstack_count_mismatch",
                f"model={len(deepstack_names)} declared={expected}",
            )


def compare_manifest_parameter_table(manifest: dict, model_table: Mapping[str, dict]) -> None:
    """Manifest parameter table (name/shape/dtype/numel) must match the base
    model's enumerated merger keys exactly. manifest 参数表必须与 base 模型
    枚举的 merger key 完全一致。"""
    entries = manifest.get("merger", {}).get("parameters")
    if not isinstance(entries, list):
        raise MergerLoadError("manifest_merger_parameters_missing", "merger.parameters")
    if len(entries) != len(model_table):
        raise MergerLoadError(
            "merger_parameter_count_mismatch",
            f"manifest={len(entries)} model={len(model_table)}",
        )
    manifest_by_name = {entry.get("name"): entry for entry in entries}
    manifest_names = set(manifest_by_name)
    model_names = set(model_table)
    if manifest_names != model_names:
        missing = sorted(model_names - manifest_names)[:5]
        unexpected = sorted(manifest_names - model_names)[:5]
        raise MergerLoadError(
            "merger_key_set_mismatch",
            f"missing={missing} unexpected={unexpected}",
        )
    for name, entry in manifest_by_name.items():
        meta = model_table[name]
        if entry.get("shape") != meta["shape"] or entry.get("dtype") != meta["dtype"] or entry.get("numel") != meta["numel"]:
            raise MergerLoadError("merger_parameter_meta_mismatch", name)


def load_merger_strict(
    base_model: Any,
    merger_path: str | Path,
    manifest: dict,
    model_table: Mapping[str, dict],
) -> tuple[dict[str, Any], list[dict]]:
    """Strictly load merger_model.safetensors into the base model.

    Three-way comparison (doc 04 section 6): the expected keys/shapes/dtypes
    from the live base model, the manifest parameter table, and the actual
    safetensors content. Missing key / unexpected key / shape mismatch /
    non-float dtype / count mismatch are hard failures. An explicit float
    conversion to the model parameter dtype is allowed and recorded.
    """
    torch = _torch()
    import safetensors.torch  # noqa: PLC0415

    try:
        state = safetensors.torch.load_file(Path(merger_path))
    except Exception as error:  # noqa: BLE001 - stable error surface
        raise MergerLoadError("merger_file_unreadable", MERGER_STATE_FILENAME) from error

    missing = sorted(set(model_table) - set(state))
    unexpected = sorted(set(state) - set(model_table))
    if missing or unexpected:
        raise MergerLoadError(
            "merger_key_mismatch",
            f"missing={missing[:5]} unexpected={unexpected[:5]}",
        )
    if len(state) != len(model_table):
        raise MergerLoadError(
            "merger_count_mismatch", f"file={len(state)} model={len(model_table)}"
        )

    conversions: list[dict] = []
    loaded: dict[str, Any] = {}
    parameter_map = {name: parameter for name, parameter in base_model.named_parameters()}
    buffer_map = {name: buffer for name, buffer in base_model.named_buffers()}
    for name, tensor in state.items():
        meta = model_table[name]
        if list(tensor.shape) != meta["shape"]:
            raise MergerLoadError("merger_shape_mismatch", name)
        if not torch.is_floating_point(tensor):
            raise MergerLoadError("merger_unsafe_dtype", f"{name}: {tensor.dtype}")
        target = parameter_map.get(name)
        if target is None:
            target = buffer_map.get(name)
        if target is None:
            raise MergerLoadError("merger_target_missing", name)
        if tensor.dtype != target.dtype:
            if not torch.is_floating_point(target):
                raise MergerLoadError("merger_target_unsafe_dtype", name)
            tensor = tensor.to(dtype=target.dtype)
            conversions.append(
                {"name": name, "source_dtype": str(tensor.dtype), "target_dtype": str(target.dtype)}
            )
        if list(tensor.shape) != list(target.shape):
            raise MergerLoadError("merger_shape_mismatch", name)
        loaded[name] = tensor.detach().clone()
        target.data.copy_(tensor)

    # Manifest dtype cross-check against the file content (three-way).
    # manifest 记录的 dtype 与文件内容三方比对。
    entries = {entry["name"]: entry for entry in (manifest.get("merger", {}).get("parameters") or [])}
    for name, tensor in state.items():
        recorded = entries.get(name)
        if recorded is not None and recorded.get("dtype") != str(tensor.dtype):
            raise MergerLoadError("merger_dtype_mismatch", name)
    return loaded, conversions


# ---------------------------------------------------------------------------
# 5/6/7. LLM LoRA attach, verify, merge_and_unload.
# ---------------------------------------------------------------------------


def _lora_parent_module(logical_name: str) -> str:
    segments = logical_name.split(".")
    cut = len(segments)
    for index, segment in enumerate(segments):
        if segment.startswith("lora_"):
            cut = index
            break
    return ".".join(segments[:cut])


def _has_lora_segment(logical_name: str) -> bool:
    return any(segment.startswith("lora_") for segment in logical_name.split("."))


def attach_and_verify_adapter(
    base_model: Any,
    adapter_dir: str | Path,
    manifest: dict,
    merger_names: Sequence[str],
    local_files_only: bool,
) -> tuple[Any, dict]:
    """Attach the LLM LoRA adapter and verify it against the manifest:

    - adapter_config peft_type must be LORA;
    - adapter-declared base identity must match the training manifest;
    - adapter target module set must equal the manifest target set exactly;
    - every adapter tensor must be consumed by the attached peft model;
    - no visual/merger LoRA targets.
    """
    del local_files_only  # PeftModel.from_pretrained uses the local dir only
    adapter_dir = Path(adapter_dir)
    try:
        adapter_config = json.loads(
            (adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise LoRAValidationError("adapter_config_unreadable", "adapter/adapter_config.json") from error

    peft_type = str(adapter_config.get("peft_type") or "").lower()
    if peft_type != "lora":
        raise LoRAValidationError("adapter_not_lora", peft_type or "missing")

    manifest_targets = sorted(manifest.get("lora", {}).get("target_modules") or [])
    if not manifest_targets:
        raise LoRAValidationError("manifest_lora_targets_missing", "lora.target_modules")

    # Adapter-declared base identity must agree with the training manifest.
    declared_base = adapter_config.get("base_model_name_or_path")
    manifest_base = manifest.get("model_id_as_given")
    if declared_base and manifest_base and declared_base != manifest_base:
        raise LoRAValidationError(
            "adapter_base_identity_mismatch",
            "adapter base_model_name_or_path differs from manifest model_id_as_given",
        )

    # Manifest targets themselves must stay inside the LLM subtree.
    for target in manifest_targets:
        segments = target.split(".")
        if "visual" in segments or _under_any(target, merger_names):
            raise LoRAValidationError("manifest_lora_target_outside_llm", target)

    peft_model = _peft_from_pretrained(base_model, adapter_dir)

    attached_targets = sorted(peft_model.peft_config[peft_model.active_adapter].target_modules or [])
    if attached_targets != manifest_targets:
        raise LoRAValidationError(
            "lora_target_mismatch",
            f"adapter={attached_targets} manifest={manifest_targets}",
        )

    # Every adapter tensor must be consumed by the attached model.
    import safetensors  # noqa: PLC0415

    try:
        with safetensors.safe_open(adapter_dir / "adapter_model.safetensors", framework="pt") as handle:
            adapter_keys = list(handle.keys())
    except Exception as error:  # noqa: BLE001 - stable error surface
        raise LoRAValidationError("adapter_file_unreadable", "adapter/adapter_model.safetensors") from error
    peft_state = {_normalize_lora_key(_strip_peft_prefix(key)) for key in peft_model.state_dict().keys()}
    unconsumed: list[str] = []
    violations: list[str] = []
    for key in adapter_keys:
        logical = _normalize_lora_key(_strip_peft_prefix(key))
        if logical not in peft_state:
            unconsumed.append(key)
            continue
        if not _has_lora_segment(logical):
            violations.append(f"non-lora adapter key: {key}")
            continue
        parent = _lora_parent_module(logical)
        if parent not in set(manifest_targets):
            violations.append(f"adapter key outside LLM targets: {key}")
        if "visual" in parent.split(".") or _under_any(parent, merger_names):
            violations.append(f"visual/merger LoRA target: {key}")
    if unconsumed:
        raise LoRAValidationError("adapter_tensors_not_consumed", "; ".join(unconsumed[:10]))
    if violations:
        raise LoRAValidationError("adapter_key_violation", "; ".join(violations[:10]))

    return peft_model, {
        "peft_type": peft_type,
        "rank": int(adapter_config.get("r") or manifest.get("lora", {}).get("rank")),
        "alpha": int(adapter_config.get("lora_alpha") or manifest.get("lora", {}).get("alpha")),
        "dropout": float(adapter_config.get("lora_dropout") or manifest.get("lora", {}).get("dropout")),
        "bias": str(adapter_config.get("bias") or manifest.get("lora", {}).get("bias")),
        "target_modules": manifest_targets,
        "adapter_key_count": len(adapter_keys),
    }


def merge_and_audit(
    peft_model: Any,
    merger_names: Sequence[str],
    expected_merger_tensors: Mapping[str, Any],
) -> tuple[Any, dict]:
    """merge_and_unload the LoRA, then audit the final model:

    - no lora_A/lora_B or PEFT wrapper-only keys remain;
    - no module whose class name contains 'lora' remains;
    - every merger tensor still equals the trained checkpoint values;
    - deepstack merger count is unchanged.
    """
    merged_model = peft_model.merge_and_unload()
    final_state = set(merged_model.state_dict().keys())
    lora_residue = sorted(
        key for key in final_state
        if "lora_" in key or key.startswith("base_model.")
    )
    if lora_residue:
        raise LoRAValidationError("lora_residue_after_merge", "; ".join(lora_residue[:10]))
    lora_modules = sorted(
        name for name, module in merged_model.named_modules()
        if "lora" in module.__class__.__name__.lower()
    )
    if lora_modules:
        raise LoRAValidationError("lora_module_residue", "; ".join(lora_modules[:10]))

    mismatched: list[str] = []
    parameters = dict(merged_model.named_parameters())
    buffers = dict(merged_model.named_buffers())
    for name, expected in expected_merger_tensors.items():
        target = parameters.get(name)
        if target is None:
            target = buffers.get(name)
        if target is None:
            mismatched.append(f"{name}: missing")
            continue
        if not bool((target.detach() == expected).all()):
            mismatched.append(f"{name}: value drift")
    if mismatched:
        raise LoRAValidationError("merger_value_drift_after_merge", "; ".join(mismatched[:10]))

    deepstack_count = len(
        _merger_root(merged_model).deepstack_merger_list
    )
    return merged_model, {
        "lora_residue_count": len(lora_residue),
        "lora_module_residue_count": len(lora_modules),
        "merger_value_verified_count": len(expected_merger_tensors),
        "deepstack_count": deepstack_count,
    }


def _merger_root(model: Any) -> Any:
    """The vision module owning merger + deepstack_merger_list.
    拥有 merger 与 deepstack_merger_list 的视觉模块。"""
    torch = _torch()
    for _name, module in model.named_modules():
        if isinstance(getattr(module, "deepstack_merger_list", None), torch.nn.ModuleList) and isinstance(
            getattr(module, "merger", None), torch.nn.Module
        ):
            return module
    raise LoRAValidationError("vision_root_not_found", "after merge")


# ---------------------------------------------------------------------------
# 9/10. Save model + processor, copy auxiliary files.
# ---------------------------------------------------------------------------


def save_full_checkpoint(merged_model: Any, processor: Any, output_dir: str | Path) -> None:
    """Save the full merged model and the training processor with
    safe_serialization=True. 使用 safe_serialization=True 保存完整合并模型
    与训练 processor。"""
    try:
        merged_model.save_pretrained(output_dir, safe_serialization=True)
    except Exception as error:  # noqa: BLE001 - stable error surface
        raise SaveError("model_save_failed", "save_pretrained") from error
    try:
        processor.save_pretrained(output_dir)
    except Exception as error:  # noqa: BLE001 - stable error surface
        raise SaveError("processor_save_failed", "processor.save_pretrained") from error


def copy_auxiliary_files(model_id: str | Path, output_dir: str | Path) -> list[str]:
    """Copy small base-side config files that save_pretrained does not emit;
    never overwrite an existing output file. 复制 save_pretrained 不会产出的
    辅助配置；绝不覆盖输出目录已有文件。"""
    copied: list[str] = []
    for filename in _AUXILIARY_FILES:
        source = Path(model_id) / filename
        target = Path(output_dir) / filename
        if source.is_file() and not target.exists():
            shutil.copy2(source, target)
            copied.append(filename)
    return copied


def verify_shard_files(output_dir: str | Path) -> dict:
    """Every shard referenced by model.safetensors.index.json must exist;
    without an index, model.safetensors itself must exist.
    index.json 引用的全部 shard 必须存在；无 index 时 model.safetensors
    必须存在。"""
    output_dir = Path(output_dir)
    index_path = output_dir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ReloadValidationError("weight_index_unreadable", "model.safetensors.index.json") from error
        weight_map = index.get("weight_map") or {}
        missing = sorted(
            {path for path in weight_map.values() if not (output_dir / path).is_file()}
        )
        if missing:
            raise ReloadValidationError("shard_missing", "; ".join(missing[:10]))
        return {"indexed": True, "shard_count": len(set(weight_map.values()))}
    if not (output_dir / "model.safetensors").is_file():
        raise ReloadValidationError("model_weights_missing", "model.safetensors")
    return {"indexed": False, "shard_count": 1}


# ---------------------------------------------------------------------------
# 11. Offline reload validation + minimal render check.
# ---------------------------------------------------------------------------


def _synthetic_image(size: int = 64, seed: int = 42) -> Any:
    """Small deterministic RGB image for render/forward checks (never reads
    training images). 程序生成的小型确定性 RGB 图片（不读取训练图片）。"""
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    array = (np.random.RandomState(seed).rand(size, size, 3) * 255).astype(np.uint8)
    return Image.fromarray(array)


def minimal_render_check(processor: Any) -> dict:
    """The processor must render a minimal image+text chat template offline.
    processor 必须能离线渲染最小 image+text chat template。"""
    image = _synthetic_image()
    text = "Describe the image."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]
    try:
        rendered = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        chat_template_ok = rendered is not None
    except (TypeError, ValueError):
        try:
            rendered = processor.apply_chat_template(messages)
            chat_template_ok = rendered is not None
        except (TypeError, ValueError):
            chat_template_ok = False
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    required = ("input_ids", "pixel_values")
    if isinstance(inputs, dict):
        present = set(inputs)
    else:
        present = {name for name in required if hasattr(inputs, name)}
    missing = [name for name in required if name not in present]
    if missing:
        raise ReloadValidationError("processor_render_missing", "; ".join(missing))
    if not chat_template_ok:
        raise ReloadValidationError("processor_chat_template_failed", "minimal image+text render")
    return {"chat_template_ok": True, "input_keys": [name for name in required]}


def reload_validate(
    output_dir: str | Path,
    base_config: Any,
    expected_deepstack: int | None,
    torch_dtype: Any,
    device: str,
    local_files_only: bool,
) -> tuple[dict, Any, Any]:
    """Offline reload validation of the temp export directory:
    AutoConfig / AutoProcessor / AutoModelForImageTextToText with
    local_files_only=True; model_type, config identity, merger counts,
    no PEFT/LoRA residue, minimal render, weight shards. Returns the
    validation result plus the reloaded (model, processor) so the optional
    forward check can reuse them instead of loading a second copy.
    从临时导出目录以 local_files_only=True 重新加载并校验；返回校验结果
    与重新加载的 (model, processor)，供可选 forward 校验复用。"""
    config = _auto_config_from_pretrained(output_dir, local_files_only)
    if config.model_type != "qwen3_vl":
        raise ReloadValidationError("reload_model_type", str(config.model_type))
    base_identity = model_logical_identity(base_config)
    reload_identity = model_logical_identity(config)
    if reload_identity["fingerprint"] != base_identity["fingerprint"]:
        raise ReloadValidationError("reload_config_drift", "exported config differs from base")
    model = _auto_model_from_pretrained(output_dir, torch_dtype, device, local_files_only)
    processor = _auto_processor_from_pretrained(output_dir, local_files_only)

    vision_root = _merger_root(model)
    deepstack_count = len(vision_root.deepstack_merger_list)
    if expected_deepstack is not None and deepstack_count != expected_deepstack:
        raise ReloadValidationError(
            "reload_deepstack_count",
            f"found={deepstack_count} expected={expected_deepstack}",
        )
    lora_modules = sorted(
        name for name, _module in model.named_modules()
        if "lora" in _module.__class__.__name__.lower()
    )
    if lora_modules:
        raise ReloadValidationError("reload_lora_residue", "; ".join(lora_modules[:10]))

    render = minimal_render_check(processor)
    shards = verify_shard_files(output_dir)

    torch = _torch()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result = {
        "model_type": str(config.model_type),
        "config_fingerprint_matches_base": True,
        "deepstack_count": deepstack_count,
        "lora_module_residue": [],
        "render": render,
        "shards": shards,
    }
    return result, model, processor


# ---------------------------------------------------------------------------
# 12. Optional real forward verification (--verify-forward).
# ---------------------------------------------------------------------------


def run_forward_verification(model: Any, processor: Any, device: str, seed: int = 1234) -> dict:
    """One no-grad forward on a program-generated tiny image with a fixed
    short prompt. Only interface/logits finiteness is asserted, never model
    quality. 对程序生成的小图与固定短 prompt 做一次无梯度 forward；
    只验证接口与 logits 有限性，不宣称模型质量。"""
    torch = _torch()
    image = _synthetic_image(size=96, seed=seed)
    inputs = processor(text=["Describe the image."], images=[image], return_tensors="pt")
    prepared = {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in inputs.items()
    }
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(**prepared)
    except Exception as error:  # noqa: BLE001 - stable error surface
        raise ReloadValidationError("forward_failed", "forward pass raised") from error
    logits = getattr(outputs, "logits", None)
    if logits is None and isinstance(outputs, dict):
        logits = outputs.get("logits")
    if logits is None:
        raise ReloadValidationError("forward_no_logits", "forward returned no logits")
    finite = bool(torch.isfinite(logits).all().item())
    if not finite:
        raise ReloadValidationError("forward_non_finite_logits", "logits contain non-finite values")
    return {
        "seed": int(seed),
        "logits_shape": list(logits.shape),
        "logits_finite": True,
    }


# ---------------------------------------------------------------------------
# 13. Atomic publish (temp dir + same-filesystem rename).
# ---------------------------------------------------------------------------


def _make_temp_dir(output_path: Path) -> Path:
    parent = output_path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:  # noqa: BLE001
        raise PublishError("output_parent_unavailable", "cannot create parent directory") from error
    for index in range(100):
        candidate = parent / f"{output_path.name}{TEMP_DIR_SUFFIX}-{os.getpid()}-{index}"
        if not candidate.exists():
            candidate.mkdir()
            return candidate
    raise PublishError("temp_dir_exhausted", "no free temp directory name")


def _cleanup_temp_dir(temp_dir: Path | None) -> None:
    """Remove ONLY the temp directory this process created; never touch any
    user-given path. 只清理本进程创建的临时目录，绝不触碰用户路径。"""
    if temp_dir is None:
        return
    name = temp_dir.name
    if TEMP_DIR_SUFFIX not in name:
        return
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:  # noqa: BLE001 - best effort cleanup
        pass


def _publish(temp_dir: Path, output_path: Path) -> None:
    if output_path.exists():
        raise PublishError("output_exists_at_publish", "final output appeared during export")
    try:
        os.replace(temp_dir, output_path)
    except OSError as error:  # noqa: BLE001
        raise PublishError("publish_rename_failed", "atomic rename failed") from error


def compute_file_checksums(output_dir: str | Path) -> list[dict]:
    """Relative path / size / sha256 of every file (excluding the export
    manifest itself, which is written afterwards).
    全部文件的相对路径/size/sha256（不含随后写入的导出 manifest）。"""
    output_dir = Path(output_dir)
    entries: list[dict] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(output_dir).as_posix()
        if rel == EXPORT_MANIFEST_FILENAME:
            continue
        entries.append(
            {
                "path": rel,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return entries


# ---------------------------------------------------------------------------
# Orchestration (fixed order; every gate gates the publish).
# ---------------------------------------------------------------------------


def export_phase2(
    *,
    model_id: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    torch_dtype_name: str,
    device: str,
    local_files_only: bool,
    verify_forward: bool,
    repo_root: str | Path,
) -> dict:
    """Run the full export pipeline. Raises ExportError subclasses on any
    failure; the final output path is never created on failure. All heavy
    model references live in _export_body's frame, so they are released as
    soon as it returns or raises. 执行完整导出流水线；任何失败抛出
    ExportError 子类，且最终输出不创建。

    Fixed order (doc 04 section 5): validate manifest -> load base ->
    enumerate expected merger keys -> strict merger load -> attach LLM LoRA
    -> verify adapter -> merge_and_unload -> audit -> save -> copy aux ->
    offline reload -> optional forward -> export manifest -> atomic publish.
    """
    torch = _torch()
    output_path = Path(output_path)
    if output_path.exists():
        raise ExportError("output_exists", "choose a new --output-path")
    try:
        return _export_body(
            model_id=model_id,
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            torch_dtype_name=torch_dtype_name,
            device=device,
            local_files_only=local_files_only,
            verify_forward=verify_forward,
            repo_root=repo_root,
        )
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _export_body(
    *,
    model_id: str | Path,
    checkpoint_path: str | Path,
    output_path: Path,
    torch_dtype_name: str,
    device: str,
    local_files_only: bool,
    verify_forward: bool,
    repo_root: str | Path,
) -> dict:
    torch = _torch()

    # 1. read + validate the training manifest (no big weights loaded yet).
    manifest = validate_checkpoint_layout(checkpoint_path)
    validate_manifest_safety(manifest)
    checksums = verify_checkpoint_checksums(checkpoint_path, manifest)
    source_manifest_sha256 = sha256_file(
        Path(checkpoint_path) / TRAINING_MANIFEST_FILENAME
    )

    # 2. load base config + model + processor (offline by default).
    dtype = resolve_dtype(torch, torch_dtype_name)
    config, base_model, base_processor = load_base_components(
        model_id, dtype, device, local_files_only
    )
    base_identity = verify_base_identity(config, manifest)
    recorded_processor = manifest.get("processor") or {}
    if recorded_processor.get("fingerprint") and recorded_processor["fingerprint"] != processor_identity(base_processor)["fingerprint"]:
        raise CheckpointValidationError(
            "processor_identity_mismatch", "manifest processor differs from base"
        )

    # 3. enumerate expected merger keys from the live base model and compare
    #    with the manifest declaration.
    vision_root_name, _text_root_name = locate_roots(base_model)
    merger_names = enumerate_merger_names(base_model, vision_root_name)
    verify_manifest_merger_layout(manifest, merger_names, config)
    model_table = expected_merger_table(base_model, merger_names)
    compare_manifest_parameter_table(manifest, model_table)

    # 4. strict merger load (three-way model/manifest/file comparison).
    loaded_merger, merger_conversions = load_merger_strict(
        base_model, Path(checkpoint_path) / MERGER_STATE_FILENAME, manifest, model_table
    )

    # 5/6/7. attach the LLM LoRA adapter, verify, merge_and_unload.
    peft_model, adapter_info = attach_and_verify_adapter(
        base_model,
        Path(checkpoint_path) / "adapter",
        manifest,
        merger_names,
        local_files_only,
    )
    merged_model, merge_audit = merge_and_audit(peft_model, merger_names, loaded_merger)

    # The checkpoint processor must carry the same identity as the base one.
    checkpoint_processor = load_processor_from_dir(
        Path(checkpoint_path) / "processor", local_files_only
    )
    if processor_identity(checkpoint_processor)["fingerprint"] != processor_identity(base_processor)["fingerprint"]:
        raise CheckpointValidationError(
            "processor_identity_mismatch", "checkpoint processor differs from base"
        )

    # 9. save into the temp directory; 10. copy auxiliary files.
    temp_dir = None
    try:
        temp_dir = _make_temp_dir(output_path)
        save_full_checkpoint(merged_model, checkpoint_processor, temp_dir)
        copied_aux = copy_auxiliary_files(model_id, temp_dir)

        # 11. offline reload validation of the exported directory.
        expected_deepstack = expected_deepstack_count(config)
        reload_result, reloaded_model, reloaded_processor = reload_validate(
            temp_dir, config, expected_deepstack, dtype, device, local_files_only
        )

        # 12. optional real forward verification (reuses the reloaded model).
        forward_result = None
        if verify_forward:
            forward_result = run_forward_verification(
                reloaded_model, reloaded_processor, device
            )
        del reloaded_model, reloaded_processor

        # 13. export manifest + atomic publish.
        files = compute_file_checksums(temp_dir)
        export_manifest = {
            "schema_version": SCHEMA_VERSION,
            "export_type": "phase2_complete_deployment",
            "base_model": {
                "fields": base_identity["fields"],
                "fingerprint": base_identity["fingerprint"],
                "revision": base_identity.get("revision"),
            },
            "model_id_basename": _path_basename(model_id),
            "source_training_checkpoint": {
                "type": str(manifest.get("checkpoint_type")),
                "step": manifest.get("step"),
                "epoch": manifest.get("epoch"),
                "training_manifest_sha256": source_manifest_sha256,
            },
            "adapter": {
                "file": "adapter/adapter_model.safetensors",
                "file_sha256": checksums["adapter"]["file_sha256"],
                "rank": int(adapter_info["rank"]),
                "alpha": int(adapter_info["alpha"]),
                "dropout": float(adapter_info["dropout"]),
                "bias": str(adapter_info["bias"]),
                "target_modules": list(adapter_info["target_modules"]),
                "adapter_key_count": int(adapter_info["adapter_key_count"]),
            },
            "merger": {
                "modules": list(merger_names),
                "parameters": manifest.get("merger", {}).get("parameters"),
                "file": MERGER_STATE_FILENAME,
                "file_sha256": checksums["merger"]["file_sha256"],
                "parameter_count": len(loaded_merger),
                "verified_after_merge_count": int(merge_audit["merger_value_verified_count"]),
            },
            "lora_merged": True,
            "merge_audit": merge_audit,
            "output_dtype": torch_dtype_name,
            "merger_dtype_conversions": list(merger_conversions),
            "device": str(device),
            "verify_forward": bool(verify_forward),
            "auxiliary_files_copied": sorted(copied_aux),
            "environment": {
                "git_head": git_head(repo_root),
                "transformers_version": _package_version("transformers"),
                "torch_version": _package_version("torch"),
                "peft_version": _package_version("peft"),
                "python_version": sys.version.split()[0],
            },
            "files": files,
            "reload_validation": reload_result,
            "forward_validation": forward_result,
        }
        _atomic_write_json(temp_dir / EXPORT_MANIFEST_FILENAME, export_manifest)

        _publish(temp_dir, output_path)
        temp_dir = None  # published; nothing to clean
        return export_manifest
    finally:
        _cleanup_temp_dir(temp_dir)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _path_basename(value: str | Path) -> str:
    """Non-identity label for the local loading hint (never a logical id).
    本地加载提示的非身份标签（不作为逻辑身份）。"""
    path = Path(value)
    if path.is_dir():
        return path.name
    return str(value)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the export CLI. / 构建导出 CLI。"""
    parser = argparse.ArgumentParser(
        description=(
            "Export a Phase 2 composite training checkpoint (base + LLM LoRA "
            "adapter + trained merger states) into a complete deployable "
            "Qwen3-VL checkpoint."
        )
    )
    parser.add_argument(
        "--model-id",
        type=Path,
        default=Path("models/qwen3_vl_8b/weights"),
        help="Original Qwen3-VL-8B base checkpoint directory.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Phase 2 composite training checkpoint directory (adapter/ + merger_model.safetensors + processor/ + manifest).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Complete export directory (must not already exist).",
    )
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=("float32", "float16", "bfloat16", "auto"),
        help="Output dtype; default bfloat16. Float conversions of merger "
             "state are explicit and recorded in the export manifest.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for loading/merging/reload validation (cpu or cuda:0).",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refuse network access while loading checkpoints; default on for offline safety.",
    )
    parser.add_argument(
        "--verify-forward",
        action="store_true",
        help="Run one tiny no-grad forward on a program-generated image before publishing.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry; public stderr only carries the stable stage and the
    exception type, never raw exception text, paths or secrets.
    CLI 入口；公共 stderr 只输出稳定阶段与异常类型。"""
    args = build_parser().parse_args(argv)
    try:
        export_phase2(
            model_id=args.model_id,
            checkpoint_path=args.checkpoint_path,
            output_path=args.output_path,
            torch_dtype_name=args.torch_dtype,
            device=args.device,
            local_files_only=args.local_files_only,
            verify_forward=args.verify_forward,
            repo_root=Path(__file__).resolve().parents[1],
        )
    except KeyboardInterrupt:
        print("export interrupted; no final directory was published", file=sys.stderr)
        return 130
    except ExportError as error:
        print(f"export failed: {type(error).__name__}", file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001 - stable public surface
        print("export failed: unexpected error", file=sys.stderr)
        return 1
    print(
        f"export complete: {args.output_path} "
        f"(local_files_only={args.local_files_only}, dtype={args.torch_dtype})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
