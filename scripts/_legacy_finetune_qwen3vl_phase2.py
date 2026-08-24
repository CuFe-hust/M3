#!/usr/bin/env python3
"""Phase 2 Qwen3-VL-8B SFT training: full merger tuning + LLM LoRA.

Strategy / 训练策略:
- Vision Encoder (``model.visual``): frozen, except:
- ``model.visual.merger`` + ``model.visual.deepstack_merger_list.*``: FULLY
  trainable (no LoRA) with ``--merger-lr``;
- LLM (``model.language_model``): base weights frozen, LoRA attached to every
  decoder layer's ``self_attn.{q,k,v,o}_proj`` and ``mlp.{gate,up,down}_proj``
  with ``--lora-lr``.

The two parameter families use different learning rates: the optimizer is
explicitly built as four groups (merger/lora x decay/no_decay), and the
cosine scheduler scales every group by the same lambda so the LR ratio is
preserved. 两类参数使用不同学习率：显式构造四个 optimizer group，cosine
scheduler 对所有组使用同一缩放比例，保持两套 LR 的比值。

The script consumes ``scripts/qwen3vl_phase2_data.py``
(Phase2EpisodeDataset / Phase2DataCollator / AugmentationConfig /
DatasetRootConfig) and never re-implements data semantics, augmentation or
prompt rendering. 本脚本只消费数据管线文件提供的 Dataset/Collator/配置对象，
不复制数据语义、增强或 prompt 逻辑。

The ONLY persistent product is a resumable composite checkpoint:

    checkpoint-N/
    ├── adapter/adapter_config.json        # LLM LoRA adapter
    ├── adapter/adapter_model.safetensors
    ├── merger_model.safetensors           # full merger + deepstack states
    ├── processor/
    ├── phase2_training_manifest.json      # completion marker, written last
    ├── trainer_state.json
    ├── optimizer.pt / scheduler.pt / scaler.pt / rng_state.pth

The deployment model is produced later by the independent exporter
(docs/train/04); this script never merges LoRA into the base.

Importing this module loads no weights: torch/transformers/peft are imported
lazily inside the execution path. 本模块 import 不加载权重：
torch/transformers/peft 都在执行路径内惰性导入。

Model structure facts (verified against transformers 5.14.1 / peft 0.20.0):
    Qwen3VLForConditionalGeneration
    ├── model.visual                 # Qwen3VLVisionModel (vision encoder)
    │   ├── merger                   #   Qwen3VLVisionPatchMerger
    │   └── deepstack_merger_list.*  #   3 more mergers (deepstack fusion)
    ├── model.language_model         # Qwen3VLTextModel (LLM)
    │   ├── embed_tokens
    │   └── layers.*.self_attn.{q,k,v,o}_proj
    │   └── layers.*.mlp.{gate,up,down}_proj
    └── lm_head
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# Stable constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1
CHECKPOINT_PREFIX = "checkpoint-"
MERGER_STATE_FILENAME = "merger_model.safetensors"
TRAINING_MANIFEST_FILENAME = "phase2_training_manifest.json"
PARAMETER_AUDIT_FILENAME = "parameter_audit.json"
SAVE_ERROR_FILENAME = "save_error.json"
DEFAULT_AUG_SEED = "phase2-default-seed"
DEFAULT_IMAGE_MIN_PIXELS = 256 * 32 * 32
DEFAULT_IMAGE_MAX_PIXELS = 1280 * 32 * 32
DATA_PROFILES = ("phase2", "change_agent")

# LLM projection modules that receive LoRA (7 per decoder layer).
# 每层语言模型接收 LoRA 的七个 projection。
_ATTN_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
EXPECTED_PROJECTIONS_PER_LAYER = len(_ATTN_PROJECTIONS) + len(_MLP_PROJECTIONS)

# Explicit LoRA no-decay policy (test-fixed, never name-sniffing by luck):
# only the LoRA A/B matrices decay; any other LoRA parameter (bias variants,
# embedding adapters) is no-decay.
# 明确的 LoRA no-decay 策略：只有 lora_A/lora_B 矩阵参与 weight decay，
# 其余 LoRA 参数一律不 decay。
_LORA_MATRIX_RE = re.compile(r"\.(lora_A|lora_B)\.default\.weight$")

# Merger/LLM no-decay markers: bias suffix + normalization modules.
# merger/LLM 的 no-decay 标记：bias 后缀与归一化模块。
_NORM_SEGMENT_MARKERS = ("norm", "norm1", "norm2", "layernorm", "rmsnorm")

# PEFT name prefixes produced by get_peft_model (generic PeftModel wrapper).
_PEFT_NAME_PREFIXES = ("base_model.model.", "base_model.")

logger = logging.getLogger("finetune_qwen3vl_phase2")


# ---------------------------------------------------------------------------
# Stable error types (messages never carry credentials or raw exception
# dumps; artifact paths are kept relative).
# 稳定错误类型（错误信息不携带凭据或原始异常全文）。
# ---------------------------------------------------------------------------


class Phase2TrainError(Exception):
    """Base class for training-script errors. / 训练脚本错误基类。"""


class StructureError(Phase2TrainError):
    """Model tree does not match the expected Qwen3-VL layout.
    模型树结构与预期 Qwen3-VL 布局不符。"""


class ParameterAuditError(Phase2TrainError):
    """Parameter freeze/LoRA/merger audit failed.
    参数冻结/LoRA/merger 审计失败。"""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"parameter audit error {code}: {detail}")
        self.code = code
        self.detail = detail


class ResumeConflictError(Phase2TrainError):
    """Persisted checkpoint is incompatible with the current explicit
    request; training refuses to guess. 持久化 checkpoint 与当前显式请求
    不兼容；训练稳定拒绝而不是猜测继续。"""

    def __init__(self, code: str, fields: Sequence[str] = ()) -> None:
        super().__init__(f"resume conflict {code}: {', '.join(fields)}")
        self.code = code
        self.fields = list(fields)


class CheckpointError(Phase2TrainError):
    """Composite checkpoint read/verify failure.
    复合 checkpoint 读取/校验失败。"""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"checkpoint error {code}: {detail}")
        self.code = code
        self.detail = detail


class Phase2SaveError(Phase2TrainError):
    """Checkpoint saving failed; no completion marker is faked.
    保存 checkpoint 失败；不伪造完成标记。"""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"save error {code}: {detail}")
        self.code = code
        self.detail = detail


class ConfigurationError(Phase2TrainError):
    """Invalid CLI configuration. / CLI 配置不合法。"""


# ---------------------------------------------------------------------------
# CLI argument groups (stdlib dataclasses; no heavy imports).
# ---------------------------------------------------------------------------


@dataclass
class ModelArguments:
    """Model loading arguments. / 模型加载参数。"""

    model_id: str = field(
        default="Qwen/Qwen3-VL-8B-Instruct",
        metadata={"help": "Hugging Face id or local checkpoint directory."},
    )
    merger_lora_adapter: str | None = field(
        default=None,
        metadata={"help": "Optional phase-1 merger LoRA adapter directory; the "
                          "adapter is merged into the base weights before "
                          "phase-2 training starts."},
    )
    local_files_only: bool = field(
        default=True,
        metadata={"help": "Refuse network access while loading the checkpoint."},
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"choices": ("float32", "float16", "bfloat16", "auto")},
    )
    attn_implementation: str = field(
        default="sdpa",
        metadata={"choices": ("sdpa", "eager", "flash_attention_2")},
    )


@dataclass
class DataArguments:
    """Dataset arguments. / 数据参数。"""

    train_file: str = field(metadata={"help": "Phase 2 canonical train episodes JSONL."})
    data_profile: str = field(
        default="phase2",
        metadata={"choices": DATA_PROFILES, "help": "Data contract profile (default: phase2)."},
    )
    eval_file: str | None = field(
        default=None, metadata={"help": "Optional Phase 2 validation episodes JSONL."}
    )
    image_root: list[str] = field(
        default_factory=list,
        metadata={
            "help": "image_source=path pairs, space separated in one flag "
            "(e.g. --image-root vrsbench=/data/VRSBench-full geochat=/data/GeoChat/images)."
        },
    )
    max_seq_length: int = field(
        default=4096, metadata={"help": "Maximum tokenized sequence length."}
    )
    image_min_pixels: int = field(
        default=DEFAULT_IMAGE_MIN_PIXELS,
        metadata={"help": "Minimum image pixels (only applied when the processor supports it)."},
    )
    image_max_pixels: int = field(
        default=DEFAULT_IMAGE_MAX_PIXELS,
        metadata={"help": "Maximum image pixels (only applied when the processor supports it)."},
    )
    aug_seed: str = field(
        default=DEFAULT_AUG_SEED,
        metadata={"help": "Deterministic augmentation group seed (string)."},
    )
    repeat_group_key: str = field(
        default="task_kind",
        metadata={"choices": ("task_kind", "source_task", "task"), "help": "Episode field used as the repeat group key."},
    )
    repeat_weights: list[str] = field(
        default_factory=list,
        metadata={"help": "Deterministic group repeat weights, e.g. group=2 (repeat each episode of that group twice)."},
    )
    max_train_samples: int | None = field(
        default=None, metadata={"help": "Cap the train dataset for smoke runs."}
    )
    max_eval_samples: int | None = field(
        default=None, metadata={"help": "Cap the eval dataset for smoke runs."}
    )
    preflight_limit: int = field(
        default=0,
        metadata={"help": "Preflight check N episodes before training (0=skip, negative=all)."},
    )
    change_prompt_file: str | None = field(
        default=None,
        metadata={"help": "Production ChangeAgent prompt text used by the change_agent profile."},
    )


@dataclass
class LoRAArguments:
    """LLM LoRA hyperparameters. / LLM LoRA 超参数。"""

    lora_rank: int = field(default=64, metadata={"help": "LoRA rank r."})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_lr: float = field(default=1e-4, metadata={"help": "Learning rate for LLM LoRA parameters."})
    merger_lr: float = field(default=1e-5, metadata={"help": "Learning rate for full merger base parameters."})
    lora_bias: str = field(
        default="none",
        metadata={"choices": ("none", "all", "lora_only"), "help": "LoRA bias mode."},
    )


@dataclass
class AugmentationArguments:
    """Online augmentation surface (delegated to AugmentationConfig of the
    data pipeline file; only a curated subset is exposed on the CLI).
    在线增强参数（全部委托给数据管线文件的 AugmentationConfig）。"""

    aug_enabled: bool = field(default=True, metadata={"help": "Enable online augmentation on train."})
    aug_rotate90_prob: float = field(default=0.5, metadata={"help": "90-degree rotation probability."})
    aug_affine_rotation_prob: float = field(default=0.3, metadata={"help": "Small affine rotation probability."})
    aug_scale_prob: float = field(default=0.3, metadata={"help": "Scale probability."})
    aug_translate_prob: float = field(default=0.2, metadata={"help": "Translation probability."})
    aug_perspective_prob: float = field(default=0.2, metadata={"help": "Mild perspective probability."})
    aug_degradation_probability: float = field(default=0.45, metadata={"help": "Imaging degradation probability."})
    aug_min_degradations: int = field(default=1, metadata={"help": "Minimum degradations per degraded sample."})
    aug_max_degradations: int = field(default=3, metadata={"help": "Maximum degradations per degraded sample."})


@dataclass
class OptimizationArguments:
    """Optimizer / scheduler / trainer settings (per-GPU batch size, GPU
    count, accumulation, DeepSpeed/FSDP are never hardcoded).
    优化器/调度器/Trainer 设置（GPU 数、per-device batch、accumulation、
    DeepSpeed/FSDP 均不硬编码）。"""

    epochs: float = field(default=2.0, metadata={"help": "Number of training epochs."})
    per_device_train_batch_size: int = field(default=1, metadata={"help": "Per-device train batch size."})
    per_device_eval_batch_size: int = field(default=1, metadata={"help": "Per-device eval batch size."})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": "Gradient accumulation steps."})
    weight_decay: float = field(default=0.01, metadata={"help": "Weight decay for decay-enabled groups."})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Cosine warmup ratio of total steps."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Gradient clipping max norm."})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "Enable gradient checkpointing."})
    seed: int = field(default=42, metadata={"help": "RNG seed (training + distributed sampler)."})
    dataloader_num_workers: int = field(default=0, metadata={"help": "Dataloader worker processes."})
    logging_steps: int = field(default=50, metadata={"help": "Log every N steps."})
    eval_steps: int = field(default=500, metadata={"help": "Evaluate every N steps (eval file required)."})
    deepspeed: str | None = field(default=None, metadata={"help": "DeepSpeed config JSON path (passthrough)."})
    fsdp: str | None = field(default=None, metadata={"help": "FSDP flags (passthrough)."})
    fsdp_config: str | None = field(default=None, metadata={"help": "FSDP config JSON path (passthrough)."})
    smoke_gradients: bool = field(
        default=False,
        metadata={"help": "Run one forward/backward smoke check before training "
                         "(requires --max-train-samples)."},
    )


@dataclass
class CheckpointArguments:
    """Checkpoint / resume arguments. / checkpoint 与 resume 参数。"""

    output_dir: str = field(metadata={"help": "Run output directory."})
    resume_from_checkpoint: str | None = field(
        default=None,
        metadata={"help": "Explicit checkpoint dir; omit to auto-resume the latest complete "
                         "checkpoint; pass empty string to force a fresh run."},
    )
    save_steps: int = field(default=1000, metadata={"help": "Save a checkpoint every N steps."})
    save_total_limit: int = field(default=3, metadata={"help": "Rotate to keep at most N complete checkpoints."})


# ---------------------------------------------------------------------------
# Lazy heavy-dependency loaders (module import must stay weight-free).
# ---------------------------------------------------------------------------


def _load_data_module(profile: str = "phase2") -> Any:
    """Load the sibling data-pipeline module by file path (it imports
    torch/opencv/PIL, so it must stay out of the module import path).
    按文件路径加载数据管线模块（它 import torch/opencv/PIL，必须保持惰性）。"""
    if profile not in DATA_PROFILES:
        raise ConfigurationError(f"unknown data profile: {profile!r}")
    filename = "qwen3vl_phase2_data.py" if profile == "phase2" else "change_qwen_sft_data.py"
    path = Path(__file__).resolve().parent / filename
    if not path.is_file():
        raise ConfigurationError(f"data pipeline module missing: {path.name}")
    module_name = "qwen3vl_phase2_data" if profile == "phase2" else "change_qwen_sft_data"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ConfigurationError(f"cannot load data pipeline module: {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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
# Files, checksums, runtime checks
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


def check_runtime() -> None:
    """Verify the installed Transformers provides the Qwen3-VL deepstack
    fusion path used by this build. 校验 Transformers 具备 Qwen3-VL
    deepstack fusion 路径。"""
    from packaging.version import Version  # noqa: PLC0415

    transformers_mod = _transformers()
    if Version(transformers_mod.__version__) < Version("5.6.0"):
        raise ConfigurationError(
            "Phase 2 training requires transformers>=5.6.0; "
            f"found {transformers_mod.__version__} (reference env pins 5.14.1)."
        )
    try:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel  # noqa: PLC0415
    except ImportError as error:
        raise ConfigurationError("This Transformers build does not provide Qwen3-VL.") from error
    if not hasattr(Qwen3VLTextModel, "_deepstack_process"):
        raise ConfigurationError(
            "This Transformers build lacks the Qwen3-VL deepstack fusion path; "
            "upgrade to transformers==5.14.1."
        )


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
        raise ConfigurationError(f"Unsupported torch_dtype: {name}")
    return mapping[name]


def load_base_model(model_args: ModelArguments) -> tuple[Any, Any, Any]:
    """Load config, model and processor from the same checkpoint; validates
    model_type == 'qwen3_vl'. Never touches the network by default.
    从同一 checkpoint 加载 config/model/processor；校验 model_type。"""
    transformers_mod = _transformers()
    torch = _torch()
    config = transformers_mod.AutoConfig.from_pretrained(
        model_args.model_id, local_files_only=model_args.local_files_only
    )
    if config.model_type != "qwen3_vl":
        raise ConfigurationError(
            f"This script supports qwen3_vl only, got model_type={config.model_type!r}."
        )
    dtype = resolve_dtype(torch, model_args.torch_dtype)
    model = transformers_mod.AutoModelForImageTextToText.from_pretrained(
        model_args.model_id,
        config=config,
        torch_dtype=dtype,
        attn_implementation=model_args.attn_implementation,
        local_files_only=model_args.local_files_only,
    )
    processor = transformers_mod.AutoProcessor.from_pretrained(
        model_args.model_id, local_files_only=model_args.local_files_only
    )
    if model_args.merger_lora_adapter:
        model = apply_merger_lora_adapter(model, model_args.merger_lora_adapter)
    return model, processor, config


def apply_merger_lora_adapter(model: Any, adapter_path: str | Path) -> Any:
    """Load a phase-1 merger LoRA adapter and merge it into the base weights
    before phase-2 training. The adapter must target only merger modules; any
    LLM or vision projection target is a hard error, because phase-2 attaches
    LLM LoRA and trains merger base parameters afterwards (doc 03 section 5).
    加载 phase1 merger LoRA adapter 并合并进 base 权重。adapter 必须只作用于
    merger 模块；任何 LLM/视觉投影 target 都是硬错误（phase2 随后要挂 LLM
    LoRA 并全参训练 merger base 参数）。"""
    peft_mod = _peft()
    config_path = Path(adapter_path) / "adapter_config.json"
    if not config_path.is_file():
        raise ConfigurationError(f"merger LoRA adapter config missing: {adapter_path}")
    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    if adapter_config.get("peft_type") != "LORA":
        raise ConfigurationError(
            f"merger LoRA adapter must be a PEFT LORA adapter, got "
            f"{adapter_config.get('peft_type')!r}"
        )

    roots = locate_roots(model)
    merger_names = enumerate_merger_names(roots)
    merger_leaf_names = {
        name.rsplit(".", 1)[-1]
        for name, _module in model.named_modules()
        if _under_any(name, merger_names)
    }
    declared_targets = set(adapter_config.get("target_modules") or [])
    if not declared_targets:
        raise ConfigurationError("merger LoRA adapter declares no target_modules")
    for target in sorted(declared_targets):
        full = target in merger_names or _under_any(target, merger_names)
        short = target.rsplit(".", 1)[-1] in merger_leaf_names
        if not (full or short):
            raise ConfigurationError(
                f"merger LoRA target outside merger subtrees: {target}"
            )

    peft_model = peft_mod.PeftModel.from_pretrained(model, str(adapter_path))
    merged = peft_model.merge_and_unload()
    logger.info("merged phase-1 merger LoRA adapter into base weights: %s", adapter_path)
    return merged


# ---------------------------------------------------------------------------
# Model structure location (structure-based, never fuzzy "visual" matching).
# 模型结构定位（基于结构，不使用模糊字符串匹配）。
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelRoots:
    """Located structural roots of the Qwen3-VL tree.
    定位到的 Qwen3-VL 模型结构根。"""

    vision_root: Any
    vision_name: str
    text_root: Any
    text_name: str

    @property
    def lm_head_name(self) -> str:
        # lm_head lives on the top-level model, not inside the text root.
        # lm_head 位于顶层模型，不在文本根内部。
        return "lm_head"


def locate_roots(model: Any) -> ModelRoots:
    """Locate the vision transformer and the language model structurally.

    Vision root: the module owning both ``merger`` and
    ``deepstack_merger_list``. Text root: the module owning ``layers``
    (ModuleList of decoder layers) and ``embed_tokens``. Ambiguous or
    missing roots raise StructureError.
    """
    torch = _torch()
    vision_root, vision_name, text_root, text_name = None, None, None, None
    for name, module in model.named_modules():
        if vision_root is None and isinstance(
            getattr(module, "deepstack_merger_list", None), torch.nn.ModuleList
        ) and isinstance(getattr(module, "merger", None), torch.nn.Module):
            vision_root, vision_name = module, name
        if text_root is None and isinstance(
            getattr(module, "layers", None), torch.nn.ModuleList
        ) and isinstance(getattr(module, "embed_tokens", None), torch.nn.Module):
            text_root, text_name = module, name
    if vision_root is None or vision_name is None:
        raise StructureError("vision root not found (expected module with merger + deepstack_merger_list)")
    if text_root is None or text_name is None:
        raise StructureError("text root not found (expected module with layers + embed_tokens)")
    if not text_name:
        raise StructureError("text root must be a named subtree")
    return ModelRoots(vision_root, vision_name, text_root, text_name)


def enumerate_llm_lora_targets(roots: ModelRoots) -> list[str]:
    """Full module paths of every LLM Attention/MLP projection to attach
    LoRA to: 7 per decoder layer, all asserted inside the text subtree.
    枚举需要挂 LoRA 的 LLM Attention/MLP projection 全路径。"""
    torch = _torch()
    targets: list[str] = []
    layers = roots.text_root.layers
    for layer_index, layer in enumerate(layers):
        self_attn = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        if self_attn is None or mlp is None:
            raise StructureError(f"decoder layer {layer_index} missing self_attn/mlp")
        for projection in _ATTN_PROJECTIONS:
            module = getattr(self_attn, projection, None)
            if not isinstance(module, torch.nn.Linear):
                raise StructureError(f"layer {layer_index} self_attn.{projection} is not nn.Linear")
            targets.append(f"{roots.text_name}.layers.{layer_index}.self_attn.{projection}")
        for projection in _MLP_PROJECTIONS:
            module = getattr(mlp, projection, None)
            if not isinstance(module, torch.nn.Linear):
                raise StructureError(f"layer {layer_index} mlp.{projection} is not nn.Linear")
            targets.append(f"{roots.text_name}.layers.{layer_index}.mlp.{projection}")
    expected = EXPECTED_PROJECTIONS_PER_LAYER * len(layers)
    if len(targets) != expected:
        raise StructureError(
            f"expected {expected} LLM LoRA targets, enumerated {len(targets)}"
        )
    for target in targets:
        segments = target.split(".")
        if "visual" in segments or "merger" in segments:
            raise StructureError(f"LoRA target escaped the LLM subtree: {target}")
        if not target.startswith(roots.text_name + "."):
            raise StructureError(f"LoRA target outside text root: {target}")
    return targets


def enumerate_merger_names(roots: ModelRoots) -> list[str]:
    """Full module paths of the main merger and every deepstack merger.
    主 merger 与全部 deepstack merger 的全路径。"""
    names = [f"{roots.vision_name}.merger"]
    for index, _merger in enumerate(roots.vision_root.deepstack_merger_list):
        names.append(f"{roots.vision_name}.deepstack_merger_list.{index}")
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


# ---------------------------------------------------------------------------
# Freeze + LoRA injection + merger unfreeze (fixed order, doc 03 section 5).
# ---------------------------------------------------------------------------


def freeze_all(model: Any) -> None:
    """Set requires_grad=False on every parameter.
    全部参数 requires_grad=False。"""
    for parameter in model.parameters():
        parameter.requires_grad_(False)


def inject_lora(
    model: Any,
    target_modules: Sequence[str],
    rank: int,
    alpha: int,
    dropout: float,
    bias: str,
) -> Any:
    """Attach LoRA to the explicitly enumerated LLM projections.

    target_modules is the FULL list of module paths (never a bare
    ["q_proj", ...] assumption); the resulting hits are audited afterwards.
    """
    peft_mod = _peft()
    lora_config = peft_mod.LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=list(target_modules),
        lora_dropout=dropout,
        bias=bias,
    )
    return peft_mod.get_peft_model(model, lora_config)


def unwrap_base(peft_model: Any) -> Any:
    """Unwrap the PEFT wrapper back to the original model.

    get_peft_model() wraps the model in a generic PeftModel whose
    get_base_model() returns the original nn.Module (verified on peft
    0.20.0). A structural fallback walks base_model.model for older builds.
    """
    getter = getattr(peft_model, "get_base_model", None)
    if getter is not None:
        try:
            base = getter()
        except Exception:  # noqa: BLE001 - fall back below
            base = None
        if base is not None and base is not peft_model:
            return base
    base_model = getattr(peft_model, "base_model", None)
    if base_model is not None:
        inner = getattr(base_model, "model", None)
        if inner is not None and inner is not peft_model and inner is not base_model:
            return inner
    return peft_model


def _under_any(name: str, roots: Sequence[str]) -> bool:
    """True when name == root or name.startswith(root + '.') for some root.
    名称是否属于任一 root 子树。"""
    return any(name == root or name.startswith(root + ".") for root in roots)


def unfreeze_merger_base(base_model: Any, merger_names: Sequence[str]) -> None:
    """Fully unfreeze the main merger and every deepstack merger base
    parameter (Linear weights/biases, LayerNorm weights/biases, any other
    floating-point params in the merger subtrees). No LoRA on mergers.
    全量解冻主 merger 与 deepstack merger 的 base 参数；merger 不挂 LoRA。"""
    for name, parameter in base_model.named_parameters():
        if _under_any(name, merger_names):
            parameter.requires_grad = True


# ---------------------------------------------------------------------------
# Parameter classification, audit, optimizer groups
# ---------------------------------------------------------------------------


def _has_lora_segment(logical_name: str) -> bool:
    return any(segment.startswith("lora_") for segment in logical_name.split("."))


def _is_lora_matrix(logical_name: str) -> bool:
    """Explicit LoRA decay policy: only lora_A/lora_B default weights decay.
    明确的 LoRA decay 策略：只有 lora_A/lora_B 权重 decay。"""
    return bool(_LORA_MATRIX_RE.search(logical_name))


def _is_bias_or_norm(logical_name: str) -> bool:
    """No-decay markers for base params: bias suffix and normalization
    modules (norm/norm1/norm2/LayerNorm/RMSNorm/ln_*).
    base 参数的 no-decay 标记：bias 后缀与归一化模块。"""
    segments = logical_name.split(".")
    if segments and segments[-1] == "bias":
        return True
    for segment in segments:
        low = segment.lower()
        if low in _NORM_SEGMENT_MARKERS or low.startswith("ln_") or "layernorm" in low or "rmsnorm" in low:
            return True
    return False


def classify_parameter(logical_name: str, merger_names: Sequence[str]) -> tuple[str, bool]:
    """Classify one trainable parameter into exactly one family:
    ('llm_lora'|'merger_base', decay). Raises for anything unclassifiable.
    将单个可训练参数精确分类到 llm_lora 或 merger_base，并给出 decay 策略。"""
    if _has_lora_segment(logical_name):
        return "llm_lora", _is_lora_matrix(logical_name)
    if _under_any(logical_name, merger_names):
        return "merger_base", not _is_bias_or_norm(logical_name)
    raise ParameterAuditError("unclassified_trainable", logical_name)


def _strip_peft_prefix(name: str) -> str:
    for prefix in _PEFT_NAME_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _param_meta(name: str, parameter: Any) -> dict:
    return {
        "name": name,
        "shape": list(parameter.shape),
        "dtype": str(parameter.dtype),
        "numel": int(parameter.numel()),
    }


def audit_trainable_parameters(
    peft_model: Any,
    roots: ModelRoots,
    merger_names: Sequence[str],
    llm_targets: Sequence[str],
    expected_deepstack: int | None,
) -> dict:
    """Hard startup audit (doc 03 section 5.5).

    Verifies: main merger + declared deepstack count found; every expected
    LLM projection hit by LoRA; no visual projection has LoRA; no merger has
    LoRA; no non-merger vision parameter is trainable; no LLM base parameter
    is trainable; every requires_grad=True parameter is classified into
    exactly one of {merger_base, llm_lora} and the two sets do not overlap.
    """
    lora_trainable: list[dict] = []
    merger_trainable: list[dict] = []
    frozen_vision_non_merger = 0
    frozen_vision_total = 0
    frozen_text = 0
    frozen_lm_head = 0
    lora_parents: set[str] = set()
    total_parameters = 0
    trainable_parameters = 0

    for name, parameter in peft_model.named_parameters():
        logical = _strip_peft_prefix(name)
        total_parameters += int(parameter.numel())
        if parameter.requires_grad:
            trainable_parameters += int(parameter.numel())
            family, _decay = classify_parameter(logical, merger_names)
            if family == "llm_lora":
                lora_parents.add(_lora_parent_module(logical))
                lora_trainable.append(_param_meta(logical, parameter))
            else:
                merger_trainable.append(_param_meta(logical, parameter))
        else:
            if _under_any(logical, [roots.vision_name]):
                frozen_vision_total += int(parameter.numel())
                if not _under_any(logical, merger_names):
                    frozen_vision_non_merger += int(parameter.numel())
            elif logical.startswith(roots.text_name + "."):
                frozen_text += int(parameter.numel())
            elif logical == roots.lm_head_name or logical.startswith(roots.lm_head_name + "."):
                frozen_lm_head += int(parameter.numel())

    deepstack_found = len(roots.vision_root.deepstack_merger_list)
    checks: dict[str, Any] = {
        "main_merger_found": True,
        "deepstack_found": deepstack_found,
        "deepstack_expected": expected_deepstack,
        "deepstack_count_matches": expected_deepstack is None or deepstack_found == expected_deepstack,
        "seven_projections_per_layer": True,
        "lora_targets_hit_all": set(llm_targets) == lora_parents,
        "no_visual_lora": not any("visual" in target.split(".") for target in lora_parents),
        "no_merger_lora": not any(_under_any(target, merger_names) for target in lora_parents),
        "no_non_merger_vision_trainable": True,
        "no_llm_base_trainable": True,
        "no_lm_head_trainable": True,
        "classification_closed": True,
        "families_disjoint": True,
    }
    errors: list[str] = []
    if expected_deepstack is not None and deepstack_found != expected_deepstack:
        errors.append(
            f"deepstack mergers {deepstack_found} != declared {expected_deepstack}"
        )
    missing_targets = sorted(set(llm_targets) - lora_parents)
    if missing_targets:
        errors.append(f"LoRA missing targets: {missing_targets}")
    for target in lora_parents:
        segments = target.split(".")
        if "visual" in segments:
            errors.append(f"visual projection got LoRA: {target}")
        if _under_any(target, merger_names):
            errors.append(f"merger got LoRA: {target}")

    base = unwrap_base(peft_model)
    for name, parameter in base.named_parameters():
        if parameter.requires_grad:
            if _under_any(name, [roots.vision_name]) and not _under_any(name, merger_names):
                errors.append(f"non-merger vision parameter trainable: {name}")
            if name.startswith(roots.text_name + ".") and not _has_lora_segment(name):
                errors.append(f"LLM base parameter trainable: {name}")
            if name == roots.lm_head_name or name.startswith(roots.lm_head_name + "."):
                errors.append(f"lm_head parameter trainable: {name}")

    lora_names = {entry["name"] for entry in lora_trainable}
    merger_names_set = {entry["name"] for entry in merger_trainable}
    if lora_names & merger_names_set:
        errors.append("trainable families overlap")
    for name, parameter in peft_model.named_parameters():
        if not parameter.requires_grad:
            continue
        logical = _strip_peft_prefix(name)
        if logical not in lora_names and logical not in merger_names_set:
            errors.append(f"trainable parameter unclassified: {logical}")

    if errors:
        raise ParameterAuditError("audit_failed", "; ".join(errors[:10]))

    checks["no_non_merger_vision_trainable"] = True
    checks["no_llm_base_trainable"] = True
    checks["no_lm_head_trainable"] = True
    checks["classification_closed"] = True
    checks["families_disjoint"] = True
    if not checks["deepstack_count_matches"] or not checks["lora_targets_hit_all"]:
        raise ParameterAuditError("audit_failed", "structure checks failed")

    trainable_fraction = (trainable_parameters / total_parameters) if total_parameters else 0.0
    return {
        "structure": {
            "vision_root": roots.vision_name,
            "text_root": roots.text_name,
            "merger_names": list(merger_names),
            "llm_target_count": len(llm_targets),
        },
        "checks": checks,
        "llm_lora_trainable": sorted(lora_trainable, key=lambda entry: entry["name"]),
        "merger_trainable": sorted(merger_trainable, key=lambda entry: entry["name"]),
        "frozen_vision_non_merger_parameters": int(frozen_vision_non_merger),
        "frozen_vision_total_parameters": int(frozen_vision_total),
        "frozen_llm_base_parameters": int(frozen_text),
        "frozen_lm_head_parameters": int(frozen_lm_head),
        "total_parameters": int(total_parameters),
        "trainable_parameters": int(trainable_parameters),
        "trainable_fraction": float(trainable_fraction),
    }


def _lora_parent_module(logical_name: str) -> str:
    """Strip trailing lora segments to get the target module path.
    去掉尾部 lora 段得到目标模块路径。"""
    segments = logical_name.split(".")
    cut = len(segments)
    for index, segment in enumerate(segments):
        if segment.startswith("lora_"):
            cut = index
            break
    return ".".join(segments[:cut])


def build_optimizer_groups(
    peft_model: Any,
    merger_names: Sequence[str],
    lora_lr: float,
    merger_lr: float,
    weight_decay: float,
) -> tuple[list[dict], dict]:
    """Explicit four-group optimizer layout (doc 03 section 6):

    1. merger_base + decay        (lr=merger_lr, wd=weight_decay)
    2. merger_base + no_decay     (lr=merger_lr, wd=0)
    3. llm_lora + decay           (lr=lora_lr,   wd=weight_decay)
    4. llm_lora + no_decay        (lr=lora_lr,   wd=0)

    Every trainable parameter lands in exactly one group; group topology is
    returned as deterministic stats for the manifest and resume validation.
    """
    groups: list[dict] = [
        {"name": "merger_base+decay", "lr": float(merger_lr), "weight_decay": float(weight_decay), "params": []},
        {"name": "merger_base+no_decay", "lr": float(merger_lr), "weight_decay": 0.0, "params": []},
        {"name": "llm_lora+decay", "lr": float(lora_lr), "weight_decay": float(weight_decay), "params": []},
        {"name": "llm_lora+no_decay", "lr": float(lora_lr), "weight_decay": 0.0, "params": []},
    ]
    for name, parameter in peft_model.named_parameters():
        if not parameter.requires_grad:
            continue
        logical = _strip_peft_prefix(name)
        family, decay = classify_parameter(logical, merger_names)
        group_name = f"{family}+{'decay' if decay else 'no_decay'}"
        for group in groups:
            if group["name"] == group_name:
                group["params"].append(parameter)
                break
        else:  # pragma: no cover - classification invariant
            raise ParameterAuditError("unclassified_trainable", logical)

    stats = {"groups": []}
    for group in groups:
        names = sorted(
            _strip_peft_prefix(name)
            for name, parameter in peft_model.named_parameters()
            if any(parameter is item for item in group["params"])
        )
        stats["groups"].append(
            {
                "name": group["name"],
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "param_count": len(group["params"]),
                "params": names,
            }
        )
    stats["total_trainable"] = sum(len(g["params"]) for g in groups)
    return groups, stats


def build_cosine_scheduler(optimizer: Any, num_training_steps: int, warmup_ratio: float) -> Any:
    """Cosine schedule with warmup; one shared lambda scales every group so
    the per-group initial LR ratio (lora_lr / merger_lr) is preserved.
    cosine + warmup 调度：同一 lambda 作用于所有组，保持两套 LR 比值。"""
    torch = _torch()
    if num_training_steps <= 0:
        raise ConfigurationError(f"num_training_steps must be positive, got {num_training_steps}")
    if not (0.0 <= warmup_ratio < 1.0):
        raise ConfigurationError(f"warmup_ratio must be in [0, 1), got {warmup_ratio}")
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if num_warmup_steps <= 0:
            return 1.0
        if current_step < num_warmup_steps:
            return float(current_step) / float(num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(
            1.0, float(num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Dataset wiring (consumes the data pipeline module, never duplicates it).
# ---------------------------------------------------------------------------


def parse_image_roots(image_root_flags: Sequence[str]) -> dict[str, str]:
    """Parse --image-root source=path pairs; roots must exist locally.
    解析 image_source=path 映射；本地 root 目录必须存在。"""
    roots: dict[str, str] = {}
    for flag in image_root_flags:
        if "=" not in flag:
            raise ConfigurationError(f"--image-root must be source=path, got {flag!r}")
        source, path = flag.split("=", 1)
        source, path = source.strip(), path.strip()
        if not source or not path:
            raise ConfigurationError(f"--image-root must be source=path, got {flag!r}")
        if source in roots and roots[source] != path:
            raise ConfigurationError(f"duplicate image root for source {source!r}")
        if not os.path.isdir(path):
            raise ConfigurationError(f"image root for {source!r} is not a directory: {path}")
        roots[source] = path
    if not roots:
        raise ConfigurationError("at least one --image-root source=path is required")
    return roots


def augmentation_config_from_args(aug_args: AugmentationArguments) -> Any:
    """Build AugmentationConfig (data pipeline module) from CLI overrides.
    由 CLI 参数构造数据管线模块的 AugmentationConfig。"""
    data_module = _load_data_module()
    config = data_module.AugmentationConfig()
    return replace(
        config,
        enabled=aug_args.aug_enabled,
        rotate90_prob=aug_args.aug_rotate90_prob,
        affine_rotation_prob=aug_args.aug_affine_rotation_prob,
        scale_prob=aug_args.aug_scale_prob,
        translate_prob=aug_args.aug_translate_prob,
        perspective_prob=aug_args.aug_perspective_prob,
        degradation_probability=aug_args.aug_degradation_probability,
        min_degradations=aug_args.aug_min_degradations,
        max_degradations=aug_args.aug_max_degradations,
    )


def parse_repeat_weights(flags: Sequence[str]) -> dict[str, int]:
    """Parse --repeat-weights group=weight entries; weights must be >= 1.
    解析确定性 group repeat weight；权重必须 >= 1。"""
    weights: dict[str, int] = {}
    for flag in flags:
        if "=" not in flag:
            raise ConfigurationError(f"--repeat-weights must be group=weight, got {flag!r}")
        group, raw_weight = flag.split("=", 1)
        group, raw_weight = group.strip(), raw_weight.strip()
        try:
            weight = int(raw_weight)
        except ValueError as error:
            raise ConfigurationError(f"--repeat-weights weight must be an int, got {raw_weight!r}") from error
        if weight < 1:
            raise ConfigurationError(f"--repeat-weights weight must be >= 1, got {weight}")
        weights[group] = weight
    return weights


class GroupRepeatDataset:
    """Deterministic dataset wrapper that repeats each episode of a group
    `weight` times in stable episode order.

    - every episode is seen at least once per epoch (no random sampling);
    - default weight 1 => identity schedule;
    - set_epoch delegates to the wrapped dataset (augmentation epochs stay
      in sync);
    - preflight delegates to the wrapped dataset.

    The group id is read from the canonical episode JSONL (group key field)
    via the data pipeline module's LazyJsonLines reader.
    """

    def __init__(
        self,
        base: Any,
        episode_jsonl: str | Path,
        group_key: str,
        weights: Mapping[str, int],
        max_samples: int | None = None,
    ) -> None:
        data_module = _load_data_module()
        self._base = base
        self._group_key = group_key
        self._weights = dict(weights)
        store = data_module.LazyJsonLines(Path(episode_jsonl))
        group_keys: list[str] = []
        for index in range(len(store)):
            episode = store[index]
            group = episode.get(group_key)
            if not isinstance(group, str) or not group:
                raise ConfigurationError(
                    f"episode {index} missing group key {group_key!r}"
                )
            group_keys.append(group)
        self._group_counts = {name: group_keys.count(name) for name in sorted(set(group_keys))}
        schedule: list[int] = []
        for index, group in enumerate(group_keys):
            schedule.extend([index] * self._weights.get(group, 1))
        if max_samples is not None:
            schedule = schedule[:max_samples]
        self._schedule = schedule
        self._data_module = data_module

    def __len__(self) -> int:
        return len(self._schedule)

    def __getitem__(self, index: int) -> dict:
        return self._base[self._schedule[index]]

    def set_epoch(self, epoch: int) -> None:
        """Delegate to the wrapped dataset; augmentation seed stays
        epoch-driven. 委托给底层 dataset，增强 seed 保持由 epoch 驱动。"""
        self._base.set_epoch(epoch)

    @property
    def epoch(self) -> int:
        return self._base.epoch

    def preflight(self, limit: int | None = None) -> dict:
        return self._base.preflight(limit)

    def group_counts(self) -> dict[str, int]:
        return dict(self._group_counts)


class ModelBatchCollator:
    """Trainer-facing collator: strips the (batch, meta) tuple returned by
    Phase2DataCollator down to the model batch; episode metadata never
    reaches the model forward pass. 面向 Trainer 的 collator：丢弃
    Phase2DataCollator 返回的 meta，只把模型 batch 传给 forward。"""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __call__(self, features: Sequence[dict]) -> dict:
        batch, _meta = self._inner(features)
        return batch


class PixelBoundedProcessor:
    """Thin processor wrapper that injects min_pixels/max_pixels into every
    processor call. Only used when the pinned Transformers build declares
    those kwargs on its image processor (5.14.1 Qwen3-VL does not).
    处理器包装：仅在处理器支持时注入 min_pixels/max_pixels。"""

    def __init__(self, processor: Any, min_pixels: int, max_pixels: int) -> None:
        self._processor = processor
        self._min_pixels = int(min_pixels)
        self._max_pixels = int(max_pixels)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("min_pixels", self._min_pixels)
        kwargs.setdefault("max_pixels", self._max_pixels)
        return self._processor(*args, **kwargs)

    def save_pretrained(self, output_dir: str | Path) -> None:
        return self._processor.save_pretrained(output_dir)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._processor, name)


def _processor_supports_pixel_bounds(processor: Any) -> bool:
    image_processor = getattr(processor, "image_processor", None)
    valid_kwargs = getattr(image_processor, "valid_kwargs", None)
    if valid_kwargs is None:
        return False
    annotations = getattr(valid_kwargs, "__annotations__", {}) or {}
    return "min_pixels" in annotations and "max_pixels" in annotations


# ---------------------------------------------------------------------------
# Epoch synchronization (augmentation epochs + distributed sampler).
# ---------------------------------------------------------------------------


def _make_epoch_sync_callback(dataset: Any, sampler_provider: Any = None) -> Any:
    """Build the epoch-sync TrainerCallback (imports transformers lazily).

    Keeps the dataset augmentation epoch and the distributed sampler epoch
    in sync with Trainer's state (also after resume, where state.epoch may
    be fractional and int() floors it back to the epoch whose seeds the
    current steps were generated with).
    保持 dataset 增强 epoch 与 distributed sampler epoch 与 Trainer 同步。
    """
    from transformers import TrainerCallback  # noqa: PLC0415

    class EpochSyncCallback(TrainerCallback):
        def __init__(self, dataset: Any, sampler_provider: Any = None) -> None:
            self._dataset = dataset
            self._sampler_provider = sampler_provider

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, control, kwargs
            self._sync(state)

        def on_epoch_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            del args, control, kwargs
            self._sync(state)

        def _sync(self, state: Any) -> None:
            epoch = int(state.epoch)
            self._dataset.set_epoch(epoch)
            if self._sampler_provider is not None:
                sampler = self._sampler_provider()
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)

    return EpochSyncCallback(dataset, sampler_provider)


# ---------------------------------------------------------------------------
# Merger state (safetensors, logical keys, strict read-back).
# ---------------------------------------------------------------------------


def collect_merger_state(base_model: Any, merger_names: Sequence[str]) -> dict[str, Any]:
    """All merger parameters and persistent buffers as detached contiguous
    tensors keyed by the unwrapped base model's stable logical names.
    merger 全部参数与 persistent buffer，使用 unwrap 后 base model 的
    稳定逻辑 key。"""
    state: dict[str, Any] = {}
    for name, parameter in base_model.named_parameters():
        if _under_any(name, merger_names):
            state[name] = parameter.detach().clone().contiguous()
    for name, buffer in base_model.named_buffers():
        if _under_any(name, merger_names) and buffer is not None:
            state[name] = buffer.detach().clone().contiguous()
    if not state:
        raise CheckpointError("empty_merger_state", "no merger parameters found")
    return state


def merger_state_meta(base_model: Any, merger_names: Sequence[str]) -> list[dict]:
    """Name/shape/dtype/numel meta of the merger state (no tensor copies).
    merger state 的 name/shape/dtype/numel 元数据（不复制张量）。"""
    torch = _torch()
    meta: list[dict] = []
    for name, parameter in base_model.named_parameters():
        if _under_any(name, merger_names):
            meta.append(
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "dtype": str(parameter.dtype),
                    "numel": int(parameter.numel()),
                }
            )
    for name, buffer in base_model.named_buffers():
        if _under_any(name, merger_names) and buffer is not None and isinstance(buffer, torch.Tensor):
            meta.append(
                {
                    "name": name,
                    "shape": list(buffer.shape),
                    "dtype": str(buffer.dtype),
                    "numel": int(buffer.numel()),
                }
            )
    meta.sort(key=lambda entry: entry["name"])
    return meta


def save_merger_state(base_model: Any, output_path: str | Path, merger_names: Sequence[str]) -> dict:
    """Save the full merger state to safetensors, then read it back and
    verify keys/shapes/dtypes. Returns file checksum + parameter meta.
    保存 merger state 到 safetensors，读回核对 key/shape/dtype。"""
    import safetensors.torch  # noqa: PLC0415

    state = collect_merger_state(base_model, merger_names)
    output_path = Path(output_path)
    try:
        safetensors.torch.save_file(state, output_path)
    except Exception as error:  # noqa: BLE001 - stable error surface
        raise CheckpointError("merger_save_failed", str(error)) from error
    try:
        with safetensors.safe_open(output_path, framework="pt") as handle:
            read_back = set(handle.keys())
            for name, tensor in state.items():
                if name not in read_back:
                    raise CheckpointError("merger_save_failed", f"key missing on read-back: {name}")
                info = handle.get_slice(name)
                shape = list(info.get_shape())
                if shape != list(tensor.shape) or str(tensor.dtype) != str(state[name].dtype):
                    raise CheckpointError(
                        "merger_save_failed", f"read-back mismatch for {name}"
                    )
    except CheckpointError:
        raise
    except Exception as error:  # noqa: BLE001
        raise CheckpointError("merger_save_failed", str(error)) from error
    return {
        "file": MERGER_STATE_FILENAME,
        "file_sha256": sha256_file(output_path),
        "parameters": merger_state_meta(base_model, merger_names),
    }


def load_merger_state_strict(base_model: Any, path: str | Path, merger_names: Sequence[str]) -> None:
    """Strictly load merger state into the base model: missing/unexpected
    keys and shape/dtype mismatches are hard failures.
    严格加载 merger state：缺失/意外 key 与 shape/dtype 不匹配都失败。"""
    import safetensors.torch  # noqa: PLC0415

    expected = {entry["name"] for entry in merger_state_meta(base_model, merger_names)}
    state = safetensors.torch.load_file(Path(path))
    missing = sorted(expected - set(state))
    unexpected = sorted(set(state) - expected)
    if missing or unexpected:
        raise CheckpointError(
            "merger_load_mismatch",
            f"missing={missing[:5]} unexpected={unexpected[:5]}",
        )
    for name, tensor in state.items():
        parameter = dict(base_model.named_parameters()).get(name)
        if parameter is None:
            buffer = dict(base_model.named_buffers()).get(name)
            if buffer is None:
                raise CheckpointError("merger_load_mismatch", f"no parameter/buffer for {name}")
            if list(buffer.shape) != list(tensor.shape) or buffer.dtype != tensor.dtype:
                raise CheckpointError("merger_load_mismatch", f"shape/dtype mismatch for {name}")
            buffer.data.copy_(tensor)
        else:
            if list(parameter.shape) != list(tensor.shape) or parameter.dtype != tensor.dtype:
                raise CheckpointError("merger_load_mismatch", f"shape/dtype mismatch for {name}")
            parameter.data.copy_(tensor)


def load_composite_weights(
    peft_model: Any,
    base_model: Any,
    checkpoint_dir: str | Path,
    merger_names: Sequence[str],
) -> None:
    """Load the persisted LLM LoRA adapter and merger state from a validated
    composite checkpoint before resuming. transformers' own PEFT checkpoint
    loading is disabled on the trainer (it would freeze the merger base
    parameters); this explicit load does not touch requires_grad flags.
    从校验通过的复合 checkpoint 加载持久化的 LLM LoRA adapter 与 merger
    状态；transformers 自带加载已禁用（会冻结 merger base 参数）；本显式
    加载不修改 requires_grad 标志。"""
    import safetensors.torch  # noqa: PLC0415

    checkpoint_dir = Path(checkpoint_dir)
    adapter_dir = checkpoint_dir / "adapter"
    adapter_path = adapter_dir / "adapter_model.safetensors"
    if not adapter_path.is_file():
        raise CheckpointError("adapter_files_missing", str(adapter_dir))
    state = safetensors.torch.load_file(adapter_path)
    # PEFT persists default-adapter keys without the ".default." segment
    # ("...lora_A.weight"); the live model names them "...lora_A.default.weight".
    # 恢复保存时省略的 ".default." 段（保存用 "...lora_A.weight"，模型用
    # "...lora_A.default.weight"）。
    mapped: dict[str, Any] = {}
    for name, tensor in state.items():
        head, _sep, tail = name.rpartition(".")
        if head.endswith("lora_A") or head.endswith("lora_B"):
            mapped[f"{head}.default.{tail}"] = tensor
        else:
            mapped[name] = tensor
    result = peft_model.load_state_dict(mapped, strict=False)
    # Non-LoRA model keys are legitimately absent from an adapter state dict;
    # every LoRA key of the live model must be covered, and no unexpected key
    # may remain. 非 LoRA 模型 key 本就不在 adapter state 中；但模型的所有
    # LoRA key 必须被覆盖，且不得有意外 key。
    missing_lora = [key for key in result.missing_keys if "lora_" in key]
    if missing_lora or result.unexpected_keys:
        raise CheckpointError(
            "adapter_state_mismatch",
            f"missing_lora={missing_lora[:5]} unexpected={result.unexpected_keys[:5]}",
        )
    load_merger_state_strict(base_model, checkpoint_dir / MERGER_STATE_FILENAME, merger_names)
    logger.info("loaded composite weights from checkpoint: %s", checkpoint_dir)


def verify_adapter_keys(adapter_dir: str | Path, llm_targets: Sequence[str]) -> dict:
    """Read back the saved adapter and verify every key is an LLM LoRA key
    whose parent module is inside llm_targets (no visual/merger LoRA).
    读回 adapter，校验所有 key 都是 LLM LoRA 且父模块在 target 集合内。"""
    import safetensors  # noqa: PLC0415

    adapter_dir = Path(adapter_dir)
    weights_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    if not weights_path.is_file() or not config_path.is_file():
        raise CheckpointError("adapter_files_missing", str(adapter_dir))
    with safetensors.safe_open(weights_path, framework="pt") as handle:
        keys = list(handle.keys())
    violations: list[str] = []
    weight_parents: set[str] = set()
    for key in keys:
        logical = _strip_peft_prefix(key)
        if not _has_lora_segment(logical):
            violations.append(f"non-lora adapter key: {key}")
            continue
        parent = _lora_parent_module(logical)
        if parent not in llm_targets:
            violations.append(f"adapter key outside LLM targets: {key}")
        weight_parents.add(parent)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    declared_targets = set(config.get("target_modules") or [])
    # PEFT persists target_modules as short projection names (e.g. "q_proj")
    # while llm_targets are full module paths; normalize both sides to short
    # names before comparing so the strict check works for either format.
    # PEFT 持久化的 target_modules 是短投影名（如 "q_proj"），而 llm_targets
    # 是完整路径；比较前统一归一化为短名集合，两种格式都保持严格校验。
    declared_short = {name.rsplit(".", 1)[-1] for name in declared_targets}
    weight_short = {parent.rsplit(".", 1)[-1] for parent in weight_parents}
    if not declared_short.issubset(weight_short):
        violations.append("adapter_config target_modules exceed the audited LLM targets")
    if not weight_short.issubset(declared_short):
        violations.append(
            "adapter_config target_modules miss LoRA modules present in adapter weights"
        )
    if violations:
        raise CheckpointError("adapter_key_violation", "; ".join(violations[:10]))
    return {
        "file": "adapter/adapter_model.safetensors",
        "file_sha256": sha256_file(weights_path),
        "key_count": len(keys),
        "target_module_count": len(set(_lora_parent_module(_strip_peft_prefix(k)) for k in keys)),
    }


# ---------------------------------------------------------------------------
# Checkpoint layout: composite write, completeness, resume validation.
# ---------------------------------------------------------------------------


def checkpoint_complete(checkpoint_dir: str | Path) -> bool:
    """A checkpoint is resumable only when the full composite layout exists
    (adapter + merger + processor + manifest + trainer state + optimizer +
    scheduler). The manifest is written last, so partial writes are never
    mistaken for successful checkpoints. 只有完整复合布局才算可恢复
    checkpoint；manifest 最后写，半成品不会被当成成功目录。"""
    checkpoint_dir = Path(checkpoint_dir)
    required = [
        checkpoint_dir / "adapter" / "adapter_config.json",
        checkpoint_dir / "adapter" / "adapter_model.safetensors",
        checkpoint_dir / MERGER_STATE_FILENAME,
        checkpoint_dir / "processor",
        checkpoint_dir / TRAINING_MANIFEST_FILENAME,
        checkpoint_dir / "trainer_state.json",
        checkpoint_dir / "optimizer.pt",
        checkpoint_dir / "scheduler.pt",
    ]
    if not all(path.exists() for path in required):
        return False
    try:
        manifest = json.loads(
            (checkpoint_dir / TRAINING_MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("schema_version") == SCHEMA_VERSION


def _checkpoint_step(checkpoint_dir: Path) -> int | None:
    name = checkpoint_dir.name
    if not name.startswith(CHECKPOINT_PREFIX):
        return None
    suffix = name[len(CHECKPOINT_PREFIX):]
    if not suffix.isdigit():
        return None
    return int(suffix)


def find_complete_checkpoints(output_dir: str | Path) -> list[tuple[int, Path]]:
    """All complete composite checkpoints sorted by step.
    全部完整复合 checkpoint，按 step 升序。"""
    output_dir = Path(output_dir)
    found: list[tuple[int, Path]] = []
    if not output_dir.is_dir():
        return found
    for candidate in output_dir.iterdir():
        if not candidate.is_dir() or not candidate.name.startswith(CHECKPOINT_PREFIX):
            continue
        step = _checkpoint_step(candidate)
        if step is None:
            continue
        if checkpoint_complete(candidate):
            found.append((step, candidate))
    return sorted(found, key=lambda item: item[0])


def resolve_resume_target(output_dir: str | Path, explicit: str | None) -> str | None:
    """Resolve the resume target:

    - explicit non-empty path: must be a complete composite checkpoint;
    - explicit empty string: force fresh run (refused if checkpoints exist);
    - None: auto-resume the latest complete checkpoint;
    - fresh start with any existing checkpoint-* dir (complete or not) is
      refused so a new run never silently overwrites or mixes old state.
    """
    output_dir = str(output_dir)
    if explicit is not None and explicit.strip() == "":
        if any(Path(output_dir).glob(f"{CHECKPOINT_PREFIX}*")):
            raise ResumeConflictError(
                "fresh_start_with_existing_checkpoints", [output_dir]
            )
        return None
    if explicit is not None:
        target = Path(explicit)
        if not target.is_dir():
            raise ResumeConflictError("resume_target_missing", [explicit])
        if not checkpoint_complete(target):
            raise ResumeConflictError("resume_target_incomplete", [explicit])
        return explicit
    complete = find_complete_checkpoints(output_dir)
    if complete:
        return str(complete[-1][1])
    partial = [
        str(path)
        for path in Path(output_dir).iterdir()
        if path.is_dir() and path.name.startswith(CHECKPOINT_PREFIX)
    ] if Path(output_dir).is_dir() else []
    if partial:
        raise ResumeConflictError("incomplete_checkpoints_exist", partial[:5])
    return None


def _compare(value_name: str, expected: Any, found: Any, mismatches: list[str]) -> None:
    if expected != found:
        mismatches.append(f"{value_name}: manifest={found!r} request={expected!r}")


def _jsonable(value: Any) -> Any:
    """Recursively convert a value to its JSON-safe canonical form
    (tuples -> lists, sets -> sorted lists). Used only for comparison of
    persisted vs in-memory configuration. 递归转换为 JSON 安全规范形式
    （元组转列表、集合转排序列表），仅用于持久化与内存配置的比较。"""
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    return value


def validate_resume_checkpoint(checkpoint_dir: str | Path, context: dict) -> None:
    """Validate a persisted composite manifest against the current explicit
    request; any conflict is a stable refusal (no guessing).
    校验持久化 manifest 与当前显式请求；冲突稳定拒绝，不猜测。"""
    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_dir / TRAINING_MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResumeConflictError("manifest_unreadable", [str(manifest_path)]) from error
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ResumeConflictError("schema_version", [str(manifest.get("schema_version"))])

    mismatches: list[str] = []
    # Missing profile is only backward-compatible with the original Phase2
    # layout; Change checkpoints never silently cross-resume.
    # 缺失 profile 仅兼容原始 Phase2；Change checkpoint 绝不跨 profile 恢复。
    _compare("training_profile", context.get("training_profile", "phase2"),
             manifest.get("training_profile", "phase2"), mismatches)
    if context.get("training_profile") == "change_agent":
        _compare("data_contract", context.get("data_contract"), manifest.get("data_contract"), mismatches)
        _compare("change_prompt.sha256", context.get("change_prompt", {}).get("sha256"),
                 manifest.get("change_prompt", {}).get("sha256"), mismatches)
    _compare("base_model.fingerprint", context["base_model"]["fingerprint"],
             manifest.get("base_model", {}).get("fingerprint"), mismatches)
    _compare("base_model.revision", context["base_model"].get("revision"),
             manifest.get("base_model", {}).get("revision"), mismatches)
    _compare("base_model.merger_lora_adapter",
             context["base_model"].get("merger_lora_adapter"),
             manifest.get("base_model", {}).get("merger_lora_adapter"), mismatches)
    _compare("processor.fingerprint", context["processor"]["fingerprint"],
             manifest.get("processor", {}).get("fingerprint"), mismatches)
    _compare("data.train_sha256", context["data"]["train_sha256"],
             manifest.get("data", {}).get("train_sha256"), mismatches)
    _compare("data.eval_sha256", context["data"].get("eval_sha256"),
             manifest.get("data", {}).get("eval_sha256"), mismatches)
    _compare("data.train_upstream_manifest_sha256",
             context["data"].get("train_upstream_manifest_sha256"),
             manifest.get("data", {}).get("train_upstream_manifest_sha256"), mismatches)
    lora = manifest.get("lora", {})
    _compare("lora.rank", context["lora"]["rank"], lora.get("rank"), mismatches)
    _compare("lora.alpha", context["lora"]["alpha"], lora.get("alpha"), mismatches)
    _compare("lora.dropout", context["lora"]["dropout"], lora.get("dropout"), mismatches)
    _compare("lora.bias", context["lora"]["bias"], lora.get("bias"), mismatches)
    _compare("lora.target_modules", sorted(context["lora"]["target_modules"]),
             sorted(lora.get("target_modules") or []), mismatches)
    _compare("merger.parameters", context["merger"]["parameters"],
             manifest.get("merger", {}).get("parameters"), mismatches)
    # The per-group LR is runtime scheduler state restored from
    # optimizer.pt on resume, not a stable request field; compare the group
    # topology (name/weight_decay/params) only. This also keeps older
    # checkpoints saved with the live warmup LR resumable.
    # 每组 LR 是运行时调度状态（resume 时由 optimizer.pt 恢复），不是稳定
    # 请求字段；只比较分组拓扑（name/weight_decay/params）。这也使保存了
    # 实时 warmup LR 的旧 checkpoint 可以 resume。
    def _strip_lr(groups: Any) -> list:
        return [
            {key: value for key, value in group.items() if key != "lr"}
            for group in groups
        ]

    _compare("optimizer.groups", _strip_lr(context["optimizer"]["groups"]),
             _strip_lr(manifest.get("optimizer", {}).get("groups") or []), mismatches)
    _compare("augmentation.seed", context["augmentation"]["seed"],
             manifest.get("augmentation", {}).get("seed"), mismatches)
    _compare("augmentation.config", _jsonable(context["augmentation"]["config"]),
             _jsonable(manifest.get("augmentation", {}).get("config")), mismatches)
    training = manifest.get("training", {})
    _compare("training.max_seq_length", context["training"]["max_seq_length"],
             training.get("max_seq_length"), mismatches)
    _compare("training.image_min_pixels", context["training"]["image_min_pixels"],
             training.get("image_min_pixels"), mismatches)
    _compare("training.image_max_pixels", context["training"]["image_max_pixels"],
             training.get("image_max_pixels"), mismatches)
    _compare("training.image_pixels_applied", context["training"]["image_pixels_applied"],
             training.get("image_pixels_applied"), mismatches)
    _compare("training.torch_dtype", context["training"]["torch_dtype"],
             training.get("torch_dtype"), mismatches)
    sampling = manifest.get("data_sampling", {})
    _compare("data_sampling.group_key", context["data_sampling"]["group_key"],
             sampling.get("group_key"), mismatches)
    _compare("data_sampling.repeat_weights", context["data_sampling"]["repeat_weights"],
             sampling.get("repeat_weights"), mismatches)

    # File integrity: checksums recorded at save time must still match.
    # 文件完整性：保存时记录的 checksum 必须仍然匹配。
    merger_meta = manifest.get("merger", {})
    merger_path = checkpoint_dir / MERGER_STATE_FILENAME
    if merger_path.is_file():
        _compare("merger.file_sha256", sha256_file(merger_path),
                 merger_meta.get("file_sha256"), mismatches)
    else:
        mismatches.append("merger_model.safetensors missing")
    adapter_meta = manifest.get("adapter", {})
    adapter_path = checkpoint_dir / "adapter" / "adapter_model.safetensors"
    if adapter_path.is_file():
        _compare("adapter.file_sha256", sha256_file(adapter_path),
                 adapter_meta.get("file_sha256"), mismatches)
    else:
        mismatches.append("adapter/adapter_model.safetensors missing")

    if mismatches:
        raise ResumeConflictError("resume_conflict", mismatches)


# ---------------------------------------------------------------------------
# Identities and manifest
# ---------------------------------------------------------------------------


def model_logical_identity(config: Any) -> dict:
    """Path-independent logical identity of the base model.

    The local checkpoint path is only a loading hint and never part of the
    persisted identity; the fingerprint covers stable config fields.
    与机器路径无关的 base model 逻辑身份；指纹覆盖稳定配置字段。"""
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


def merger_lora_adapter_identity(adapter_path: str | Path) -> dict:
    """Path-independent identity of the phase-1 merger LoRA adapter used to
    seed the base weights; resume requires the same adapter (a different
    adapter silently changes the initial weights, which must not happen).
    用于初始化 base 权重的 phase1 merger LoRA adapter 的无路径身份；resume
    必须使用同一 adapter（不同 adapter 会静默改变初始权重，绝不允许）。"""
    adapter_path = Path(adapter_path)
    config_path = adapter_path / "adapter_config.json"
    weights_path = adapter_path / "adapter_model.safetensors"
    for path in (config_path, weights_path):
        if not path.is_file():
            raise ConfigurationError(f"merger LoRA adapter file missing: {path}")
    try:
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigurationError(f"merger LoRA adapter config unreadable: {config_path}") from error
    return {
        "config_sha256": sha256_file(config_path),
        "weights_sha256": sha256_file(weights_path),
        "target_modules": sorted(adapter_config.get("target_modules") or []),
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


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


def build_training_context(
    *,
    config: Any,
    processor: Any,
    model_args: ModelArguments,
    data_args: DataArguments,
    lora_args: LoRAArguments,
    aug_config: Any,
    aug_seed: str,
    optimizer_stats: dict,
    merger_names: Sequence[str],
    merger_meta: list[dict],
    llm_targets: Sequence[str],
    image_pixels_applied: bool,
    train_sha256: str,
    eval_sha256: str | None,
    train_upstream_manifest_sha256: str | None,
    train_episode_count: int,
    eval_episode_count: int | None,
    group_counts: dict[str, int],
    repeat_weights: dict[str, int],
    image_sources: Sequence[str],
    repo_root: str | Path,
    training_profile: str = "phase2",
    change_prompt: dict | None = None,
) -> dict:
    """Everything needed to (a) build the training manifest at save time and
    (b) validate a resume candidate before training.
    构建训练 manifest 与 resume 校验所需的全部上下文。"""
    import dataclasses  # noqa: PLC0415

    training = {
        "epochs": None,
        "max_seq_length": int(data_args.max_seq_length),
        "image_min_pixels": int(data_args.image_min_pixels),
        "image_max_pixels": int(data_args.image_max_pixels),
        "image_pixels_applied": bool(image_pixels_applied),
        "torch_dtype": model_args.torch_dtype,
        "attn_implementation": model_args.attn_implementation,
        "local_files_only": bool(model_args.local_files_only),
    }
    base_identity = model_logical_identity(config)
    if model_args.merger_lora_adapter:
        base_identity["merger_lora_adapter"] = merger_lora_adapter_identity(
            model_args.merger_lora_adapter
        )
    if training_profile not in DATA_PROFILES:
        raise ConfigurationError(f"unknown data profile: {training_profile!r}")
    context = {
        "training_profile": training_profile,
        "base_model": base_identity,
        "processor": processor_identity(processor),
        "model_id_as_given": model_args.model_id,
        "data": {
            "train_file": Path(data_args.train_file).name,
            "train_sha256": train_sha256,
            "train_episode_count": int(train_episode_count),
            "eval_file": Path(data_args.eval_file).name if data_args.eval_file else None,
            "eval_sha256": eval_sha256,
            "eval_episode_count": int(eval_episode_count) if eval_episode_count is not None else None,
            "train_upstream_manifest_sha256": train_upstream_manifest_sha256,
            "image_sources": sorted(image_sources),
        },
        "lora": {
            "rank": int(lora_args.lora_rank),
            "alpha": int(lora_args.lora_alpha),
            "dropout": float(lora_args.lora_dropout),
            "bias": lora_args.lora_bias,
            "target_modules": list(llm_targets),
        },
        "merger": {
            "modules": list(merger_names),
            "parameters": merger_meta,
        },
        "optimizer": optimizer_stats,
        "augmentation": {
            "seed": str(aug_seed),
            "enabled": bool(aug_config.enabled),
            "config": dataclasses.asdict(aug_config),
        },
        "training": training,
        "data_sampling": {
            "group_key": data_args.repeat_group_key,
            "repeat_weights": dict(sorted(repeat_weights.items())),
            "group_counts": dict(sorted(group_counts.items())),
        },
        "environment": {
            "git_head": git_head(repo_root),
            "transformers_version": _package_version("transformers"),
            "torch_version": _package_version("torch"),
            "peft_version": _package_version("peft"),
            "python_version": sys.version.split()[0],
        },
    }
    if training_profile == "change_agent":
        if not change_prompt or not change_prompt.get("sha256"):
            raise ConfigurationError("change_agent profile requires a prompt sha256")
        context["data_contract"] = {
            "name": "change_qwen_sft", "schema_version": 1,
            "ordered_multi_image": True,
            "required_leading_roles": ["raw_full_t1", "raw_full_t2"],
            "target_schema": "ChangeInitialResult",
        }
        context["change_prompt"] = dict(change_prompt)
    return context


def build_manifest(
    context: dict,
    *,
    step: int,
    epoch: float,
    merger_meta: dict,
    adapter_meta: dict,
    optimizer_groups: Sequence[dict],
    train_episodes_after_repeat: int,
    eval_episodes: int | None,
) -> dict:
    """Final checkpoint manifest; written last as the completion marker.
    最终 checkpoint manifest；作为完成标记最后写入。"""
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_type": "phase2_composite",
        "training_profile": context["training_profile"],
        "step": int(step),
        "epoch": float(epoch),
        "base_model": context["base_model"],
        "processor": context["processor"],
        "model_id_as_given": context["model_id_as_given"],
        "data": {
            **context["data"],
            "train_episodes_after_repeat": int(train_episodes_after_repeat),
            "eval_episodes_after_cap": int(eval_episodes) if eval_episodes is not None else None,
        },
        "lora": context["lora"],
        "merger": {**context["merger"], **merger_meta},
        "adapter": adapter_meta,
        "optimizer": {"groups": list(optimizer_groups)},
        "augmentation": context["augmentation"],
        "training": context["training"],
        "data_sampling": context["data_sampling"],
        "environment": context["environment"],
        **({"data_contract": context["data_contract"], "change_prompt": context["change_prompt"]}
           if context["training_profile"] == "change_agent" else {}),
    }


# ---------------------------------------------------------------------------
# Phase2Trainer (lazy mixin; the concrete class is built by
# _phase2_trainer_class() inside the execution path so that importing this
# module never imports transformers).
# ---------------------------------------------------------------------------


class _Phase2TrainerMixin:
    """Trainer overrides for Phase 2:

    - explicit four-group optimizer (merger vs LoRA learning rates);
    - cosine scheduler with a shared lambda (LR ratio preserved);
    - composite checkpoint saving (adapter/ + merger_model.safetensors +
      processor/ + manifest) for every checkpoint dir and the final root.

    Mixed into transformers.Trainer via _phase2_trainer_class(); methods are
    written against the transformers 5.14.1 Trainer API (reference env pins
    that version). Gradient checkpointing needs no forwarding: peft's
    PeftModel instance already delegates gradient_checkpointing_enable to
    the base model (verified on peft 0.20.0).
    """

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model: Any | None = None) -> None:
        """Composite checkpoint weights (LLM LoRA adapter + merger state) are
        loaded explicitly by the phase-2 script before training starts.
        transformers' own PEFT adapter loading would freeze every non-adapter
        parameter (including the merger base parameters we train), so it is
        disabled. 复合 checkpoint 权重（LLM LoRA adapter + merger 状态）由
        phase2 脚本在训练前显式加载；transformers 自带的 PEFT adapter 加载
        会冻结所有非 adapter 参数（包括我们要训练的 merger base 参数），
        因此禁用它。"""
        del resume_from_checkpoint, model
        logger.info("skipping transformers model checkpoint load (composite layout handled by script)")

    def __init__(
        self,
        *,
        model: Any,        args: Any,
        data_collator: Any,
        train_dataset: Any,
        eval_dataset: Any,
        processing_class: Any,
        callbacks: Sequence[Any],
        merger_lr: float,
        lora_lr: float,
        weight_decay: float,
        warmup_ratio: float,
        merger_names: Sequence[str],
        llm_targets: Sequence[str],
        manifest_context: dict,
    ) -> None:
        self._merger_lr = float(merger_lr)
        self._lora_lr = float(lora_lr)
        self._weight_decay = float(weight_decay)
        self._warmup_ratio = float(warmup_ratio)
        self._merger_names = list(merger_names)
        self._llm_targets = list(llm_targets)
        self._manifest_context = manifest_context
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=list(callbacks),
        )

    # -- optimizer / scheduler ----------------------------------------------

    def create_optimizer(self, model: Any = None) -> Any:
        """Build the explicit four-group AdamW (custom optimizer is created
        before any scheduler; Trainer calls create_optimizer first).
        显式四组 AdamW（自定义 optimizer 在 scheduler 之前创建）。"""
        del model
        if self.optimizer is None:
            groups, stats = build_optimizer_groups(
                self.model, self._merger_names, self._lora_lr, self._merger_lr, self._weight_decay
            )
            # Remember the initial per-group LR from the independent stats
            # dicts (AdamW mutates the passed groups in place, so the groups
            # list itself cannot be reused later). The manifest/resume
            # comparison must stay on the configured initial values, never
            # the live warmup LR. 从独立的 stats 记住各组初始 LR（AdamW 会
            # 原地修改传入的 groups，因此不能复用该列表）；manifest 与 resume
            # 校验必须基于配置的初始值，而不是实时的 warmup LR。
            self._initial_optimizer_groups = stats["groups"]
            torch = _torch()
            self.optimizer = torch.optim.AdamW(
                groups, lr=self._lora_lr, betas=(0.9, 0.999), eps=1e-8
            )
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: Any = None) -> Any:
        """Cosine + warmup scheduler over the custom optimizer; the shared
        lambda keeps the merger/LoRA LR ratio constant.
        cosine + warmup 调度；共享 lambda 保持两套 LR 比值。"""
        if self.lr_scheduler is None:
            if optimizer is None:
                optimizer = self.optimizer
            self.lr_scheduler = build_cosine_scheduler(
                optimizer, num_training_steps, self._warmup_ratio
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def _optimizer_group_stats_for_manifest(self) -> list[dict]:
        """Optimizer group topology for the manifest. Before training starts
        there is no optimizer yet; after create_optimizer() the live groups
        are derived from the actual param_groups.
        manifest 用的 optimizer group 拓扑。"""
        if getattr(self, "optimizer", None) is None:
            return []
        id_to_name = {
            id(parameter): _strip_peft_prefix(name)
            for name, parameter in self.model.named_parameters()
        }
        initial_lrs = {
            group.get("name"): float(group["lr"])
            for group in getattr(self, "_initial_optimizer_groups", [])
        }
        stats = []
        for group in self.optimizer.param_groups:
            names = sorted(id_to_name[id(parameter)] for parameter in group["params"])
            stats.append(
                {
                    "name": group.get("name", "unknown"),
                    # Persist the configured initial LR, not the live (warmup-
                    # adjusted) value; resume validation compares against the
                    # initial configuration. 持久化配置的初始 LR 而非实时的
                    # warmup 值；resume 校验与初始配置比较。
                    "lr": float(initial_lrs.get(group.get("name"), group["lr"])),
                    "weight_decay": float(group.get("weight_decay", 0.0)),
                    "param_count": len(names),
                    "params": names,
                }
            )
        return stats

    # -- composite checkpoint saving -----------------------------------------

    def _composite_model(self) -> Any:
        model = self.model
        torch = _torch()
        if isinstance(model, torch.nn.DataParallel):
            return model.module
        return model

    def _write_composite(self, output_dir: str | Path) -> None:
        """Write the complete composite layout; the manifest is written last
        and acts as the completion marker. On failure a stable save_error
        file is written and Phase2SaveError is raised (no fake completion).
        写入完整复合布局；manifest 最后写入作为完成标记。"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            if getattr(self, "is_deepspeed_enabled", False):
                deepspeed_config = getattr(getattr(self, "deepspeed", None), "config", None) or {}
                if deepspeed_config.get("zero_optimization", {}).get("stage") == 3:
                    raise Phase2SaveError(
                        "deepspeed_zero3_composite_unsupported",
                        "ZeRO-3 shards weights; composite saving would be silently wrong",
                    )
            if getattr(self, "is_fsdp_enabled", False):
                raise Phase2SaveError(
                    "fsdp_composite_unsupported",
                    "FSDP shards weights; composite saving would be silently wrong",
                )

            model = self._composite_model()
            adapter_dir = output_dir / "adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(adapter_dir, safe_serialization=True)
            adapter_meta = verify_adapter_keys(adapter_dir, self._llm_targets)

            base = unwrap_base(model)
            merger_meta = save_merger_state(
                base, output_dir / MERGER_STATE_FILENAME, self._merger_names
            )

            processor_dir = output_dir / "processor"
            self.processing_class.save_pretrained(processor_dir)

            optimizer_groups = self._optimizer_group_stats_for_manifest()
            manifest = build_manifest(
                self._manifest_context,
                step=int(self.state.global_step),
                epoch=float(self.state.epoch),
                merger_meta=merger_meta,
                adapter_meta=adapter_meta,
                optimizer_groups=optimizer_groups,
                train_episodes_after_repeat=len(self.train_dataset) if self.train_dataset is not None else 0,
                eval_episodes=len(self.eval_dataset) if self.eval_dataset is not None else None,
            )
            _atomic_write_json(output_dir / TRAINING_MANIFEST_FILENAME, manifest)
        except Phase2TrainError:
            self._record_save_error(output_dir)
            raise
        except Exception as error:  # noqa: BLE001 - stable error surface
            self._record_save_error(output_dir)
            raise Phase2SaveError("checkpoint_save_failed", str(error)) from error

    def _record_save_error(self, output_dir: Path) -> None:
        """Stable save-failure marker; the dir is then never treated as a
        complete checkpoint. 稳定的保存失败标记；该目录不会被当作完整
        checkpoint。"""
        try:
            _atomic_write_json(
                output_dir / SAVE_ERROR_FILENAME,
                {
                    "schema_version": SCHEMA_VERSION,
                    "code": "checkpoint_save_failed",
                    "step": int(getattr(self.state, "global_step", -1)),
                    "completion_marker": "not written",
                },
            )
        except Exception:  # noqa: BLE001 - best effort marker
            pass

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        """Save the full composite checkpoint (adapter + merger + processor +
        manifest). Used for every checkpoint-N dir and the final run root.
        保存完整复合 checkpoint（adapter + merger + processor + manifest）。"""
        if output_dir is None:
            output_dir = self.args.output_dir
        if self.args.should_save:
            self._write_composite(output_dir)
        if self.args.push_to_hub and not _internal_call:
            self.push_to_hub(commit_message="Model save", revision=self.args.hub_revision)

    def finalize_root_checkpoint(self) -> None:
        """Complete the run root so the final composite is also a valid
        resume target: optimizer / scheduler / scaler / RNG / trainer state
        are written at output_dir (same helpers used by _save_checkpoint,
        guarded for API drift). 补全 run root：optimizer/scheduler/scaler/
        RNG/trainer state 写入 output_dir，使最终根目录也可作为 resume 目标。"""
        if not getattr(self.args, "should_save", False):
            return
        output_dir = self.args.output_dir
        for helper in ("_save_optimizer_and_scheduler", "_save_scaler", "_save_rng_state"):
            method = getattr(self, helper, None)
            if method is not None:
                method(output_dir)
        self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))


# ---------------------------------------------------------------------------
# Gradient smoke check (doc 03 section 7 / test item 14).
# ---------------------------------------------------------------------------


def run_gradient_smoke_check(
    trainer: Any, batch: Mapping[str, Any]
) -> dict:
    """Small forward/backward smoke check verifying that LoRA and merger
    parameters both receive non-zero gradients and frozen parameters receive
    none.

    peft initializes lora_B to zeros, so on the very first backward lora_A's
    gradient is exactly zero by design; the check therefore runs one warm-up
    optimizer step first, then performs the verification backward.
    peft 将 lora_B 初始化为零，首次 backward 时 lora_A 梯度按设计为零；
    因此先做一步 warm-up 更新，再做验证性 backward。
    """
    torch = _torch()
    model = trainer.model
    device = next(model.parameters()).device
    prepared = {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in batch.items()
    }
    model.train()

    def _loss() -> Any:
        model.zero_grad(set_to_none=True)
        outputs = model(**prepared)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        loss.backward()
        return loss

    # Warm-up step: makes lora_B non-zero so the verification backward gives
    # lora_A a real gradient. 预热一步，使 lora_B 非零。
    _loss()
    if trainer.optimizer is not None:
        trainer.optimizer.step()
    model.zero_grad(set_to_none=True)
    _loss()

    lora_missing: list[str] = []
    merger_missing: list[str] = []
    frozen_with_grad: list[str] = []
    lora_checked = 0
    merger_checked = 0
    for name, parameter in model.named_parameters():
        logical = _strip_peft_prefix(name)
        if not parameter.requires_grad:
            if parameter.grad is not None:
                frozen_with_grad.append(logical)
            continue
        gradient = parameter.grad
        has_gradient = gradient is not None and bool(torch.count_nonzero(gradient))
        if _has_lora_segment(logical):
            lora_checked += 1
            if not has_gradient:
                lora_missing.append(logical)
        elif _under_any(logical, trainer._merger_names):
            merger_checked += 1
            if not has_gradient:
                merger_missing.append(logical)
        else:
            raise ParameterAuditError(
                "smoke_unclassified_trainable", logical
            )

    model.zero_grad(set_to_none=True)
    summary = {
        "lora_checked": lora_checked,
        "merger_checked": merger_checked,
        "lora_missing_gradient": lora_missing[:10],
        "merger_missing_gradient": merger_missing[:10],
        "frozen_with_grad": frozen_with_grad[:10],
    }
    if lora_missing or merger_missing or frozen_with_grad:
        raise ParameterAuditError(
            "gradient_smoke_failed",
            json.dumps(summary, ensure_ascii=False),
        )
    return summary


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def _build_training_arguments(
    model_args: ModelArguments,
    optim_args: OptimizationArguments,
    ckpt_args: CheckpointArguments,
    lora_args: LoRAArguments,
    use_bf16: bool,
    use_fp16: bool,
    has_eval: bool,
) -> Any:
    """Programmatic TrainingArguments; GPU count / batch / accumulation /
    DeepSpeed / FSDP come from the CLI (or torchrun env), never hardcoded.
    程序化构造 TrainingArguments；GPU 数/批次/accumulation/DeepSpeed/FSDP
    全部来自 CLI 或 torchrun 环境，不硬编码。"""
    transformers_mod = _transformers()
    return transformers_mod.TrainingArguments(
        output_dir=ckpt_args.output_dir,
        per_device_train_batch_size=optim_args.per_device_train_batch_size,
        per_device_eval_batch_size=optim_args.per_device_eval_batch_size,
        gradient_accumulation_steps=optim_args.gradient_accumulation_steps,
        num_train_epochs=optim_args.epochs,
        learning_rate=lora_args.lora_lr,  # informational; real LRs are per-group
        weight_decay=optim_args.weight_decay,  # informational; real WDs are per-group
        warmup_ratio=optim_args.warmup_ratio,
        lr_scheduler_type="cosine",  # informational; a custom scheduler is built
        bf16=use_bf16,
        fp16=use_fp16,
        max_grad_norm=optim_args.max_grad_norm,
        gradient_checkpointing=optim_args.gradient_checkpointing,
        logging_steps=optim_args.logging_steps,
        save_strategy="steps",
        save_steps=ckpt_args.save_steps,
        save_total_limit=ckpt_args.save_total_limit,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=optim_args.eval_steps,
        dataloader_num_workers=optim_args.dataloader_num_workers,
        remove_unused_columns=False,
        report_to="none",
        local_rank=int(os.environ.get("LOCAL_RANK", "-1")),
        seed=optim_args.seed,
        deepspeed=optim_args.deepspeed,
        fsdp=optim_args.fsdp,
        fsdp_config=optim_args.fsdp_config,
        optim="adamw_torch",
    )


def _get_train_sampler(trainer: Any) -> Any:
    """Return Trainer's sampler across supported Transformers seams.
    兼容不同 Transformers seam，返回 Trainer 的 sampler。
    """
    dataloader = getattr(trainer, "_train_dataloader", None)
    if dataloader is None:
        # Transformers 5.x exposes the loader through this method, not a
        # public ``train_dataloader`` property. / Transformers 5.x 通过
        # 该方法提供 loader，而不是公开的 ``train_dataloader`` 属性。
        getter = getattr(trainer, "get_train_dataloader", None)
        if callable(getter):
            dataloader = getter()
        else:
            # Keep compatibility with older Trainer seams used by tests.
            # 兼容测试中使用的旧版 Trainer seam。
            dataloader = getattr(trainer, "train_dataloader", None)
    return getattr(dataloader, "sampler", None)


def _phase2_trainer_class() -> type:
    """Build the concrete Trainer subclass lazily (imports transformers on
    first call). 惰性构造 Trainer 子类（首次调用才 import transformers）。"""
    class _Phase2Trainer(_Phase2TrainerMixin, _transformers().Trainer):
        """Concrete Phase 2 trainer. / Phase 2 具体 Trainer。"""

    return _Phase2Trainer


def main(argv: Sequence[str] | None = None) -> None:
    transformers_mod = _transformers()
    torch = _torch()
    check_runtime()

    parser = transformers_mod.HfArgumentParser(
        (
            ModelArguments,
            DataArguments,
            LoRAArguments,
            AugmentationArguments,
            OptimizationArguments,
            CheckpointArguments,
        )
    )
    (
        model_args,
        data_args,
        lora_args,
        aug_args,
        optim_args,
        ckpt_args,
    ) = parser.parse_args_into_dataclasses(args=list(argv) if argv is not None else None)

    if lora_args.lora_rank < 1 or lora_args.lora_alpha < 1:
        raise ConfigurationError("lora_rank and lora_alpha must be >= 1")
    if not (0.0 <= lora_args.lora_dropout < 1.0):
        raise ConfigurationError(f"lora_dropout must be in [0, 1), got {lora_args.lora_dropout}")
    if lora_args.lora_lr <= 0.0 or lora_args.merger_lr <= 0.0:
        raise ConfigurationError("lora_lr and merger_lr must be positive")
    if data_args.image_min_pixels < 1 or data_args.image_max_pixels < data_args.image_min_pixels:
        raise ConfigurationError("image_min_pixels/image_max_pixels invalid")
    if data_args.max_seq_length < 16:
        raise ConfigurationError(f"max_seq_length too small: {data_args.max_seq_length}")
    if data_args.data_profile not in DATA_PROFILES:
        raise ConfigurationError(f"unknown data profile: {data_args.data_profile!r}")
    if data_args.data_profile == "change_agent" and data_args.repeat_group_key == "task_kind":
        data_args.repeat_group_key = "task"
    if optim_args.smoke_gradients and data_args.max_train_samples is None:
        raise ConfigurationError("--smoke-gradients requires --max-train-samples")
    if optim_args.deepspeed is not None:
        raise ConfigurationError(
            "--deepspeed is not supported: transformers.Trainer's DeepSpeed path "
            "replaces the explicit four-group optimizer (merger vs LoRA LRs), "
            "which violates the dual-LR contract"
        )
    if optim_args.fsdp is not None or optim_args.fsdp_config is not None:
        raise ConfigurationError(
            "--fsdp is not supported: composite checkpoint saving (adapter + "
            "merger) requires full weights on the saving rank; FSDP would "
            "silently write sharded weights"
        )

    train_file = Path(data_args.train_file)
    if not train_file.is_file():
        raise ConfigurationError(f"train file not found: {train_file}")
    train_sha256 = sha256_file(train_file)
    eval_file = Path(data_args.eval_file) if data_args.eval_file else None
    if eval_file is not None and not eval_file.is_file():
        raise ConfigurationError(f"eval file not found: {eval_file}")
    eval_sha256 = sha256_file(eval_file) if eval_file is not None else None
    upstream_manifest = train_file.parent / "manifest.json"
    upstream_sha256 = sha256_file(upstream_manifest) if upstream_manifest.is_file() else None

    image_roots = parse_image_roots(data_args.image_root)
    repeat_weights = parse_repeat_weights(data_args.repeat_weights)
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    if data_args.data_profile == "change_agent" and any(
        flag in {"--aug_enabled", "--aug-enabled"} for flag in raw_args
    ) and aug_args.aug_enabled:
        raise ConfigurationError("CHANGE_PAIR_AUGMENTATION_UNSUPPORTED")
    aug_config = augmentation_config_from_args(aug_args)
    if data_args.data_profile == "change_agent":
        # Temporal pairs must never receive independent legacy augmentation.
        # 时相图对绝不能使用旧的独立单图增强。
        aug_config = replace(aug_config, enabled=False)
    data_module = _load_data_module(data_args.data_profile)
    change_prompt = None
    if data_args.data_profile == "change_agent":
        if not data_args.change_prompt_file:
            raise ConfigurationError("change_agent profile requires --change-prompt-file")
        prompt_path = Path(data_args.change_prompt_file)
        if not prompt_path.is_file():
            raise ConfigurationError("change prompt file not found")
        prompt_text = prompt_path.read_text(encoding="utf-8")
        change_prompt = {"ref": prompt_path.name, "sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()}

    logger.info("loading base model (offline: %s)", model_args.local_files_only)
    model, processor, config = load_base_model(model_args)
    use_bf16 = use_fp16 = False
    if model_args.torch_dtype == "bfloat16":
        use_bf16 = True
    elif model_args.torch_dtype == "float16":
        use_fp16 = True
    elif model_args.torch_dtype == "auto":
        dtype = next(model.parameters()).dtype
        use_bf16 = dtype == torch.bfloat16
        use_fp16 = dtype == torch.float16

    # -- freeze + LoRA + merger unfreeze + audit (fixed order) ---------------
    roots = locate_roots(model)
    llm_targets = enumerate_llm_lora_targets(roots)
    merger_names = enumerate_merger_names(roots)
    expected_deepstack = expected_deepstack_count(config)
    freeze_all(model)
    peft_model = inject_lora(
        model, llm_targets, lora_args.lora_rank, lora_args.lora_alpha,
        lora_args.lora_dropout, lora_args.lora_bias,
    )
    base = unwrap_base(peft_model)
    unfreeze_merger_base(base, merger_names)

    audit = audit_trainable_parameters(
        peft_model, roots, merger_names, llm_targets, expected_deepstack
    )
    Path(ckpt_args.output_dir).mkdir(parents=True, exist_ok=True)
    _atomic_write_json(Path(ckpt_args.output_dir) / PARAMETER_AUDIT_FILENAME, audit)
    checks = audit["checks"]
    logger.info(
        "audit: %d LLM LoRA targets, %d deepstack mergers (declared %s), "
        "trainable %d / %d params (%.4f%%)",
        len(llm_targets), checks["deepstack_found"], checks["deepstack_expected"],
        audit["trainable_parameters"], audit["total_parameters"],
        audit["trainable_fraction"] * 100.0,
    )

    if optim_args.gradient_checkpointing:
        base.enable_input_require_grads()
        base.gradient_checkpointing_enable()

    # -- datasets -----------------------------------------------------------
    image_pixels_applied = _processor_supports_pixel_bounds(processor)
    if not image_pixels_applied:
        logger.warning(
            "pinned processor does not declare min_pixels/max_pixels; image "
            "pixel bounds are recorded in the manifest but NOT applied "
            "(image_pixels_applied=false)"
        )
        dataset_processor = processor
    else:
        dataset_processor = PixelBoundedProcessor(
            processor, data_args.image_min_pixels, data_args.image_max_pixels
        )
    roots_config = data_module.DatasetRootConfig(image_roots)
    train_dataset = data_module.Phase2EpisodeDataset(
        episode_jsonl=str(train_file),
        roots=roots_config,
        processor=dataset_processor,
        aug_config=aug_config,
        max_seq_length=data_args.max_seq_length,
        seed=data_args.aug_seed,
        split="train",
        start_epoch=0,
        **({"prompt_text": prompt_text} if data_args.data_profile == "change_agent" else {}),
    )
    eval_dataset = None
    if eval_file is not None:
        eval_dataset = data_module.Phase2EpisodeDataset(
            episode_jsonl=str(eval_file),
            roots=roots_config,
            processor=dataset_processor,
            aug_config=aug_config,
            max_seq_length=data_args.max_seq_length,
            seed=data_args.aug_seed,
            split="validation",
            start_epoch=0,
            **({"prompt_text": prompt_text} if data_args.data_profile == "change_agent" else {}),
        )
    train_store = data_module.LazyJsonLines(train_file)
    train_wrapped = GroupRepeatDataset(
        base=train_dataset,
        episode_jsonl=str(train_file),
        group_key=data_args.repeat_group_key,
        weights=repeat_weights,
        max_samples=data_args.max_train_samples,
    )
    eval_wrapped = None
    if eval_dataset is not None:
        eval_wrapped = GroupRepeatDataset(
            base=eval_dataset,
            episode_jsonl=str(eval_file),
            group_key=data_args.repeat_group_key,
            weights={},
            max_samples=data_args.max_eval_samples,
        )
    collator = data_module.Phase2DataCollator()
    model_collator = ModelBatchCollator(collator)

    # -- optimizer topology (for manifest + resume validation) ---------------
    _optimizer_groups, optimizer_stats = build_optimizer_groups(
        peft_model, merger_names, lora_args.lora_lr, lora_args.merger_lr,
        optim_args.weight_decay,
    )
    merger_meta = merger_state_meta(base, merger_names)
    context = build_training_context(
        config=config,
        processor=processor,
        model_args=model_args,
        data_args=data_args,
        lora_args=lora_args,
        aug_config=aug_config,
        aug_seed=data_args.aug_seed,
        optimizer_stats=optimizer_stats,
        merger_names=merger_names,
        merger_meta=merger_meta,
        llm_targets=llm_targets,
        image_pixels_applied=image_pixels_applied,
        train_sha256=train_sha256,
        eval_sha256=eval_sha256,
        train_upstream_manifest_sha256=upstream_sha256,
        train_episode_count=len(train_store),
        eval_episode_count=len(eval_wrapped) if eval_wrapped is not None else None,
        group_counts=train_wrapped.group_counts(),
        repeat_weights=repeat_weights,
        image_sources=list(image_roots.keys()),
        repo_root=Path(__file__).resolve().parents[1],
        training_profile=data_args.data_profile,
        change_prompt=change_prompt,
    )

    # -- resume target resolution + validation -------------------------------
    resume_dir = resolve_resume_target(ckpt_args.output_dir, ckpt_args.resume_from_checkpoint)
    if resume_dir is not None:
        validate_resume_checkpoint(resume_dir, context)
        load_composite_weights(peft_model, base, resume_dir, merger_names)
        logger.info("resuming from validated checkpoint: %s", resume_dir)

    training_arguments = _build_training_arguments(
        model_args, optim_args, ckpt_args, lora_args, use_bf16, use_fp16,
        eval_wrapped is not None,
    )

    def _sampler_provider() -> Any:
        return _get_train_sampler(trainer)

    trainer = _phase2_trainer_class()(
        model=peft_model,
        args=training_arguments,
        data_collator=model_collator,
        train_dataset=train_wrapped,
        eval_dataset=eval_wrapped,
        processing_class=processor,
        callbacks=[_make_epoch_sync_callback(train_wrapped, sampler_provider=_sampler_provider)],
        merger_lr=lora_args.merger_lr,
        lora_lr=lora_args.lora_lr,
        weight_decay=optim_args.weight_decay,
        warmup_ratio=optim_args.warmup_ratio,
        merger_names=merger_names,
        llm_targets=llm_targets,
        manifest_context=context,
    )

    # -- preflight ------------------------------------------------------------
    if data_args.preflight_limit != 0:
        counts = trainer.train_dataset.preflight(
            None if data_args.preflight_limit < 0 else data_args.preflight_limit
        )
        logger.info("preflight: %s", json.dumps(counts, ensure_ascii=False))
        if counts.get("episode_too_long", 0):
            logger.warning(
                "%d episode(s) exceed max_seq_length as a single turn pair; "
                "raise --max-seq-length or filter the episodes",
                counts["episode_too_long"],
            )

    # -- gradient smoke check -------------------------------------------------
    if optim_args.smoke_gradients:
        smoke_batch, _meta = collator(
            [trainer.train_dataset[index] for index in range(min(2, len(trainer.train_dataset)))]
        )
        smoke_summary = run_gradient_smoke_check(trainer, smoke_batch)
        _atomic_write_json(
            Path(ckpt_args.output_dir) / "smoke_gradients.json", smoke_summary
        )
        logger.info("gradient smoke check passed: %s", json.dumps(smoke_summary, ensure_ascii=False))

    # -- train ----------------------------------------------------------------
    trainer.train(resume_from_checkpoint=resume_dir)
    trainer.save_model()
    trainer.finalize_root_checkpoint()
    logger.info("training finished; final composite checkpoint at %s", ckpt_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()


