#!/usr/bin/env python3
"""LLaMA-Factory LoRA fine-tuning entry for remote-sensing VLM models.
LLaMA-Factory LoRA 微调入口：支持遥感多模态大模型（InternVL3.5-8B / Qwen3-VL-4B）。

Features / 功能:
- LoRA SFT via ``llamafactory-cli``; dataset registration and train YAML are
  generated at runtime under ``.cache/llamafactory/``.
  通过 llamafactory-cli 运行 LoRA SFT，数据集注册与训练 YAML 均在运行时生成。
- Auto-resume from the latest checkpoint; completed runs skip training.
  自动从最新 checkpoint 断点续训；已完成的训练直接跳过。
- Export the best LoRA adapter and the merged base+LoRA full model.
  导出表现最佳的 LoRA 适配器以及基座+LoRA 合并后的完整模型。
- Plot training curves by reusing ``scripts/plot_train_curves.py``.
  复用 scripts/plot_train_curves.py 绘制训练曲线。

Usage / 用法示例:
  python scripts/finetune_vlm_lora.py --model internvl3.5-8b
  python scripts/finetune_vlm_lora.py --model qwen3-vl-4b \
      --model-name-or-path /path/to/Qwen3-VL-4B-Instruct
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "微调数据集" / "merged"
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache" / "llamafactory"
PLOT_SCRIPT = PROJECT_ROOT / "scripts" / "plot_train_curves.py"
STATE_FILE = ".finetune_state.json"

# Accepted CLI names for each model family.
# 每个模型家族接受的命令行名称。
MODEL_ALIASES = {
    "internvl3.5-8b": "internvl3.5-8b",
    "internvl3_5-8b": "internvl3.5-8b",
    "internvl3-8b": "internvl3.5-8b",
    "qwen3-vl-4b": "qwen3-vl-4b",
    "qwen3-vl-4b-instruct": "qwen3-vl-4b",
    "qwen3vl-4b": "qwen3-vl-4b",
}

# Per-model LLaMA-Factory defaults. Paths are relative to the repo root and
# can be overridden with --model-name-or-path.
# 每个模型的 LLaMA-Factory 默认配置；路径相对于仓库根目录，可用参数覆盖。
MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "internvl3.5-8b": {
        "default_model_path": "models/InternVL3_5-8B",
        "template": "intern_vl",
        "trust_remote_code": True,
        "lora_target": "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
        "freeze_vision_tower": True,
        "freeze_multi_modal_projector": False,
        "server_allowed": True,
    },
    "qwen3-vl-4b": {
        "default_model_path": "Qwen/Qwen3-VL-4B-Instruct",
        "template": "qwen3_vl",
        "trust_remote_code": False,
        "lora_target": "all",
        "freeze_vision_tower": True,
        "freeze_multi_modal_projector": True,
        "server_allowed": False,
    },
}

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def local_now() -> str:
    """Return the current local time in ISO format for the run state.
    返回用于运行状态的本地 ISO 时间。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def normalize_model_name(raw: str) -> str:
    """Normalize a user-supplied model name to a canonical key.
    将用户输入的模型名归一化为规范键。"""
    key = raw.strip().lower()
    if key not in MODEL_ALIASES:
        choices = ", ".join(sorted(MODEL_ALIASES))
        raise argparse.ArgumentTypeError(
            f"unknown model {raw!r}; choose from {choices}  未知模型，可选值：{choices}"
        )
    return MODEL_ALIASES[key]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments; ``argv`` is injectable for tests.
    解析命令行参数；argv 可注入以便测试。"""
    parser = argparse.ArgumentParser(
        description=(
            "LLaMA-Factory LoRA fine-tune, best-checkpoint export and curves. "
            "LLaMA-Factory LoRA 微调、最佳 checkpoint 导出与训练曲线。"
        )
    )
    parser.add_argument(
        "--model",
        type=normalize_model_name,
        required=True,
        help="internvl3.5-8b or qwen3-vl-4b (aliases accepted) 模型名称",
    )
    parser.add_argument(
        "--model-name-or-path",
        default=None,
        help="base model path or Hugging Face id; default per model profile 基座模型路径或 HF 模型 ID",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="merged ShareGPT dataset directory 合并后的 ShareGPT 数据集目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="training/export output directory 训练与导出输出目录",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="runtime config cache directory 运行时配置缓存目录",
    )
    parser.add_argument(
        "--llamafactory-cli",
        default=None,
        help="llamafactory-cli command; auto-detect when omitted llamafactory-cli 命令，缺省时自动探测",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="limit samples for a smoke run; omit for the real run 冒烟测试样本上限，正式运行不要传",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from the latest checkpoint; auto-enabled when artifacts exist 从最新 checkpoint 断点续训",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="start fresh; refuses a non-empty output dir (no automatic deletion) 从头训练；拒绝非空输出目录（不自动删除）",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="skip training and only export/plot from existing outputs 跳过训练，仅基于已有产物导出/绘图",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="skip best-LoRA and merged-model export 跳过最佳 LoRA 与合并模型导出",
    )
    parser.add_argument(
        "--skip-curves",
        action="store_true",
        help="skip training-curve plotting 跳过训练曲线绘制",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help=(
            "run on the dedicated InternVL-only server; qwen3-vl-4b is rejected here "
            "在专用 InternVL 服务器上运行；此处拒绝 qwen3-vl-4b"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="generate configs and print the plan without running anything 只生成配置并打印计划，不执行训练/导出",
    )
    parser.add_argument(
        "--metric",
        default="eval_loss",
        help="metric name in trainer_log.jsonl used to pick the best checkpoint 选择最佳 checkpoint 的指标名",
    )
    parser.add_argument(
        "--higher-is-better",
        action="store_true",
        help="treat the selected metric as higher-is-better (e.g. predict_bleu-4) 所选指标越大越好",
    )

    # Common training hyperparameters. 常用训练超参数。
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target",
        default=None,
        help="LoRA target modules; default comes from the model profile LoRA 目标模块，缺省使用模型默认",
    )
    parser.add_argument("--cutoff-len", type=int, default=2048)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--extra-yaml",
        type=Path,
        default=None,
        help=(
            "extra LLaMA-Factory YAML merged into the train config "
            "合并进训练配置的附加 LLaMA-Factory YAML"
        ),
    )
    return parser.parse_args(argv)


def resolve_model_ref(ref: str) -> str:
    """Resolve a repo-relative local path; otherwise keep the original string.
    若指向仓库内本地模型目录则转为绝对路径；否则保留原字符串（HF ID 或服务器绝对路径）。"""
    candidate = PROJECT_ROOT / ref
    if candidate.exists():
        return str(candidate.resolve())
    absolute = Path(ref).expanduser()
    if absolute.is_absolute() and absolute.exists():
        return str(absolute.resolve())
    return ref


def build_fingerprint(model_ref: str, data_dir: Path, args: argparse.Namespace) -> str:
    """Build a fingerprint of the run so mismatched resumes fail early.
    生成运行指纹，配置不一致时拒绝续训。"""
    payload = {
        "model_ref": model_ref,
        "data_dir": str(data_dir),
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_target": args.lora_target,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "cutoff_len": args.cutoff_len,
        "max_samples": args.max_samples,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def load_state(output_dir: Path) -> dict[str, Any]:
    """Load the run state JSON; return {} when absent.
    读取运行状态 JSON；不存在时返回空字典。"""
    path = output_dir / STATE_FILE
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_state(output_dir: Path, state: dict[str, Any]) -> Path:
    """Atomically persist the run state JSON.
    原子化保存运行状态 JSON。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / STATE_FILE
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)
    return path


def build_dataset_info(data_dir: Path, cache_dir: Path) -> Path:
    """Generate LLaMA-Factory dataset_info.json for the merged ShareGPT splits.
    为合并 ShareGPT 数据切分生成 LLaMA-Factory dataset_info.json。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {}
    for split, file_name in (
        ("train", "train.json"),
        ("val", "val.json"),
        ("test", "test.json"),
    ):
        data_file = data_dir / file_name
        if not data_file.is_file():
            if split == "train":
                raise FileNotFoundError(
                    f"training dataset not found: {data_file}  未找到训练数据集"
                )
            continue
        info[f"merged_{split}"] = {
            "file_name": str(data_file.resolve()),
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images"},
            "media_dir": str(data_dir.resolve()),
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }
    out_path = cache_dir / "dataset_info.json"
    out_path.write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out_path


def latest_checkpoint(output_dir: Path) -> Optional[Path]:
    """Return the checkpoint directory with the largest step, or None.
    返回 step 最大的 checkpoint 目录；不存在时返回 None。"""
    if not output_dir.is_dir():
        return None
    found: list[tuple[int, Path]] = []
    for child in output_dir.iterdir():
        match = _CHECKPOINT_RE.match(child.name)
        if match and child.is_dir():
            found.append((int(match.group(1)), child))
    if not found:
        return None
    return max(found, key=lambda item: item[0])[1]


def training_completed(output_dir: Path, state: dict[str, Any]) -> bool:
    """Decide whether training is already finished.
    判断训练是否已经完成。"""
    return bool(state.get("train_completed")) or (output_dir / "all_results.json").is_file()


def build_train_config(
    profile: dict[str, Any],
    model_ref: str,
    data_dir: Path,
    dataset_info_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    resume_checkpoint: Optional[Path],
) -> dict[str, Any]:
    """Build the LLaMA-Factory SFT+LoRA training config dict.
    构建 LLaMA-Factory SFT+LoRA 训练配置字典。"""
    has_val = (data_dir / "val.json").is_file()
    config: dict[str, Any] = {
        "model_name_or_path": model_ref,
        "template": profile["template"],
        "trust_remote_code": profile["trust_remote_code"],
        "stage": "sft",
        "finetuning_type": "lora",
        "lora_target": args.lora_target or profile["lora_target"],
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "freeze_vision_tower": profile["freeze_vision_tower"],
        "freeze_multi_modal_projector": profile["freeze_multi_modal_projector"],
        "dataset": "merged_train",
        "dataset_dir": str(dataset_info_path.parent.resolve()),
        "media_dir": str(data_dir.resolve()),
        "cutoff_len": args.cutoff_len,
        "packing": False,
        "do_train": True,
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "gradient_checkpointing": True,
        "bf16": True,
        "optim": "adamw_torch",
        "report_to": "tensorboard",
        "seed": args.seed,
        "output_dir": str(output_dir.resolve()),
    }
    if has_val:
        config.update(
            {
                "do_eval": True,
                "eval_dataset": "merged_val",
                "per_device_eval_batch_size": args.per_device_batch_size,
                "eval_strategy": "steps",
                "eval_steps": args.eval_steps,
                "predict_with_generate": False,
            }
        )
    if args.max_samples is not None:
        config["max_samples"] = args.max_samples
    if resume_checkpoint is not None:
        config["resume_from_checkpoint"] = str(resume_checkpoint.resolve())
    if args.extra_yaml is not None:
        extra = yaml.safe_load(args.extra_yaml.read_text(encoding="utf-8")) or {}
        config.update(extra)
    return config


def write_yaml(config: dict[str, Any], path: Path) -> Path:
    """Write a YAML config file.
    写入 YAML 配置文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def detect_cli(override: Optional[str]) -> tuple[list[str], str]:
    """Detect the LLaMA-Factory CLI command.
    探测 LLaMA-Factory CLI 命令。"""
    if override:
        return shlex.split(override), override
    exe = shutil.which("llamafactory-cli")
    if exe:
        return [exe], exe
    fallback = [sys.executable, "-m", "llamafactory.cli"]
    return fallback, " ".join(fallback)


def run_cli(cli: list[str], subcommand: str, config_path: Path, dry_run: bool = False) -> None:
    """Run a LLaMA-Factory CLI subcommand with a generated YAML config.
    使用生成的 YAML 配置运行 LLaMA-Factory CLI 子命令。"""
    cmd = [*cli, subcommand, str(config_path)]
    print(f"[finetune] run: {' '.join(cmd)}")
    if dry_run:
        return
    try:
        proc = subprocess.run(cmd, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"LLaMA-Factory CLI not found ({exc}); install it with: pip install llamafactory "
            "未找到 LLaMA-Factory CLI，请先 pip install llamafactory"
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(
            f"llamafactory-cli {subcommand} failed with exit code {proc.returncode}; "
            "check the command output above  命令失败，请查看上方输出"
        )


def _load_log_rows(log_path: Path) -> list[dict[str, Any]]:
    """Load trainer_log.jsonl into a list of dicts.
    将 trainer_log.jsonl 读取为字典列表。"""
    rows: list[dict[str, Any]] = []
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _metric_key(metric: str) -> str:
    """Resolve a user metric name to a trainer_log.jsonl key.
    将用户指标名解析为 trainer_log.jsonl 中的键名。"""
    if metric.startswith(("eval_", "predict_")) or metric == "loss":
        return metric
    return f"eval_{metric}"


def select_best_checkpoint(
    output_dir: Path,
    metric: str,
    higher_is_better: bool,
) -> dict[str, Any]:
    """Select the best checkpoint from trainer_log.jsonl.
    从 trainer_log.jsonl 中选择表现最佳的 checkpoint。

    Prefers the requested eval metric; falls back to the lowest train loss,
    then the latest checkpoint, then the final adapter at the output root.
    优先使用指定验证指标；依次回退到最低训练损失、最新 checkpoint、输出根目录的最终适配器。
    """
    log_path = output_dir / "trainer_log.jsonl"
    rows = _load_log_rows(log_path) if log_path.is_file() else []
    metric_key = _metric_key(metric)
    candidates: list[tuple[int, float]] = []
    for row in rows:
        step = row.get("current_steps", row.get("step"))
        value = row.get(metric_key)
        if step is None or value is None:
            continue
        candidates.append((int(step), float(value)))

    source = metric_key
    if not candidates and metric_key != "loss":
        metric_key = "loss"
        for row in rows:
            step = row.get("current_steps", row.get("step"))
            value = row.get(metric_key)
            if step is None or value is None:
                continue
            candidates.append((int(step), float(value)))
        source = "train_loss"

    if candidates:
        use_min = metric_key == "loss" or not higher_is_better
        chosen_step, chosen_value = (
            min(candidates, key=lambda item: item[1])
            if use_min
            else max(candidates, key=lambda item: item[1])
        )
        checkpoint = output_dir / f"checkpoint-{chosen_step}"
        if checkpoint.is_dir():
            return {
                "checkpoint": checkpoint,
                "step": chosen_step,
                "value": chosen_value,
                "metric": source,
            }
        print(
            f"WARNING: best checkpoint {checkpoint.name} was pruned; using the latest one. "
            "最佳 checkpoint 已被清理，改用最新 checkpoint。",
            file=sys.stderr,
        )
        latest = latest_checkpoint(output_dir)
        if latest is not None:
            match = _CHECKPOINT_RE.match(latest.name)
            return {
                "checkpoint": latest,
                "step": int(match.group(1)) if match else None,
                "value": chosen_value,
                "metric": f"{source}->latest",
            }

    latest = latest_checkpoint(output_dir)
    if latest is not None:
        match = _CHECKPOINT_RE.match(latest.name)
        return {
            "checkpoint": latest,
            "step": int(match.group(1)) if match else None,
            "value": None,
            "metric": "latest",
        }

    # LLaMA-Factory also writes the final adapter at the output root.
    # LLaMA-Factory 训练结束后还会在输出根目录写入最终适配器。
    if (output_dir / "adapter_config.json").is_file():
        return {
            "checkpoint": output_dir,
            "step": None,
            "value": None,
            "metric": "final_adapter",
        }
    return {"checkpoint": None, "step": None, "value": None, "metric": source}


def copy_adapter_files(src: Path, dst: Path) -> None:
    """Copy only adapter weights (not optimizer/scheduler state) to dst.
    仅复制适配器权重（不含优化器/调度器状态）到目标目录。"""
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for pattern in ("adapter_config.json", "adapter_model*"):
        for src_file in src.glob(pattern):
            if src_file.is_file():
                shutil.copy2(src_file, dst / src_file.name)
                copied += 1
    if copied == 0:
        raise FileNotFoundError(
            f"no adapter weights found in {src}  未在 {src} 找到适配器权重"
        )


def build_export_config(
    profile: dict[str, Any],
    model_ref: str,
    adapter_path: Path,
    export_dir: Path,
) -> dict[str, Any]:
    """Build the LLaMA-Factory merge/export config dict.
    构建 LLaMA-Factory 合并导出配置字典。"""
    return {
        "model_name_or_path": model_ref,
        "adapter_name_or_path": str(adapter_path.resolve()),
        "template": profile["template"],
        "trust_remote_code": profile["trust_remote_code"],
        "finetuning_type": "lora",
        "export_dir": str(export_dir.resolve()),
        "export_size": 5,
        "export_device": "cpu",
        "export_legacy_format": False,
    }


def plot_curves(output_dir: Path) -> tuple[bool, str]:
    """Plot training curves with the existing plot script.
    复用现有绘图脚本绘制训练曲线。"""
    out_png = output_dir / "train_curves.png"
    try:
        subprocess.run(
            [sys.executable, str(PLOT_SCRIPT), str(output_dir), "--out", str(out_png)],
            check=True,
        )
        return True, str(out_png)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return (
            False,
            f"failed to plot curves: {exc}; install matplotlib and rerun "
            "训练曲线绘制失败；请安装 matplotlib 后重试",
        )


def probe_tensorboard() -> None:
    """Warn early when tensorboard is missing for report_to=tensorboard.
    在 report_to=tensorboard 需要 tensorboard 时提前告警。"""
    if importlib.util.find_spec("tensorboard") is None:
        print(
            "WARNING: tensorboard is not importable; training with report_to=tensorboard "
            "will fail unless it is installed (pip install tensorboard). "
            "未检测到 tensorboard，report_to=tensorboard 会报错，请先安装。",
            file=sys.stderr,
        )


def print_config(name: str, config: dict[str, Any]) -> None:
    """Pretty-print a generated config for --dry-run.
    在 --dry-run 下友好打印生成的配置。"""
    print(f"===== {name} =====")
    print(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), end="")


def main(argv: Optional[list[str]] = None) -> int:
    """Orchestrate fine-tune, resume, best-export and curves.
    编排微调、断点续训、最佳导出与曲线绘制流程。"""
    args = parse_args(argv)
    model_key = normalize_model_name(args.model)
    profile = MODEL_PROFILES[model_key]

    if args.server and not profile["server_allowed"]:
        print(
            "ERROR: the dedicated server only fine-tunes InternVL; qwen3-vl-4b is rejected "
            "with --server.  专用服务器只负责 InternVL 微调，qwen3-vl-4b 不允许在服务器上运行。",
            file=sys.stderr,
        )
        return 2

    model_ref = resolve_model_ref(args.model_name_or_path or profile["default_model_path"])
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = (
        args.output_dir
        or PROJECT_ROOT / "outputs" / "finetune" / f"{model_key}_lora"
    ).expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()

    fingerprint = build_fingerprint(model_ref, data_dir, args)
    state = load_state(output_dir)
    if state and state.get("fingerprint") not in (None, fingerprint):
        print(
            "ERROR: existing run state does not match the current arguments (model/data/"
            "hyperparameters). Use a new --output-dir or move the existing one manually. "
            "已有运行状态与当前参数不一致（模型/数据/超参数）；请换输出目录或手动移走旧目录。",
            file=sys.stderr,
        )
        return 2
    state.setdefault("model_key", model_key)
    state.setdefault("model_ref", model_ref)
    state.setdefault("data_dir", str(data_dir))
    state.setdefault("fingerprint", fingerprint)

    completed = training_completed(output_dir, state)
    latest = latest_checkpoint(output_dir)
    trainer_log = output_dir / "trainer_log.jsonl"
    artifacts_exist = latest is not None or trainer_log.is_file()

    if args.force_restart:
        if output_dir.is_dir() and any(output_dir.iterdir()):
            print(
                "ERROR: --force-restart requires an empty output dir; move or remove existing "
                "files manually (no automatic deletion). "
                "--force-restart 要求输出目录为空；请手动移走或删除已有文件（脚本不会自动删除）。",
                file=sys.stderr,
            )
            return 2
        state = {
            "model_key": model_key,
            "model_ref": model_ref,
            "data_dir": str(data_dir),
            "fingerprint": fingerprint,
            "train_completed": False,
            "updated_at": local_now(),
        }
        save_state(output_dir, state)
        completed = False
        latest = None
        artifacts_exist = False

    resume_checkpoint: Optional[Path] = None
    need_train = not args.skip_train and not completed
    if need_train and (args.resume or artifacts_exist):
        if latest is not None:
            resume_checkpoint = latest
            print(f"[finetune] resume from checkpoint: {latest}  从 checkpoint 断点续训")
        else:
            print(
                "WARNING: no checkpoint found to resume; starting a fresh run. "
                "未找到可续训的 checkpoint，从头开始训练。",
                file=sys.stderr,
            )

    if args.dry_run:
        if need_train:
            dataset_info_path = build_dataset_info(data_dir, cache_dir)
            train_config = build_train_config(
                profile,
                model_ref,
                data_dir,
                dataset_info_path,
                output_dir,
                args,
                resume_checkpoint,
            )
            print_config(f"train_{model_key}.yaml", train_config)
            print_config("dataset_info.json", json.loads(dataset_info_path.read_text(encoding="utf-8")))
        else:
            print("[finetune] dry-run: training already completed; training will be skipped.")
        best = select_best_checkpoint(output_dir, args.metric, args.higher_is_better)
        if not args.skip_export and best["checkpoint"] is not None:
            export_config = build_export_config(
                profile, model_ref, best["checkpoint"], output_dir / "merged"
            )
            print_config("export_merged.yaml", export_config)
            print(f"[finetune] planned best LoRA -> {output_dir / 'best_lora'}")
        print(f"[finetune] planned output dir: {output_dir}")
        return 0

    cli, cli_source = detect_cli(args.llamafactory_cli)
    print(f"[finetune] LLaMA-Factory CLI: {cli_source}")

    if need_train:
        if not (data_dir / "train.json").is_file():
            print(
                f"ERROR: training dataset not found: {data_dir / 'train.json'}  未找到训练数据集",
                file=sys.stderr,
            )
            return 2
        probe_tensorboard()
        dataset_info_path = build_dataset_info(data_dir, cache_dir)
        train_config = build_train_config(
            profile,
            model_ref,
            data_dir,
            dataset_info_path,
            output_dir,
            args,
            resume_checkpoint,
        )
        train_yaml = write_yaml(train_config, cache_dir / f"train_{model_key}.yaml")
        print(f"[finetune] train config: {train_yaml}")
        try:
            run_cli(cli, "train", train_yaml)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 3
        state["train_completed"] = True
        state["updated_at"] = local_now()
        save_state(output_dir, state)
    elif args.skip_train:
        print("[finetune] --skip-train; skipping training.  已跳过训练。")
        if completed:
            state["train_completed"] = True
            state["updated_at"] = local_now()
            save_state(output_dir, state)
    else:
        print("[finetune] training already completed; skipping training.  训练已完成，跳过。")
        state["train_completed"] = True
        state["updated_at"] = local_now()
        save_state(output_dir, state)

    if args.skip_export:
        print("[finetune] --skip-export; not exporting LoRA / merged model.  已跳过导出。")
    else:
        best = select_best_checkpoint(output_dir, args.metric, args.higher_is_better)
        if best["checkpoint"] is None:
            print(
                "ERROR: no LoRA checkpoint found to export; run training first or check the "
                "output dir.  未找到可导出的 LoRA checkpoint；请先训练或检查输出目录。",
                file=sys.stderr,
            )
            return 3
        state.update(
            {
                "best_checkpoint": str(best["checkpoint"].resolve()),
                "best_step": best["step"],
                "best_metric": best["metric"],
                "best_value": best["value"],
            }
        )
        best_key = str(best["checkpoint"].resolve())
        best_lora_dir = output_dir / "best_lora"
        if state.get("lora_exported_checkpoint") != best_key or not (
            best_lora_dir / "adapter_config.json"
        ).is_file():
            copy_adapter_files(best["checkpoint"], best_lora_dir)
            state["lora_exported_checkpoint"] = best_key
            print(f"[finetune] exported best LoRA -> {best_lora_dir}  已导出最佳 LoRA")
        else:
            print("[finetune] best LoRA already exported; skipping.  最佳 LoRA 已导出，跳过。")

        merged_dir = output_dir / "merged"
        if state.get("merged_exported_checkpoint") != best_key or not (
            merged_dir / "config.json"
        ).is_file():
            if merged_dir.is_dir() and any(merged_dir.iterdir()):
                print(
                    "ERROR: merged dir already exists from another step; move or remove it "
                    "manually, then rerun (no automatic deletion). "
                    "合并目录已存在且来自其他 checkpoint；请手动移走或删除后重试。",
                    file=sys.stderr,
                )
                return 3
            export_config = build_export_config(profile, model_ref, best["checkpoint"], merged_dir)
            export_yaml = write_yaml(export_config, cache_dir / f"export_{model_key}.yaml")
            print(f"[finetune] export config: {export_yaml}")
            try:
                run_cli(cli, "export", export_yaml)
            except RuntimeError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 3
            state["merged_exported_checkpoint"] = best_key
            print(f"[finetune] exported merged model -> {merged_dir}  已导出合并模型")
        else:
            print("[finetune] merged model already exported; skipping.  合并模型已导出，跳过。")
        state["updated_at"] = local_now()
        save_state(output_dir, state)

    if args.skip_curves:
        print("[finetune] --skip-curves; skipping curve plotting.  已跳过曲线绘制。")
    else:
        ok, message = plot_curves(output_dir)
        state["curves_plotted"] = ok
        state["updated_at"] = local_now()
        save_state(output_dir, state)
        if ok:
            print(f"[finetune] training curves -> {message}  训练曲线已导出")
        else:
            print(f"WARNING: {message}", file=sys.stderr)

    print("\n===== summary 汇总 =====")
    print(f"model: {model_key} ({model_ref})")
    print(f"output dir: {output_dir}")
    print(f"state file: {output_dir / STATE_FILE}")
    if not args.skip_export:
        print(f"best checkpoint: {state.get('best_checkpoint')} (metric={state.get('best_metric')}, value={state.get('best_value')})")
        print(f"best LoRA: {output_dir / 'best_lora'}")
        print(f"merged model: {output_dir / 'merged'}")
    if not args.skip_curves:
        print(f"curves: {output_dir / 'train_curves.png'} (plotted={state.get('curves_plotted')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
