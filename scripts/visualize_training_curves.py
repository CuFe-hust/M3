#!/usr/bin/env python3
"""Visualize Qwen3-VL phase-2 SFT training curves from a transformers TrainerState.

Primary data source is the authoritative `trainer_state.json` written by the
Trainer (per-step loss / learning rate / grad norm, plus eval metrics). The raw
training log is optional and only used to print the run timeline (start,
preflight, audit, finish) to the console, since its tqdm progress bars carry no
loss values.

Usage:
  python scripts/visualize_training_curves.py \
      --state local_outputs/trainer_state.json \
      --log local_outputs/qwen3-vl-8b-phase2-20260814.log \
      --output local_outputs/training_curves.png \
      --csv local_outputs/training_curves.csv

Dependencies: stdlib + matplotlib only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def load_state(path: Path) -> dict:
    """Load a TrainerState JSON file."""
    # 加载 TrainerState JSON 文件。
    with path.open("r", encoding="utf-8") as fh:
        state = json.load(fh)
    return state


def load_log_events(path: Path | None) -> list[str]:
    """Extract `INFO finetune_qwen3vl_phase2:` timeline lines from the raw log."""
    # 从原始日志提取 INFO 时间线（进度条中不含 loss 值，仅作事件参考）。
    if path is None:
        return []
    events: list[str] = []
    pattern = re.compile(r"INFO finetune_qwen3vl_phase2: (.*)")
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                match = pattern.search(line)
                if match:
                    events.append(match.group(1).strip())
    except OSError as exc:
        print(f"[warn] cannot read log {path}: {exc}", file=sys.stderr)
    return events


def extract_series(state: dict) -> dict:
    """Split log_history into train rows and eval rows, deduped by step."""
    # 将 log_history 拆分为训练行与 eval 行，并按 step 去重（保留每步最后一条）。
    train_rows: dict[int, dict] = {}
    eval_rows: dict[int, dict] = {}
    for entry in state.get("log_history", []):
        step = entry.get("step")
        if step is None:
            continue
        if "eval_loss" in entry:
            eval_rows[step] = entry
        elif "loss" in entry:
            train_rows[step] = entry
    return {
        "train": [train_rows[s] for s in sorted(train_rows)],
        "eval": [eval_rows[s] for s in sorted(eval_rows)],
    }


def moving_average(values: list[float], window: int) -> list[float]:
    """Simple centered moving average; returns None (skipped) near edges."""
    # 简单滑动平均，窗口边缘位置返回 NaN 以便绘图跳过。
    if window <= 1 or len(values) < window:
        return list(values)
    half = window // 2
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #


def plot_curves(state: dict, series: dict, output: Path, title: str | None) -> None:
    """Render the 2x2 training-curve figure."""
    # 绘制 2x2 训练曲线图。
    train = series["train"]
    eval_rows = series["eval"]

    max_steps = state.get("max_steps") or (train[-1]["step"] if train else 0)
    num_epochs = state.get("num_train_epochs") or state.get("epoch") or 1.0
    epochs_per_step = num_epochs / max_steps if max_steps else 0.0

    def step_to_epoch(step: int) -> float:
        return step * epochs_per_step

    steps = [r["step"] for r in train]
    loss = [r["loss"] for r in train]
    lr = [r.get("learning_rate") for r in train]
    grad_norm = [r.get("grad_norm") for r in train]
    eval_steps = [r["step"] for r in eval_rows]
    eval_loss = [r["eval_loss"] for r in eval_rows]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    fig.suptitle(
        title or f"Qwen3-VL-8B Phase-2 SFT  ({max_steps} steps, {num_epochs:g} epochs)",
        fontsize=15,
        fontweight="bold",
    )

    # --- Panel 1: train loss + eval loss ------------------------------- #
    ax = axes[0, 0]
    ax.plot(steps, loss, color="#4C72B0", lw=1.0, alpha=0.55, label="train loss")
    smooth = moving_average(loss, 21)
    ax.plot(steps, smooth, color="#4C72B0", lw=2.0, label="train loss (smoothed, w=21)")
    ax.plot(
        eval_steps,
        eval_loss,
        marker="o",
        ms=5,
        ls="--",
        lw=1.2,
        color="#C44E52",
        label="eval loss",
    )
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training / Eval Loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # secondary epoch axis on top
    # 顶部附加 epoch 副坐标轴。
    ax2 = ax.twiny()
    epoch_ticks = [e / 4 * num_epochs for e in range(5)]
    ax2.set_xticks([int(e / epochs_per_step) for e in epoch_ticks])
    ax2.set_xticklabels([f"{e:g}" for e in epoch_ticks])
    ax2.set_xlabel("epoch")

    # --- Panel 2: learning rate ---------------------------------------- #
    ax = axes[0, 1]
    ax.plot(steps, lr, color="#55A868", lw=1.8)
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- Panel 3: gradient norm ---------------------------------------- #
    ax = axes[1, 0]
    if grad_norm:
        ax.plot(steps, grad_norm, color="#8172B2", lw=1.0, alpha=0.6)
        ax.plot(
            steps,
            moving_average(grad_norm, 21),
            color="#8172B2",
            lw=1.8,
            label="grad norm (smoothed, w=21)",
        )
        ax.legend(loc="upper right")
    ax.set_xlabel("step")
    ax.set_ylabel("grad norm")
    ax.set_title("Gradient Norm")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # --- Panel 4: eval loss zoom --------------------------------------- #
    ax = axes[1, 1]
    ax.plot(eval_steps, eval_loss, marker="o", ms=6, lw=1.8, color="#C44E52")
    for s, v in zip(eval_steps, eval_loss):
        ax.annotate(
            f"{v:.4f}",
            (s, v),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7.5,
        )
    ax.set_xlabel("step")
    ax.set_ylabel("eval loss")
    ax.set_title("Eval Loss (every eval step)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.savefig(output, dpi=150)
    print(f"figure saved -> {output}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Qwen3-VL phase-2 SFT training curves from trainer_state.json."
    )
    parser.add_argument(
        "--state",
        type=Path,
        required=True,
        help="path to trainer_state.json",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="optional raw training log for the run timeline (not for curves)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output PNG path (default: <state_dir>/training_curves.png)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="optional CSV export of the curves",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="optional figure title",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=21,
        help="moving-average window for loss/grad norm (default: 21)",
    )
    return parser.parse_args(argv)


def export_csv(series: dict, path: Path) -> None:
    """Export step/epoch/loss/lr/grad_norm/eval_loss to a CSV file."""
    # 将曲线数据导出为 CSV，便于后续自行分析。
    import csv

    train = series["train"]
    eval_by_step = {r["step"]: r["eval_loss"] for r in series["eval"]}
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["step", "epoch", "loss", "learning_rate", "grad_norm", "eval_loss"]
        )
        for row in train:
            epoch = row.get("epoch", "")
            if isinstance(epoch, (int, float)):
                epoch = f"{epoch:g}"
            writer.writerow(
                [
                    row["step"],
                    epoch,
                    row.get("loss", ""),
                    row.get("learning_rate", ""),
                    row.get("grad_norm", ""),
                    eval_by_step.get(row["step"], ""),
                ]
            )
    print(f"csv saved -> {path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    state = load_state(args.state)
    series = extract_series(state)

    if not series["train"]:
        print("error: no per-step training rows (loss) found in log_history", file=sys.stderr)
        return 1

    train = series["train"]
    eval_rows = series["eval"]
    print(
        f"run: {args.state.parent.name or args.state.name}  "
        f"steps logged={len(train)}  evals={len(eval_rows)}"
    )
    print(f"global_step={state.get('global_step')}  epoch={state.get('epoch'):g}")

    # run timeline from the raw log (events only, no curve data)
    # 从原始日志打印运行时间线（仅事件，不参与曲线绘制）。
    for event in load_log_events(args.log):
        print(f"  [log] {event[:180]}")

    last = train[-1]
    best = min(eval_rows, key=lambda r: r["eval_loss"]) if eval_rows else None
    print(f"final train loss: {last['loss']:.4f} (step {last['step']})")
    if best:
        print(
            f"best eval loss: {best['eval_loss']:.4f} at step {best['step']} "
            f"(epoch {best.get('epoch', 0.0):.3f})"
        )
    last_entry = state["log_history"][-1] if state.get("log_history") else {}
    if last_entry.get("train_runtime"):
        print(
            f"train runtime: {last_entry['train_runtime'] / 3600:.2f} h  "
            f"({last_entry.get('train_samples_per_second', 0):.2f} samples/s)"
        )

    output = args.output or args.state.parent / "training_curves.png"
    plot_curves(state, series, output, args.title)
    if args.csv:
        export_csv(series, args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
