#!/usr/bin/env python3
"""Plot LLaMA-Factory training curves from trainer_log.jsonl.
从 LLaMA-Factory 训练输出目录中的 trainer_log.jsonl 绘制训练曲线。

Usage:
  python3 scripts/plot_train_curves.py <output_dir> [--out curves.png]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional


def load_log(log_path: str) -> list[dict]:
    """Load trainer_log.jsonl lines into a list of dicts.
    读取 trainer_log.jsonl 的每一行 JSON。"""
    rows: list[dict] = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_series(rows: list[dict]) -> dict[str, list]:
    """Extract train/eval loss and learning-rate series by step.
    按 step 提取训练损失、验证损失与学习率序列。"""
    series: dict[str, list] = {
        "train_steps": [], "train_loss": [],
        "eval_steps": [], "eval_loss": [],
        "lr_steps": [], "lr": [],
    }
    for row in rows:
        step = row.get("current_steps", row.get("step"))
        if step is None:
            continue
        if row.get("loss") is not None:
            series["train_steps"].append(step)
            series["train_loss"].append(float(row["loss"]))
        if row.get("eval_loss") is not None:
            series["eval_steps"].append(step)
            series["eval_loss"].append(float(row["eval_loss"]))
        if row.get("lr") is not None:
            series["lr_steps"].append(step)
            series["lr"].append(float(row["lr"]))
    return series


def _cjk_font() -> Optional[str]:
    """Return an available CJK font family name, or None when none exists.
    返回可用的中文字体名；若系统无中文字体则返回 None。"""
    import matplotlib.font_manager as fm

    for name in (
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei",
        "PingFang SC",
        "Hiragino Sans GB",
        "SimHei",
        "Microsoft YaHei",
    ):
        try:
            if fm.findfont(name, fallback_to_default=False):
                return name
        except ValueError:
            continue
    return None


def plot(series: dict[str, list], out_path: str) -> None:
    """Render loss and learning-rate curves to a PNG.
    将损失与学习率曲线渲染为 PNG。"""
    import matplotlib

    matplotlib.use("Agg")  # headless backend; 无显示环境的后端
    import matplotlib.pyplot as plt

    cjk_font = _cjk_font()
    if cjk_font is not None:
        # Use the CJK font so bilingual labels render correctly.
        # 使用中文字体，保证中英双语标签正常显示。
        matplotlib.rcParams["font.family"] = cjk_font
        xlabel = "step  步数"
        title = "Training Curves  训练曲线"
    else:
        # Fall back to ASCII-only labels when no CJK font exists.
        # 系统无中文字体时退回纯英文标签，避免乱码。
        xlabel = "step"
        title = "Training Curves"

    fig, ax_loss = plt.subplots(figsize=(9, 5))
    if series["train_loss"]:
        ax_loss.plot(series["train_steps"], series["train_loss"],
                     label="train loss", color="#1f77b4")
    if series["eval_loss"]:
        ax_loss.plot(series["eval_steps"], series["eval_loss"],
                     label="eval loss", color="#ff7f0e", marker="o",
                     markersize=4, linestyle="--")
    ax_loss.set_xlabel(xlabel)
    ax_loss.set_ylabel("loss")
    ax_loss.legend(loc="upper right")
    ax_loss.set_title(title)
    ax_loss.grid(True, alpha=0.3)

    if series["lr"]:
        ax_lr = ax_loss.twinx()
        ax_lr.plot(series["lr_steps"], series["lr"],
                   label="learning rate", color="#2ca02c", alpha=0.7)
        ax_lr.set_ylabel("learning rate")
        ax_lr.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved curves: {out_path}  曲线已保存")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot LLaMA-Factory training curves. 绘制 LLaMA-Factory 训练曲线。"
    )
    parser.add_argument("output_dir", help="training output dir containing trainer_log.jsonl 训练输出目录")
    parser.add_argument("--out", default=None, help="output PNG path 输出 PNG 路径")
    args = parser.parse_args()

    log_path = os.path.join(args.output_dir, "trainer_log.jsonl")
    if not os.path.isfile(log_path):
        print(f"ERROR: {log_path} not found. Did training finish and write logs?  未找到训练日志。", file=sys.stderr)
        return 1

    try:
        series = extract_series(load_log(log_path))
    except json.JSONDecodeError as exc:
        print(f"ERROR: failed to parse {log_path}: {exc}  日志解析失败。", file=sys.stderr)
        return 1

    if not series["train_loss"] and not series["eval_loss"]:
        print("ERROR: no loss entries found in trainer_log.jsonl.  日志中没有损失记录。", file=sys.stderr)
        return 1

    try:
        out_path = args.out or os.path.join(args.output_dir, "train_curves.png")
        plot(series, out_path)
    except ImportError:
        print("ERROR: matplotlib is required. Install it with: pip install matplotlib", file=sys.stderr)
        print("提示：需要 matplotlib，请执行 pip install matplotlib", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
