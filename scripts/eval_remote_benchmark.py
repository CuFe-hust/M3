#!/usr/bin/env python3
"""Benchmark evaluation for the remote 评测数据集 with Qwen3-VL/InternVL.
在远端评测数据集上运行 Qwen3-VL / InternVL 基准评测并导出指标与图表。
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.loaders import REMOTE_BENCHMARK_DATASETS, load_remote_benchmark_samples
from eval.audit_report import build_audit_report, report_dir_for_result, write_deepseek_audit
from eval.metrics import _box_iou, _normalize_text, evaluate_records
from models.benchmark_vlm import BenchmarkVLM, MODEL_TYPES, resolve_load_paths


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "eval_remote_benchmark"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen3-VL-4B or InternVL3.5-8B (base or LoRA) on the remote "
            "评测数据集 and export metrics/figures. "
            "在远端评测数据集上运行 Qwen3-VL-4B / InternVL3.5-8B（基座或 LoRA）并导出指标与图表。"
        )
    )
    parser.add_argument("--model-type", required=True, choices=MODEL_TYPES, help="Model family. 模型类型")
    parser.add_argument("--model-path", type=Path, required=True, help="Base model directory. 基座模型目录")
    parser.add_argument(
        "--lora-path",
        type=Path,
        default=None,
        help=(
            "Optional LoRA dir: PEFT adapter (adapter_config.json) or a merged full model. "
            "可选的 LoRA 目录：PEFT 适配器或合并后的完整模型。"
        ),
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Remote 评测数据集 root. 数据集根目录")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=f"Output parent; a timestamped run dir is created inside. 输出父目录，内部会创建带时间戳的运行目录（默认 {DEFAULT_OUTPUT_ROOT}）",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=REMOTE_BENCHMARK_DATASETS,
        default=list(REMOTE_BENCHMARK_DATASETS),
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional per-dataset sample limit. 每个数据集可选样本上限")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to shuffle each dataset before applying --limit. 截取 --limit 前用于打乱每个数据集的随机种子",
    )
    parser.add_argument(
        "--deepseek-proxy",
        action="store_true",
        help=(
            "Enable the optional non-official DeepSeek VQA semantic proxy metric; "
            "the key is read from --deepseek-api-key when provided, otherwise from "
            "DEEPSEEK_API_KEY. 启用可选的非官方 DeepSeek VQA 语义代理指标；"
            "密钥优先取 --deepseek-api-key，否则从环境变量 DEEPSEEK_API_KEY 读取。"
        ),
    )
    parser.add_argument(
        "--deepseek-api-key",
        default=None,
        help=(
            "Optional DeepSeek API key for the proxy metric; takes precedence over "
            "DEEPSEEK_API_KEY. Note that CLI values may appear in shell history or "
            "process listings. 可选的 DeepSeek API 密钥，优先于环境变量 "
            "DEEPSEEK_API_KEY；注意命令行传参会出现在 shell 历史或进程列表中。"
        ),
    )
    parser.add_argument(
        "--deepseek-model",
        default="deepseek-chat",
        help="DeepSeek model name used by the proxy metric. DeepSeek 代理指标使用的模型名（默认 deepseek-chat）",
    )
    parser.add_argument(
        "--deepseek-base-url",
        default="https://api.deepseek.com/v1",
        help="DeepSeek OpenAI-compatible base URL. DeepSeek OpenAI 兼容接口地址",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-tiles", type=int, default=12, help="InternVL dynamic tiles per image. InternVL 单图最大分块数")
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "float16", "bfloat16", "float32"))
    parser.add_argument("--device", default=None, help="torch device string; auto when omitted. torch 设备字符串，缺省自动")
    parser.add_argument(
        "--local-files-only",
        dest="local_files_only",
        action="store_true",
        default=True,
        help="Load model/tokenizer only from local paths (default). 仅从本地路径加载模型/分词器（默认）",
    )
    parser.add_argument("--no-local-files-only", dest="local_files_only", action="store_false")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing result files; without it completed labels resume-skip. 覆盖已有结果文件；未指定时已完成标签会跳过续跑",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Also build the default HTML audit report per result file. 同时为每个结果文件生成默认 HTML 审计报告",
    )
    parser.add_argument("--report-max-samples", type=int, default=100)
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip matplotlib figure export. 跳过 matplotlib 图表导出",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    effective_model_path, adapter_path = resolve_load_paths(
        model_path,
        args.lora_path.expanduser().resolve() if args.lora_path else None,
    )

    run_dir = _create_run_dir(args.output_root)
    results_dir = run_dir / "results"
    figures_dir = run_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    print(f"Loading {args.model_type} from {effective_model_path}", flush=True)
    model = BenchmarkVLM(
        model_type=args.model_type,
        model_path=effective_model_path,
        lora_path=adapter_path,
        dtype=args.dtype,
        device=args.device,
        local_files_only=args.local_files_only,
        max_new_tokens=args.max_new_tokens,
        max_tiles=args.max_tiles,
    )
    print(f"Model loaded on {model.device}", flush=True)

    all_metrics: dict[str, dict[str, Any]] = {}
    per_sample: dict[str, list[dict[str, Any]]] = {}
    inference_seconds: dict[str, float] = {}
    label_files: dict[str, str] = {}
    dataset_labels: dict[str, list[str]] = {}

    for dataset in args.datasets:
        samples = _limit_samples(
            list(load_remote_benchmark_samples(dataset_root, dataset)),
            args.limit,
            args.seed,
        )
        if not samples:
            print(f"[{dataset}] no samples", flush=True)
            continue
        groups = _group_by_task(samples)
        print(
            f"[{dataset}] {len(samples)} samples -> {', '.join(f'{label}={len(group)}' for label, group in groups.items())}",
            flush=True,
        )
        for label, group in groups.items():
            result_path = results_dir / f"{label}.jsonl"
            if result_path.exists() and not args.overwrite:
                print(f"[{label}] result exists, resume-skip: {result_path}", flush=True)
                continue
            records = _infer_group(model, group, label)
            result_path.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
                encoding="utf-8",
            )
            deepseek_audit: list[dict[str, Any]] | None = [] if args.deepseek_proxy else None
            deepseek_config: dict[str, Any] = {
                "model": args.deepseek_model,
                "base_url": args.deepseek_base_url,
            }
            if args.deepseek_api_key:
                deepseek_config["api_key"] = args.deepseek_api_key
            metrics = evaluate_records(
                records,
                use_deepseek=args.deepseek_proxy,
                deepseek_config=deepseek_config,
                deepseek_audit=deepseek_audit,
            )
            metric_path = result_path.with_suffix(".metrics.json")
            metric_path.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            audit_path: Path | None = None
            if deepseek_audit is not None:
                audit_path = report_dir_for_result(result_path) / "deepseek_audit.jsonl"
                write_deepseek_audit(audit_path, deepseek_audit)
            label_seconds = sum(
                float(record["prediction"].get("meta", {}).get("latency_seconds", 0.0))
                for record in records
            )
            all_metrics[label] = {"metrics": metrics, "samples": len(records)}
            per_sample[label] = _sample_stats(records)
            inference_seconds[label] = round(label_seconds, 6)
            label_files[label] = str(metric_path)
            dataset_labels.setdefault(dataset, []).append(label)
            if args.report:
                report_path = build_audit_report(result_path, metric_path, audit_path)
                print(f"[{label}] audit report: {report_path}", flush=True)

    records_by_label = {
        label: _read_result_records(results_dir / f"{label}.jsonl")
        for label in per_sample
    }
    combined_results = _build_combined_results(records_by_label, dataset_labels)

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_metrics_csv(run_dir / "metrics.csv", all_metrics)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            _build_summary(
                args,
                model,
                all_metrics,
                per_sample,
                inference_seconds,
                label_files,
                run_dir,
                started_at,
                combined_results,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    figure_paths: dict[str, str] = {}
    if not args.skip_figures:
        figure_paths = plot_figures(all_metrics, per_sample, figures_dir)
    (figures_dir / "figures_manifest.json").write_text(
        json.dumps(figure_paths, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))
    _print_results(all_metrics, combined_results)
    print(f"Run directory: {run_dir}")
    print(f"Metrics: {metrics_path}")
    print(f"Summary: {summary_path}")
    for name, path in figure_paths.items():
        print(f"Figure: {path}")
    return 0


def _limit_samples(samples: list[Any], limit: int | None, seed: int) -> list[Any]:
    """Shuffle deterministically with the configured seed before taking the limit.
    按配置的随机种子做确定性打乱后，再截取每个数据集的样本上限。
    """

    sampled = list(samples)
    if limit is not None:
        random.Random(seed).shuffle(sampled)
        sampled = sampled[:limit]
    return sampled


def _infer_group(model: BenchmarkVLM, samples: list[Any], label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, 1):
        prediction = model.predict(sample)
        prediction.validate()
        records.append({"sample": sample.serializable(), "prediction": prediction.serializable()})
        if index % 10 == 0 or index == len(samples):
            print(f"[{label}] {index}/{len(samples)} completed", flush=True)
    return records


def _group_by_task(samples: Iterable[Any]) -> dict[str, list[Any]]:
    groups: dict[str, list[Any]] = defaultdict(list)
    for sample in samples:
        label = str(sample.meta.get("benchmark_task") or sample.task_type)
        groups[label].append(sample)
    return dict(groups)


def _sample_stats(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats = []
    for record in records:
        sample = record["sample"]
        prediction = record["prediction"]
        task_type = sample["task_type"]
        stat: dict[str, Any] = {
            "id": sample["id"],
            "task_type": task_type,
            "question_type": sample.get("meta", {}).get("question_type", ""),
            "difficulty": sample.get("meta", {}).get("difficulty", ""),
            "latency_seconds": prediction.get("meta", {}).get("latency_seconds"),
            "correct": None,
            "iou": None,
        }
        if task_type == "vqa":
            predicted = _normalize_text(prediction.get("answer") or prediction["text"])
            references = {_normalize_text(answer) for answer in sample.get("answers", [])}
            stat["correct"] = predicted in references
        elif task_type == "grounding":
            predicted_boxes = prediction.get("boxes", [])
            expected_boxes = sample.get("boxes", [])
            if predicted_boxes and expected_boxes:
                stat["iou"] = round(_box_iou(predicted_boxes[0], expected_boxes[0]), 6)
            else:
                stat["iou"] = 0.0
        stats.append(stat)
    return stats


def _read_result_records(path: Path) -> list[dict[str, Any]]:
    """Read persisted canonical JSONL records.
    读取已持久化的统一 JSONL 评测记录。
    """

    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


_AGGREGATE_TASKS = {
    "vqa": "vqa",
    "caption": "caption",
    "change_caption": "caption",
    "grounding": "grounding",
}


def _build_combined_results(
    records_by_label: dict[str, list[dict[str, Any]]],
    dataset_labels: dict[str, list[str]],
) -> dict[str, Any]:
    """Aggregate per-dataset and overall metrics by metric family.
    按指标族汇总每个数据集以及全部数据集的测试结果。
    """

    combined: dict[str, Any] = {
        "datasets": {},
        "overall": {"task_type_metrics": {}, "samples": 0},
    }
    overall_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    overall_samples = 0
    for dataset in sorted(dataset_labels):
        dataset_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
        dataset_samples = 0
        for label in dataset_labels[dataset]:
            records = records_by_label.get(label, [])
            if not records:
                continue
            namespaced_records = _namespaced_for_aggregation(records, dataset)
            sample_task = records[0]["sample"].get("task_type", "")
            aggregate_task = _AGGREGATE_TASKS.get(sample_task, sample_task)
            dataset_records[aggregate_task].extend(namespaced_records)
            overall_records[aggregate_task].extend(namespaced_records)
            dataset_samples += len(records)
        combined["datasets"][dataset] = {
            "labels": list(dataset_labels[dataset]),
            "samples": dataset_samples,
            "task_type_metrics": {
                task_type: evaluate_records(records)
                for task_type, records in dataset_records.items()
            },
        }
        overall_samples += dataset_samples
    combined["overall"]["samples"] = overall_samples
    combined["overall"]["task_type_metrics"] = {
        task_type: evaluate_records(records)
        for task_type, records in overall_records.items()
    }
    return combined


def _namespaced_for_aggregation(
    records: list[dict[str, Any]],
    namespace: str,
) -> list[dict[str, Any]]:
    """Copy records with dataset-prefixed sample ids for collision-free aggregation.
    复制记录并把样本 id 加上数据集前缀，避免合并指标时不同数据集 id 冲突。
    """

    namespaced: list[dict[str, Any]] = []
    for record in records:
        sample = dict(record["sample"])
        sample["id"] = f"{namespace}:{sample['id']}"
        namespaced.append({"sample": sample, "prediction": record["prediction"]})
    return namespaced


def _print_results(
    all_metrics: dict[str, dict[str, Any]],
    combined_results: dict[str, Any],
) -> None:
    """Print per-label, per-dataset and overall combined results.
    打印每个标签、每个数据集以及全部数据集的汇总结果。
    """

    print("=== Per-label results ===")
    for label in sorted(all_metrics):
        payload = all_metrics[label]
        print(
            f"{label}: samples={payload['samples']}, "
            f"primary={_primary_score(payload['metrics']):.4f}"
        )
    print("=== Per-dataset combined results (by metric family) ===")
    for dataset, payload in combined_results["datasets"].items():
        print(f"{dataset}: samples={payload['samples']}")
        for task_type, metrics in payload["task_type_metrics"].items():
            _print_combined_metric(task_type, metrics)
    print("=== Overall combined results (by metric family) ===")
    overall = combined_results["overall"]
    print(f"all: samples={overall['samples']}")
    for task_type, metrics in overall["task_type_metrics"].items():
        _print_combined_metric(task_type, metrics)


def _print_combined_metric(task_type: str, metrics: dict[str, Any]) -> None:
    """Print one combined metric family in a compact line.
    用一行紧凑格式打印一个合并指标族。
    """

    if metrics.get("metric") == "exact_match_accuracy":
        print(
            f"  {task_type}: exact_match_accuracy="
            f"{metrics.get('score', 0.0):.4f} "
            f"({metrics.get('correct', 0)}/{metrics.get('total', 0)})"
        )
    elif metrics.get("metric") == "axis_aligned_iou_at_0_5":
        print(
            f"  {task_type}: mean_iou={metrics.get('mean_iou', 0.0):.4f}, "
            f"accuracy={metrics.get('accuracy', 0.0):.4f} "
            f"({metrics.get('total', 0)})"
        )
    elif "CIDEr" in metrics:
        print(
            f"  {task_type}: CIDEr={metrics.get('CIDEr', 0.0):.4f}, "
            f"BLEU_4={metrics.get('BLEU_4', 0.0):.4f}, "
            f"METEOR={metrics.get('METEOR', 0.0):.4f}, "
            f"ROUGE_L={metrics.get('ROUGE_L', 0.0):.4f} "
            f"({metrics.get('total', 0)})"
        )
    else:
        print(f"  {task_type}: {json.dumps(metrics, ensure_ascii=False)}")


def _create_run_dir(output_root: Path | None) -> Path:
    base = (output_root or DEFAULT_OUTPUT_ROOT).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base / stamp
    counter = 1
    while candidate.exists():
        candidate = base / f"{stamp}_{counter}"
        counter += 1
    candidate.mkdir()
    return candidate


def _write_metrics_csv(path: Path, all_metrics: dict[str, dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "metric", "value"])
        for label, payload in all_metrics.items():
            for metric, value in payload["metrics"].items():
                if isinstance(value, (int, float)):
                    writer.writerow([label, metric, value])


def _build_summary(
    args: argparse.Namespace,
    model: BenchmarkVLM,
    all_metrics: dict[str, dict[str, Any]],
    per_sample: dict[str, list[dict[str, Any]]],
    inference_seconds: dict[str, float],
    label_files: dict[str, str],
    run_dir: Path,
    started_at: str,
    combined_results: dict[str, Any],
) -> dict[str, Any]:
    env = {
        "python": platform.python_version(),
        "device": model.device,
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "peft": _package_version("peft"),
        "datasets": _package_version("datasets"),
        "matplotlib": _package_version("matplotlib"),
    }
    labels = {
        label: {
            "samples": len(stats),
            "inference_seconds": inference_seconds.get(label, 0.0),
            "metrics_file": label_files.get(label, ""),
            "primary_score": _primary_score(all_metrics[label]["metrics"]) if label in all_metrics else None,
        }
        for label, stats in per_sample.items()
    }
    return {
        "model_type": args.model_type,
        "model_path": str(model.effective_model_path),
        "lora_path": str(model.adapter_path) if model.adapter_path else None,
        "dataset_root": str(args.dataset_root.expanduser().resolve()),
        "datasets": args.datasets,
        "limit": args.limit,
        "seed": args.seed,
        "seed_applied": args.limit is not None,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "max_tiles": args.max_tiles,
        "local_files_only": args.local_files_only,
        "deepseek_proxy": args.deepseek_proxy,
        "deepseek_model": args.deepseek_model if args.deepseek_proxy else None,
        "deepseek_base_url": args.deepseek_base_url if args.deepseek_proxy else None,
        "deepseek_api_key_source": "cli" if args.deepseek_api_key else "environment",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "environment": env,
        "labels": labels,
        "combined_results": combined_results,
        "run_dir": str(run_dir),
    }


def _package_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "installed"))
    except Exception:
        return None


def plot_figures(
    all_metrics: dict[str, dict[str, Any]],
    per_sample: dict[str, list[dict[str, Any]]],
    figures_dir: Path,
) -> dict[str, str]:
    """Export metric figures; requires matplotlib (Agg backend).
    导出指标图表；需要 matplotlib（Agg 后端）。
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install matplotlib to export benchmark figures. 导出图表需要安装 matplotlib。") from error

    figures: dict[str, str] = {}
    labels = sorted(all_metrics)
    if labels:
        path = figures_dir / "metrics_overview.png"
        _plot_overview(all_metrics, labels, plt, path)
        figures["metrics_overview"] = str(path)

    caption_labels = [label for label in labels if "CIDEr" in all_metrics[label]["metrics"]]
    if caption_labels:
        path = figures_dir / "caption_scores.png"
        _plot_caption_scores(all_metrics, caption_labels, plt, path)
        figures["caption_scores"] = str(path)

    vqa_labels = [
        label
        for label in labels
        if any(stat["correct"] is not None for stat in per_sample.get(label, []))
    ]
    if vqa_labels:
        path = figures_dir / "vqa_accuracy_by_type.png"
        _plot_vqa_accuracy(per_sample, vqa_labels, plt, path)
        figures["vqa_accuracy_by_type"] = str(path)

    grounding_labels = [label for label in labels if label in per_sample and any(stat["iou"] is not None for stat in per_sample[label])]
    if grounding_labels:
        path = figures_dir / "grounding_iou_histogram.png"
        _plot_iou_histogram(per_sample, grounding_labels, plt, path)
        figures["grounding_iou_histogram"] = str(path)

    plt.close("all")
    return figures


def _primary_score(metrics: dict[str, Any]) -> float:
    if metrics.get("metric") == "exact_match_accuracy":
        return float(metrics.get("score", 0.0))
    if "mean_iou" in metrics:
        return float(metrics["mean_iou"])
    if "CIDEr" in metrics:
        return float(metrics["CIDEr"])
    return 0.0


def _plot_overview(
    all_metrics: dict[str, dict[str, Any]],
    labels: list[str],
    plt: Any,
    path: Path,
) -> None:
    short = [_short_label(label) for label in labels]
    scores = [_primary_score(all_metrics[label]["metrics"]) for label in labels]
    figure, axis = plt.subplots(figsize=(max(8, 1.2 * len(labels)), 5))
    axis.bar(short, scores, color="#4C78A8")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Primary score")
    axis.set_title("Remote benchmark overview")
    for index, value in enumerate(scores):
        axis.text(index, value + 0.02, f"{value:.3f}", ha="center", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_caption_scores(
    all_metrics: dict[str, dict[str, Any]],
    labels: list[str],
    plt: Any,
    path: Path,
) -> None:
    metrics_names = ("BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4", "METEOR", "ROUGE_L", "CIDEr")
    index = range(len(metrics_names))
    width = 0.8 / len(labels)
    figure, axis = plt.subplots(figsize=(max(9, 2 * len(labels)), 5))
    for offset, label in enumerate(labels):
        values = [float(all_metrics[label]["metrics"].get(name, 0.0)) for name in metrics_names]
        axis.bar([position + offset * width for position in index], values, width, label=_short_label(label))
    axis.set_xticks([position + width * (len(labels) - 1) / 2 for position in index])
    axis.set_xticklabels(metrics_names)
    axis.set_ylabel("Score")
    axis.set_title("Caption metrics")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_vqa_accuracy(
    per_sample: dict[str, list[dict[str, Any]]],
    labels: list[str],
    plt: Any,
    path: Path,
) -> None:
    matrix: dict[str, dict[str, float]] = defaultdict(dict)
    dataset_shorts: set[str] = set()
    for label in labels:
        by_type: dict[str, list[bool]] = defaultdict(list)
        for stat in per_sample[label]:
            if stat["correct"] is None:
                continue
            key = str(stat.get("question_type") or label)
            by_type[key].append(bool(stat["correct"]))
        for key, values in by_type.items():
            short = _short_label(label)
            matrix[key][short] = sum(values) / len(values)
            dataset_shorts.add(short)
    group_names = sorted(matrix)
    dataset_shorts = sorted(dataset_shorts)
    width = 0.8 / max(len(dataset_shorts), 1)
    figure, axis = plt.subplots(figsize=(max(10, 1.8 * len(group_names)), 5))
    for offset, short in enumerate(dataset_shorts):
        axis.bar(
            [position + offset * width for position in range(len(group_names))],
            [matrix.get(group, {}).get(short, 0.0) for group in group_names],
            width,
            label=short,
        )
    axis.set_xticks(
        [position + width * (len(dataset_shorts) - 1) / 2 for position in range(len(group_names))]
    )
    axis.set_xticklabels(group_names, rotation=20, fontsize=8)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Accuracy")
    axis.set_title("VQA accuracy by question type")
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_iou_histogram(
    per_sample: dict[str, list[dict[str, Any]]],
    labels: list[str],
    plt: Any,
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(max(9, 2.2 * len(labels)), 5))
    bins = [value / 20 for value in range(21)]
    for label in labels:
        ious = [float(stat["iou"]) for stat in per_sample[label] if stat["iou"] is not None]
        if ious:
            axis.hist(ious, bins=bins, alpha=0.55, label=_short_label(label))
    axis.set_xlabel("IoU")
    axis.set_ylabel("Samples")
    axis.set_title("Grounding IoU distribution")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _short_label(label: str) -> str:
    replacements = {
        "vrsbench_caption": "VRS-cap",
        "vrsbench_vqa": "VRS-vqa",
        "vrsbench_grounding": "VRS-gnd",
        "mme_real_rs_vqa": "MME-vqa",
        "xlrs_vqa_lite": "XLRS-vqa",
        "xlrs_caption_en": "XLRS-cap",
        "xlrs_grounding_condition": "XLRS-gnd-c",
        "xlrs_grounding_fine": "XLRS-gnd-f",
        "levir_cc_change_caption": "LEVIR-cap",
    }
    return replacements.get(label, label)


if __name__ == "__main__":
    raise SystemExit(main())
