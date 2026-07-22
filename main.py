"""Colab-ready commands for the Qwen3-VL-4B remote-sensing baseline.
Qwen3-VL-4B 遥感基线的 Colab 可运行命令。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.loaders import DATASET_REPOS, download_datasets, load_samples
from data.schema import CanonicalPrediction, CanonicalSample
from eval.audit_report import AuditReportWriter, build_audit_report, report_dir_for_result, write_deepseek_audit
from eval.metrics import evaluate_records
from models.qwen3vl import Qwen3VLBaseline, Qwen3VLSettings


EVALUATION_TARGETS = (
    "vrsbench_caption",
    "vrsbench_vqa",
    "vrsbench_grounding",
    "mme_real_rs",
    "xlrs_caption_en",
    "xlrs_grounding_en",
    "xlrs_vqa_lite",
    "levir_cc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3-VL-4B remote-sensing baseline")
    parser.add_argument("--config", type=Path, required=True, help="Path to a JSON experiment configuration.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download official dataset releases.")
    download.add_argument("--datasets", nargs="+", choices=sorted(DATASET_REPOS), default=sorted(DATASET_REPOS))

    inspect = subparsers.add_parser("inspect", help="Print canonical samples without loading the model.")
    inspect.add_argument("--dataset", choices=EVALUATION_TARGETS, required=True)
    inspect.add_argument("--limit", type=int, default=3)

    infer = subparsers.add_parser("infer", help="Run Qwen3-VL inference and save canonical JSONL records.")
    infer.add_argument("--dataset", choices=(*EVALUATION_TARGETS, "all"), required=True)
    infer.add_argument("--limit", type=int, default=None, help="Optional smoke-test sample limit.")
    infer.add_argument("--overwrite", action="store_true")

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a saved JSONL result file.")
    evaluate.add_argument("--result", type=Path, required=True)
    evaluate.add_argument("--deepseek-proxy", action="store_true", help="Use the optional non-official DeepSeek VQA proxy.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    required_paths = {"data_root", "output_root"}
    missing = required_paths - set(config.get("paths", {}))
    if missing:
        raise ValueError(f"Missing config paths: {', '.join(sorted(missing))}")
    return config


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_root = Path(config["paths"]["data_root"]).expanduser()
    output_root = Path(config["paths"]["output_root"]).expanduser()
    if args.command == "download":
        downloaded = download_datasets(args.datasets, data_root)
        print(json.dumps({name: str(path) for name, path in downloaded.items()}, indent=2))
        return
    if args.command == "inspect":
        _inspect(args.dataset, data_root, args.limit)
        return
    if args.command == "infer":
        targets = EVALUATION_TARGETS if args.dataset == "all" else (args.dataset,)
        model_load_started = time.perf_counter()
        model = _load_model(config)
        model_load_seconds = time.perf_counter() - model_load_started
        for target in targets:
            _infer_target(
                target,
                data_root,
                output_root,
                model,
                args.limit,
                args.overwrite,
                config,
                model_load_seconds,
            )
        return
    records = _read_jsonl(args.result)
    deepseek_audit: list[dict[str, Any]] | None = [] if args.deepseek_proxy else None
    metrics = evaluate_records(
        records,
        use_deepseek=args.deepseek_proxy,
        deepseek_config=config.get("deepseek", {}),
        deepseek_audit=deepseek_audit,
    )
    metric_path = args.result.with_suffix(".metrics.json")
    metric_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_path = None
    if deepseek_audit is not None:
        audit_path = report_dir_for_result(args.result) / "deepseek_audit.jsonl"
        write_deepseek_audit(audit_path, deepseek_audit)
    report_path = build_audit_report(args.result, metric_path, audit_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved evaluation metrics to {metric_path.resolve()}")
    if report_path is not None:
        print(f"Saved default audit report to {report_path}")
    else:
        print("Default audit report unavailable; run inference again with report.enabled=true.")


def _load_model(config: dict[str, Any]) -> Qwen3VLBaseline:
    model_config = config.get("model", {})
    settings = Qwen3VLSettings(
        model_id=model_config.get("id", "Qwen/Qwen3-VL-4B-Instruct"),
        dtype=model_config.get("dtype", "auto"),
        device_map=model_config.get("device_map", "auto"),
        max_new_tokens=int(model_config.get("max_new_tokens", 256)),
        min_pixels=model_config.get("min_pixels"),
        max_pixels=model_config.get("max_pixels"),
        local_files_only=bool(model_config.get("local_files_only", False)),
    )
    return Qwen3VLBaseline(settings)


def _inspect(dataset_name: str, data_root: Path, limit: int) -> None:
    count = 0
    for sample in load_samples(dataset_name, data_root):
        sample.validate()
        print(json.dumps(sample.serializable(), ensure_ascii=False, indent=2))
        count += 1
        if count >= limit:
            break
    if not count:
        raise RuntimeError(f"No samples loaded for {dataset_name}.")


def _infer_target(
    dataset_name: str,
    data_root: Path,
    output_root: Path,
    model: Qwen3VLBaseline,
    limit: int | None,
    overwrite: bool,
    config: dict[str, Any],
    model_load_seconds: float = 0.0,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / f"{dataset_name}.jsonl"
    if result_path.exists() and not overwrite:
        raise FileExistsError(f"{result_path} already exists. Use --overwrite to replace it.")
    metadata_path = output_root / f"{dataset_name}.metadata.json"
    metadata = {
        "dataset": dataset_name,
        "model": config.get("model", {}),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "scope_note": _scope_note(dataset_name),
        "model_load_seconds": round(model_load_seconds, 6),
    }
    report_config = config.get("report", {})
    report_enabled = bool(report_config.get("enabled", True))
    report_max_samples = int(report_config.get("max_samples", 200))
    report_context = AuditReportWriter(result_path, report_max_samples) if report_enabled else nullcontext(None)
    completed = 0
    official_mme_records = []
    inference_started = time.perf_counter()
    with result_path.open("w", encoding="utf-8") as result_file, report_context as report_writer:
        for sample in load_samples(dataset_name, data_root):
            sample_started = time.perf_counter()
            prediction = model.predict(sample)
            sample_seconds = time.perf_counter() - sample_started
            prediction.validate()
            record = {"sample": sample.serializable(), "prediction": prediction.serializable()}
            result_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            if report_writer is not None:
                default_agent_trace = {
                    "agent_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
                    "entrypoint": "predict",
                    "route": "direct_baseline",
                    "router_used": False,
                    "task_type": sample.task_type,
                }
                agent_trace = prediction.meta.get("agent_trace", default_agent_trace)
                report_writer.capture(sample, prediction, sample_seconds, agent_trace)
            if dataset_name == "mme_real_rs":
                official_record = dict(sample.meta["record"])
                official_record["Output"] = prediction.answer
                official_mme_records.append(official_record)
            completed += 1
            if completed % 10 == 0:
                print(f"{dataset_name}: completed {completed} samples")
            if limit is not None and completed >= limit:
                break
    metadata["completed_samples"] = completed
    metadata["inference_seconds"] = round(time.perf_counter() - inference_started, 6)
    metadata["report_enabled"] = report_enabled
    metadata["report_max_samples"] = report_max_samples if report_enabled else 0
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if official_mme_records:
        official_path = output_root / "mme_real_rs.official.json"
        official_path.write_text(json.dumps(official_mme_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {completed} predictions to {result_path}")
    if report_enabled:
        report_path = build_audit_report(result_path)
        if report_path is not None:
            print(f"Saved default audit report to {report_path}")


def _scope_note(dataset_name: str) -> str:
    if dataset_name == "xlrs_vqa_lite":
        return "Official XLRS-Bench Lite VQA release; report separately from full caption and grounding releases."
    if dataset_name == "xlrs_caption_en":
        return "Official full English caption release exposes only a train-named split; this is an evaluation-only baseline run."
    if dataset_name == "mme_real_rs":
        return "MME-RealWorld Remote Sensing subdomain only."
    return "Official released evaluation scope."


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    if not records:
        raise ValueError(f"No prediction records found in {path}")
    return records


if __name__ == "__main__":
    main()
