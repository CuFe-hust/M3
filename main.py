"""Colab-ready commands for the Qwen3-VL-4B remote-sensing baseline.
Qwen3-VL-4B 遥感基线的 Colab 可运行命令。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.loaders import DATASET_REPOS, download_datasets, load_samples
from data.schema import CanonicalPrediction, CanonicalSample
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
        model = _load_model(config)
        for target in targets:
            _infer_target(target, data_root, output_root, model, args.limit, args.overwrite, config)
        return
    records = _read_jsonl(args.result)
    metrics = evaluate_records(records, use_deepseek=args.deepseek_proxy, deepseek_config=config.get("deepseek", {}))
    metric_path = args.result.with_suffix(".metrics.json")
    metric_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def _load_model(config: dict[str, Any]) -> Qwen3VLBaseline:
    model_config = config.get("model", {})
    settings = Qwen3VLSettings(
        model_id=model_config.get("id", "Qwen/Qwen3-VL-4B-Instruct"),
        dtype=model_config.get("dtype", "auto"),
        device_map=model_config.get("device_map", "auto"),
        max_new_tokens=int(model_config.get("max_new_tokens", 256)),
        min_pixels=model_config.get("min_pixels"),
        max_pixels=model_config.get("max_pixels"),
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
    }
    completed = 0
    official_mme_records = []
    with result_path.open("w", encoding="utf-8") as result_file:
        for sample in load_samples(dataset_name, data_root):
            prediction = model.predict(sample)
            prediction.validate()
            record = {"sample": sample.serializable(), "prediction": prediction.serializable()}
            result_file.write(json.dumps(record, ensure_ascii=False) + "\n")
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
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if official_mme_records:
        official_path = output_root / "mme_real_rs.official.json"
        official_path.write_text(json.dumps(official_mme_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {completed} predictions to {result_path}")


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
