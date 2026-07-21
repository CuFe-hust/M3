"""Colab-ready commands for the Qwen3-VL-4B remote-sensing baseline.
Qwen3-VL-4B 遥感基线的 Colab 可运行命令。
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
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
    infer.add_argument("--resume", action="store_true", help="Continue an interrupted result file.")

    agent_infer = subparsers.add_parser(
        "agent-infer", help="Run the fixed LangGraph workflow and save canonical JSONL records."
    )
    agent_infer.add_argument("--dataset", choices=(*EVALUATION_TARGETS, "all"), required=True)
    agent_infer.add_argument("--limit", type=int, default=None, help="Optional smoke-test sample limit.")
    agent_infer.add_argument("--overwrite", action="store_true")
    agent_infer.add_argument("--resume", action="store_true", help="Continue an interrupted result file.")

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
    if args.command in {"infer", "agent-infer"}:
        if args.overwrite and args.resume:
            raise ValueError("--overwrite and --resume cannot be used together.")
        targets = EVALUATION_TARGETS if args.dataset == "all" else (args.dataset,)
        model = _load_model(config)
        agent = None
        if args.command == "agent-infer":
            from agents.langgraph_qwen import LangGraphQwenAgent

            agent = LangGraphQwenAgent(model)
        for target in targets:
            _infer_target(
                target,
                data_root,
                output_root,
                model,
                args.limit,
                args.overwrite,
                args.resume,
                config,
                agent,
            )
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
    resume: bool,
    config: dict[str, Any],
    agent: Any | None = None,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    suffix = ".agent" if agent is not None else ""
    workflow = "langgraph" if agent is not None else "baseline"
    result_path = output_root / f"{dataset_name}{suffix}.jsonl"
    failure_path = output_root / f"{dataset_name}{suffix}.failures.jsonl"
    metadata_path = output_root / f"{dataset_name}{suffix}.metadata.json"
    if result_path.exists() and not (overwrite or resume):
        raise FileExistsError(f"{result_path} already exists. Use --resume or --overwrite.")
    if overwrite:
        result_path.unlink(missing_ok=True)
        failure_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    completed_ids = _completed_ids(result_path) if resume else set()
    metadata = {
        "dataset": dataset_name,
        "model": config.get("model", {}),
        "workflow": workflow,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "scope_note": _scope_note(dataset_name),
        "resumed": resume,
    }
    completed_this_run = 0
    failed_this_run = 0
    skipped_existing = 0
    scoped_samples = 0
    started = time.perf_counter()
    _reset_peak_memory()
    result_mode = "a" if resume else "w"
    failure_mode = "a" if resume else "w"
    with (
        result_path.open(result_mode, encoding="utf-8") as result_file,
        failure_path.open(failure_mode, encoding="utf-8") as failure_file,
    ):
        for sample in load_samples(dataset_name, data_root):
            if limit is not None and scoped_samples >= limit:
                break
            scoped_samples += 1
            if sample.id in completed_ids:
                skipped_existing += 1
                _close_sample_images(sample)
                continue
            try:
                def save_result(
                    current_sample: CanonicalSample,
                    prediction: CanonicalPrediction,
                    model_elapsed_seconds: float,
                ) -> None:
                    _validate_prediction(current_sample, prediction)
                    record = {
                        "sample": current_sample.serializable(),
                        "prediction": prediction.serializable(),
                        "runtime": {"model_elapsed_seconds": model_elapsed_seconds},
                    }
                    result_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    result_file.flush()

                if agent is None:
                    model_started = time.perf_counter()
                    prediction = model.predict(sample)
                    model_elapsed = time.perf_counter() - model_started
                    save_result(sample, prediction, model_elapsed)
                else:
                    agent.run(sample, save_result)
                completed_this_run += 1
                completed_ids.add(sample.id)
            except Exception as error:
                failed_this_run += 1
                failure = {
                    "sample_id": sample.id,
                    "task_type": sample.task_type,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                failure_file.write(json.dumps(failure, ensure_ascii=False) + "\n")
                failure_file.flush()
                print(f"{dataset_name}: failed sample {sample.id}: {error}")
            finally:
                _close_sample_images(sample)
            attempted = completed_this_run + failed_this_run
            if attempted % 10 == 0:
                print(
                    f"{dataset_name}: completed {completed_this_run}, "
                    f"failed {failed_this_run}, skipped {skipped_existing} this run"
                )
                _write_run_metadata(
                    metadata_path,
                    metadata,
                    len(completed_ids),
                    completed_this_run,
                    failed_this_run,
                    skipped_existing,
                    time.perf_counter() - started,
                )
    elapsed = time.perf_counter() - started
    _write_run_metadata(
        metadata_path,
        metadata,
        len(completed_ids),
        completed_this_run,
        failed_this_run,
        skipped_existing,
        elapsed,
    )
    if dataset_name == "mme_real_rs":
        official_path = output_root / f"mme_real_rs{suffix}.official.json"
        _write_mme_official_output(result_path, official_path)
    print(
        f"Saved {completed_this_run} new predictions to {result_path}; "
        f"failed {failed_this_run}; skipped {skipped_existing}."
    )


def _validate_prediction(sample: CanonicalSample, prediction: CanonicalPrediction) -> None:
    prediction.validate()
    if prediction.id != sample.id:
        raise ValueError("Prediction id does not match the sample id.")
    if prediction.task_type != sample.task_type:
        raise ValueError("Prediction task type does not match the sample task type.")


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                completed.add(str(record["sample"]["id"]))
            except (json.JSONDecodeError, KeyError) as error:
                raise ValueError(f"Invalid existing result at {path}:{line_number}") from error
    return completed


def _close_sample_images(sample: CanonicalSample) -> None:
    for image in sample.images:
        close = getattr(image, "close", None)
        if callable(close):
            close()


def _reset_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        return


def _peak_memory_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**3
    except ImportError:
        return None
    return None


def _write_run_metadata(
    path: Path,
    base: dict[str, Any],
    completed_total: int,
    completed_this_run: int,
    failed_this_run: int,
    skipped_existing: int,
    elapsed_seconds: float,
) -> None:
    current = {
        **base,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed_samples": completed_total,
        "completed_this_run": completed_this_run,
        "failed_this_run": failed_this_run,
        "skipped_existing": skipped_existing,
        "elapsed_seconds_this_run": elapsed_seconds,
        "peak_memory_gb": _peak_memory_gb(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_mme_official_output(result_path: Path, official_path: Path) -> None:
    official_records = []
    for record in _read_jsonl(result_path):
        source = record["sample"].get("meta", {}).get("record")
        if not isinstance(source, dict):
            continue
        official_record = dict(source)
        official_record["Output"] = record["prediction"].get("answer")
        official_records.append(official_record)
    official_path.write_text(
        json.dumps(official_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


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
