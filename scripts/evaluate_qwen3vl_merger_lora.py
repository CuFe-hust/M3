"""Run VRSBench test-set inference with a Qwen3-VL LoRA adapter.
使用 Qwen3-VL LoRA 适配器在 VRSBench 测试集上执行推理。

The script loads the base Qwen3-VL checkpoint, optionally wraps it with a
LoRA adapter (``--adapter-path``), and generates greedy predictions for the
official/local VRSBench test JSONL (caption and/or VQA), limited to the first
``--max-images`` test images by default. Caption results report
BLEU-1..4 / METEOR / ROUGE-L / CIDEr; VQA results report overall exact-match
accuracy plus per-question-type accuracy. Results are written as canonical
sample/prediction JSONL plus a JSON summary.
脚本加载基础 Qwen3-VL 权重，可选挂载 LoRA 适配器（``--adapter-path``），
并在 VRSBench 测试 JSONL（caption 和/或 VQA）上做贪心推理，默认只使用前
``--max-images`` 张测试图片。caption 报告 BLEU-1..4 / METEOR / ROUGE-L /
CIDEr；VQA 报告整体精确匹配准确率与按问题类型准确率。结果以规范化
sample/prediction JSONL 以及 JSON 摘要写出。
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import torch
from PIL import Image

# Make the repository root importable when this script runs directly.
# 直接运行本脚本时确保仓库根目录可导入。
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.schema import CanonicalPrediction, CanonicalSample


TASK_MAX_NEW_TOKENS = {
    "caption": 512,
    "vqa": 64,
}


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI. / 构建评测 CLI。"""
    parser = argparse.ArgumentParser(
        description="Run VRSBench test-set inference with a Qwen3-VL LoRA adapter."
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Base Qwen3-VL checkpoint (local directory or Hugging Face id).",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Optional LoRA adapter directory; when omitted the base model runs.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="VRSBench dataset root containing Images_* and VRSBench_test_*.jsonl.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=("caption", "vqa"),
        default=("caption", "vqa"),
        help="Test tasks to run; default caption vqa.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/eval/qwen3-vl-8b-merger-lora/vrsbench_test.jsonl"),
        help="Canonical prediction JSONL output path.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap samples per task for smoke runs.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=200,
        help="Use only the first N test images (ordered by the caption "
        "annotation file) for caption and VQA; 0 means no image cap. "
        "Default 200.",
    )
    parser.add_argument(
        "--max-new-tokens-caption",
        type=int,
        default=512,
        help="Generation token ceiling for caption; default 512.",
    )
    parser.add_argument(
        "--max-new-tokens-vqa",
        type=int,
        default=64,
        help="Generation token ceiling for VQA; default 64.",
    )
    parser.add_argument(
        "--image-min-pixels",
        type=int,
        default=256 * 32 * 32,
        help="Minimum image pixels; Qwen3-VL uses multiples of 32.",
    )
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=1280 * 32 * 32,
        help="Maximum image pixels; Qwen3-VL uses multiples of 32.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=("float32", "float16", "bfloat16", "auto"),
        help="Dtype for loading the base model; default bfloat16.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=("sdpa", "flash_attention_2", "eager"),
        help="Attention implementation; default sdpa.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device map; default auto.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Refuse network access while loading the checkpoint.",
    )
    return parser


def check_runtime() -> None:
    """Verify the Transformers build provides the Qwen3-VL deepstack path.
    校验 Transformers 构建包含 Qwen3-VL deepstack 路径。
    """
    from packaging.version import Version

    import transformers

    if Version(transformers.__version__) < Version("5.6.0"):
        raise RuntimeError(
            "Qwen3-VL LoRA inference requires transformers>=5.6.0; "
            f"found {transformers.__version__}."
        )


def resolve_dtype(name: str) -> Any:
    """Resolve the CLI dtype name to a torch dtype.
    将 CLI dtype 名称解析为 torch dtype。
    """
    if name == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {name}")
    return mapping[name]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a VRSBench JSONL annotation file.
    读取一个 VRSBench JSONL 标注文件。
    """
    if not path.is_file():
        raise SystemExit(f"Annotation file not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}: {error}") from error
    return records


def record_to_sample(record: dict[str, Any], task: str) -> CanonicalSample:
    """Convert one VRSBench test record into a canonical sample.
    将一条 VRSBench 测试记录转换为统一样本。
    """
    image_path = record.get("image")
    if not image_path:
        raise RuntimeError(f"Record {record.get('id')!r} has no image field.")
    if task == "caption":
        prompt = record.get("instruction")
        answers = [record.get("caption")]
    elif task == "vqa":
        prompt = record.get("question")
        answers = [record.get("answer")]
    else:
        raise ValueError(f"Unsupported task: {task}")
    if not prompt or not answers[0]:
        raise RuntimeError(f"Record {record.get('id')!r} misses {task} text fields.")
    return CanonicalSample(
        id=str(record["id"]),
        task_type=task,
        images=[str(image_path)],
        prompt=prompt,
        answers=[str(answers[0])],
        meta={
            "source": "VRSBench",
            "split": "test",
            "task": task,
            "question_type": record.get("task"),
            "original_type": (record.get("source") or {}).get("original_type"),
            "image_path": str(image_path),
            "annotation": record.get("source"),
        },
    )


def ordered_test_image_ids(data_root: Path, max_images: int) -> list[str]:
    """Return the first ``max_images`` unique test image ids in caption order.
    按 caption 标注文件顺序返回前 ``max_images`` 个不重复测试图片 ID。
    """
    caption_path = data_root / "VRSBench_test_caption.jsonl"
    if not caption_path.is_file():
        raise SystemExit(
            f"Caption annotation file not found: {caption_path}; the caption "
            "file defines the test image order."
        )
    image_ids: list[str] = []
    seen: set[str] = set()
    for record in read_jsonl(caption_path):
        image_id = record.get("image_id") or Path(record["image"]).name
        if image_id in seen:
            continue
        seen.add(image_id)
        image_ids.append(image_id)
        if max_images and len(image_ids) >= max_images:
            break
    return image_ids


def load_task_records(
    data_root: Path, task: str, image_ids: list[str] | None
) -> list[dict[str, Any]]:
    """Load one task's test records, optionally keeping only selected images.
    读取一个任务的测试记录，可只保留指定图片对应的记录。
    """
    records = read_jsonl(data_root / f"VRSBench_test_{task}.jsonl")
    if image_ids is None:
        return records
    allowed = set(image_ids)
    return [
        record
        for record in records
        if (record.get("image_id") or Path(record["image"]).name) in allowed
    ]


def compute_meteor(gts: dict[str, list[str]], res: dict[str, list[str]]) -> float:
    """Compute METEOR via the official Java scorer when Java exists, else nltk.
    Java 可用时使用官方 pycocoevalcap METEOR（Java），否则回退到 nltk METEOR。
    """
    if shutil.which("java"):
        from pycocoevalcap.meteor.meteor import Meteor

        scorer = Meteor()
        try:
            score, _ = scorer.compute_score(gts, res)
            return float(score)
        finally:
            scorer.__del__()
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
    except ImportError as error:
        raise RuntimeError(
            "METEOR requires either java (official pycocoevalcap scorer) or "
            "nltk (Python fallback); install nltk and its wordnet data."
        ) from error
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError as error:
        raise RuntimeError(
            "nltk wordnet data is missing; run nltk.download('wordnet')."
        ) from error
    scores = []
    for image_id in gts:
        hypothesis = res[image_id][0].split()
        references = [reference.split() for reference in gts[image_id]]
        scores.append(meteor_score(references, hypothesis))
    return float(sum(scores) / len(scores))


def compute_caption_metrics(
    sample_answer_pairs: list[tuple[CanonicalSample, str]],
) -> dict[str, float]:
    """Compute BLEU-1..4 / METEOR / ROUGE-L / CIDEr over succeeded captions.
    对成功的 caption 样本计算 BLEU-1..4 / METEOR / ROUGE-L / CIDEr。
    """
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    gts: dict[str, list[str]] = {}
    res: dict[str, list[str]] = {}
    for sample, answer in sample_answer_pairs:
        gts[sample.id] = list(sample.answers)
        res[sample.id] = [answer]
    if not gts:
        return {}
    bleu_scores, _ = Bleu(4).compute_score(gts, res)
    rouge_score, _ = Rouge().compute_score(gts, res)
    cider_score, _ = Cider().compute_score(gts, res)
    meteor_score = compute_meteor(gts, res)
    return {
        "bleu_1": round(float(bleu_scores[0]), 6),
        "bleu_2": round(float(bleu_scores[1]), 6),
        "bleu_3": round(float(bleu_scores[2]), 6),
        "bleu_4": round(float(bleu_scores[3]), 6),
        "meteor": round(float(meteor_score), 6),
        "rouge_l": round(float(rouge_score), 6),
        "cider": round(float(cider_score), 6),
    }


def compute_vqa_accuracy_by_type(
    sample_exact_pairs: list[tuple[CanonicalSample, bool]],
) -> dict[str, dict[str, float | int]]:
    """Aggregate VQA exact-match accuracy per question type.
    按问题类型汇总 VQA 精确匹配准确率。
    """
    by_type: dict[str, dict[str, int]] = {}
    for sample, exact in sample_exact_pairs:
        question_type = str(sample.meta.get("question_type") or "unknown")
        stats = by_type.setdefault(question_type, {"total": 0, "exact_matches": 0})
        stats["total"] += 1
        if exact:
            stats["exact_matches"] += 1
    result: dict[str, dict[str, float | int]] = {}
    for question_type, stats in by_type.items():
        result[question_type] = {
            "total": stats["total"],
            "exact_matches": stats["exact_matches"],
            "accuracy": round(stats["exact_matches"] / stats["total"], 4),
        }
    return result


def normalize_answer(text: str) -> str:
    """Normalize an answer for deterministic exact matching.
    规范化答案以进行确定性精确匹配。
    """
    normalized = unicodedata.normalize("NFKC", text or "").strip().lower()
    normalized = re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)
    return normalized


def build_messages(sample: CanonicalSample) -> list[dict[str, Any]]:
    """Build the OpenAI-style chat message for one sample.
    为一条样本构建 OpenAI 风格对话消息。
    """
    content: list[dict[str, Any]] = [{"type": "image", "image": sample.images[0]}]
    content.append({"type": "text", "text": sample.prompt})
    return [{"role": "user", "content": content}]


def resolve_image_path(image_path: str, image_folder: Path) -> Path:
    """Resolve a possibly relative image path against the dataset root.
    将可能为相对路径的图片路径拼接到数据集根目录。
    """
    path = Path(image_path)
    if not path.is_absolute():
        path = image_folder / path
    return path


def infer_one(
    model: Any,
    processor: Any,
    sample: CanonicalSample,
    max_new_tokens: int,
    image_min_pixels: int,
    image_max_pixels: int,
    image_folder: Path,
    device: str,
) -> tuple[str, float]:
    """Generate one greedy prediction and return (text, duration_seconds).
    生成一条贪心预测并返回 (文本, 耗时秒)。
    """
    image_path = resolve_image_path(sample.images[0], image_folder)
    image = Image.open(image_path).convert("RGB")
    messages = build_messages(sample)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
        min_pixels=image_min_pixels,
        max_pixels=image_max_pixels,
    )
    inputs = {key: value.to(device) for key, value in inputs.items() if isinstance(value, torch.Tensor)}
    prompt_length = inputs["input_ids"].shape[1]
    start = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    duration = time.perf_counter() - start
    output_ids = generated[0][prompt_length:]
    answer = processor.batch_decode(
        output_ids.unsqueeze(0),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return answer, duration


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    """Atomically write one JSON object. / 原子写入一个 JSON 对象。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def main() -> int:
    args = build_parser().parse_args()
    check_runtime()
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    dtype = resolve_dtype(args.torch_dtype)
    config = AutoConfig.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
    )
    if config.model_type != "qwen3_vl":
        raise RuntimeError(
            f"This script supports qwen3_vl only, got model_type={config.model_type!r}."
        )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        config=config,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        device_map=args.device,
        local_files_only=args.local_files_only,
    )
    if args.adapter_path is not None:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = next(model.parameters()).device
    if device.type == "meta":
        raise RuntimeError(
            "Model parameters are on the meta device; pass an explicit device "
            "such as --device cuda:0 or --device cpu."
        )
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
    )

    image_ids = ordered_test_image_ids(args.data_root, args.max_images)
    print(
        f"Selected {len(image_ids)} test images "
        f"(max_images={args.max_images or 'no cap'})"
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "model_id": args.model_id,
        "adapter_path": args.adapter_path,
        "data_root": str(args.data_root),
        "tasks": list(args.tasks),
        "max_images": args.max_images,
        "num_images": len(image_ids),
        "output_path": str(args.output_path),
        "results": {},
    }
    total_records = 0
    total_failures = 0
    caption_pairs: list[tuple[CanonicalSample, str]] = []
    vqa_exact_pairs: list[tuple[CanonicalSample, bool]] = []
    with args.output_path.open("w", encoding="utf-8") as output:
        for task in args.tasks:
            records = load_task_records(args.data_root, task, image_ids)
            if args.max_samples is not None:
                records = records[: args.max_samples]
            max_new_tokens = (
                args.max_new_tokens_caption
                if task == "caption"
                else args.max_new_tokens_vqa
            )
            task_stats: dict[str, Any] = {
                "total": len(records),
                "succeeded": 0,
                "failed": 0,
                "exact_match": None,
            }
            task_durations: list[float] = []
            exact_matches = 0
            for record in records:
                total_records += 1
                sample = record_to_sample(record, task)
                try:
                    answer, duration = infer_one(
                        model=model,
                        processor=processor,
                        sample=sample,
                        max_new_tokens=max_new_tokens,
                        image_min_pixels=args.image_min_pixels,
                        image_max_pixels=args.image_max_pixels,
                        image_folder=args.data_root,
                        device=str(device),
                    )
                except Exception as error:  # noqa: BLE001 - failures are persisted per sample
                    total_failures += 1
                    task_stats["failed"] += 1
                    prediction = CanonicalPrediction(
                        id=sample.id,
                        task_type=task,
                        text="",
                        meta={
                            "error": f"{type(error).__name__}: {error}",
                            "model_id": args.model_id,
                            "adapter_path": args.adapter_path,
                        },
                    )
                    output.write(
                        json.dumps(
                            {
                                "sample": sample.serializable(),
                                "prediction": prediction.serializable(),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue
                task_stats["succeeded"] += 1
                task_durations.append(duration)
                if task == "caption":
                    caption_pairs.append((sample, answer))
                elif task == "vqa":
                    reference = sample.answers[0]
                    exact = normalize_answer(answer) == normalize_answer(reference)
                    if exact:
                        exact_matches += 1
                    vqa_exact_pairs.append((sample, exact))
                prediction = CanonicalPrediction(
                    id=sample.id,
                    task_type=task,
                    text=answer,
                    answer=answer if task == "vqa" else None,
                    meta={
                        "model_id": args.model_id,
                        "adapter_path": args.adapter_path,
                        "raw_text": answer,
                        "max_new_tokens": max_new_tokens,
                        "inference_seconds": round(duration, 4),
                    },
                )
                output.write(
                    json.dumps(
                        {
                            "sample": sample.serializable(),
                            "prediction": prediction.serializable(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if task == "caption" and task_stats["succeeded"]:
                task_stats["metrics"] = compute_caption_metrics(caption_pairs)
            elif task == "vqa" and task_stats["succeeded"]:
                task_stats["exact_match"] = round(
                    exact_matches / task_stats["succeeded"], 4
                )
                task_stats["all_accuracy"] = task_stats["exact_match"]
                task_stats["accuracy_by_type"] = compute_vqa_accuracy_by_type(
                    vqa_exact_pairs
                )
            if task_durations:
                task_stats["mean_inference_seconds"] = round(
                    sum(task_durations) / len(task_durations), 4
                )
            summary["results"][task] = task_stats
    summary["total_records"] = total_records
    summary["total_failures"] = total_failures
    write_json_atomic(summary, args.output_path.with_suffix(".summary.json"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
