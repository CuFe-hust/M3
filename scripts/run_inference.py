"""System-under-test CLI: request JSONL -> model inference -> prediction JSONL.

Invoked by m3rs-eval as ``system.command`` with ``{input_jsonl}`` and
``{output_jsonl}`` placeholders. Loads the LoRA fine-tuned Qwen3-VL-4B or
InternVL3.5-8B and produces one prediction record per request.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

MODEL_PATHS = {
    "qwen3vl": "/home/user/models/Qwen3_vl_4b_instruct",
    "internvl": "/home/user/models/InternVL3_5-8B",
}

_BOX_RE = re.compile(r"\[?\s*([\d.]+)\s*[,;\s]+([\d.]+)\s*[,;\s]+([\d.]+)\s*[,;\s]+([\d.]+)\s*\]?")
# VRSBench official ground-truth style: "{<86><82><95><90>}"
_VRS_BOX_RE = re.compile(r"\{<([\d.]+)><([\d.]+)><([\d.]+)><([\d.]+)>\}")
_CHOICE_RE = re.compile(r"\b([A-Ea-e])\b")


def _load_model(model_name: str, weights_dir: str, lora_dir: str | None, device: str):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    base_model = AutoModelForImageTextToText.from_pretrained(
        weights_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto" if device == "auto" else {"": device},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if lora_dir:
        base_model = PeftModel.from_pretrained(base_model, lora_dir)
    base_model.eval()
    processor = AutoProcessor.from_pretrained(weights_dir, trust_remote_code=True)
    return base_model, processor


def _build_messages(row: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[Any]]:
    images = [Image.open(path).convert("RGB") for path in row["images"]]
    prompt = row["prompt"]
    content: list[dict[str, Any]] = []
    for image in images:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    return messages, images


def _generate(
    model: Any,
    processor: Any,
    row: Mapping[str, Any],
    max_new_tokens: int,
    device: str,
) -> tuple[str, float]:
    import torch

    messages, images = _build_messages(row)
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # images must be a list even for single-image samples so the
        # placeholder count matches the token stream
        inputs = processor(text=[text], images=images, return_tensors="pt")
    except TypeError:
        # internvl-style processor: images passed separately
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    started = time.perf_counter()
    with torch.inference_mode():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                thinking=False,  # Qwen3-style models: emit the answer directly
            )
        except (TypeError, ValueError):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = processor.decode(generated, skip_special_tokens=True).strip()
    # defensive: strip any residual thinking block
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text, elapsed_ms


def _clamp_box(box: list[float]) -> list[float]:
    """Clamp model coordinates into the normalized inclusive 0-100 domain."""
    return [min(100.0, max(0.0, value)) for value in box]


def _parse_prediction(text: str, expected_output: str) -> tuple[str | None, list[list[float]] | None]:
    """Parse the model output into the prediction contract fields."""
    if expected_output == "boxes":
        match = _VRS_BOX_RE.search(text)
        if match:
            box = [float(match.group(i)) for i in range(1, 5)]
            return None, [_clamp_box(box)]
        match = _BOX_RE.search(text)
        if match:
            box = [float(match.group(i)) for i in range(1, 5)]
            return None, [_clamp_box(box)]
        return None, None
    if expected_output == "choice":
        match = _CHOICE_RE.search(text)
        if match:
            return match.group(1).upper(), None
        return text, None
    return text, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_inference")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_PATHS))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--weights-dir", type=Path, default=None)
    parser.add_argument("--lora-dir", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    weights_dir = str(args.weights_dir or MODEL_PATHS[args.model])
    lora_dir = str(args.lora_dir) if args.lora_dir else str(Path(weights_dir) / "lora")
    device = args.device if args.device != "auto" else "auto"
    model, processor = _load_model(args.model, weights_dir, lora_dir, device)

    with args.input.open("r", encoding="utf-8") as handle:
        requests = [json.loads(line) for line in handle if line.strip()]

    predictions: list[dict[str, Any]] = []
    for index, row in enumerate(requests, start=1):
        sample_id = row.get("sample_id")
        try:
            text, latency_ms = _generate(model, processor, row, args.max_new_tokens, device)
            prediction, boxes = _parse_prediction(text, row.get("expected_output", "text"))
            if prediction is None and boxes is None:
                predictions.append(
                    {
                        "sample_id": sample_id,
                        "status": "error",
                        "error_code": "parse_error",
                        "error": f"could not parse output: {text[:200]!r}",
                        "raw_output": text,
                        "latency_ms": latency_ms,
                    }
                )
            else:
                record: dict[str, Any] = {
                    "sample_id": sample_id,
                    "status": "ok",
                    "raw_output": text,
                    "latency_ms": latency_ms,
                }
                if prediction is not None:
                    record["prediction"] = prediction
                if boxes is not None:
                    record["boxes"] = boxes
                predictions.append(record)
        except Exception as error:  # per-sample isolation: failures never kill the run
            predictions.append(
                {
                    "sample_id": sample_id,
                    "status": "error",
                    "error_code": "inference_error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        if index % 10 == 0:
            print(f"processed {index}/{len(requests)}", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in predictions:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
