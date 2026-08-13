"""Single-image or interactive inference CLI for a Qwen3-VL LoRA adapter.
面向 Qwen3-VL LoRA 适配器的单图 / 交互式推理 CLI。

The script loads one base Qwen3-VL checkpoint, attaches one LoRA adapter, and
answers either a single image + prompt from the command line or a stdin-driven
prompt loop in ``--interactive`` mode. It is designed to run on the remote GPU
node (``100.88.222.9``, host ``qi2``) where the checkpoint and adapter are
already local; model and adapter paths come from CLI arguments or from the
``MODEL_ID`` / ``ADAPTER_PATH`` environment variables.
脚本加载一个基础 Qwen3-VL 权重并挂载一个 LoRA 适配器，既可从命令行回答单张
图片 + 提示词，也可在 ``--interactive`` 模式下从标准输入读取提示词循环推理。
脚本面向远端 GPU 节点（``100.88.222.9``，主机名 ``qi2``）运行，权重与适配器
均已在远端本地；模型与适配器路径来自 CLI 参数或 ``MODEL_ID`` /
``ADAPTER_PATH`` 环境变量。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    """Build the inference CLI. / 构建推理 CLI。"""
    parser = argparse.ArgumentParser(
        description="Run single-shot or interactive Qwen3-VL LoRA inference."
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("MODEL_ID"),
        help="Base Qwen3-VL checkpoint (local directory or Hugging Face id); "
        "falls back to MODEL_ID.",
    )
    parser.add_argument(
        "--adapter-path",
        default=os.environ.get("ADAPTER_PATH"),
        help="LoRA adapter directory produced by "
        "finetune_qwen3vl_merger_lora.py; falls back to ADAPTER_PATH.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Image path as seen by this process; the SSH launcher treats "
        "--image as a local path and uploads it automatically. In interactive "
        "mode the path can also be entered at the prompt.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt for one-shot inference; also runs first in interactive mode.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep the model loaded and read prompts from stdin.",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Load the model once and serve line-delimited JSON commands from "
        "stdin; used by the SSH launcher over one persistent connection.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Generation token ceiling; default 512.",
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
        help="Torch device map; use cuda:0 on the remote 4090 node.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Refuse network access while loading the checkpoint.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON file to persist answers and latencies.",
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


def resolve_image_path(image: Path) -> Path:
    """Resolve a CLI image path to an absolute path.
    将 CLI 图片路径解析为绝对路径。
    """
    return image.resolve()


def validate_args(args: argparse.Namespace) -> None:
    """Validate required CLI fields before loading the model.
    加载模型前校验 CLI 必填字段。
    """
    if not args.model_id:
        raise SystemExit("--model-id is required (or set MODEL_ID).")
    if not args.adapter_path:
        raise SystemExit("--adapter-path is required (or set ADAPTER_PATH).")
    if args.server:
        return
    if not args.prompt and not args.interactive:
        raise SystemExit("--prompt is required unless --interactive is used.")
    if args.prompt and not args.interactive and not args.image:
        raise SystemExit("--image is required for one-shot mode (or use --interactive).")


def build_messages(image_path: Path, prompt: str) -> list[dict[str, Any]]:
    """Build the OpenAI-style chat message for one image + prompt.
    为单张图片 + 提示词构建 OpenAI 风格聊天消息。
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def infer_one(
    model: Any,
    processor: Any,
    image_path: Path,
    prompt: str,
    max_new_tokens: int,
    image_min_pixels: int,
    image_max_pixels: int,
    device: str,
) -> tuple[str, float]:
    """Generate one greedy answer and return (text, duration_seconds).
    生成一条贪心回答并返回 (文本, 耗时秒)。
    """
    image = Image.open(image_path).convert("RGB")
    messages = build_messages(image_path, prompt)
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
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }
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


def load_model_and_processor(
    model_id: str,
    adapter_path: str,
    torch_dtype: str,
    attn_implementation: str,
    device: str,
    local_files_only: bool,
) -> tuple[Any, Any, str]:
    """Load the base Qwen3-VL model, attach the LoRA adapter, and freeze it.
    加载基础 Qwen3-VL 模型、挂载 LoRA 适配器并冻结全部参数。
    """
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    dtype = resolve_dtype(torch_dtype)
    config = AutoConfig.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )
    if config.model_type != "qwen3_vl":
        raise RuntimeError(
            f"This script supports qwen3_vl only, got model_type={config.model_type!r}."
        )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        config=config,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        device_map=device,
        local_files_only=local_files_only,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
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
        model_id,
        local_files_only=local_files_only,
    )
    return model, processor, str(device)


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    """Atomically write one JSON object. / 原子写入一个 JSON 对象。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def emit_json(payload: dict[str, Any], output_stream: Any = sys.stdout) -> None:
    """Write one JSON protocol line and flush it.
    写出一条 JSON 协议行并立即刷新。
    """
    print(json.dumps(payload, ensure_ascii=False), file=output_stream, flush=True)


def handle_infer_command(
    model: Any,
    processor: Any,
    device: str,
    image_b64: str,
    prompt: str,
    max_new_tokens: int,
    image_min_pixels: int,
    image_max_pixels: int,
) -> dict[str, Any]:
    """Run one inference from a base64 image and return a protocol response.
    根据 base64 图片执行一次推理并返回协议响应。
    """
    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
        with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as handle:
            handle.write(image_bytes)
            temp_path = Path(handle.name)
        try:
            answer, duration = infer_one(
                model=model,
                processor=processor,
                image_path=temp_path,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                image_min_pixels=image_min_pixels,
                image_max_pixels=image_max_pixels,
                device=device,
            )
        finally:
            temp_path.unlink(missing_ok=True)
        return {
            "type": "result",
            "answer": answer,
            "inference_seconds": round(duration, 4),
        }
    except Exception as error:  # noqa: BLE001 - protocol errors must reach the client
        return {
            "type": "error",
            "message": f"{type(error).__name__}: {error}",
        }


def run_server(
    model: Any,
    processor: Any,
    device: str,
    args: argparse.Namespace,
    input_stream: Any = None,
    output_stream: Any = None,
) -> int:
    """Serve JSON commands over stdin/stdout with one loaded model.
    通过 stdin/stdout 提供 JSON 指令服务，模型只加载一次。
    """
    input_stream = input_stream if input_stream is not None else sys.stdin
    output_stream = output_stream if output_stream is not None else sys.stdout
    print("Model loaded. Ready.", file=sys.stderr, flush=True)
    for line in input_stream:
        line = line.strip()
        if not line:
            continue
        try:
            command = json.loads(line)
        except json.JSONDecodeError as error:
            emit_json(
                {"type": "error", "message": f"Invalid JSON command: {error}"},
                output_stream,
            )
            continue
        if command.get("type") == "exit":
            break
        if command.get("type") != "infer":
            emit_json(
                {"type": "error", "message": f"Unsupported command: {command}"},
                output_stream,
            )
            continue
        response = handle_infer_command(
            model=model,
            processor=processor,
            device=device,
            image_b64=command.get("image_b64", ""),
            prompt=command.get("prompt", ""),
            max_new_tokens=args.max_new_tokens,
            image_min_pixels=args.image_min_pixels,
            image_max_pixels=args.image_max_pixels,
        )
        emit_json(response, output_stream)
    return 0


def read_image_path_interactively() -> Path:
    """Read one image path from stdin until an existing file is entered.
    从标准输入读取图片路径，直到输入一个存在的文件。
    """
    while True:
        if sys.stdin.isatty():
            print("Image path: ", end="", flush=True)
        line = sys.stdin.readline()
        if line == "":
            raise SystemExit("No image path provided.")
        image_path = resolve_image_path(Path(line.strip()))
        if image_path.is_file():
            return image_path
        print(f"Image file not found: {image_path}", file=sys.stderr)


def change_image_command(line: str) -> Path | None:
    """Parse ``!image <path>`` and return the new path, else None.
    解析 ``!image <path>`` 并返回新路径；不是该命令时返回 None。
    """
    prefix = "!image "
    if not line.startswith(prefix):
        return None
    raw_path = line[len(prefix) :].strip()
    if not raw_path:
        raise ValueError("!image requires a path.")
    return resolve_image_path(Path(raw_path))


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)

    check_runtime()
    model, processor, device = load_model_and_processor(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    if args.server:
        return run_server(model, processor, device, args)
    image_path = resolve_image_path(args.image) if args.image is not None else None

    results: list[dict[str, Any]] = []

    if args.prompt:
        if image_path is None:
            image_path = read_image_path_interactively()
        answer, duration = infer_one(
            model=model,
            processor=processor,
            image_path=image_path,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            image_min_pixels=args.image_min_pixels,
            image_max_pixels=args.image_max_pixels,
            device=device,
        )
        print(answer)
        results.append(
            {
                "prompt": args.prompt,
                "answer": answer,
                "inference_seconds": round(duration, 4),
            }
        )

    if args.interactive:
        if image_path is None:
            image_path = read_image_path_interactively()
        print(
            "Interactive mode; type a prompt and press Enter, or send EOF / "
            "'exit' to stop. Use '!image <path>' to switch the image.",
            file=sys.stderr,
        )
        while True:
            if sys.stdin.isatty():
                print(">>> ", end="", flush=True)
            line = sys.stdin.readline()
            if line == "":
                break
            prompt = line.strip()
            if prompt.startswith("!image "):
                try:
                    new_image_path = change_image_command(prompt)
                except ValueError as error:
                    print(str(error), file=sys.stderr)
                    continue
                if new_image_path.is_file():
                    image_path = new_image_path
                    print(f"Switched image to {image_path}", file=sys.stderr)
                else:
                    print(f"Image file not found: {new_image_path}", file=sys.stderr)
                continue
            if not prompt or prompt.lower() in {"exit", "quit"}:
                break
            answer, duration = infer_one(
                model=model,
                processor=processor,
                image_path=image_path,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                image_min_pixels=args.image_min_pixels,
                image_max_pixels=args.image_max_pixels,
                device=device,
            )
            print(answer)
            results.append(
                {
                    "prompt": prompt,
                    "answer": answer,
                    "inference_seconds": round(duration, 4),
                }
            )

    if args.output_json is not None:
        write_json_atomic(
            {
                "model_id": args.model_id,
                "adapter_path": args.adapter_path,
                "image_path": str(image_path),
                "max_new_tokens": args.max_new_tokens,
                "results": results,
            },
            args.output_json,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
