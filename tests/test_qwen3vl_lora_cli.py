"""Unit tests for pure helpers of the Qwen3-VL LoRA CLI.
Qwen3-VL LoRA CLI 纯辅助函数的单元测试。
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import qwen3vl_lora_cli as cli  # noqa: E402


def test_parser_defaults_and_required_fields() -> None:
    args = cli.build_parser().parse_args(
        [
            "--model-id",
            "/models/qwen3_vl_8b",
            "--adapter-path",
            "/outputs/lora",
            "--image",
            "remote.png",
            "--prompt",
            "How many ships?",
        ]
    )
    assert args.model_id == "/models/qwen3_vl_8b"
    assert args.adapter_path == "/outputs/lora"
    assert args.interactive is False
    assert args.max_new_tokens == 512
    assert args.torch_dtype == "bfloat16"
    assert args.attn_implementation == "sdpa"


def test_validate_args_rejects_promptless_non_interactive() -> None:
    args = cli.build_parser().parse_args(
        ["--model-id", "m", "--adapter-path", "a", "--image", "i.png"]
    )
    with pytest.raises(SystemExit):
        cli.validate_args(args)


def test_validate_args_requires_model_and_adapter() -> None:
    args = cli.build_parser().parse_args(
        [
            "--image",
            "i.png",
            "--prompt",
            "q",
            "--interactive",
        ]
    )
    with pytest.raises(SystemExit):
        cli.validate_args(args)


def test_validate_args_requires_image_for_one_shot() -> None:
    args = cli.build_parser().parse_args(
        ["--model-id", "m", "--adapter-path", "a", "--prompt", "q"]
    )
    with pytest.raises(SystemExit):
        cli.validate_args(args)


def test_validate_args_allows_interactive_without_image() -> None:
    args = cli.build_parser().parse_args(
        ["--model-id", "m", "--adapter-path", "a", "--interactive"]
    )
    cli.validate_args(args)


def test_resolve_dtype() -> None:
    assert cli.resolve_dtype("bfloat16") == torch.bfloat16
    assert cli.resolve_dtype("auto") == "auto"
    with pytest.raises(ValueError):
        cli.resolve_dtype("not-a-dtype")


def test_resolve_image_path(tmp_path: Path) -> None:
    absolute = tmp_path / "a.png"
    assert cli.resolve_image_path(absolute) == absolute.resolve()
    relative = Path("relative.png")
    assert cli.resolve_image_path(relative) == relative.resolve()


def test_build_messages_contains_image_and_text(tmp_path: Path) -> None:
    image_path = tmp_path / "a.png"
    messages = cli.build_messages(image_path, "What is this?")
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["type"] == "image"
    assert messages[0]["content"][0]["image"] == str(image_path)
    assert messages[0]["content"][1]["type"] == "text"
    assert messages[0]["content"][1]["text"] == "What is this?"


def test_change_image_command_parses_and_validates() -> None:
    assert cli.change_image_command("What is this?") is None
    path = cli.change_image_command("!image /tmp/a.png")
    assert path == Path("/tmp/a.png").resolve()
    with pytest.raises(ValueError):
        cli.change_image_command("!image  ")


def test_handle_infer_command_returns_result(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color="blue").save(image_path)

    class FakeProcessor:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            assert messages[0]["content"][1]["text"] == "How many ships?"
            return "chat"

        def __call__(
            self,
            text=None,
            images=None,
            padding=True,
            return_tensors="pt",
            min_pixels=0,
            max_pixels=0,
        ):
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "pixel_values": torch.zeros(1, 3, 4, 4),
            }

        def batch_decode(
            self,
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ):
            return ["two ships"]

    class FakeModel:
        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3, 9, 9]])

    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    response = cli.handle_infer_command(
        model=FakeModel(),
        processor=FakeProcessor(),
        device="cpu",
        image_b64=image_b64,
        prompt="How many ships?",
        max_new_tokens=64,
        image_min_pixels=256 * 32 * 32,
        image_max_pixels=1280 * 32 * 32,
    )
    assert response["type"] == "result"
    assert response["answer"] == "two ships"
    assert response["inference_seconds"] >= 0


def test_handle_infer_command_returns_error_for_bad_base64() -> None:
    response = cli.handle_infer_command(
        model=None,
        processor=None,
        device="cpu",
        image_b64="not-base64!",
        prompt="?",
        max_new_tokens=64,
        image_min_pixels=256 * 32 * 32,
        image_max_pixels=1280 * 32 * 32,
    )
    assert response["type"] == "error"
    assert "message" in response


def test_run_server_loop_handles_exit_and_bad_json() -> None:
    stream_in = io.StringIO('not-json\n{"type":"exit"}\n')
    stream_out = io.StringIO()
    args = cli.build_parser().parse_args(
        [
            "--model-id",
            "m",
            "--adapter-path",
            "a",
            "--server",
        ]
    )
    cli.run_server(
        model=None,
        processor=None,
        device="cpu",
        args=args,
        input_stream=stream_in,
        output_stream=stream_out,
    )
    lines = [line for line in stream_out.getvalue().splitlines() if line.strip()]
    assert json.loads(lines[0])["type"] == "error"
    assert len(lines) == 1


def test_infer_one_uses_greedy_generation(tmp_path: Path) -> None:
    image_path = tmp_path / "img.png"
    Image.new("RGB", (8, 8), color="red").save(image_path)

    class FakeProcessor:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            assert messages[0]["content"][1]["text"] == "How many ships?"
            return "chat"

        def __call__(
            self,
            text=None,
            images=None,
            padding=True,
            return_tensors="pt",
            min_pixels=0,
            max_pixels=0,
        ):
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "pixel_values": torch.zeros(1, 3, 4, 4),
            }

        def batch_decode(
            self,
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ):
            assert tuple(ids.shape) == (1, 2)
            return ["two ships"]

    class FakeModel:
        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            return torch.tensor([[1, 2, 3, 9, 9]])

    answer, duration = cli.infer_one(
        model=FakeModel(),
        processor=FakeProcessor(),
        image_path=image_path,
        prompt="How many ships?",
        max_new_tokens=64,
        image_min_pixels=256 * 32 * 32,
        image_max_pixels=1280 * 32 * 32,
        device="cpu",
    )
    assert answer == "two ships"
    assert duration >= 0


def test_write_json_atomic(tmp_path: Path) -> None:
    output = tmp_path / "sub" / "result.json"
    cli.write_json_atomic({"answer": "a", "latency": 1.25}, output)
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "answer": "a",
        "latency": 1.25,
    }
