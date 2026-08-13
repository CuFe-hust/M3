"""Unit tests for the persistent SSH client of the Qwen3-VL LoRA server.
Qwen3-VL LoRA 服务持久 SSH 客户端的单元测试。
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import qwen3vl_lora_remote as remote  # noqa: E402


def test_build_ssh_command_uses_one_persistent_server_session(monkeypatch) -> None:
    monkeypatch.setenv("REMOTE_USER", "lijia")
    monkeypatch.setenv("REMOTE_HOST", "100.88.222.9")
    monkeypatch.setenv("REMOTE_PORT", "1522")
    monkeypatch.setenv("REMOTE_REPO", "/home/lijia/M3")
    monkeypatch.setenv("REMOTE_PYTHON", "/home/lijia/miniconda3/envs/m3/bin/python")
    monkeypatch.setenv("MODEL_ID", "/models/qwen3_vl_8b")
    monkeypatch.setenv("ADAPTER_PATH", "/outputs/lora")
    args = remote.build_parser().parse_args(
        ["--image", "a.png", "--prompt", "q", "--max-new-tokens", "128"]
    )
    command = remote.build_ssh_command(args)
    assert command[0] == "ssh"
    assert "-T" in command
    assert "-p" in command
    assert command[command.index("-p") + 1] == "1522"
    assert "scripts/qwen3vl_lora_cli.py" in command[-1]
    assert "--server" in command[-1]
    assert "--local-files-only" in command[-1]
    assert "--max-new-tokens 128" in command[-1]
    assert "sshpass" not in command


def test_build_ssh_command_uses_sshpass_when_sshpss_set(monkeypatch) -> None:
    monkeypatch.setenv("SSHPASS", "secret")
    args = remote.build_parser().parse_args(
        ["--image", "a.png", "--prompt", "q"]
    )
    command = remote.build_ssh_command(args)
    assert command[:2] == ["sshpass", "-e"]


def test_build_ssh_command_uses_explicit_password(monkeypatch) -> None:
    monkeypatch.delenv("SSHPASS", raising=False)
    args = remote.build_parser().parse_args(
        ["--image", "a.png", "--prompt", "q"]
    )
    command = remote.build_ssh_command(args, password="secret")
    assert command[:2] == ["sshpass", "-e"]


def test_resolve_password_prompts_on_tty(monkeypatch) -> None:
    fake_stdin = io.StringIO()
    fake_stdin.isatty = lambda: True
    monkeypatch.setattr(sys, "stdin", fake_stdin)
    monkeypatch.setattr(remote.getpass, "getpass", lambda prompt: "s3cret")
    assert remote.resolve_password("lijia", "100.88.222.9", "1522") == "s3cret"


def test_resolve_password_requires_tty(monkeypatch) -> None:
    fake_stdin = io.StringIO()
    fake_stdin.isatty = lambda: False
    monkeypatch.setattr(sys, "stdin", fake_stdin)
    with pytest.raises(SystemExit):
        remote.resolve_password("lijia", "100.88.222.9", "1522")


def test_wait_until_ready_returns_after_model_marker() -> None:
    class FakeProc:
        stderr = io.StringIO("Loading weights...\nModel loaded. Ready.\n")

        def poll(self) -> None:
            return None

    remote.wait_until_ready(FakeProc())


def test_wait_until_ready_raises_when_process_exits() -> None:
    class FakeProc:
        stderr = io.StringIO("Permission denied, please try again.\n")

        def poll(self) -> int:
            return 255

    with pytest.raises(RuntimeError, match="exited before the model was ready"):
        remote.wait_until_ready(FakeProc())


def test_read_image_b64_roundtrip(tmp_path: Path) -> None:
    image = tmp_path / "a.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    import base64

    assert base64.b64decode(remote.read_image_b64(str(image))) == image.read_bytes()
    with pytest.raises(FileNotFoundError):
        remote.read_image_b64(str(tmp_path / "missing.png"))


def test_send_infer_roundtrip(tmp_path: Path) -> None:
    image = tmp_path / "a.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    class FakeProc:
        def __init__(self) -> None:
            self.stdin = io.StringIO()
            self.stdout = io.StringIO(
                json.dumps({"type": "result", "answer": "two ships"}) + "\n"
            )

    proc = FakeProc()
    response = remote.send_infer(proc, str(image), "How many?")
    assert response["answer"] == "two ships"
    sent = json.loads(proc.stdin.getvalue())
    assert sent["type"] == "infer"
    assert sent["prompt"] == "How many?"
    assert sent["image_b64"]
