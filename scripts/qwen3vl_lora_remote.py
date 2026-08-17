#!/usr/bin/env python3
"""Persistent SSH client for the remote Qwen3-VL LoRA server.
远程 Qwen3-VL LoRA 服务的持久 SSH 客户端。

The client establishes ONE ssh session, prompts for the SSH password once,
asks the remote process to load the model once, waits for the remote
"Model loaded. Ready." marker, then opens a CLI dialog that repeatedly sends
base64 image + prompt commands over the same session and prints the returned
answers. This avoids repeated password prompts and repeated model loads.
客户端只建立一次 SSH 会话，远端进程只加载一次模型；之后在同一会话内反复发送
base64 图片 + 提示词指令，并打印返回的回答，避免重复输密码和重复加载模型。
客户端启动时若未设置 SSHPASS，会在本机 TTY 上提示输入一次密码；远端模型加载
完成（stderr 出现 “Model loaded. Ready.”）后才弹出 CLI 对话框，输入本地图片
路径与问题后发送到远端推理，并等待下一次输入。
"""

from __future__ import annotations

import argparse
import base64
import getpass
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any


def get_env(name: str, default: str) -> str:
    """Read an environment variable with a fallback default.
    读取环境变量，缺失时使用默认值。
    """
    return os.environ.get(name) or default


def resolve_password(user: str, host: str, port: str) -> str:
    """Prompt once on a TTY for the SSH password.
    在本机 TTY 上提示输入一次 SSH 密码。
    """
    if not sys.stdin.isatty():
        raise SystemExit(
            "SSH password prompt requires a terminal; set SSHPASS to provide "
            "the password non-interactively, or run from a terminal."
        )
    return getpass.getpass(f"SSH password for {user}@{host} (port {port}): ")


def build_parser() -> argparse.ArgumentParser:
    """Build the persistent client CLI. / 构建持久客户端 CLI。"""
    parser = argparse.ArgumentParser(
        description="Persistent SSH client for the remote Qwen3-VL LoRA server."
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Local image path for one-shot mode.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Question text for one-shot mode.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Loop locally: enter an image path and a question each round.",
    )
    parser.add_argument("--model-id", default=None, help="Override MODEL_ID.")
    parser.add_argument("--adapter-path", default=None, help="Override ADAPTER_PATH.")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--image-min-pixels", type=int)
    parser.add_argument("--image-max-pixels", type=int)
    parser.add_argument(
        "--torch-dtype",
        choices=("float32", "float16", "bfloat16", "auto"),
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2", "eager"),
    )
    parser.add_argument("--device")
    return parser


def build_ssh_command(
    args: argparse.Namespace, *, password: str | None = None
) -> list[str]:
    """Build one ssh command that starts the remote server.
    构建一条启动远端服务器的 ssh 命令。
    """
    user = get_env("REMOTE_USER", "lijia")
    host = get_env("REMOTE_HOST", "100.88.222.9")
    port = get_env("REMOTE_PORT", "1522")
    repo = get_env("REMOTE_REPO", "/home/lijia/M3")
    python = get_env("REMOTE_PYTHON", "/home/lijia/miniconda3/envs/m3/bin/python")
    model = args.model_id or get_env(
        "MODEL_ID", "/home/lijia/M3/models/qwen3_vl_8b/weights"
    )
    adapter = args.adapter_path or get_env(
        "ADAPTER_PATH", "/home/lijia/M3/outputs/finetune/qwen3-vl-8b-merger-lora"
    )

    ssh_cmd: list[str] = []
    if password or os.environ.get("SSHPASS"):
        ssh_cmd += ["sshpass", "-e"]
    ssh_cmd += [
        "ssh",
        "-T",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-p",
        port,
        f"{user}@{host}",
    ]

    remote_parts = [
        "cd",
        shlex.quote(repo),
        "&&",
        "exec",
        shlex.quote(python),
        "scripts/qwen3vl_lora_cli.py",
        "--server",
        "--model-id",
        shlex.quote(model),
        "--adapter-path",
        shlex.quote(adapter),
        "--local-files-only",
    ]
    server_flags = (
        ("--max-new-tokens", args.max_new_tokens),
        ("--image-min-pixels", args.image_min_pixels),
        ("--image-max-pixels", args.image_max_pixels),
        ("--torch-dtype", args.torch_dtype),
        ("--attn-implementation", args.attn_implementation),
        ("--device", args.device),
    )
    for flag, value in server_flags:
        if value is not None:
            remote_parts += [flag, shlex.quote(str(value))]
    ssh_cmd.append(" ".join(remote_parts))
    return ssh_cmd


def wait_until_ready(proc: subprocess.Popen[str]) -> None:
    """Wait for the remote model-loaded marker on stderr.
    等待远端在 stderr 上输出模型加载完成标记。
    """
    ready = threading.Event()
    stderr_lines: list[str] = []

    def read_stderr() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            text = line.rstrip("\n")
            stderr_lines.append(text)
            print(text, file=sys.stderr, flush=True)
            if "Model loaded. Ready." in text:
                ready.set()

    thread = threading.Thread(target=read_stderr, name="remote-stderr", daemon=True)
    thread.start()
    while not ready.is_set():
        if proc.poll() is not None or not thread.is_alive():
            thread.join(timeout=2)
            tail = "\n".join(stderr_lines[-15:])
            raise RuntimeError(
                "Remote process exited before the model was ready; "
                f"last stderr lines:\n{tail or '(none)'}"
            )
        ready.wait(timeout=0.2)
    thread.join(timeout=0.01)


def read_image_b64(path_text: str) -> str:
    """Read a local image and return its base64 payload.
    读取本地图片并返回 base64 内容。
    """
    path = Path(path_text).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Local image not found: {path}")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def send_infer(
    proc: subprocess.Popen[str],
    image_path: str,
    prompt: str,
) -> dict[str, Any]:
    """Send one infer command and read its protocol response.
    发送一条推理指令并读取协议响应。
    """
    assert proc.stdin is not None and proc.stdout is not None
    command = {
        "type": "infer",
        "image_b64": read_image_b64(image_path),
        "prompt": prompt,
    }
    proc.stdin.write(json.dumps(command, ensure_ascii=False) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("Remote process closed unexpectedly.")
    return json.loads(line)


def run_interactive(proc: subprocess.Popen[str]) -> int:
    """Run the local input loop against one persistent remote server.
    针对同一个持久远端服务器运行本地输入循环。
    """
    print(
        "Model loaded. CLI dialog ready.",
        file=sys.stderr,
    )
    print(
        "Enter a local image path and a question each round; "
        "type 'exit' or 'quit' to stop.",
        file=sys.stderr,
    )
    while True:
        try:
            image_path = input("Image path (local): ").strip()
        except EOFError:
            break
        if not image_path or image_path.lower() in {"exit", "quit"}:
            break
        try:
            question = input("Question: ").strip()
        except EOFError:
            break
        if not question or question.lower() in {"exit", "quit"}:
            break
        try:
            response = send_infer(proc, image_path, question)
        except Exception as error:  # noqa: BLE001 - keep the loop alive
            print(f"Error: {error}", file=sys.stderr)
            continue
        if response.get("type") == "error":
            print(
                f"Remote error: {response.get('message', '')}",
                file=sys.stderr,
            )
        else:
            print(response.get("answer", ""))
    return 0


def run_one_shot(proc: subprocess.Popen[str], args: argparse.Namespace) -> int:
    """Send one infer command and print the answer.
    发送一条推理指令并打印回答。
    """
    try:
        response = send_infer(proc, args.image, args.prompt)
    except Exception as error:  # noqa: BLE001 - report and fail visibly
        print(f"Error: {error}", file=sys.stderr)
        return 1
    if response.get("type") == "error":
        print(f"Remote error: {response.get('message', '')}", file=sys.stderr)
        return 1
    print(response.get("answer", ""))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.interactive and not (args.image and args.prompt):
        parser.error("Provide --image and --prompt, or use --interactive.")

    user = get_env("REMOTE_USER", "lijia")
    host = get_env("REMOTE_HOST", "100.88.222.9")
    port = get_env("REMOTE_PORT", "1522")
    dry_run = os.environ.get("DRY_RUN") == "1"

    password = os.environ.get("SSHPASS")
    if password is None and not dry_run:
        password = resolve_password(user, host, port)

    ssh_cmd = build_ssh_command(args, password=password)
    if dry_run:
        print(" ".join(shlex.quote(part) for part in ssh_cmd))
        return 0

    if password and shutil.which("sshpass") is None:
        raise SystemExit(
            "sshpass is required to supply the SSH password automatically; "
            "install it (for example: brew install sshpass) or use "
            "SSH key authentication with an empty SSHPASS variable."
        )

    env = os.environ.copy()
    if password:
        env["SSHPASS"] = password

    print(f"Connecting to {user}@{host}:{port} ...", file=sys.stderr)
    proc = subprocess.Popen(
        ssh_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    try:
        wait_until_ready(proc)
        if args.interactive:
            return run_interactive(proc)
        return run_one_shot(proc, args)
    finally:
        try:
            if proc.stdin is not None:
                proc.stdin.write(json.dumps({"type": "exit"}) + "\n")
                proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
