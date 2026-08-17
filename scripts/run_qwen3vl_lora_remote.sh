#!/usr/bin/env bash
set -euo pipefail
# Persistent SSH client entry for the Qwen3-VL LoRA server.
# Qwen3-VL LoRA 服务的持久 SSH 客户端入口。
#
# One SSH session is established once (one password prompt), the remote model
# is loaded once, then the CLI dialog opens only after the remote reports
# "Model loaded. Ready."; the local Python client repeatedly sends local
# image + question and prints remote answers. All connection parameters come
# from environment variables; no credentials are stored in this file.
# 只建立一次 SSH 会话（只需输入一次密码），远端模型只加载一次；远端报告
# “Model loaded. Ready.” 后才弹出 CLI 对话框，由本地 Python 客户端反复发送
# 本地图片 + 问题并打印远端回答。所有连接参数均来自环境变量；本文件不保存
# 任何凭据。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${LOCAL_PYTHON:-python3}" "${SCRIPT_DIR}/qwen3vl_lora_remote.py" "$@"
