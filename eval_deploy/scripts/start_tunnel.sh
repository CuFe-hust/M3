#!/usr/bin/env bash
# Cloudflare quick tunnel for the participant API. Run inside tmux: bash start_tunnel.sh
# 快速隧道（公网 HTTPS）；建议在 tmux 会话中运行，URL 见控制台输出。
set -euo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost

# --protocol http2 is more reliable than quic from mainland networks.
# 从大陆网络连 Cloudflare 时 http2 通常比 quic 稳。
exec /usr/local/bin/cloudflared tunnel --no-autoupdate --protocol http2 \
  --url "http://127.0.0.1:${EVAL_PORT:-9001}"
