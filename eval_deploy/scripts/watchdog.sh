#!/usr/bin/env bash
# Watchdog: 5-minute self-check loop (local API / public tunnel / GPU / disk).
# Keep-alive: restarts cloudflared if it died (quick-tunnel URL will change; logged).
# Run inside tmux: bash ~/eval/scripts/watchdog.sh
# 巡检：5 分钟一轮；隧道进程消失时自动重启并记录新 URL（新 URL 需在评测网站重新登记）。
set -u
# .bashrc exports a local mihomo proxy for interactive shells; curl to the
# public tunnel must NOT go through it (direct works; proxy breaks TLS).
# 交互 shell 会继承本机代理变量；公网探测必须直连。
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost
EVAL=/home/tongyi/eval
LOG=$EVAL/logs/watch.log
KEY="$(cat "$EVAL/secrets/model_api_key")"
last_url="https://dealers-shower-accessories-stores.trycloudflare.com"

while true; do
  ts="$(date '+%F %T')"
  health="$(curl -s -m 8 -H "Authorization: Bearer $KEY" http://127.0.0.1:9001/healthz | head -c 80)"
  gpu="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 1 2>/dev/null)"
  disk="$(df -h / | awk 'NR==2{print $4}')"

  if ! pgrep -f "cloudflared.*--url http://127.0.0.1:9001" >/dev/null; then
    echo "$ts ALERT tunnel process missing; restarting cloudflared" >> "$LOG"
    nohup bash "$EVAL/scripts/start_tunnel.sh" >> "$EVAL/logs/tunnel.log" 2>&1 &
    sleep 25
    new_url="$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$EVAL/logs/tunnel.log" | tail -1)"
    echo "$ts ALERT tunnel restarted url=$new_url (需在评测网站重新登记 Endpoint)" >> "$LOG"
    last_url="$new_url"
  fi

  pub="$(curl -s -m 10 "$last_url/healthz" | head -c 80)"
  echo "$ts | local=${health:0:48} | tunnel=${pub:0:48} | gpu1=${gpu} | disk_free=${disk}" >> "$LOG"

  if [ "$(wc -l < "$LOG")" -gt 1200 ]; then
    tail -n 600 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
  fi
  sleep 300
done
