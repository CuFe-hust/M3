#!/usr/bin/env bash
set -euo pipefail
: "${QWEN_PORT:?Set QWEN_PORT}"
# Stop only the process listening on the configured local port. / 仅停止配置本地端口上的进程。
pids="$(ss -ltnp "sport = :${QWEN_PORT}" | awk -F'pid=' 'NR>1 {split($2,a,","); print a[1]}' | sort -u)"
if [[ -n "${pids}" ]]; then kill ${pids}; fi
