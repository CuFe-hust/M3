#!/usr/bin/env bash
set -euo pipefail
# Print observed server facts before choosing a runtime. / 选择运行时前输出实际服务器信息。
uname -a
cat /etc/os-release
nvidia-smi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"${PYTHON_BIN:?Set PYTHON_BIN}" --version
df -h
free -h
