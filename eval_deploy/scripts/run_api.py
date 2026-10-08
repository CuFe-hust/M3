#!/usr/bin/env python3
"""Entry point: run the official participant API shell with the M3 bridge.

入口脚本：以 M3 桥接运行官方协议壳（participant_api_starter/server.py）。

Responsibilities / 职责:
- put the starter checkout, the deployment root (m3_bridge) and M3_ROOT on
  sys.path; 将官方壳、部署根目录（m3_bridge）与 M3_ROOT 加入 sys.path；
- chdir into M3_ROOT so M3 relative config paths resolve as on its own CLI;
  切换到 M3_ROOT，使 M3 相对配置路径与其 CLI 行为一致；
- configure INFO logging (bridge logs contain no secrets) and delegate to
  server.main(). 配置 INFO 日志（桥接日志不含敏感信息）并进入 server.main()。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def main() -> int:
    eval_root = Path(__file__).resolve().parent.parent
    starter = eval_root / "participant_api_starter"
    m3_root = Path(os.environ.get("M3_ROOT", "/home/tongyi/M3")).resolve()
    for entry in (str(m3_root), str(eval_root), str(starter)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    os.chdir(m3_root)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    from server import main as server_main

    server_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
