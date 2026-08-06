from __future__ import annotations

import os
import subprocess
import sys
import time

import psutil
import pytest

from m3rs_eval.command_adapter import run_system
from m3rs_eval.config import SystemCommandConfig
from m3rs_eval.windows_job import WindowsJobController


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows Job Objects require Windows")


def test_job_close_kills_owned_process():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], shell=False)
    controller = WindowsJobController.create()
    try:
        controller.assign(process)
        controller.close()
        process.wait(timeout=2)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)

    assert process.returncode is not None


def test_timeout_job_owns_descendant_processes(tmp_path):
    script = tmp_path / "spawn_child.py"
    child_pid_path = tmp_path / "child.pid"
    script.write_text(
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "open(sys.argv[1], 'w', encoding='utf-8').write(str(child.pid))\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    config = SystemCommandConfig(
        command=(sys.executable, str(script), str(child_pid_path)),
        working_directory=tmp_path,
        timeout_seconds=1,
        environment={},
    )
    child_pid: int | None = None
    try:
        result = run_system(config, tmp_path / "in", tmp_path / "out", tmp_path / "logs")
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 2
        while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
            time.sleep(0.02)

        assert result.timed_out
        assert not psutil.pid_exists(child_pid)
    finally:
        if child_pid is not None and psutil.pid_exists(child_pid):
            psutil.Process(child_pid).kill()
