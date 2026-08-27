#!/usr/bin/env python3
"""GPU memory monitor for M3 data preparation.

Two usages:
1) External monitor (standalone):
   python scripts/gpu_memory_monitor.py --interval 1 \
       --progress-log outputs/.../prepare_cpu.log \
       --output outputs/.../gpu_monitor.csv

2) In-process CUDA memory monitor (used by finetune script when
   M3_GPU_MONITOR=1):
   from scripts.gpu_memory_monitor import CudaMemoryMonitor
   mon = CudaMemoryMonitor(output_dir=..., tag="train")
   mon.start()
   ...
   mon.sample("after_agent_run")
   mon.stop()
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")




def log_cuda_memory_event(kind: str, event: str, **extra: Any) -> None:
    """Write a lightweight CUDA memory event from model code.

    Used by YOLO/SegFormer hooks when M3_GPU_MONITOR=1 and
    M3_GPU_MONITOR_DIR points to the monitor output directory.
    """
    if os.environ.get("M3_GPU_MONITOR") != "1":
        return
    monitor_dir = os.environ.get("M3_GPU_MONITOR_DIR")
    if not monitor_dir:
        return
    try:
        import torch  # noqa: PLC0415
        if not torch.cuda.is_available():
            return
        stats = torch.cuda.memory_stats()
        row: dict[str, Any] = {
            "timestamp": _now(),
            "kind": kind,
            "event": event,
            "allocated_bytes": stats.get("allocated_bytes.all.current"),
            "reserved_bytes": stats.get("reserved_bytes.all.current"),
            "active_bytes": stats.get("active_bytes.all.current"),
            "peak_allocated_bytes": stats.get("allocated_bytes.all.peak"),
            "peak_reserved_bytes": stats.get("reserved_bytes.all.peak"),
            "num_alloc_retries": stats.get("num_alloc_retries"),
            "num_ooms": stats.get("num_ooms"),
            "cuda_device": torch.cuda.current_device(),
        }
        row.update(extra)
        path = Path(monitor_dir) / "model_calls.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _run_nvidia_smi(args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["nvidia-smi", *args],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return proc.stdout.strip()
    except Exception as exc:
        return f"ERROR:{exc}"


def _parse_csv_line(line: str) -> list[str]:
    return next(csv.reader([line]))


def collect_gpu_snapshot() -> dict[str, Any]:
    """Collect one nvidia-smi snapshot (GPU + per-process compute apps)."""
    row: dict[str, Any] = {"timestamp": _now()}

    gpu_out = _run_nvidia_smi([
        "--query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ])
    if gpu_out and not gpu_out.startswith("ERROR"):
        parts = _parse_csv_line(gpu_out.splitlines()[0])
        if len(parts) >= 5:
            row.update({
                "gpu_index": parts[0].strip(),
                "gpu_memory_used_mb": parts[1].strip(),
                "gpu_memory_total_mb": parts[2].strip(),
                "gpu_util_percent": parts[3].strip(),
                "gpu_mem_util_percent": parts[4].strip(),
            })

    proc_out = _run_nvidia_smi([
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ])
    processes: list[dict[str, str]] = []
    if proc_out and not proc_out.startswith("ERROR") and "No running processes" not in proc_out:
        for line in proc_out.splitlines():
            parts = _parse_csv_line(line)
            if len(parts) >= 3:
                processes.append({
                    "pid": parts[0].strip(),
                    "process_name": parts[1].strip(),
                    "used_memory_mb": parts[2].strip(),
                })
    row["processes"] = json.dumps(processes, ensure_ascii=False)
    return row


def tail_progress(log_path: str | Path, pattern: str = r"prepared\s+(\w+)\s+(\d+)/(\d+)") -> str:
    """Return the last matching progress line from a log file."""
    path = Path(log_path)
    if not path.is_file():
        return ""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > 65536:
                f.seek(size - 65536)
            tail = f.read().decode(errors="replace")
        matches = re.findall(pattern, tail)
        if matches:
            split, current, total = matches[-1]
            return f"{split} {current}/{total}"
    except Exception:
        pass
    return ""


def external_monitor_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Poll nvidia-smi and log GPU memory.")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--output", required=True, help="CSV output path")
    parser.add_argument("--progress-log", default=None, help="Optional data-prep log to correlate progress")
    parser.add_argument("--no-header", action="store_true")
    args = parser.parse_args(argv)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.no_header and (not out_path.exists() or out_path.stat().st_size == 0)

    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "timestamp",
            "gpu_index",
            "gpu_memory_used_mb",
            "gpu_memory_total_mb",
            "gpu_util_percent",
            "gpu_mem_util_percent",
            "processes",
            "progress",
        ]
        if write_header:
            writer.writerow(header)
            f.flush()
        while True:
            row = collect_gpu_snapshot()
            row["progress"] = (
                tail_progress(args.progress_log) if args.progress_log else ""
            )
            writer.writerow([row.get(h, "") for h in header])
            f.flush()
            time.sleep(max(0.1, args.interval))
    return 0


class CudaMemoryMonitor:
    """In-process PyTorch CUDA memory sampling.

    Only active when torch.cuda.is_available(). Writes JSONL rows with
    allocated/reserved/peak memory and can optionally record allocation
    history for stack traces.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        tag: str = "default",
        interval: float = 5.0,
        record_history: bool = False,
    ) -> None:
        self.tag = tag
        self.interval = interval
        self.record_history = record_history
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._torch = None
        self._enabled = False
        self._output_path: Path | None = None
        self._history_started = False

        try:
            import torch  # noqa: PLC0415
            self._torch = torch
            self._enabled = torch.cuda.is_available()
        except Exception:
            self._enabled = False

        if self._enabled:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            self._output_path = out_dir / f"cuda_memory_{tag}.jsonl"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _stats_dict(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        torch = self._torch
        stats = torch.cuda.memory_stats()
        row: dict[str, Any] = {
            "timestamp": _now(),
            "tag": self.tag,
            "allocated_bytes": stats.get("allocated_bytes.all.current"),
            "reserved_bytes": stats.get("reserved_bytes.all.current"),
            "active_bytes": stats.get("active_bytes.all.current"),
            "peak_allocated_bytes": stats.get("allocated_bytes.all.peak"),
            "peak_reserved_bytes": stats.get("reserved_bytes.all.peak"),
            "num_alloc_retries": stats.get("num_alloc_retries"),
            "num_ooms": stats.get("num_ooms"),
            "cuda_device": torch.cuda.current_device(),
        }
        if extra:
            row.update(extra)
        return row

    def _write(self, row: dict[str, Any]) -> None:
        if self._output_path is None:
            return
        with self._lock:
            with self._output_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def reset_peak(self) -> None:
        if self._enabled:
            self._torch.cuda.reset_peak_memory_stats()

    def start(self) -> None:
        if not self._enabled or self._thread is not None:
            return
        if self.record_history and hasattr(self._torch.cuda.memory, "_record_memory_history"):
            try:
                self._torch.cuda.memory._record_memory_history(
                    enabled="all",
                    stack="python",
                    device="cuda",
                )
                self._history_started = True
            except Exception:
                self._history_started = False
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample("periodic")
            self._stop.wait(self.interval)

    def sample(self, event: str = "manual", **extra: Any) -> dict[str, Any] | None:
        if not self._enabled:
            return None
        row = self._stats_dict({"event": event, **extra})
        self._write(row)
        return row

    def summary(self, path: str | Path | None = None) -> None:
        if not self._enabled:
            return
        text = self._torch.cuda.memory_summary()
        target = Path(path) if path is not None else (
            self._output_path.with_suffix(".summary.txt") if self._output_path else None
        )
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as f:
                f.write(f"\n===== {_now()} {self.tag} =====\n")
                f.write(text)
                f.write("\n")

    def dump_snapshot(self, path: str | Path | None = None) -> None:
        """Dump a PyTorch CUDA memory snapshot for offline analysis."""
        if not self._enabled or not self._history_started:
            return
        target = Path(path) if path is not None else (
            self._output_path.with_suffix(".snapshot.pkl.gz") if self._output_path else None
        )
        if target is None:
            return
        try:
            self._torch.cuda.memory._dump_snapshot(str(target))
        except Exception:
            pass

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2)
            self._thread = None
        if self._history_started:
            self.dump_snapshot()
            self.summary()
        if self._history_started and hasattr(self._torch.cuda.memory, "_record_memory_history"):
            try:
                self._torch.cuda.memory._record_memory_history(enabled=None)
            except Exception:
                pass
            self._history_started = False

    def __enter__(self) -> "CudaMemoryMonitor":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


if __name__ == "__main__":
    raise SystemExit(external_monitor_main())
