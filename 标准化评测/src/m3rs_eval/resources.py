"""Race-safe process-tree resource sampling and latency summaries."""

from __future__ import annotations

import math
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable

import psutil


@dataclass(frozen=True)
class LatencySummary:
    successes: int
    failures: int
    failure_rate: float
    p50_ms: float | None
    p95_ms: float | None


@dataclass(frozen=True)
class ResourceSummary:
    started_at: str
    finished_at: str
    duration_seconds: float
    sample_count: int
    peak_cpu_percent: float | None
    peak_rss_bytes: int | None
    peak_gpu_memory_bytes: int | None
    warnings: tuple[str, ...]


class ResourceSampler:
    """Sample a root PID and all currently reachable descendants at a fixed cadence."""

    def __init__(self, sample_interval_seconds: float = 1.0) -> None:
        if not isinstance(sample_interval_seconds, (int, float)) or sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be positive")
        self._interval = float(sample_interval_seconds)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._pid: int | None = None
        self._started_at: str | None = None
        self._started_monotonic: float | None = None
        self._sample_count = 0
        self._peak_cpu_percent: float | None = None
        self._peak_rss_bytes: int | None = None
        self._peak_gpu_memory_bytes: int | None = None
        self._warnings: set[str] = set()
        self._summary: ResourceSummary | None = None
        self._state = "idle"
        self._cpu_baselines: dict[tuple[int, float], tuple[float, float]] = {}
        self._gpu_probe = NvidiaSmiProbe()

    def start(self, pid: int) -> None:
        with self._condition:
            if self._summary is not None:
                raise RuntimeError("ResourceSampler cannot be restarted after stop")
            if self._thread is not None:
                if self._pid != pid:
                    raise RuntimeError("ResourceSampler is already sampling another PID")
                return
            self._pid = pid
            self._started_at = datetime.now(UTC).isoformat()
            self._started_monotonic = time.monotonic()
            self._state = "running"
            self._take_sample()
            self._thread = threading.Thread(target=self._run, name="m3rs-resource-sampler", daemon=True)
            self._thread.start()

    def stop(self) -> ResourceSummary:
        with self._condition:
            if self._summary is not None:
                return self._summary
            if self._thread is None or self._started_at is None or self._started_monotonic is None:
                raise RuntimeError("ResourceSampler has not been started")
            if self._state == "stopping":
                while self._summary is None:
                    self._condition.wait()
                return self._summary
            self._state = "stopping"
            thread = self._thread
            self._stop_event.set()
        thread.join(timeout=self._interval * 2 + 1.0)
        with self._condition:
            if self._summary is not None:
                return self._summary
            self._take_sample()
            self._summary = ResourceSummary(
                started_at=self._started_at,
                finished_at=datetime.now(UTC).isoformat(),
                duration_seconds=time.monotonic() - self._started_monotonic,
                sample_count=self._sample_count,
                peak_cpu_percent=self._peak_cpu_percent,
                peak_rss_bytes=self._peak_rss_bytes,
                peak_gpu_memory_bytes=self._peak_gpu_memory_bytes,
                warnings=tuple(sorted(self._warnings)),
            )
            self._state = "stopped"
            self._condition.notify_all()
            return self._summary

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval):
            with self._condition:
                self._take_sample()

    def _take_sample(self) -> None:
        if self._pid is None:
            return
        processes = _process_tree(self._pid)
        if not processes:
            self._warnings.add("process tree unavailable")
            return
        now = time.monotonic()
        cpu_percent = 0.0
        rss_bytes = 0
        live_pids: set[int] = set()
        seen_baselines: set[tuple[int, float]] = set()
        for process in processes:
            try:
                identity = (process.pid, process.create_time())
                cpu_times = process.cpu_times()
                cpu_total = cpu_times.user + cpu_times.system
                previous = self._cpu_baselines.get(identity)
                if previous is not None and now > previous[1]:
                    cpu_percent += (cpu_total - previous[0]) / (now - previous[1]) * 100
                self._cpu_baselines[identity] = (cpu_total, now)
                seen_baselines.add(identity)
                rss_bytes += process.memory_info().rss
                live_pids.add(process.pid)
            except (psutil.Error, OSError):
                continue
        self._cpu_baselines = {
            identity: baseline
            for identity, baseline in self._cpu_baselines.items()
            if identity in seen_baselines
        }
        self._sample_count += 1
        self._peak_cpu_percent = _maximum(self._peak_cpu_percent, cpu_percent)
        self._peak_rss_bytes = max(self._peak_rss_bytes or 0, rss_bytes)
        gpu_memory = self._gpu_probe.memory_for_pids(live_pids)
        self._warnings.update(self._gpu_probe.warnings)
        if gpu_memory is not None:
            self._peak_gpu_memory_bytes = max(self._peak_gpu_memory_bytes or 0, gpu_memory)


def summarize_latencies(latencies_ms: Iterable[float], failures: int) -> LatencySummary:
    """Summarize successful per-sample latencies with linear-interpolated quantiles."""
    values = sorted(float(value) for value in latencies_ms)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("latencies_ms must contain finite nonnegative values")
    if not isinstance(failures, int) or isinstance(failures, bool) or failures < 0:
        raise ValueError("failures must be a nonnegative integer")
    total = len(values) + failures
    return LatencySummary(
        successes=len(values),
        failures=failures,
        failure_rate=(failures / total) if total else 0.0,
        p50_ms=_quantile(values, 0.50),
        p95_ms=_quantile(values, 0.95),
    )


def _quantile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _process_tree(pid: int) -> list[psutil.Process]:
    try:
        root = psutil.Process(pid)
        return [root, *root.children(recursive=True)]
    except (psutil.Error, OSError):
        return []


class NvidiaSmiProbe:
    """Query per-process GPU memory once per sample and cache unsupported tooling."""

    def __init__(self) -> None:
        self._unsupported = False
        self.warnings: set[str] = set()

    def memory_for_pids(self, pids: set[int]) -> int | None:
        if not pids or self._unsupported:
            return None
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                check=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
                timeout=1,
            )
        except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
            self._unsupported = True
            self.warnings.add("nvidia-smi unavailable")
            return None
        memory, warnings = parse_gpu_process_rows(completed.stdout, pids)
        self.warnings.update(warnings)
        return memory


def parse_gpu_process_rows(output: str, pids: set[int]) -> tuple[int, set[str]]:
    """Parse nvidia-smi process CSV deterministically without retaining raw output."""
    total_mib = 0
    warnings: set[str] = set()
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            warnings.add("nvidia-smi returned an unparseable process row")
            continue
        try:
            if int(parts[0]) in pids:
                total_mib += int(parts[1])
        except ValueError:
            warnings.add("nvidia-smi returned an unparseable process memory value")
    return total_mib * 1024 * 1024, warnings


def _maximum(current: float | None, candidate: float) -> float:
    return candidate if current is None else max(current, candidate)
