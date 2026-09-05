"""Restartable GPU worker proxies for VQA detector and segmenter clients.

VQA 检测与分割客户端的可重启 GPU worker 代理。每个 worker 拥有独立 CUDA
context；达到按 PID 计算的显存上限时只退出对应 worker，绝不操作 Qwen 进程。
"""

from __future__ import annotations

import multiprocessing as mp
import logging
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from models.base import (
    ModelCacheIdentity,
    ObjectDetectionOutput,
    SemanticMaskOutput,
)

LOGGER = logging.getLogger("evidence_gpu_worker")


class EvidenceGpuWorkerError(RuntimeError):
    """Stable isolated-worker failure. / 稳定的隔离 worker 失败。"""


@dataclass(frozen=True)
class EvidenceGpuWorkerPolicy:
    """Per-worker memory limits in binary GiB. / 单 worker 二进制 GiB 限制。"""

    soft_limit_gib: float
    hard_limit_gib: float
    device_free_floor_gib: float
    poll_interval_seconds: float = 1.0
    max_retries: int = 1

    def __post_init__(self) -> None:
        if not 0 < self.soft_limit_gib < self.hard_limit_gib:
            raise ValueError("worker limits require 0 < soft < hard")
        if self.device_free_floor_gib <= 0:
            raise ValueError("device_free_floor_gib must be positive")
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if self.max_retries != 1:
            raise ValueError("evidence worker retries are frozen at one")


def _nvidia_smi_memory(pid: int) -> tuple[int, int]:
    """Return worker-used and device-free bytes without importing CUDA libs.
    不导入 CUDA 库，返回 worker 占用与设备空闲字节。
    """
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    used_mib = 0
    worker_gpu_uuids: set[str] = set()
    for line in apps.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3 and fields[0].isdigit() and int(fields[0]) == pid:
            used_mib += int(fields[1])
            worker_gpu_uuids.add(fields[2])
    if not worker_gpu_uuids:
        # A freshly spawned lazy worker has no CUDA context before its first
        # model call. / 新 spawn 的惰性 worker 在首次模型调用前没有 CUDA context。
        return 0, 2**63 - 1
    free = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    free_values = []
    for line in free.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] in worker_gpu_uuids:
            free_values.append(int(fields[1]))
    if not free_values:
        raise EvidenceGpuWorkerError("gpu_memory_probe_unavailable")
    return used_mib * 1024**2, min(free_values) * 1024**2


def _worker_main(connection: Any, kind: str, specification: Mapping[str, Any]) -> None:
    """Construct one lazy runtime and serve CPU/PIL IPC requests.
    构造一个惰性 runtime，并处理 CPU/PIL IPC 请求。
    """
    client: Any = None
    try:
        if kind == "yolo":
            from agents.counting.backends.yolo_model_store import YoloModelStore
            from agents.counting.settings import YoloDetectorSettings
            from models.base import RuntimeObjectDetectionClient

            detector = YoloDetectorSettings.model_validate(specification["detector"])
            runtime = YoloModelStore().get(detector)
            client = RuntimeObjectDetectionClient(
                runtime,
                logical_model_id=detector.model_id,
                weights_sha256=detector.sha256,
            )
        elif kind == "segformer":
            from models.segformer_transformers import SegFormerTransformersClient
            from models.settings import SegFormerSettings

            settings = SegFormerSettings.model_validate(specification["settings"])
            labels = {int(key): value for key, value in specification["labels"].items()}
            client = SegFormerTransformersClient(settings, id_to_label=labels)
        else:
            raise ValueError("unsupported evidence worker kind")
        connection.send(("ready", None))
        while True:
            request = connection.recv()
            if request is None:
                return
            operation, args, kwargs = request
            try:
                result = getattr(client, operation)(*args, **kwargs)
            except Exception as error:
                connection.send(("error", type(error).__name__))
            else:
                connection.send(("ok", result))
    except Exception as error:
        try:
            connection.send(("startup_error", type(error).__name__))
        except Exception:
            pass
    finally:
        connection.close()


class _RestartableEvidenceWorker:
    """Synchronous, thread-safe proxy around one spawned GPU worker.
    单个 spawn GPU worker 的同步线程安全代理。
    """

    def __init__(
        self,
        *,
        kind: Literal["yolo", "segformer"],
        specification: Mapping[str, Any],
        identity: ModelCacheIdentity,
        policy: EvidenceGpuWorkerPolicy,
    ) -> None:
        self._kind = kind
        self._specification = dict(specification)
        self._identity = identity
        self._policy = policy
        self._context = mp.get_context("spawn")
        self._lock = threading.Lock()
        self._process: Any = None
        self._connection: Any = None

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return self._identity

    def close(self) -> None:
        """Release only this evidence worker's process and CUDA context.
        仅释放本 evidence worker 的进程与 CUDA context。
        """
        with self._lock:
            self._stop()

    def _start(self) -> None:
        parent, child = self._context.Pipe()
        process = self._context.Process(
            target=_worker_main,
            args=(child, self._kind, self._specification),
            daemon=True,
            name=f"m3-{self._kind}-gpu-worker",
        )
        process.start()
        child.close()
        state, detail = parent.recv()
        if state != "ready":
            process.join(timeout=5)
            parent.close()
            raise EvidenceGpuWorkerError(f"{self._kind}_worker_startup:{detail}")
        self._process = process
        self._connection = parent

    def _stop(self) -> None:
        process, connection = self._process, self._connection
        self._process = None
        self._connection = None
        if connection is not None:
            try:
                connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
        if process is not None:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        if connection is not None:
            connection.close()

    def _memory_exceeded(self) -> bool:
        if self._process is None:
            return False
        used, free = _nvidia_smi_memory(self._process.pid)
        gib = 1024**3
        return (
            used >= int(self._policy.soft_limit_gib * gib)
            or free < int(self._policy.device_free_floor_gib * gib)
        )

    def _watch(self, stop: threading.Event, exceeded: list[bool]) -> None:
        gib = 1024**3
        while not stop.wait(self._policy.poll_interval_seconds):
            process = self._process
            if process is None or not process.is_alive():
                return
            try:
                used, free = _nvidia_smi_memory(process.pid)
            except (OSError, subprocess.SubprocessError, EvidenceGpuWorkerError):
                continue
            if (
                used >= int(self._policy.hard_limit_gib * gib)
                or free < int(self._policy.device_free_floor_gib * gib / 2)
            ):
                exceeded.append(True)
                process.terminate()
                return

    def _call(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            for attempt in range(self._policy.max_retries + 1):
                if self._process is None or not self._process.is_alive():
                    self._stop()
                    self._start()
                stop = threading.Event()
                exceeded: list[bool] = []
                watcher = threading.Thread(
                    target=self._watch, args=(stop, exceeded), daemon=True
                )
                watcher.start()
                try:
                    self._connection.send((operation, args, kwargs))
                    state, payload = self._connection.recv()
                except (BrokenPipeError, EOFError, OSError):
                    state, payload = "worker_lost", None
                finally:
                    stop.set()
                    watcher.join(timeout=self._policy.poll_interval_seconds + 1)
                should_retry = state != "ok" or bool(exceeded)
                if should_retry:
                    LOGGER.warning(
                        "%s worker attempt %d failed with stable code %s",
                        self._kind,
                        attempt + 1,
                        "memory_limit" if exceeded else str(payload or state),
                    )
                    self._stop()
                    if attempt < self._policy.max_retries:
                        continue
                    reason = "memory_limit" if exceeded else str(payload or state)
                    raise EvidenceGpuWorkerError(f"{self._kind}_worker_failed:{reason}")
                soft_exceeded = False
                try:
                    soft_exceeded = self._memory_exceeded()
                except (OSError, subprocess.SubprocessError) as error:
                    self._stop()
                    raise EvidenceGpuWorkerError(
                        f"{self._kind}_worker_failed:gpu_memory_probe"
                    ) from error
                if soft_exceeded:
                    # The completed result is valid; recycle before the next
                    # request without rerunning this inference.
                    # 已完成结果仍有效；在下一次请求前回收，禁止重复本次推理。
                    self._stop()
                return payload
        raise EvidenceGpuWorkerError(f"{self._kind}_worker_failed:retry_exhausted")


class ProcessObjectDetectionClient(_RestartableEvidenceWorker):
    """ObjectDetectionClient hosted in a restartable CUDA worker."""

    def detect(self, image: Any, **kwargs: Any) -> list[ObjectDetectionOutput]:
        return self._call("detect", image, **kwargs)


class ProcessSemanticMaskClient(_RestartableEvidenceWorker):
    """SemanticMaskClient hosted in a restartable CUDA worker."""

    def segment(self, image: Any) -> SemanticMaskOutput:
        return self._call("segment", image)


def yolo_worker_client(
    detector: Any, policy: EvidenceGpuWorkerPolicy
) -> ProcessObjectDetectionClient:
    """Build a lazy path-private YOLO worker proxy. / 构造惰性且路径私有的 YOLO worker。"""
    return ProcessObjectDetectionClient(
        kind="yolo",
        specification={"detector": detector.model_dump(mode="python")},
        identity=ModelCacheIdentity(
            model=detector.model_id,
            generation={"weights_sha256": detector.sha256},
            client_version="yolo-detection-runtime-v1",
        ),
        policy=policy,
    )


def segformer_worker_client(
    settings: Any,
    labels: Mapping[int, str],
    policy: EvidenceGpuWorkerPolicy,
) -> ProcessSemanticMaskClient:
    """Build a lazy path-private SegFormer worker proxy. / 构造惰性且路径私有的 SegFormer worker。"""
    return ProcessSemanticMaskClient(
        kind="segformer",
        specification={
            "settings": settings.model_dump(mode="python"),
            "labels": dict(labels),
        },
        identity=ModelCacheIdentity(
            model=settings.logical_model_id,
            generation={"backend": "segformer_transformers", "dtype": settings.dtype},
            client_version="dense-v1",
            revision=settings.revision,
        ),
        policy=policy,
    )


__all__ = [
    "EvidenceGpuWorkerError",
    "EvidenceGpuWorkerPolicy",
    "ProcessObjectDetectionClient",
    "ProcessSemanticMaskClient",
    "segformer_worker_client",
    "yolo_worker_client",
]
