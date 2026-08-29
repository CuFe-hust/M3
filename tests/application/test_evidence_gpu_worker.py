"""Offline tests for isolated evidence GPU worker guards.

隔离 evidence GPU worker 保护的离线测试。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from application.evidence_gpu_worker import (
    EvidenceGpuWorkerPolicy,
    ProcessObjectDetectionClient,
    _nvidia_smi_memory,
)
from models.base import ModelCacheIdentity


def test_worker_policy_requires_ordered_limits_and_one_retry() -> None:
    assert EvidenceGpuWorkerPolicy(6, 8, 8).max_retries == 1
    with pytest.raises(ValueError, match="soft < hard"):
        EvidenceGpuWorkerPolicy(8, 8, 8)
    with pytest.raises(ValueError, match="frozen at one"):
        EvidenceGpuWorkerPolicy(6, 8, 8, max_retries=2)


def test_nvidia_smi_memory_attributes_only_the_worker_gpu(monkeypatch) -> None:
    responses = iter(
        [
            "10, 1024, GPU-a\n20, 4096, GPU-b\n10, 512, GPU-a\n",
            "GPU-a, 16384\nGPU-b, 2048\n",
        ]
    )

    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout=next(responses))

    monkeypatch.setattr("application.evidence_gpu_worker.subprocess.run", fake_run)

    used, free = _nvidia_smi_memory(10)

    assert used == 1536 * 1024**2
    assert free == 16384 * 1024**2


def test_soft_limit_recycles_after_returning_completed_result(monkeypatch) -> None:
    client = ProcessObjectDetectionClient(
        kind="yolo",
        specification={},
        identity=ModelCacheIdentity(model="test", generation={}, client_version="1"),
        policy=EvidenceGpuWorkerPolicy(6, 8, 8),
    )

    class FakeProcess:
        @staticmethod
        def is_alive() -> bool:
            return True

    class FakeConnection:
        @staticmethod
        def send(value) -> None:
            assert value[0] == "detect"

        @staticmethod
        def recv():
            return "ok", ["completed"]

    client._process = FakeProcess()
    client._connection = FakeConnection()
    stopped: list[bool] = []
    monkeypatch.setattr(client, "_watch", lambda stop, exceeded: None)
    monkeypatch.setattr(client, "_memory_exceeded", lambda: True)
    monkeypatch.setattr(client, "_stop", lambda: stopped.append(True))

    result = client.detect(
        "image",
        confidence=0.5,
        iou=0.5,
        image_size=1024,
        device="0",
        max_detections=10,
    )

    assert result == ["completed"]
    assert stopped == [True]


def test_close_releases_only_the_owned_worker(monkeypatch) -> None:
    client = ProcessObjectDetectionClient(
        kind="yolo",
        specification={},
        identity=ModelCacheIdentity(model="test", generation={}, client_version="1"),
        policy=EvidenceGpuWorkerPolicy(6, 8, 8),
    )
    stopped: list[bool] = []
    monkeypatch.setattr(client, "_stop", lambda: stopped.append(True))

    client.close()

    assert stopped == [True]
