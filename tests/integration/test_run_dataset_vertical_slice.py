"""Vertical slice: main.py run-dataset end to end with a fake model client.

垂直切片：main.py run-dataset 全链路（fake 模型客户端）。覆盖：解析→配置→
运行时（Qwen 注入一次）→DatasetRunOptions→DatasetRunner→汇总输出→退出码；
产物落在 run 目录且报告可构建。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

import main as main_module
from application.runtime import Runtime
from application.settings import AppSettings, RunSettings
from data.adapters.manifest import ManifestDraftAdapter
from data.registry import DatasetRegistry
from models.base import ModelCacheIdentity


class _FakeQwenClient:
    """Branches on the response model; records calls so the slice can assert
    single-client reuse. 按 response model 分支；记录调用以断言单客户端复用。"""

    def __init__(self) -> None:
        self.calls = 0

    @property
    def cache_identity(self) -> ModelCacheIdentity:
        return ModelCacheIdentity(
            model="fake",
            generation={"temperature": 0.0},
            client_version="1",
        )

    async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
        self.calls += 1
        return response_model.model_validate(
            {"agent_name": "general_vqa_agent", "answer": "yes", "status": "completed"}
        )


def _make_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (1, 2, 3)).save(root / "img.png", format="PNG")
    (root / "spacers_adapter.json").write_text(
        json.dumps(
            {
                "dataset": "auto-demo",
                "version": "1",
                "samples_file": "samples.jsonl",
                "fields": {
                    "id": "id",
                    "split": "split",
                    "question": "question",
                    "images": "images",
                    "task": "task",
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "samples.jsonl").write_text(
        json.dumps(
            {
                "id": "a1",
                "split": "test",
                "question": "Is there a road?",
                "images": ["img.png"],
                "task": "general_vqa",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_run_dataset_vertical_slice(tmp_path: Path, monkeypatch, capsys) -> None:
    data_root = tmp_path / "data"
    _make_dataset(data_root)
    settings = AppSettings(runs=RunSettings(root=tmp_path / "runs"))
    client = _FakeQwenClient()
    base = main_module.Runtime.create(
        settings=settings,
        project_root=tmp_path,
        prompts_root=Path(__file__).resolve().parents[2] / "prompts",
        qwen_client=client,
    )
    registry = DatasetRegistry()
    registry.register(
        "auto-demo", lambda: ManifestDraftAdapter("auto-demo", {"general_vqa", "caption"})
    )
    runtime = Runtime(
        settings=base.settings, components=base.components, registry=registry
    )
    monkeypatch.setattr(
        main_module.Runtime, "create", classmethod(lambda cls, **kwargs: runtime)
    )

    code = main_module.main(
        [
            "--config",
            str(tmp_path / "missing.yaml"),  # absent config falls back to defaults
            "run-dataset",
            "--dataset",
            "auto-demo",
            "--root",
            str(data_root),
            "--split",
            "test",
            "--auto-task",
            "--run-id",
            "slice-run",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    assert out["summaries"]["auto"]["succeeded"] == 1
    assert "slice-run" in out["run_dir"]
    run_dir = tmp_path / "runs" / "slice-run"
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "predictions.jsonl").is_file()
    assert (run_dir / "tasks" / "auto" / "dataset_probe.json").is_file()
    # The one runtime serves the whole run: Qwen loaded once, reused.
    # 单一运行时服务整个运行：Qwen 加载一次并复用。
    assert client.calls >= 1
    # The report is buildable from the run directory.
    # 报告可从 run 目录构建。
    report = runtime.build_report("slice-run")
    assert report.total == 1
    assert report.samples[0].prediction == "yes"


def test_run_dataset_slice_rejects_auto_task_conflict(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        main_module.Runtime, "create", classmethod(lambda **kwargs: object())
    )
    code = main_module.main(
        [
            "run-dataset",
            "--dataset",
            "d",
            "--root",
            str(tmp_path),
            "--split",
            "test",
            "--task",
            "caption",
            "--auto-task",
        ]
    )
    assert code == 2
    assert "mutually exclusive" in json.loads(capsys.readouterr().err)["error"]
