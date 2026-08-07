"""Unit tests for pure helpers of the merger LoRA training script.
merger LoRA 训练脚本纯辅助函数的单元测试。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import finetune_qwen3vl_merger_lora as trainer_script  # noqa: E402


def test_load_records_json(tmp_path: Path) -> None:
    path = tmp_path / "train.json"
    records = [{"id": "a", "image": "a.png", "conversations": []}]
    path.write_text(json.dumps(records), encoding="utf-8")
    assert trainer_script.load_records(str(path)) == records


def test_load_records_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    records = [{"id": "a"}, {"id": "b"}]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    assert trainer_script.load_records(str(path)) == records


def test_find_merger_linear_names() -> None:
    class FakeVisual(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.merger = nn.Linear(4, 4)
            self.deepstack_merger_list = nn.ModuleList(
                [nn.Linear(4, 4) for _ in range(3)]
            )

    class FakeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = FakeVisual()

    names = trainer_script.find_merger_linear_names(FakeModel())
    assert len(names) == 4
    assert all("merger" in name for name in names)
    assert any("deepstack_merger_list" in name for name in names)


def test_apply_merger_lora_rejects_too_few_merger_linears() -> None:
    class SmallModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = nn.Linear(4, 4)

    try:
        trainer_script.apply_merger_lora(
            SmallModel(),
            rank=4,
            alpha=8,
            dropout=0.05,
            bias="none",
            freeze_merger_base=True,
        )
    except RuntimeError as error:
        assert "at least 8 merger linear layers" in str(error)
    else:
        raise AssertionError("Expected RuntimeError for too few merger linear layers.")


def test_replace_image_tokens() -> None:
    assert trainer_script.replace_image_tokens("<image>\nWhat is this?") == (
        "<|vision_start|><|image_pad|><|vision_end|>\nWhat is this?"
    )


def test_resolve_dtype() -> None:
    assert trainer_script.resolve_dtype(torch, "bfloat16") == torch.bfloat16
    assert trainer_script.resolve_dtype(torch, "auto") == "auto"
