"""Unit tests for scripts/finetune_vlm_lora.py (no LLaMA-Factory needed).
scripts/finetune_vlm_lora.py 的单元测试（不需要安装 LLaMA-Factory）。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "finetune_vlm_lora", ROOT / "scripts" / "finetune_vlm_lora.py"
)
assert _SPEC is not None and _SPEC.loader is not None
finetune = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(finetune)


def test_model_alias_normalization() -> None:
    assert finetune.normalize_model_name("InternVL3_5-8B") == "internvl3.5-8b"
    assert finetune.normalize_model_name("qwen3-VL-4B-Instruct") == "qwen3-vl-4b"
    with pytest.raises(Exception):
        finetune.normalize_model_name("unknown-model")


def test_server_allowed_rule() -> None:
    assert finetune.MODEL_PROFILES["internvl3.5-8b"]["server_allowed"] is True
    assert finetune.MODEL_PROFILES["qwen3-vl-4b"]["server_allowed"] is False


def test_build_dataset_info(tmp_path: Path) -> None:
    data_dir = tmp_path / "merged"
    (data_dir / "train").mkdir(parents=True)
    (data_dir / "train.json").write_text("[]", encoding="utf-8")
    (data_dir / "val.json").write_text("[]", encoding="utf-8")
    (data_dir / "test.json").write_text("[]", encoding="utf-8")
    info_path = finetune.build_dataset_info(data_dir, tmp_path / "cache")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    assert set(info) == {"merged_train", "merged_val", "merged_test"}
    assert info["merged_train"]["media_dir"] == str(data_dir.resolve())
    assert info["merged_train"]["columns"]["images"] == "images"
    assert info["merged_train"]["tags"]["user_tag"] == "human"


def test_build_dataset_info_missing_train(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        finetune.build_dataset_info(tmp_path, tmp_path / "cache")


def test_build_train_config_profile_and_resume(tmp_path: Path) -> None:
    data_dir = tmp_path / "merged"
    data_dir.mkdir()
    (data_dir / "train.json").write_text("[]", encoding="utf-8")
    (data_dir / "val.json").write_text("[]", encoding="utf-8")
    args = finetune.parse_args(
        ["--model", "internvl3.5-8b", "--data-dir", str(data_dir), "--output-dir", str(tmp_path / "out")]
    )
    profile = finetune.MODEL_PROFILES["internvl3.5-8b"]
    resume = tmp_path / "out" / "checkpoint-100"
    config = finetune.build_train_config(
        profile,
        "models/InternVL3_5-8B",
        data_dir,
        tmp_path / "cache" / "dataset_info.json",
        tmp_path / "out",
        args,
        resume,
    )
    assert config["template"] == "intern_vl"
    assert config["trust_remote_code"] is True
    assert config["resume_from_checkpoint"] == str(resume.resolve())
    assert config["eval_dataset"] == "merged_val"
    assert config["dataset_dir"] == str((tmp_path / "cache").resolve())
    assert config["media_dir"] == str(data_dir.resolve())


def test_build_train_config_qwen_profile(tmp_path: Path) -> None:
    data_dir = tmp_path / "merged"
    data_dir.mkdir()
    (data_dir / "train.json").write_text("[]", encoding="utf-8")
    args = finetune.parse_args(
        ["--model", "qwen3-vl-4b", "--data-dir", str(data_dir), "--output-dir", str(tmp_path / "out")]
    )
    config = finetune.build_train_config(
        finetune.MODEL_PROFILES["qwen3-vl-4b"],
        "Qwen/Qwen3-VL-4B-Instruct",
        data_dir,
        tmp_path / "cache" / "dataset_info.json",
        tmp_path / "out",
        args,
        None,
    )
    assert config["template"] == "qwen3_vl"
    assert config["lora_target"] == "all"
    assert "eval_dataset" not in config  # no val split provided


def _write_log(output_dir: Path, rows: list[dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trainer_log.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_select_best_checkpoint_eval_loss(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    (output_dir / "checkpoint-100").mkdir(parents=True)
    (output_dir / "checkpoint-200").mkdir()
    _write_log(
        output_dir,
        [
            {"current_steps": 100, "loss": 1.0, "eval_loss": 0.8},
            {"current_steps": 200, "loss": 0.7, "eval_loss": 0.5},
        ],
    )
    best = finetune.select_best_checkpoint(output_dir, "eval_loss", False)
    assert best["step"] == 200
    assert best["value"] == 0.5
    assert best["metric"] == "eval_loss"


def test_select_best_checkpoint_higher_is_better(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    (output_dir / "checkpoint-100").mkdir(parents=True)
    (output_dir / "checkpoint-200").mkdir()
    _write_log(
        output_dir,
        [
            {"current_steps": 100, "eval_loss": 0.8},
            {"current_steps": 200, "eval_loss": 0.5},
        ],
    )
    # lower eval_loss is still better here; higher-is-better is for metrics like BLEU.
    best = finetune.select_best_checkpoint(output_dir, "eval_loss", True)
    assert best["step"] == 100


def test_select_best_checkpoint_predict_metric(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    (output_dir / "checkpoint-100").mkdir(parents=True)
    (output_dir / "checkpoint-200").mkdir()
    _write_log(
        output_dir,
        [
            {"current_steps": 100, "eval_loss": 0.8, "predict_bleu-4": 0.2},
            {"current_steps": 200, "eval_loss": 0.5, "predict_bleu-4": 0.4},
        ],
    )
    best = finetune.select_best_checkpoint(output_dir, "predict_bleu-4", True)
    assert best["step"] == 200
    assert best["value"] == 0.4
    assert best["metric"] == "predict_bleu-4"


def test_select_best_checkpoint_falls_back_to_latest(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    (output_dir / "checkpoint-300").mkdir(parents=True)
    _write_log(
        output_dir,
        [
            {"current_steps": 100, "eval_loss": 0.1},
            {"current_steps": 300, "eval_loss": 0.9},
        ],
    )
    best = finetune.select_best_checkpoint(output_dir, "eval_loss", False)
    assert best["step"] == 300
    assert best["metric"] == "eval_loss->latest"


def test_select_best_checkpoint_train_loss_fallback(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    (output_dir / "checkpoint-100").mkdir(parents=True)
    (output_dir / "checkpoint-200").mkdir()
    _write_log(
        output_dir,
        [
            {"current_steps": 100, "loss": 1.2},
            {"current_steps": 200, "loss": 0.4},
        ],
    )
    best = finetune.select_best_checkpoint(output_dir, "eval_loss", False)
    assert best["step"] == 200
    assert best["metric"] == "train_loss"


def test_select_best_checkpoint_final_adapter(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    best = finetune.select_best_checkpoint(output_dir, "eval_loss", False)
    assert best["checkpoint"] == output_dir
    assert best["metric"] == "final_adapter"


def test_copy_adapter_files(tmp_path: Path) -> None:
    src = tmp_path / "checkpoint-100"
    src.mkdir()
    (src / "adapter_config.json").write_text("{}", encoding="utf-8")
    (src / "adapter_model.safetensors").write_bytes(b"weights")
    (src / "optimizer.pt").write_bytes(b"state")
    dst = tmp_path / "best_lora"
    finetune.copy_adapter_files(src, dst)
    assert (dst / "adapter_config.json").is_file()
    assert (dst / "adapter_model.safetensors").is_file()
    assert not (dst / "optimizer.pt").exists()


def test_fingerprint_changes_with_hyperparameters() -> None:
    args = finetune.parse_args(["--model", "qwen3-vl-4b"])
    first = finetune.build_fingerprint("Qwen/Qwen3-VL-4B-Instruct", Path("data"), args)
    args.lora_rank = 8
    second = finetune.build_fingerprint("Qwen/Qwen3-VL-4B-Instruct", Path("data"), args)
    assert first != second
