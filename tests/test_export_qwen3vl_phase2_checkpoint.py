"""Tests for scripts/export_qwen3vl_phase2_checkpoint.py (task docs/train/04).

Unit tests use a tiny fake Qwen model tree (mirroring the transformers
5.14.1 Qwen3-VL layout), a fake processor, the real peft library of the M3
env (peft 0.20.0) as the injection seam, and a composite checkpoint built
with the real producer helpers (scripts/finetune_qwen3vl_phase2.py). No 8B
weights are loaded anywhere; every transformers loading seam is substituted.

测试使用微型 fake Qwen 模型树、fake processor、M3 环境真实 peft 0.20.0
注入 seam，并用训练脚本自身的辅助函数构造真实格式的复合 checkpoint；
不加载任何 8B 权重；所有 transformers 加载 seam 均被替换。
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import safetensors.torch
import torch
import torch.nn as nn
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTER_PATH = REPO_ROOT / "scripts" / "export_qwen3vl_phase2_checkpoint.py"
FINETUNE_PATH = REPO_ROOT / "scripts" / "finetune_qwen3vl_phase2.py"

_spec_exp = importlib.util.spec_from_file_location(
    "export_qwen3vl_phase2_checkpoint", EXPORTER_PATH
)
assert _spec_exp is not None and _spec_exp.loader is not None
exp = importlib.util.module_from_spec(_spec_exp)
sys.modules[_spec_exp.name] = exp
_spec_exp.loader.exec_module(exp)

_spec_ft = importlib.util.spec_from_file_location(
    "finetune_qwen3vl_phase2", FINETUNE_PATH
)
assert _spec_ft is not None and _spec_ft.loader is not None
ft = importlib.util.module_from_spec(_spec_ft)
sys.modules[_spec_ft.name] = ft
_spec_ft.loader.exec_module(ft)

ExportError = exp.ExportError
CheckpointValidationError = exp.CheckpointValidationError
BaseIdentityError = exp.BaseIdentityError
MergerLoadError = exp.MergerLoadError
LoRAValidationError = exp.LoRAValidationError
ReloadValidationError = exp.ReloadValidationError


# ---------------------------------------------------------------------------
# Fake Qwen3-VL model tree (transformers 5.14.1 layout), fake config,
# fake processor (contract mirrors the pinned Qwen3VL processor).
# ---------------------------------------------------------------------------


class FakeMerger(nn.Module):
    """Mimics Qwen3VLVisionPatchMerger: norm + linear_fc1 + linear_fc2.
    模拟 Qwen3VLVisionPatchMerger。"""

    def __init__(self, hidden: int = 8, out: int = 8) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.linear_fc1 = nn.Linear(hidden, hidden)
        self.linear_fc2 = nn.Linear(hidden, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(torch.tanh(self.linear_fc1(self.norm(x))))


class FakeVision(nn.Module):
    """Vision encoder with a same-name projection trap (q_proj inside the
    visual block must NEVER receive LoRA). 视觉编码器，含同名 projection
    陷阱。"""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(3, 8)
        self.blocks = nn.ModuleList()
        trap = nn.Module()
        trap.q_proj = nn.Linear(8, 8)
        trap.mlp = nn.Module()
        trap.mlp.gate_proj = nn.Linear(8, 8)
        self.blocks.append(trap)
        self.merger = FakeMerger()
        self.deepstack_merger_list = nn.ModuleList([FakeMerger(), FakeMerger()])


class FakeTextLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(8, 8)
        self.self_attn.k_proj = nn.Linear(8, 8)
        self.self_attn.v_proj = nn.Linear(8, 8)
        self.self_attn.o_proj = nn.Linear(8, 8)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(8, 8)
        self.mlp.up_proj = nn.Linear(8, 8)
        self.mlp.down_proj = nn.Linear(8, 8)


class FakeText(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(30000, 8)
        self.layers = nn.ModuleList([FakeTextLayer(), FakeTextLayer()])


class FakeQwen(nn.Module):
    """Top-level model: model.visual + model.language_model + lm_head, plus
    a save_pretrained() that writes safetensors + config.json (the contract
    the exporter's save step needs; no transformers is involved).
    顶层模型；实现 exporter 保存步骤所需的最小 save_pretrained 契约。"""

    VOCAB = 30000

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.visual = FakeVision()
        self.model.language_model = FakeText()
        self.lm_head = nn.Linear(8, self.VOCAB, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        **kwargs: object,
    ) -> dict:
        del attention_mask, labels, image_grid_thw, mm_token_type_ids, kwargs
        assert input_ids is not None
        hidden = self.model.language_model.embed_tokens(input_ids)
        for layer in self.model.language_model.layers:
            self_attn = layer.self_attn
            q = self_attn.q_proj(hidden)
            k = self_attn.k_proj(hidden)
            v = self_attn.v_proj(hidden)
            hidden = hidden + self_attn.o_proj(q + k + v)
            mlp = layer.mlp
            hidden = hidden + mlp.down_proj(
                torch.tanh(mlp.gate_proj(hidden)) * mlp.up_proj(hidden)
            )
        if pixel_values is not None:
            pv = pixel_values.float()
            n = pv.shape[0]
            feat = pv.reshape(n, 3, -1).mean(dim=2)
            feat = self.model.visual.patch_embed(feat)
            feat = self.model.visual.merger(feat)
            for merger in self.model.visual.deepstack_merger_list:
                feat = merger(feat)
            hidden = hidden + 0.01 * feat.mean(dim=0, keepdim=True)
        logits = self.lm_head(hidden)
        return {"logits": logits}

    def save_pretrained(self, output_dir: str | Path, safe_serialization: bool = True) -> None:
        del safe_serialization
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        state = {
            name: tensor.detach().float().contiguous()
            for name, tensor in self.state_dict().items()
        }
        safetensors.torch.save_file(state, out / "model.safetensors")
        (out / "config.json").write_text(
            json.dumps({"model_type": "qwen3_vl", "architectures": ["FakeQwen"]}),
            encoding="utf-8",
        )


class FakeConfig:
    """Minimal config matching the producer's logical-identity fields.
    与生产者逻辑身份字段一致的最小配置。"""

    model_type = "qwen3_vl"
    architectures = ["FakeQwen"]
    hidden_size = 8
    num_hidden_layers = 2
    intermediate_size = 8
    vocab_size = 30000
    _commit_hash = "fake-revision-1"

    class VisionConfig:
        hidden_size = 8
        out_hidden_size = 8
        depth = 1
        spatial_merge_size = 2
        patch_size = 14
        deepstack_visual_indexes = [1, 2]

    vision_config = VisionConfig()


_SPECIALS = {
    "<|im_start|>": 1000,
    "<|im_end|>": 1001,
    "<|endoftext|>": 1002,
    "<|vision_start|>": 1003,
    "<|vision_end|>": 1004,
    "<|image_pad|>": 1005,
}
_SPECIAL_LIST = sorted(_SPECIALS, key=len, reverse=True)
_IMAGE_TOKEN_SPAN = 16


class FakeProcessor:
    """Mimics Qwen3VLProcessor: chatml rendering, image-token expansion and
    deterministic pixel_values. 模拟 Qwen3VLProcessor。"""

    def tokenize(self, text: str) -> list[int]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            for tok in _SPECIAL_LIST:
                if text.startswith(tok, i):
                    ids.append(_SPECIALS[tok])
                    i += len(tok)
                    break
            else:
                ids.append(2000 + ord(text[i]))
                i += 1
        return ids

    def render(self, messages: list[dict], add_generation_prompt: bool = False) -> str:
        out = ""
        for message in messages:
            out += f"<|im_start|>{message['role']}\n"
            content = message["content"]
            items = content if isinstance(content, list) else [{"type": "text", "text": content}]
            for item in items:
                if item["type"] == "image":
                    out += "<|vision_start|><|image_pad|><|vision_end|>"
                else:
                    out += item["text"]
            out += "<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
        return out

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        **kwargs: object,
    ) -> object:
        del return_dict, return_assistant_tokens_mask, kwargs
        text = self.render(messages, add_generation_prompt=add_generation_prompt)
        if not tokenize:
            return text
        return self.tokenize(text)

    def __call__(
        self,
        text: list[str] | None = None,
        images: list[Image.Image] | None = None,
        return_tensors: str | None = None,
        **kwargs: object,
    ) -> dict:
        del kwargs
        assert text is not None and images is not None and return_tensors == "pt"
        ids = self.tokenize(text[0])
        expanded: list[int] = []
        for token_id in ids:
            if token_id == _SPECIALS["<|image_pad|>"]:
                expanded.extend([token_id] * _IMAGE_TOKEN_SPAN)
            else:
                expanded.append(token_id)
        input_ids = torch.tensor(expanded, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        mm_token_type_ids = torch.zeros_like(input_ids)
        pv_list: list[torch.Tensor] = []
        thw_list: list[list[int]] = []
        for image in images:
            small = image.convert("RGB").resize((2, 2))
            values = np.asarray(small, dtype=np.float32).reshape(-1) / 255.0
            pv_list.append(torch.tensor(values, dtype=torch.float32).unsqueeze(0))
            thw_list.append([1, 1, 1])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mm_token_type_ids": mm_token_type_ids,
            "pixel_values": torch.cat(pv_list, dim=0),
            "image_grid_thw": torch.tensor(thw_list, dtype=torch.long),
        }

    def save_pretrained(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "preprocessor_config.json").write_text("{}", encoding="utf-8")


# ---------------------------------------------------------------------------
# LLM target enumeration (same 7-per-layer set as the training script).
# ---------------------------------------------------------------------------

_ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP = ("gate_proj", "up_proj", "down_proj")


def llm_targets(layer_count: int = 2) -> list[str]:
    return (
        [f"model.language_model.layers.{i}.self_attn.{p}" for i in range(layer_count) for p in _ATTN]
        + [f"model.language_model.layers.{i}.mlp.{p}" for i in range(layer_count) for p in _MLP]
    )


def merger_names_of(model: FakeQwen) -> list[str]:
    vision_name, _text_name = exp.locate_roots(model)
    return exp.enumerate_merger_names(model, vision_name)


def _under_any(name: str, roots: list[str]) -> bool:
    return any(name == root or name.startswith(root + ".") for root in roots)


# ---------------------------------------------------------------------------
# Composite checkpoint builder (uses the producer's own helpers so the
# exporter is exercised against the real contract).
# ---------------------------------------------------------------------------


def build_prepared_model(
    trained_seed: int = 2024,
) -> tuple[object, FakeQwen, list[str], dict[str, torch.Tensor]]:
    """Fake tree with LoRA attached and merger params set to deterministic
    "trained" values; returns (peft_model, base, merger_names, expected).
    fake 模型树：挂 LoRA、merger 参数设为确定性的“训练后”值。"""
    model = FakeQwen()
    targets = llm_targets()
    merger_names = merger_names_of(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    from peft import LoraConfig, get_peft_model

    peft_model = get_peft_model(
        model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05, target_modules=targets, bias="none")
    )
    base = peft_model.get_base_model()
    rng = np.random.RandomState(trained_seed)
    expected: dict[str, torch.Tensor] = {}
    for name, parameter in base.named_parameters():
        if _under_any(name, merger_names):
            value = torch.tensor(rng.rand(*parameter.shape), dtype=parameter.dtype)
            parameter.data.copy_(value)
            expected[name] = value.detach().clone()
    return peft_model, base, merger_names, expected


def build_composite_checkpoint(root: Path) -> dict:
    """Write a complete composite checkpoint (adapter/ + merger_model +
    processor/ + manifest) using the producer's helper functions.
    用生产者辅助函数写出完整复合 checkpoint。"""
    ckpt = root / "checkpoint"
    adapter_dir = ckpt / "adapter"
    processor_dir = ckpt / "processor"
    adapter_dir.mkdir(parents=True)
    processor_dir.mkdir(parents=True)

    peft_model, base, merger_names, expected = build_prepared_model()
    targets = llm_targets()
    peft_model.save_pretrained(adapter_dir, safe_serialization=True)
    adapter_meta = ft.verify_adapter_keys(adapter_dir, targets)
    merger_path = ckpt / "merger_model.safetensors"
    merger_state = {
        name: parameter.detach().clone().contiguous()
        for name, parameter in base.named_parameters()
        if _under_any(name, merger_names)
    }
    safetensors.torch.save_file(merger_state, merger_path)
    merger_meta = {
        "file": "merger_model.safetensors",
        "file_sha256": ft.sha256_file(merger_path),
        "parameters": ft.merger_state_meta(base, merger_names),
    }
    FakeProcessor().save_pretrained(processor_dir)
    manifest = {
        "schema_version": 1,
        "checkpoint_type": "phase2_composite",
        "step": 100,
        "epoch": 1.5,
        "base_model": ft.model_logical_identity(FakeConfig()),
        "processor": ft.processor_identity(FakeProcessor()),
        "model_id_as_given": "fake-base",
        "data": {"train_file": "train.jsonl", "train_sha256": "fake", "train_episode_count": 4},
        "lora": {
            "rank": 4,
            "alpha": 8,
            "dropout": 0.05,
            "bias": "none",
            "target_modules": list(targets),
        },
        "merger": {"modules": list(merger_names), **merger_meta},
        "adapter": adapter_meta,
        "optimizer": {"groups": []},
        "augmentation": {"seed": "phase2-test-seed", "enabled": False, "config": {}},
        "training": {"max_seq_length": 512, "image_min_pixels": 1, "image_max_pixels": 2},
        "data_sampling": {"group_key": "task_kind", "repeat_weights": {}, "group_counts": {}},
        "environment": {"git_head": "fake", "transformers_version": "5.14.1",
                        "torch_version": "2.13.0", "peft_version": "0.20.0",
                        "python_version": "3.14"},
    }
    (ckpt / "phase2_training_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {
        "ckpt": ckpt,
        "merger_names": list(merger_names),
        "targets": list(targets),
        "expected": expected,
        "manifest": manifest,
        "adapter_config_path": adapter_dir / "adapter_config.json",
        "merger_path": merger_path,
    }


def patch_export_seams(monkeypatch: pytest.MonkeyPatch, base_dir: Path | None = None) -> None:
    """Substitute every transformers loading seam with the fake tree.
    把所有 transformers 加载 seam 替换为 fake 树。"""

    def _load_base_components(model_id, torch_dtype, device, local_files_only):
        del model_id, torch_dtype, device, local_files_only
        return FakeConfig(), FakeQwen(), FakeProcessor()

    monkeypatch.setattr(exp, "load_base_components", _load_base_components)

    def _load_processor_from_dir(path, local_files_only):
        del path, local_files_only
        return FakeProcessor()

    monkeypatch.setattr(exp, "load_processor_from_dir", _load_processor_from_dir)
    monkeypatch.setattr(exp, "_auto_config_from_pretrained", lambda path, local_files_only: FakeConfig())
    monkeypatch.setattr(exp, "_auto_model_from_pretrained", lambda path, torch_dtype, device, local_files_only: FakeQwen())
    monkeypatch.setattr(exp, "_auto_processor_from_pretrained", lambda path, local_files_only: FakeProcessor())
    if base_dir is not None:
        base_dir.mkdir(parents=True, exist_ok=True)
        for name in ("generation_config.json", "chat_template.json", "video_preprocessor_config.json"):
            (base_dir / name).write_text(json.dumps({"fake": True}), encoding="utf-8")
        (base_dir / "preprocessor_config.json").write_text(json.dumps({"fake": True}), encoding="utf-8")


def run_export(
    monkeypatch: pytest.MonkeyPatch,
    ckpt: Path,
    output_path: Path,
    base_dir: Path | None = None,
    verify_forward: bool = False,
) -> dict:
    """Full happy-path export with all seams patched.
    全部 seam 替换后的完整成功导出。"""
    patch_export_seams(monkeypatch, base_dir=base_dir)
    return exp.export_phase2(
        model_id=str(base_dir) if base_dir is not None else "fake-base",
        checkpoint_path=ckpt,
        output_path=output_path,
        torch_dtype_name="float32",
        device="cpu",
        local_files_only=True,
        verify_forward=verify_forward,
        repo_root=REPO_ROOT,
    )


# ---------------------------------------------------------------------------
# 1. import / --help never load big models
# ---------------------------------------------------------------------------


def test_import_and_help_do_not_load_weights() -> None:
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "import scripts.export_qwen3vl_phase2_checkpoint as m\n"
        "heavy = [name for name in ('torch', 'transformers', 'peft', 'safetensors',\n"
        "                            'PIL', 'numpy')\n"
        "         if name in sys.modules]\n"
        "assert not heavy, f'heavy deps imported at module level: {heavy}'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run(
        [sys.executable, str(EXPORTER_PATH), "--help"],
        capture_output=True, text=True, cwd=REPO_ROOT, check=False,
    )
    assert result.returncode == 0, result.stderr
    for flag in ("--model-id", "--checkpoint-path", "--output-path", "--verify-forward"):
        assert flag in result.stdout


# ---------------------------------------------------------------------------
# 2/3. layout + checksum failures happen before any weight loading
# ---------------------------------------------------------------------------


def test_missing_checkpoint_files_fail_before_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built = build_composite_checkpoint(tmp_path)
    ckpt = built["ckpt"]
    (ckpt / "adapter" / "adapter_model.safetensors").unlink()
    _fail_before_base_load(
        monkeypatch, ckpt, tmp_path / "out", "checkpoint_incomplete"
    )
    shutil.rmtree(ckpt)
    rebuilt = build_composite_checkpoint(tmp_path)
    (rebuilt["ckpt"] / "phase2_training_manifest.json").unlink()
    _fail_before_base_load(
        monkeypatch, rebuilt["ckpt"], tmp_path / "out2", "checkpoint_incomplete"
    )


def test_checksum_mismatch_fails_before_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built = build_composite_checkpoint(tmp_path)
    adapter_file = built["ckpt"] / "adapter" / "adapter_model.safetensors"
    adapter_file.write_bytes(adapter_file.read_bytes() + b"\x00")
    _fail_before_base_load(
        monkeypatch, built["ckpt"], tmp_path / "out", "adapter_checksum_mismatch"
    )
    # merger checksum mismatch as well
    rebuilt = build_composite_checkpoint(tmp_path / "again")
    merger_file = rebuilt["ckpt"] / "merger_model.safetensors"
    merger_file.write_bytes(merger_file.read_bytes() + b"\x00")
    _fail_before_base_load(
        monkeypatch, rebuilt["ckpt"], tmp_path / "out2", "merger_checksum_mismatch"
    )


def _fail_before_base_load(
    monkeypatch: pytest.MonkeyPatch, ckpt: Path, output: Path, expected_code: str
) -> None:
    called = {"load": False}

    def _forbidden(model_id, torch_dtype, device, local_files_only):
        del model_id, torch_dtype, device, local_files_only
        called["load"] = True
        raise AssertionError("base weights loaded before validation")

    monkeypatch.setattr(exp, "load_base_components", _forbidden)
    with pytest.raises(ExportError) as error:
        exp.export_phase2(
            model_id="fake-base",
            checkpoint_path=ckpt,
            output_path=output,
            torch_dtype_name="float32",
            device="cpu",
            local_files_only=True,
            verify_forward=False,
            repo_root=REPO_ROOT,
        )
    assert error.value.code == expected_code
    assert not called["load"], "weights were loaded before the gate failed"
    assert not output.exists()


def test_manifest_safety_rejects_secrets_and_absolute_paths(tmp_path: Path) -> None:
    built = build_composite_checkpoint(tmp_path)
    manifest = json.loads(
        (built["ckpt"] / "phase2_training_manifest.json").read_text(encoding="utf-8")
    )
    manifest["dangerous"] = {"result_path": "/abs/secret/dir", "api_key": "SIMULATED_SECRET"}
    with pytest.raises(CheckpointValidationError) as error:
        exp.validate_manifest_safety(manifest)
    assert error.value.code == "manifest_unsafe"
    # clean manifest passes the scan
    manifest.pop("dangerous")
    exp.validate_manifest_safety(manifest)


# ---------------------------------------------------------------------------
# 4. output path already exists -> refused
# ---------------------------------------------------------------------------


def test_output_exists_refused(tmp_path: Path) -> None:
    built = build_composite_checkpoint(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ExportError) as error:
        exp.export_phase2(
            model_id="fake-base",
            checkpoint_path=built["ckpt"],
            output_path=output,
            torch_dtype_name="float32",
            device="cpu",
            local_files_only=True,
            verify_forward=False,
            repo_root=REPO_ROOT,
        )
    assert error.value.code == "output_exists"


# ---------------------------------------------------------------------------
# 5. merger missing / unexpected / shape mismatch all fail
# ---------------------------------------------------------------------------


def _rewrite_merger_file(built: dict, mutate) -> None:
    state = safetensors.torch.load_file(built["merger_path"])
    mutate(state)
    safetensors.torch.save_file(state, built["merger_path"])
    # keep the manifest checksums truthful so the failure lands in the
    # merger load step itself (not the earlier checksum gate)
    manifest_path = built["ckpt"] / "phase2_training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["merger"]["file_sha256"] = ft.sha256_file(built["merger_path"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _expect_merger_load_failure(monkeypatch: pytest.MonkeyPatch, built: dict, output: Path, code: str) -> None:
    patch_export_seams(monkeypatch)
    with pytest.raises(MergerLoadError) as error:
        exp.export_phase2(
            model_id="fake-base",
            checkpoint_path=built["ckpt"],
            output_path=output,
            torch_dtype_name="float32",
            device="cpu",
            local_files_only=True,
            verify_forward=False,
            repo_root=REPO_ROOT,
        )
    assert error.value.code == code
    assert not output.exists()


def test_merger_missing_key_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built = build_composite_checkpoint(tmp_path)

    def _pop(state):
        state.pop(next(iter(state)))

    _rewrite_merger_file(built, _pop)
    _expect_merger_load_failure(monkeypatch, built, tmp_path / "out", "merger_key_mismatch")


def test_merger_unexpected_key_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built = build_composite_checkpoint(tmp_path)

    def _add(state):
        state["model.visual.merger.unexpected"] = torch.zeros(2, 2)

    _rewrite_merger_file(built, _add)
    _expect_merger_load_failure(monkeypatch, built, tmp_path / "out", "merger_key_mismatch")


def test_merger_shape_mismatch_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built = build_composite_checkpoint(tmp_path)

    def _reshape(state):
        # pick a weight (rank >= 2) so the reshape genuinely changes shape
        key = next(
            name for name, tensor in state.items() if len(tensor.shape) >= 2
        )
        state[key] = state[key].reshape(-1)

    _rewrite_merger_file(built, _reshape)
    _expect_merger_load_failure(monkeypatch, built, tmp_path / "out", "merger_shape_mismatch")


def test_merger_unsafe_dtype_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built = build_composite_checkpoint(tmp_path)

    def _intify(state):
        key = next(iter(state))
        state[key] = state[key].to(torch.int32)

    _rewrite_merger_file(built, _intify)
    _expect_merger_load_failure(monkeypatch, built, tmp_path / "out", "merger_unsafe_dtype")


# ---------------------------------------------------------------------------
# 6. fixed order: merger load strictly before LoRA attach/merge
# ---------------------------------------------------------------------------


def test_merger_load_precedes_lora_attach(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built = build_composite_checkpoint(tmp_path)
    patch_export_seams(monkeypatch)
    seen: dict[str, object] = {}
    original = exp._peft_from_pretrained

    def wrapped(base_model, adapter_dir):
        # at attach time the base model must already carry the trained
        # merger values (proves strict merger load ran first)
        for name, expected in built["expected"].items():
            parameter = dict(base_model.named_parameters()).get(name)
            assert parameter is not None, f"{name} missing at attach time"
            assert torch.equal(parameter.detach(), expected), f"{name} not loaded yet"
        seen["attach_called"] = True
        return original(base_model, adapter_dir)

    monkeypatch.setattr(exp, "_peft_from_pretrained", wrapped)
    exp.export_phase2(
        model_id="fake-base",
        checkpoint_path=built["ckpt"],
        output_path=tmp_path / "out",
        torch_dtype_name="float32",
        device="cpu",
        local_files_only=True,
        verify_forward=False,
        repo_root=REPO_ROOT,
    )
    assert seen.get("attach_called") is True


# ---------------------------------------------------------------------------
# 7. LoRA target mismatch with the manifest fails
# ---------------------------------------------------------------------------


def test_lora_target_mismatch_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built = build_composite_checkpoint(tmp_path)
    config_path = built["adapter_config_path"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["target_modules"] = config["target_modules"][:-1]  # drop one projection
    config_path.write_text(json.dumps(config), encoding="utf-8")
    patch_export_seams(monkeypatch)
    with pytest.raises(LoRAValidationError) as error:
        exp.export_phase2(
            model_id="fake-base",
            checkpoint_path=built["ckpt"],
            output_path=tmp_path / "out",
            torch_dtype_name="float32",
            device="cpu",
            local_files_only=True,
            verify_forward=False,
            repo_root=REPO_ROOT,
        )
    assert error.value.code == "lora_target_mismatch"
    assert not (tmp_path / "out").exists()


def test_adapter_base_identity_mismatch_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built = build_composite_checkpoint(tmp_path)
    config_path = built["adapter_config_path"]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = "some-other-model"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    patch_export_seams(monkeypatch)
    with pytest.raises(LoRAValidationError) as error:
        exp.export_phase2(
            model_id="fake-base",
            checkpoint_path=built["ckpt"],
            output_path=tmp_path / "out",
            torch_dtype_name="float32",
            device="cpu",
            local_files_only=True,
            verify_forward=False,
            repo_root=REPO_ROOT,
        )
    assert error.value.code == "adapter_base_identity_mismatch"


def test_base_identity_mismatch_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built = build_composite_checkpoint(tmp_path)
    manifest_path = built["ckpt"] / "phase2_training_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["base_model"]["fingerprint"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    patch_export_seams(monkeypatch)
    with pytest.raises(BaseIdentityError) as error:
        exp.export_phase2(
            model_id="fake-base",
            checkpoint_path=built["ckpt"],
            output_path=tmp_path / "out",
            torch_dtype_name="float32",
            device="cpu",
            local_files_only=True,
            verify_forward=False,
            repo_root=REPO_ROOT,
        )
    assert error.value.code == "base_fingerprint_mismatch"


# ---------------------------------------------------------------------------
# 8/9. after merge: no LoRA residue, merger values preserved
# ---------------------------------------------------------------------------


def test_merge_leaves_no_lora_residue_and_keeps_merger_values(tmp_path: Path) -> None:
    peft_model, base, merger_names, expected = build_prepared_model()
    merged, audit = exp.merge_and_audit(peft_model, merger_names, expected)
    state = merged.state_dict()
    residue = [key for key in state if "lora" in key or key.startswith("base_model.")]
    assert residue == []
    assert audit["lora_residue_count"] == 0
    assert audit["lora_module_residue_count"] == 0
    for name, value in expected.items():
        parameter = dict(merged.named_parameters()).get(name)
        assert parameter is not None, name
        assert torch.equal(parameter.detach(), value), f"merger value drift: {name}"
    assert audit["merger_value_verified_count"] == len(expected)


def test_full_export_records_merger_values_verified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built = build_composite_checkpoint(tmp_path)
    result = run_export(monkeypatch, built["ckpt"], tmp_path / "out")
    assert result["merge_audit"]["merger_value_verified_count"] == len(built["expected"])
    assert result["merge_audit"]["deepstack_count"] == 2
    assert result["lora_merged"] is True


# ---------------------------------------------------------------------------
# 10/11. output layout: processor + aux files + export manifest checksums
# ---------------------------------------------------------------------------


def test_output_layout_aux_files_and_manifest_checksums(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built = build_composite_checkpoint(tmp_path)
    base_dir = tmp_path / "base"
    result = run_export(monkeypatch, built["ckpt"], tmp_path / "out", base_dir=base_dir)

    output = tmp_path / "out"
    assert output.is_dir()
    # processor + aux files complete
    assert (output / "preprocessor_config.json").is_file()
    for name in ("generation_config.json", "chat_template.json", "video_preprocessor_config.json"):
        assert (output / name).is_file(), name
    # full model weights + config
    assert (output / "model.safetensors").is_file()
    assert (output / "config.json").is_file()
    assert (output / "phase2_export_manifest.json").is_file()

    # manifest checksums are truthful
    manifest = json.loads((output / "phase2_export_manifest.json").read_text(encoding="utf-8"))
    listed = {entry["path"]: entry for entry in manifest["files"]}
    assert "phase2_export_manifest.json" not in listed
    assert "model.safetensors" in listed
    for entry in manifest["files"]:
        path = output / entry["path"]
        assert path.is_file(), entry["path"]
        assert path.stat().st_size == entry["size"], entry["path"]
        assert exp.sha256_file(path) == entry["sha256"], entry["path"]
    assert manifest["schema_version"] == 1
    assert manifest["lora_merged"] is True
    assert manifest["output_dtype"] == "float32"
    assert manifest["reload_validation"]["model_type"] == "qwen3_vl"
    assert manifest["reload_validation"]["config_fingerprint_matches_base"]
    assert manifest["reload_validation"]["deepstack_count"] == 2
    assert manifest["reload_validation"]["lora_module_residue"] == []
    assert manifest["reload_validation"]["shards"]["shard_count"] == 1
    assert manifest["source_training_checkpoint"]["step"] == 100
    assert manifest["source_training_checkpoint"]["type"] == "phase2_composite"
    assert manifest["merger"]["parameter_count"] == len(built["expected"])
    # no temp dirs left behind
    assert list(tmp_path.glob("out.export-tmp-*")) == []


def test_forward_verification_records_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built = build_composite_checkpoint(tmp_path)
    result = run_export(monkeypatch, built["ckpt"], tmp_path / "out", verify_forward=True)
    assert result["forward_validation"] is not None
    assert result["forward_validation"]["logits_finite"] is True
    assert result["forward_validation"]["logits_shape"][-1] == FakeQwen.VOCAB


# ---------------------------------------------------------------------------
# 12/13. failure/interrupt -> no final output, temp cleaned
# ---------------------------------------------------------------------------


def test_reload_failure_does_not_publish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built = build_composite_checkpoint(tmp_path)
    patch_export_seams(monkeypatch)

    def _broken_reload(path, torch_dtype, device, local_files_only):
        del path, torch_dtype, device, local_files_only
        raise ReloadValidationError("reload_model_type", "fake")

    monkeypatch.setattr(exp, "_auto_model_from_pretrained", _broken_reload)
    with pytest.raises(ReloadValidationError):
        exp.export_phase2(
            model_id="fake-base",
            checkpoint_path=built["ckpt"],
            output_path=tmp_path / "out",
            torch_dtype_name="float32",
            device="cpu",
            local_files_only=True,
            verify_forward=False,
            repo_root=REPO_ROOT,
        )
    assert not (tmp_path / "out").exists()
    assert list(tmp_path.glob("out.export-tmp-*")) == [], "temp dir leaked"


def test_interrupt_and_error_leave_no_final_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    built = build_composite_checkpoint(tmp_path)

    def _interrupt(model_id, torch_dtype, device, local_files_only):
        del model_id, torch_dtype, device, local_files_only
        raise KeyboardInterrupt

    patch_export_seams(monkeypatch)
    monkeypatch.setattr(exp, "load_base_components", _interrupt)
    code = exp.main(
        [
            "--model-id", "fake-base",
            "--checkpoint-path", str(built["ckpt"]),
            "--output-path", str(tmp_path / "out"),
            "--torch-dtype", "float32",
        ]
    )
    assert code == 130
    assert not (tmp_path / "out").exists()

    # a generic runtime failure inside the pipeline returns 1 and publishes
    # nothing (and nothing is left in the parent directory)
    rebuilt = build_composite_checkpoint(tmp_path / "again")

    def _explode(model_id, torch_dtype, device, local_files_only):
        del model_id, torch_dtype, device, local_files_only
        raise RuntimeError("boom")

    monkeypatch.setattr(exp, "load_base_components", _explode)
    code = exp.main(
        [
            "--model-id", "fake-base",
            "--checkpoint-path", str(rebuilt["ckpt"]),
            "--output-path", str(tmp_path / "out2"),
            "--torch-dtype", "float32",
        ]
    )
    assert code == 1
    assert not (tmp_path / "out2").exists()


# ---------------------------------------------------------------------------
# 14. default local_files_only=True
# ---------------------------------------------------------------------------


def test_local_files_only_default_true() -> None:
    args = exp.build_parser().parse_args(
        ["--checkpoint-path", "ckpt", "--output-path", "out"]
    )
    assert args.local_files_only is True
    assert args.torch_dtype == "bfloat16"
    assert args.device == "cpu"
    assert args.verify_forward is False
    assert args.output_path == Path("out")


# ---------------------------------------------------------------------------
# 15. public error output carries no secret / absolute path
# ---------------------------------------------------------------------------


def test_public_error_has_no_secret_or_absolute_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    built = build_composite_checkpoint(tmp_path)
    secret = "SIMULATED_SECRET_9f3a"
    internal = "/var/secret/internal/path"

    def _explode(model_id, torch_dtype, device, local_files_only):
        del model_id, torch_dtype, device, local_files_only
        raise RuntimeError(f"{secret} {internal} raw traceback")

    monkeypatch.setattr(exp, "load_base_components", _explode)
    code = exp.main(
        [
            "--model-id", "fake-base",
            "--checkpoint-path", str(built["ckpt"]),
            "--output-path", str(tmp_path / "out"),
            "--torch-dtype", "float32",
        ]
    )
    assert code == 1
    captured = capsys.readouterr()
    assert secret not in captured.err
    assert internal not in captured.err
    assert "unexpected error" in captured.err
    assert not (tmp_path / "out").exists()


# ---------------------------------------------------------------------------
# fingerprint parity: exporter identities must agree with the producer
# ---------------------------------------------------------------------------


def test_identity_fingerprints_match_producer() -> None:
    exp_identity = exp.model_logical_identity(FakeConfig())
    ft_identity = ft.model_logical_identity(FakeConfig())
    assert exp_identity["fingerprint"] == ft_identity["fingerprint"]
    exp_processor = exp.processor_identity(FakeProcessor())
    ft_processor = ft.processor_identity(FakeProcessor())
    assert exp_processor["fingerprint"] == ft_processor["fingerprint"]
    # merger enumeration agrees with the producer on the same tree
    model = FakeQwen()
    vision_name, _text_name = exp.locate_roots(model)
    ft_roots = ft.locate_roots(model)
    assert exp.enumerate_merger_names(model, vision_name) == ft.enumerate_merger_names(ft_roots)


def test_exporter_module_surface_stable() -> None:
    """Documented stage functions exist and are import-light.
    文档约定的阶段函数存在且 import 轻量。"""
    for name in (
        "validate_checkpoint_layout",
        "validate_manifest_safety",
        "verify_checkpoint_checksums",
        "verify_base_identity",
        "load_merger_strict",
        "attach_and_verify_adapter",
        "merge_and_audit",
        "reload_validate",
        "run_forward_verification",
        "export_phase2",
    ):
        assert callable(getattr(exp, name, None)), name
