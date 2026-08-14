"""Tests for scripts/finetune_qwen3vl_phase2.py.

Unit tests use a tiny fake Qwen model tree (mirroring the transformers
5.14.1 Qwen3-VL layout), a fake processor mimicking the pinned Qwen3VL
processor contract, and the real peft library of the M3 env (peft 0.20.0)
as the injection seam. No 8B weights are loaded anywhere. Trainer-based
tests force CPU (use_cpu=True) so the suite is hermetic and offline.

测试使用微型 fake Qwen 模型树（对照 transformers 5.14.1 Qwen3-VL 布局）、
fake processor（模拟钉死的 Qwen3VL processor 契约）与 M3 环境真实 peft
0.20.0 注入 seam；不加载任何 8B 权重；Trainer 相关测试强制 CPU。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "finetune_qwen3vl_phase2.py"

spec = importlib.util.spec_from_file_location("finetune_qwen3vl_phase2", SCRIPT_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["finetune_qwen3vl_phase2"] = mod  # dataclasses need module registration
spec.loader.exec_module(mod)

# Re-export the module surface used by the tests.
# 导出测试使用的模块接口。
ft = mod
ConfigurationError = mod.ConfigurationError
ParameterAuditError = mod.ParameterAuditError
ResumeConflictError = mod.ResumeConflictError
CheckpointError = mod.CheckpointError


# ---------------------------------------------------------------------------
# Fake Qwen3-VL model tree (transformers 5.14.1 layout)
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
    """Vision encoder with a same-name projection trap: blocks contain a
    Linear named q_proj (identical to LLM projection names) that must NEVER
    receive LoRA. 视觉编码器，含同名 projection 陷阱。"""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(3, 8)
        # trap: same names as LLM projections, must stay frozen without LoRA
        self.blocks = nn.ModuleList()
        trap = nn.Module()
        trap.q_proj = nn.Linear(8, 8)
        trap.mlp = nn.Module()
        trap.mlp.gate_proj = nn.Linear(8, 8)
        self.blocks.append(trap)
        self.merger = FakeMerger()
        self.deepstack_merger_list = nn.ModuleList([FakeMerger(), FakeMerger()])

    def gradient_checkpointing_enable(self, **kwargs: object) -> None:
        del kwargs

    def enable_input_require_grads(self) -> None:
        pass


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
    """Top-level model: model.visual + model.language_model + lm_head.
    顶层模型：model.visual + model.language_model + lm_head。"""

    VOCAB = 30000

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.visual = FakeVision()
        self.model.language_model = FakeText()
        self.lm_head = nn.Linear(8, self.VOCAB, bias=False)

    def gradient_checkpointing_enable(self, **kwargs: object) -> None:
        del kwargs

    def enable_input_require_grads(self) -> None:
        pass

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
        del attention_mask, image_grid_thw, mm_token_type_ids, kwargs
        assert input_ids is not None and labels is not None
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
        logits = self.lm_head(hidden)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
        )
        if pixel_values is not None:
            pv = pixel_values.float()
            n = pv.shape[0]
            feat = pv.reshape(n, 3, -1).mean(dim=2)  # (n, 3)
            feat = self.model.visual.patch_embed(feat)
            feat = self.model.visual.merger(feat)
            for merger in self.model.visual.deepstack_merger_list:
                feat = merger(feat)
            loss = loss + 0.01 * self.lm_head(feat).pow(2).mean()
        return {"loss": loss}


class FakeConfig:
    """Minimal config for the logical-identity fingerprint.
    逻辑身份指纹用的最小配置。"""

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


# ---------------------------------------------------------------------------
# Fake processor mimicking the pinned transformers 5.14.1 Qwen3VL contract
# ---------------------------------------------------------------------------

_SPECIALS = {
    "<|im_start|>": 1000,
    "<|im_end|>": 1001,
    "<|endoftext|>": 1002,
    "<|vision_start|>": 1003,
    "<|vision_end|>": 1004,
    "<|image_pad|>": 1005,
}
_SPECIAL_LIST = sorted(_SPECIALS, key=len, reverse=True)
_IMAGE_TOKEN_SPAN = 16  # one placeholder expands to this many image tokens


class FakeProcessor:
    """Mimics Qwen3VLProcessor: chatml rendering, image-token expansion,
    assistant_masks, mm_token_type_ids and deterministic pixel_values
    derived from the (possibly augmented) image.
    模拟 Qwen3VLProcessor；pixel_values 由增强后的图像确定性派生。"""

    # -- tokenizer ---------------------------------------------------------
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

    # -- chat template -----------------------------------------------------
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
        del kwargs
        text = self.render(messages, add_generation_prompt=add_generation_prompt)
        if not tokenize:
            return text
        ids = self.tokenize(text)
        if return_assistant_tokens_mask:
            mask = [0] * len(ids)
            # assistant content + <|im_end|> are supervised (mimics 5.14.1)
            for i in range(1, len(messages), 2):
                span = self.tokenize(self.render(messages[: i + 1]))
                # ids end with the latest span; mark it
                if len(span) <= len(ids) and ids[len(ids) - len(span):] == span:
                    for pos in range(len(ids) - len(span), len(ids)):
                        mask[pos] = 1
            return {"input_ids": [ids], "assistant_masks": [mask]}

    # -- multimodal encoding ----------------------------------------------
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
        mm_token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.ones_like(input_ids)
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
# Fixtures: canonical episodes + tiny images
# ---------------------------------------------------------------------------


def make_episodes(path: Path, count: int = 4) -> None:
    rows = []
    for i in range(count):
        kind = "vqa_box_assisted" if i % 2 == 0 else "geochat_conversation"
        input_boxes = (
            []
            if kind == "geochat_conversation"
            else [
                {
                    "xyxy_999": [10, 10, 100, 100],
                    "label": "obj",
                    "description": "a small object",
                    "source_object_id": 0,
                }
            ]
        )
        rows.append(
            {
                "schema_version": 1,
                "episode_id": f"fake/train/img{i}.png/qa/0/{kind}",
                "parent_episode_id": f"fake/train/img{i}.png/qa/0",
                "dataset": "Fake",
                "split": "train",
                "image_source": "fake",
                "image": f"img{i}.png",
                "task_kind": kind,
                "source_task": "vqa" if kind == "vqa_box_assisted" else "chat",
                "turns": [
                    {
                        "user_text": f"Question number {i}?",
                        "assistant_text": f"Answer number {i}",
                        "input_boxes": input_boxes,
                        "target_boxes": [],
                    }
                ],
                "augmentation_policy": {"geometry": "geometry_safe", "reason": "test"},
                "provenance": {
                    "source_record_id": f"fake/img{i}.png",
                    "question_id": 0,
                    "view": "test",
                },
            }
        )
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@pytest.fixture(scope="module")
def phase2_workspace(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Tiny offline workspace: episodes JSONL + images + roots config.
    小型离线工作区：episodes JSONL + 图片 + root 映射。"""
    root = tmp_path_factory.mktemp("phase2_test")
    data_dir = root / "data"
    data_dir.mkdir()
    make_episodes(data_dir / "train.jsonl", count=4)
    make_episodes(data_dir / "validation.jsonl", count=2)
    image_dir = root / "images"
    image_dir.mkdir()
    for i in range(4):
        array = (np.random.RandomState(i).rand(64, 48, 3) * 255).astype(np.uint8)
        Image.fromarray(array).save(image_dir / f"img{i}.png")
    return {
        "root": root,
        "data_dir": data_dir,
        "image_dir": image_dir,
        "train_file": data_dir / "train.jsonl",
        "eval_file": data_dir / "validation.jsonl",
    }


def build_prepared_model() -> tuple[object, object, object, object, list[str], list[str]]:
    """freeze -> LoRA -> merger unfreeze on the fake tree; returns
    (peft_model, base, roots, audit, llm_targets, merger_names).
    在 fake 模型树上执行冻结/LoRA/merger 解冻并返回审计结果。"""
    model = FakeQwen()
    roots = ft.locate_roots(model)
    llm_targets = ft.enumerate_llm_lora_targets(roots)
    merger_names = ft.enumerate_merger_names(roots)
    ft.freeze_all(model)
    peft_model = ft.inject_lora(model, llm_targets, rank=4, alpha=8, dropout=0.05, bias="none")
    base = ft.unwrap_base(peft_model)
    ft.unfreeze_merger_base(base, merger_names)
    audit = ft.audit_trainable_parameters(
        peft_model, roots, merger_names, llm_targets, expected_deepstack=2
    )
    return peft_model, base, roots, audit, llm_targets, merger_names


# ---------------------------------------------------------------------------
# 1. import never loads weights (and no heavy deps at module level)
# ---------------------------------------------------------------------------


def test_import_does_not_load_weights() -> None:
    """Importing the module must not pull torch/transformers/peft into
    sys.modules (verified in a clean interpreter subprocess).
    import 模块不得把 torch/transformers/peft 拉进 sys.modules。"""
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "import scripts.finetune_qwen3vl_phase2 as m\n"
        "heavy = [name for name in ('torch', 'transformers', 'peft', 'safetensors')\n"
        "         if name in sys.modules]\n"
        "assert not heavy, f'heavy deps imported at module level: {heavy}'\n"
        "assert m.ModelArguments().local_files_only is True\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# 2/3. LoRA hits only LLM projections, never same-named visual projections
# ---------------------------------------------------------------------------


def test_lora_only_on_llm_projections() -> None:
    _peft_model, _base, roots, audit, llm_targets, _merger_names = build_prepared_model()
    lora_parents = {
        ft._lora_parent_module(entry["name"]) for entry in audit["llm_lora_trainable"]
    }
    assert lora_parents == set(llm_targets)
    assert len(llm_targets) == 2 * 7  # two layers x seven projections
    for target in llm_targets:
        assert target.startswith(roots.text_name + ".")
    assert audit["checks"]["lora_targets_hit_all"]


def test_visual_same_name_projection_not_hit() -> None:
    """The vision block contains a Linear literally named q_proj/gate_proj;
    it must never receive LoRA. 视觉块中的同名 projection 不得被挂 LoRA。"""
    _peft_model, base, roots, audit, _targets, _merger_names = build_prepared_model()
    for entry in audit["llm_lora_trainable"]:
        assert "visual" not in entry["name"].split(".")
    # the trapped vision projection stays frozen with the original weights
    trap_weight = dict(base.named_parameters())["model.visual.blocks.0.q_proj.weight"]
    assert not trap_weight.requires_grad
    assert audit["checks"]["no_visual_lora"]


# ---------------------------------------------------------------------------
# 4/5/6. merger trainable; vision rest / LLM base / embeddings / lm_head frozen
# ---------------------------------------------------------------------------


def test_merger_and_deepstack_fully_trainable() -> None:
    _peft_model, base, _roots, audit, _targets, merger_names = build_prepared_model()
    trainable = {
        name
        for name, parameter in base.named_parameters()
        if parameter.requires_grad and not ft._has_lora_segment(name)
    }
    for name in trainable:
        assert any(
            name.startswith(merger_name + ".") for merger_name in merger_names
        ), f"{name} outside every merger subtree"
    merger_param_count = len(audit["merger_trainable"])
    assert merger_param_count == 3 * 6  # 3 mergers x (norm w/b + fc1 w/b + fc2 w/b)
    assert audit["checks"]["deepstack_count_matches"]
    assert audit["checks"]["no_merger_lora"]


def test_vision_encoder_rest_frozen() -> None:
    _peft_model, base, _roots, audit, _targets, merger_names = build_prepared_model()
    for name, parameter in base.named_parameters():
        if name.startswith("model.visual.") and not ft._under_any(name, merger_names):
            assert not parameter.requires_grad, name
    assert audit["frozen_vision_non_merger_parameters"] > 0


def test_llm_base_embedding_lm_head_frozen() -> None:
    _peft_model, base, _roots, audit, _targets, _merger_names = build_prepared_model()
    for name, parameter in base.named_parameters():
        if ft._has_lora_segment(name):
            continue  # LoRA matrices live in the text subtree by design
        if name.startswith("model.language_model."):
            assert not parameter.requires_grad, name
        if name == "lm_head" or name.startswith("lm_head."):
            assert not parameter.requires_grad, name
    assert audit["frozen_llm_base_parameters"] > 0
    assert audit["frozen_lm_head_parameters"] > 0


# ---------------------------------------------------------------------------
# 7. trainable classification is closed and disjoint
# ---------------------------------------------------------------------------


def test_trainable_classification_closed() -> None:
    peft_model, _base, _roots, audit, _targets, merger_names = build_prepared_model()
    lora_names = {entry["name"] for entry in audit["llm_lora_trainable"]}
    merger_names_set = {entry["name"] for entry in audit["merger_trainable"]}
    assert not (lora_names & merger_names_set)
    trainable_logical = {
        ft._strip_peft_prefix(name)
        for name, parameter in peft_model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable_logical == (lora_names | merger_names_set)
    assert audit["checks"]["classification_closed"]
    assert audit["checks"]["families_disjoint"]


# ---------------------------------------------------------------------------
# 8. four optimizer groups: no duplicates, no gaps, right lr/decay policy
# ---------------------------------------------------------------------------


def test_four_optimizer_groups_no_duplicates_no_gaps() -> None:
    peft_model, _base, _roots, _audit, _targets, merger_names = build_prepared_model()
    groups, stats = ft.build_optimizer_groups(
        peft_model, merger_names, lora_lr=1e-4, merger_lr=1e-5, weight_decay=0.01
    )
    assert [group["name"] for group in groups] == [
        "merger_base+decay",
        "merger_base+no_decay",
        "llm_lora+decay",
        "llm_lora+no_decay",
    ]
    param_sets = [set(group["params"]) for group in groups]
    for i, group_set in enumerate(param_sets):
        for other_index, other_set in enumerate(param_sets):
            if i != other_index:
                assert not (group_set & other_set), f"group {i} overlaps {other_index}"
    union = set().union(*param_sets)
    trainable = {
        parameter
        for _name, parameter in peft_model.named_parameters()
        if parameter.requires_grad
    }
    assert union == trainable, "some trainable parameter missing from optimizer groups"
    group_lrs = {group["name"]: group["lr"] for group in stats["groups"]}
    assert group_lrs["merger_base+decay"] == 1e-5
    assert group_lrs["merger_base+no_decay"] == 1e-5
    assert group_lrs["llm_lora+decay"] == 1e-4
    assert group_lrs["llm_lora+no_decay"] == 1e-4
    # explicit decay policy: lora matrices decay; bias and norm do not
    g0 = stats["groups"][0]
    g1 = stats["groups"][1]
    assert "model.visual.merger.linear_fc1.weight" in g0["params"]
    assert "model.visual.merger.linear_fc2.bias" in g1["params"]
    assert "model.visual.merger.norm.weight" in g1["params"]
    g2 = stats["groups"][2]
    assert any(name.endswith(".lora_A.default.weight") for name in g2["params"])


# ---------------------------------------------------------------------------
# 9. merger LR != LoRA LR and the scheduler keeps the ratio
# ---------------------------------------------------------------------------


def test_dual_lr_scheduler_ratio_preserved() -> None:
    peft_model, _base, _roots, _audit, _targets, merger_names = build_prepared_model()
    groups, _stats = ft.build_optimizer_groups(
        peft_model, merger_names, lora_lr=1e-4, merger_lr=1e-5, weight_decay=0.01
    )
    optimizer = torch.optim.AdamW(groups, lr=1e-4)
    scheduler = ft.build_cosine_scheduler(optimizer, num_training_steps=100, warmup_ratio=0.03)
    ratios: set[float] = set()
    for _step in range(0, 100, 7):
        optimizer.step()
        scheduler.step()
        ratios.add(round(optimizer.param_groups[2]["lr"] / optimizer.param_groups[0]["lr"], 9))
        ratios.add(round(optimizer.param_groups[3]["lr"] / optimizer.param_groups[1]["lr"], 9))
    assert ratios == {10.0}  # lora_lr / merger_lr stays constant
    assert optimizer.param_groups[0]["lr"] != optimizer.param_groups[2]["lr"]


# ---------------------------------------------------------------------------
# 10. composite checkpoint contains adapter + merger + manifest + trainer state
# ---------------------------------------------------------------------------


def _build_trainer(workspace: dict, output_dir: Path, gradient_checkpointing: bool = False) -> object:
    """Build a Phase2Trainer over the fake tree and tiny dataset.
    在 fake 模型树与小型数据集上构造 Phase2Trainer。"""
    data_module = ft._load_data_module()
    processor = FakeProcessor()
    aug = ft.augmentation_config_from_args(ft.AugmentationArguments(aug_enabled=False))
    roots_config = data_module.DatasetRootConfig(
        ft.parse_image_roots([f"fake={workspace['image_dir']}"])
    )
    train_dataset = data_module.Phase2EpisodeDataset(
        episode_jsonl=str(workspace["train_file"]),
        roots=roots_config,
        processor=processor,
        aug_config=aug,
        max_seq_length=512,
        seed="phase2-test-seed",
        split="train",
        start_epoch=0,
    )
    wrapped = ft.GroupRepeatDataset(
        train_dataset, workspace["train_file"], "task_kind", {}, max_samples=2
    )
    collator = data_module.Phase2DataCollator()
    model_collator = ft.ModelBatchCollator(collator)

    peft_model, base, roots, _audit, llm_targets, merger_names = build_prepared_model()
    _groups, optimizer_stats = ft.build_optimizer_groups(
        peft_model, merger_names, lora_lr=1e-4, merger_lr=1e-5, weight_decay=0.01
    )
    context = ft.build_training_context(
        config=FakeConfig(),
        processor=processor,
        model_args=ft.ModelArguments(model_id="fake"),
        data_args=ft.DataArguments(train_file=str(workspace["train_file"])),
        lora_args=ft.LoRAArguments(lora_rank=4, lora_alpha=8),
        aug_config=aug,
        aug_seed="phase2-test-seed",
        optimizer_stats=optimizer_stats,
        merger_names=merger_names,
        merger_meta=ft.merger_state_meta(base, merger_names),
        llm_targets=llm_targets,
        image_pixels_applied=False,
        train_sha256=ft.sha256_file(workspace["train_file"]),
        eval_sha256=None,
        train_upstream_manifest_sha256=None,
        train_episode_count=4,
        eval_episode_count=None,
        group_counts=wrapped.group_counts(),
        repeat_weights={},
        image_sources=["fake"],
        repo_root=REPO_ROOT,
    )
    transformers_mod = __import__("transformers")
    training_arguments = transformers_mod.TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=1,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="steps",
        save_steps=1,
        save_total_limit=2,
        eval_strategy="no",
        remove_unused_columns=False,
        report_to="none",
        local_rank=-1,
        seed=42,
        bf16=False,
        fp16=False,
        optim="adamw_torch",
        gradient_checkpointing=gradient_checkpointing,
        use_cpu=True,
        dataloader_pin_memory=False,
    )
    trainer = ft._phase2_trainer_class()(
        model=peft_model,
        args=training_arguments,
        data_collator=model_collator,
        train_dataset=wrapped,
        eval_dataset=None,
        processing_class=processor,
        callbacks=[],
        merger_lr=1e-5,
        lora_lr=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        merger_names=merger_names,
        llm_targets=llm_targets,
        manifest_context=context,
    )
    trainer._test_context = context
    trainer._test_merger_names = merger_names
    return trainer


def test_composite_checkpoint_layout_complete(phase2_workspace: dict) -> None:
    output_dir = phase2_workspace["root"] / "run"
    trainer = _build_trainer(phase2_workspace, output_dir)
    trainer.train(resume_from_checkpoint=None)
    trainer.save_model()
    trainer.finalize_root_checkpoint()

    checkpoint_dir = output_dir / "checkpoint-1"
    assert ft.checkpoint_complete(checkpoint_dir)
    assert ft.checkpoint_complete(output_dir)
    required = [
        checkpoint_dir / "adapter" / "adapter_config.json",
        checkpoint_dir / "adapter" / "adapter_model.safetensors",
        checkpoint_dir / "merger_model.safetensors",
        checkpoint_dir / "processor" / "preprocessor_config.json",
        checkpoint_dir / "phase2_training_manifest.json",
        checkpoint_dir / "trainer_state.json",
        checkpoint_dir / "optimizer.pt",
        checkpoint_dir / "scheduler.pt",
    ]
    for path in required:
        assert path.is_file(), f"missing {path}"
    manifest = json.loads(
        (checkpoint_dir / "phase2_training_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 1
    assert manifest["checkpoint_type"] == "phase2_composite"
    assert manifest["lora"]["rank"] == 4
    assert manifest["merger"]["file_sha256"]
    assert manifest["adapter"]["file_sha256"]
    assert len(manifest["optimizer"]["groups"]) == 4
    # adapter keys must stay strictly inside the LLM targets
    adapter_meta = ft.verify_adapter_keys(checkpoint_dir / "adapter", manifest["lora"]["target_modules"])
    assert adapter_meta["key_count"] == 2 * len(manifest["lora"]["target_modules"])
    # merger state read-back matches the saved file
    data_module = ft._load_data_module()
    import safetensors.torch  # noqa: PLC0415

    saved = safetensors.torch.load_file(checkpoint_dir / "merger_model.safetensors")
    assert set(saved) == {entry["name"] for entry in manifest["merger"]["parameters"]}
    # no trainer-state completion marker is faked when the dir is partial
    partial_dir = output_dir / "checkpoint-99"
    partial_dir.mkdir()
    (partial_dir / "adapter").mkdir()
    (partial_dir / "adapter" / "adapter_config.json").write_text("{}")
    assert not ft.checkpoint_complete(partial_dir)


def test_verify_adapter_keys_accepts_peft_short_names(phase2_workspace: dict) -> None:
    """adapter_config target_modules persisted by PEFT as short projection
    names (e.g. "q_proj") must pass verification against full-path LLM
    targets (regression: phase2 run crashed at the first checkpoint save).
    PEFT 以短投影名持久化的 target_modules 必须通过校验（回归：phase2 运行在
    首次保存 checkpoint 时崩溃）。"""
    output_dir = phase2_workspace["root"] / "short_name_run"
    trainer = _build_trainer(phase2_workspace, output_dir)
    trainer.train(resume_from_checkpoint=None)
    trainer.save_model()
    checkpoint_dir = output_dir / "checkpoint-1"
    adapter_dir = checkpoint_dir / "adapter"
    manifest = json.loads(
        (checkpoint_dir / "phase2_training_manifest.json").read_text(encoding="utf-8")
    )
    llm_targets = manifest["lora"]["target_modules"]
    assert len(llm_targets) >= 7

    config_path = adapter_dir / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    short_names = sorted({name.rsplit(".", 1)[-1] for name in llm_targets})
    config["target_modules"] = short_names
    config_path.write_text(json.dumps(config), encoding="utf-8")

    # PEFT-style short names pass; key count still matches 2 keys per target
    meta = ft.verify_adapter_keys(adapter_dir, llm_targets)
    assert meta["key_count"] == 2 * len(llm_targets)
    # target_module_count counts full-path parents (one per llm target)
    assert meta["target_module_count"] == len(llm_targets)

    # a declared module outside the audited LLM projections is rejected
    config["target_modules"] = sorted(set(short_names) | {"visual_proj"})
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(CheckpointError) as error:
        ft.verify_adapter_keys(adapter_dir, llm_targets)
    assert error.value.code == "adapter_key_violation"
    assert "exceed" in error.value.detail

    # dropping a real LoRA module from the config is rejected as incomplete
    config["target_modules"] = short_names[:-1]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(CheckpointError) as error:
        ft.verify_adapter_keys(adapter_dir, llm_targets)
    assert error.value.code == "adapter_key_violation"
    assert "miss" in error.value.detail


# ---------------------------------------------------------------------------
# 12b. phase-1 merger LoRA adapter seeding (default launch configuration)
# ---------------------------------------------------------------------------


_MERGER_LORA_TARGETS = [
    "model.visual.merger.linear_fc1",
    "model.visual.merger.linear_fc2",
    "model.visual.deepstack_merger_list.0.linear_fc1",
    "model.visual.deepstack_merger_list.0.linear_fc2",
    "model.visual.deepstack_merger_list.1.linear_fc1",
    "model.visual.deepstack_merger_list.1.linear_fc2",
    "model.visual.deepstack_merger_list.2.linear_fc1",
    "model.visual.deepstack_merger_list.2.linear_fc2",
]


def _save_merger_lora_adapter(model: FakeQwen, adapter_dir: Path) -> None:
    """Attach a tiny LoRA to every merger projection of the fake model and
    persist it as a PEFT adapter directory (phase-1 product stand-in). LoRA B
    is perturbed so the merge actually changes weights (untrained LoRA B is
    zero and would merge to an identity).
    给 fake 模型的每个 merger projection 挂微型 LoRA 并保存为 PEFT adapter
    目录（phase1 产物的替身）。扰动 lora_B 使合并真正改变权重（未训练的
    lora_B 全零，合并结果等于恒等）。"""
    peft_mod = ft._peft()
    lora_config = peft_mod.LoraConfig(
        r=2, lora_alpha=4, target_modules=_MERGER_LORA_TARGETS
    )
    peft_model = peft_mod.get_peft_model(model, lora_config)
    with torch.no_grad():
        for name, parameter in peft_model.named_parameters():
            if name.endswith("lora_B.default.weight"):
                parameter.add_(torch.randn_like(parameter) * 0.1)
    peft_model.save_pretrained(adapter_dir)


def test_apply_merger_lora_adapter_merges_only_merger(tmp_path: Path) -> None:
    """A phase-1 merger LoRA adapter must change only the merger subtrees;
    LLM and non-merger vision weights stay byte-identical, and the result is
    a plain model ready for phase-2 freeze/LoRA injection.
    phase1 merger LoRA adapter 只改变 merger 子树权重；LLM 与非 merger 视觉
    权重保持不变，合并结果是可直接进入 phase2 冻结/LoRA 注入的普通模型。"""
    base_model = FakeQwen()
    before = {
        name: param.detach().clone() for name, param in base_model.named_parameters()
    }
    adapter_dir = tmp_path / "merger_adapter"
    _save_merger_lora_adapter(FakeQwen(), adapter_dir)

    merged = ft.apply_merger_lora_adapter(base_model, adapter_dir)
    after = dict(merged.named_parameters())
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    assert changed, "merger LoRA adapter changed nothing"
    assert all("merger" in name for name in changed), changed
    for name in before:
        if "merger" not in name:
            assert torch.equal(before[name], after[name]), name
    # merged model is a plain nn.Module (no peft wrapper remains)
    assert not hasattr(merged, "merge_and_unload")


def test_apply_merger_lora_adapter_rejects_non_merger_target(tmp_path: Path) -> None:
    """An adapter declaring an LLM projection target is a hard error: phase-2
    attaches its own LLM LoRA afterwards, so any seeded LLM LoRA would corrupt
    the audit contract. 声明 LLM projection target 的 adapter 是硬错误。"""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "target_modules": ["model.language_model.layers.0.self_attn.q_proj"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError) as error:
        ft.apply_merger_lora_adapter(FakeQwen(), adapter_dir)
    assert "outside merger" in str(error.value)


def test_resume_rejects_missing_merger_lora_adapter(phase2_workspace: dict) -> None:
    """Resuming a run seeded with a merger LoRA adapter requires the same
    adapter identity; a request without it is a stable refusal.
    带 merger LoRA adapter 初始化的 run，resume 必须提供同一 adapter 身份。"""
    output_dir = phase2_workspace["root"] / "adapter_resume_run"
    trainer = _build_trainer(phase2_workspace, output_dir)
    trainer.train(resume_from_checkpoint=None)
    trainer.save_model()
    trainer.finalize_root_checkpoint()
    checkpoint_dir = output_dir / "checkpoint-1"
    context = trainer._test_context
    ft.validate_resume_checkpoint(checkpoint_dir, context)

    mutated = json.loads(json.dumps(context))
    mutated["base_model"]["merger_lora_adapter"] = {
        "config_sha256": "deadbeef",
        "weights_sha256": "deadbeef",
        "target_modules": sorted(_MERGER_LORA_TARGETS),
    }
    with pytest.raises(ResumeConflictError) as error:
        ft.validate_resume_checkpoint(checkpoint_dir, mutated)
    assert any("merger_lora_adapter" in field for field in error.value.fields)


def test_manifest_optimizer_groups_keep_initial_lr(phase2_workspace: dict) -> None:
    """The manifest must persist the configured initial LR, never the live
    warmup-adjusted LR, otherwise resume validation always rejects a run that
    saved mid-warmup (regression: resume after checkpoint-1000 failed).
    manifest 必须持久化配置的初始 LR 而非实时 warmup LR，否则在 warmup 中途
    保存的 run 永远无法 resume（回归：checkpoint-1000 后 resume 失败）。"""
    output_dir = phase2_workspace["root"] / "lr_manifest_run"
    trainer = _build_trainer(phase2_workspace, output_dir)
    trainer.create_optimizer()
    trainer.create_scheduler(num_training_steps=100)
    initial_lrs = {group["name"]: group["lr"] for group in trainer._initial_optimizer_groups}

    # advance the scheduler inside warmup so the live LR diverges from initial
    for _ in range(2):
        trainer.lr_scheduler.step()
    live_lrs = {group["name"]: group["lr"] for group in trainer.optimizer.param_groups}
    assert any(
        abs(live_lrs[name] - initial_lrs[name]) > 1e-9 for name in live_lrs
    ), "test precondition: scheduler must move the live LR during warmup"

    stats = trainer._optimizer_group_stats_for_manifest()
    assert len(stats) == 4
    initial_counts = {
        group["name"]: group["param_count"] for group in trainer._initial_optimizer_groups
    }
    for group in stats:
        assert abs(group["lr"] - initial_lrs[group["name"]]) < 1e-12
        # llm_lora+no_decay is legitimately empty (lora_bias=none): compare
        # counts exactly instead of asserting non-empty. llm_lora+no_decay
        # 组本应为空（lora_bias=none）：精确比较而不是断言非空。
        assert group["param_count"] == initial_counts[group["name"]]


# ---------------------------------------------------------------------------
# 11. merger save / read-back: keys, shapes, dtypes consistent
# ---------------------------------------------------------------------------


def test_merger_state_roundtrip_strict(tmp_path: Path) -> None:
    peft_model, base, _roots, _audit, _targets, merger_names = build_prepared_model()
    merger_path = tmp_path / "merger_model.safetensors"
    meta = ft.save_merger_state(base, merger_path, merger_names)
    assert meta["file_sha256"] == ft.sha256_file(merger_path)
    for entry in meta["parameters"]:
        assert entry["dtype"].startswith("torch.float")
        assert entry["shape"] and entry["numel"] > 0
    # load into a fresh model of identical structure -> exact equality
    fresh_model = FakeQwen()
    fresh_roots = ft.locate_roots(fresh_model)
    fresh_merger_names = ft.enumerate_merger_names(fresh_roots)
    ft.load_merger_state_strict(fresh_model, merger_path, fresh_merger_names)
    for name, parameter in base.named_parameters():
        if ft._under_any(name, merger_names):
            fresh_parameter = dict(fresh_model.named_parameters())[name]
            assert torch.equal(parameter.detach(), fresh_parameter.detach()), name
    # missing key -> hard failure
    import safetensors.torch  # noqa: PLC0415

    broken_path = tmp_path / "broken.safetensors"
    state = ft.collect_merger_state(base, merger_names)
    state.pop(next(iter(state)))
    safetensors.torch.save_file(state, broken_path)
    with pytest.raises(CheckpointError) as error:
        ft.load_merger_state_strict(fresh_model, broken_path, fresh_merger_names)
    assert "merger_load_mismatch" in error.value.code


# ---------------------------------------------------------------------------
# 12. resume stably rejects incompatible data checksum / LoRA config /
#     merger topology
# ---------------------------------------------------------------------------


def test_resume_rejects_incompatible_requests(phase2_workspace: dict) -> None:
    output_dir = phase2_workspace["root"] / "resume_run"
    trainer = _build_trainer(phase2_workspace, output_dir)
    trainer.train(resume_from_checkpoint=None)
    trainer.save_model()
    trainer.finalize_root_checkpoint()
    checkpoint_dir = output_dir / "checkpoint-1"
    context = trainer._test_context

    # identical request passes
    ft.validate_resume_checkpoint(checkpoint_dir, context)

    def _mutate(mutator):
        mutated = json.loads(json.dumps(context))
        mutator(mutated)
        return mutated

    with pytest.raises(ResumeConflictError) as error:
        ft.validate_resume_checkpoint(
            checkpoint_dir, _mutate(lambda ctx: ctx["data"].update(train_sha256="deadbeef"))
        )
    assert any("train_sha256" in field for field in error.value.fields)

    with pytest.raises(ResumeConflictError) as error:
        ft.validate_resume_checkpoint(
            checkpoint_dir, _mutate(lambda ctx: ctx["lora"].update(rank=99))
        )
    assert any("lora.rank" in field for field in error.value.fields)

    with pytest.raises(ResumeConflictError) as error:
        ft.validate_resume_checkpoint(
            checkpoint_dir,
            _mutate(
                lambda ctx: ctx["merger"]["parameters"].__setitem__(
                    0, {**ctx["merger"]["parameters"][0], "shape": [999, 999]}
                )
            ),
        )
    assert any("merger.parameters" in field for field in error.value.fields)

    # a tampered merger file (checksum mismatch) is also rejected
    merger_path = checkpoint_dir / "merger_model.safetensors"
    original = merger_path.read_bytes()
    merger_path.write_bytes(original + b"\x00")
    try:
        with pytest.raises(ResumeConflictError) as error:
            ft.validate_resume_checkpoint(checkpoint_dir, context)
        assert any("merger.file_sha256" in field for field in error.value.fields)
    finally:
        merger_path.write_bytes(original)


# ---------------------------------------------------------------------------
# 13. resume restores the epoch -> augmentation seeds do not drift
# ---------------------------------------------------------------------------


def test_resume_epoch_augmentation_seed_no_drift(phase2_workspace: dict) -> None:
    data_module = ft._load_data_module()
    processor = FakeProcessor()
    aug = ft.augmentation_config_from_args(ft.AugmentationArguments(aug_enabled=True))
    roots_config = data_module.DatasetRootConfig(
        ft.parse_image_roots([f"fake={phase2_workspace['image_dir']}"])
    )

    def _dataset(start_epoch: int) -> object:
        return data_module.Phase2EpisodeDataset(
            episode_jsonl=str(phase2_workspace["train_file"]),
            roots=roots_config,
            processor=processor,
            aug_config=aug,
            max_seq_length=512,
            seed="drift-seed",
            split="train",
            start_epoch=start_epoch,
        )

    callback = ft._make_epoch_sync_callback(_dataset(0))
    # resume mid-epoch: state.epoch is fractional; int() floors to the epoch
    # whose seeds the current steps were generated with
    # resume 时 state.epoch 可能带小数；int() 取整回到当前步骤所属 epoch。
    callback.on_train_begin(SimpleNamespace(), SimpleNamespace(epoch=1.4), SimpleNamespace())
    assert callback._dataset.epoch == 1

    # features at the callback-driven epoch == features of a dataset whose
    # start_epoch is that same epoch (no drift), and != another epoch
    explicit_epoch_1 = _dataset(start_epoch=1)
    features_callback = [callback._dataset[i] for i in range(len(callback._dataset))]
    features_explicit = [explicit_epoch_1[i] for i in range(len(explicit_epoch_1))]
    for callback_feature, explicit_feature in zip(features_callback, features_explicit):
        assert torch.equal(callback_feature["input_ids"], explicit_feature["input_ids"])
        assert torch.equal(callback_feature["pixel_values"], explicit_feature["pixel_values"])
        assert torch.equal(callback_feature["image_grid_thw"], explicit_feature["image_grid_thw"])

    # epoch 0 and epoch 1 differ for at least one episode (geometry/degradation)
    epoch_0 = _dataset(start_epoch=0)
    drifted = False
    for index in range(len(epoch_0)):
        if not torch.equal(epoch_0[index]["pixel_values"], explicit_epoch_1[index]["pixel_values"]):
            drifted = True
            break
    assert drifted, "augmentation unexpectedly identical across epochs"


def test_sampler_provider_uses_transformers_get_train_dataloader() -> None:
    """Use the Transformers 5.x dataloader method when no cache exists.
    无缓存时使用 Transformers 5.x 的 dataloader 方法。
    """
    class Sampler:
        pass

    class Loader:
        sampler = Sampler()

    class Trainer:
        _train_dataloader = None

        def get_train_dataloader(self):
            self.called = True
            return Loader()

        train_dataloader = property(
            lambda self: (_ for _ in ()).throw(AssertionError("legacy property used"))
        )

    trainer = Trainer()
    provider = ft._get_train_sampler(trainer)
    assert trainer.called is True
    assert isinstance(provider, Sampler)


# ---------------------------------------------------------------------------
# 14. one small forward/backward: LoRA and merger both get gradients,
#     frozen parameters get none
# ---------------------------------------------------------------------------


def test_small_forward_backward_gradients(phase2_workspace: dict) -> None:
    output_dir = phase2_workspace["root"] / "grad_run"
    trainer = _build_trainer(phase2_workspace, output_dir)
    trainer.train(resume_from_checkpoint=None)
    data_module = ft._load_data_module()
    collator = data_module.Phase2DataCollator()
    smoke_batch, _meta = collator(
        [trainer.train_dataset[index] for index in range(min(2, len(trainer.train_dataset)))]
    )
    summary = ft.run_gradient_smoke_check(trainer, smoke_batch)
    assert summary["lora_checked"] > 0
    assert summary["merger_checked"] > 0
    assert summary["lora_missing_gradient"] == []
    assert summary["merger_missing_gradient"] == []
    assert summary["frozen_with_grad"] == []
    # every trainable parameter belongs to one of the two families
    assert summary["lora_checked"] + summary["merger_checked"] == 28 + 18


# ---------------------------------------------------------------------------
# 15. default local_files_only=True
# ---------------------------------------------------------------------------


def test_local_files_only_default_true() -> None:
    assert ft.ModelArguments().local_files_only is True
    assert ft.LoRAArguments().lora_rank == 64
    assert ft.LoRAArguments().lora_alpha == 128
    assert ft.LoRAArguments().lora_lr == 1e-4
    assert ft.LoRAArguments().merger_lr == 1e-5
    assert ft.CheckpointArguments(output_dir="x").save_total_limit == 3


# ---------------------------------------------------------------------------
# resume target resolution: complete dirs auto-resumed, broken refused,
# fresh start over existing checkpoints refused
# ---------------------------------------------------------------------------


def test_resume_target_resolution(phase2_workspace: dict) -> None:
    output_dir = phase2_workspace["root"] / "target_run"
    trainer = _build_trainer(phase2_workspace, output_dir)
    trainer.train(resume_from_checkpoint=None)
    trainer.save_model()
    trainer.finalize_root_checkpoint()

    latest = ft.resolve_resume_target(output_dir, None)
    assert latest == str(output_dir / "checkpoint-1")

    broken = output_dir / "checkpoint-99"
    broken.mkdir()
    (broken / "adapter").mkdir()
    (broken / "adapter" / "adapter_config.json").write_text("{}")
    # auto-resume still picks the complete checkpoint
    assert ft.resolve_resume_target(output_dir, None) == latest
    # explicit resume of the broken dir is refused
    with pytest.raises(ResumeConflictError):
        ft.resolve_resume_target(output_dir, str(broken))
    # forcing a fresh start over existing checkpoints is refused
    with pytest.raises(ResumeConflictError):
        ft.resolve_resume_target(output_dir, "")

    # an output dir containing only broken checkpoints refuses fresh start
    only_broken = phase2_workspace["root"] / "only_broken"
    only_broken.mkdir()
    (only_broken / "checkpoint-3").mkdir()
    (only_broken / "checkpoint-3" / "adapter").mkdir()
    (only_broken / "checkpoint-3" / "adapter" / "adapter_config.json").write_text("{}")
    with pytest.raises(ResumeConflictError) as error:
        ft.resolve_resume_target(only_broken, None)
    assert error.value.code == "incomplete_checkpoints_exist"


# ---------------------------------------------------------------------------
# deterministic group repeat: every episode >= 1 per epoch, stable order
# ---------------------------------------------------------------------------


def test_group_repeat_deterministic_epoch_coverage(phase2_workspace: dict) -> None:
    data_module = ft._load_data_module()
    processor = FakeProcessor()
    aug = ft.augmentation_config_from_args(ft.AugmentationArguments(aug_enabled=False))
    roots_config = data_module.DatasetRootConfig(
        ft.parse_image_roots([f"fake={phase2_workspace['image_dir']}"])
    )
    dataset = data_module.Phase2EpisodeDataset(
        episode_jsonl=str(phase2_workspace["train_file"]),
        roots=roots_config,
        processor=processor,
        aug_config=aug,
        max_seq_length=512,
        seed="repeat-seed",
        split="train",
        start_epoch=0,
    )
    weights = {"vqa_box_assisted": 2}
    wrapped = ft.GroupRepeatDataset(dataset, phase2_workspace["train_file"], "task_kind", weights)
    # 4 episodes: 2 vqa_box_assisted x2 + 2 geochat x1
    assert len(wrapped) == 2 * 2 + 2 * 1
    seen: dict[str, int] = {}
    for index in range(len(wrapped)):
        episode_id = wrapped[index]["episode_id"]
        seen[episode_id] = seen.get(episode_id, 0) + 1
    assert all(count >= 1 for count in seen.values())
    assert seen["fake/train/img0.png/qa/0/vqa_box_assisted"] == 2
    assert seen["fake/train/img1.png/qa/0/geochat_conversation"] == 1
    # schedule is deterministic: rebuilding yields identical episode order
    wrapped_again = ft.GroupRepeatDataset(dataset, phase2_workspace["train_file"], "task_kind", weights)
    assert [
        wrapped[index]["episode_id"] for index in range(len(wrapped))
    ] == [
        wrapped_again[index]["episode_id"] for index in range(len(wrapped_again))
    ]
    assert wrapped.group_counts() == {"geochat_conversation": 2, "vqa_box_assisted": 2}


# ---------------------------------------------------------------------------
# CLI surfaces: deepspeed/fsdp are refused with a stable error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extra", ["--deepspeed", "--fsdp"])
def test_deepspeed_and_fsdp_refused(extra: str, tmp_path: Path) -> None:
    """main() must refuse DeepSpeed/FSDP before any model loading.
    main() 必须在加载模型前稳定拒绝 DeepSpeed/FSDP。"""
    argv = [
        "finetune_qwen3vl_phase2.py",
        "--train-file", str(tmp_path / "train.jsonl"),
        "--image-root", f"fake={tmp_path}",
        "--output-dir", str(tmp_path / "out"),
        "--max-train-samples", "1",
        extra,
    ]
    if extra == "--deepspeed":
        argv.append(str(tmp_path / "ds.json"))
    else:
        argv.append("full_shard")
    with pytest.raises(ConfigurationError) as error:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(sys, "argv", argv)
            ft.main()
    assert "not supported" in str(error.value)
