"""Offline tests for Qwen3.5 visual-planner LoRA training utilities.
Qwen3.5 visual-planner LoRA 训练工具的离线测试。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
from PIL import Image
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "finetune_qwen35_9b_visual_planner_lora.py"
SPEC = importlib.util.spec_from_file_location("qwen35_visual_planner_lora", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ft = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ft
SPEC.loader.exec_module(ft)


def _target(task: str = "general_vqa", *, explicit: bool = False) -> str:
    return json.dumps(
        {
            "count_target": None,
            "needs_visual_assistance": False,
            "object_categories": [],
            "reason_codes": ["general_question"],
            "region_request": {
                "explicit": explicit,
                "image_index": 0 if explicit else None,
                "roi_xyxy": [100, 200, 500, 600] if explicit else None,
            },
            "task": task,
            "version": "visual-task-plan-v5",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _record(
    image: str,
    episode_id: str = "episode-1",
    task: str = "general_vqa",
    *,
    explicit: bool = False,
) -> dict:
    return {
        "episode_id": episode_id,
        "format": "visual-planner-compiled-chat-v1",
        "messages": [
            {"role": "system", "content": "system protocol"},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "What is visible?"},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": _target(task, explicit=explicit)}
                ],
            },
        ],
        "schema_version": 1,
        "source_group": "test",
        "split": "train",
    }


def _write_dataset(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "dataset"
    image_path = root / "training_images" / "train" / "image.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16), color=(10, 20, 30)).save(image_path)
    training = root / "training"
    training.mkdir()
    train = _record("training_images/train/image.png")
    (training / "train.jsonl").write_text(json.dumps(train) + "\n", encoding="utf-8")
    validation = dict(train, episode_id="episode-val", split="val")
    (training / "val.jsonl").write_text(json.dumps(validation) + "\n", encoding="utf-8")
    return root, image_path


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text: str, add_special_tokens: bool = False) -> object:
        del add_special_tokens
        return type("Tokenized", (), {"input_ids": [100 + ord(char) for char in text]})()


class FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()

    @staticmethod
    def _content(content: object) -> str:
        if isinstance(content, str):
            return content
        return "".join(
            "<image>" if item["type"] == "image" else item["text"]
            for item in content
        )

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        del tokenize, enable_thinking
        chunks = []
        for message in messages:
            prefix = f"<{message['role']}>"
            if message["role"] == "assistant":
                prefix += "<think></think>"
            chunks.append(
                f"{prefix}{self._content(message['content'])}</{message['role']}>"
            )
        text = "".join(chunks)
        if add_generation_prompt:
            text += "<assistant><think></think>"
        return text

    def __call__(
        self,
        *,
        text: list[str],
        images: list[Image.Image],
        return_tensors: str,
        padding: bool,
    ) -> dict:
        del return_tensors, padding
        ids = self.tokenizer(text[0]).input_ids
        # Add deterministic visual tokens before the assistant suffix.
        # 在 assistant 后缀之前加入确定性视觉 token。
        insert_at = text[0].index("</user>")
        ids = ids[:insert_at] + [777, 777] * len(images) + ids[insert_at:]
        length = len(ids)
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
            "mm_token_type_ids": torch.zeros((1, length), dtype=torch.long),
            "pixel_values": torch.ones((2 * len(images), 3)),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images)),
        }


class FakeLinearAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for name in ft.LINEAR_ATTN_PROJECTIONS:
            setattr(self, name, nn.Linear(4, 4, bias=False))


class FakeFullAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for name in ft.FULL_ATTN_PROJECTIONS:
            setattr(self, name, nn.Linear(4, 4, bias=False))


class FakeMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for name in ft.MLP_PROJECTIONS:
            setattr(self, name, nn.Linear(4, 4, bias=False))


class FakeLayer(nn.Module):
    def __init__(self, *, linear: bool) -> None:
        super().__init__()
        if linear:
            self.linear_attn = FakeLinearAttention()
        else:
            self.self_attn = FakeFullAttention()
        self.mlp = FakeMlp()


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.visual = nn.Module()
        self.model.visual.blocks = nn.ModuleList([nn.Linear(4, 4)])
        self.model.visual.patch_embed = nn.Linear(4, 4)
        # Same-name trap under vision must never be selected.
        # 视觉塔中的同名陷阱不得被选择。
        self.model.visual.q_proj = nn.Linear(4, 4)
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(16, 4)
        self.model.language_model.layers = nn.ModuleList(
            [FakeLayer(linear=True), FakeLayer(linear=False)]
        )


def test_index_validates_compiled_records_and_counts_tasks(tmp_path: Path) -> None:
    root, _image = _write_dataset(tmp_path)
    index = ft.CompiledChatIndex(
        root / "training" / "train.jsonl", root, expected_split="train"
    )
    assert len(index) == 1
    assert index.task_counts() == {"general_vqa": 1}
    assert index.roi_explicit_count() == 0
    assert index.read(0)["episode_id"] == "episode-1"


@pytest.mark.parametrize("unsafe", ["../escape.png", "/tmp/escape.png", "C:\\escape.png"])
def test_image_paths_fail_closed(tmp_path: Path, unsafe: str) -> None:
    root, _image = _write_dataset(tmp_path)
    with pytest.raises(ft.TrainingConfigurationError, match="unsafe_image_path"):
        ft.resolve_image_path(root, unsafe)


def test_encode_masks_everything_before_exact_assistant_target(tmp_path: Path) -> None:
    root, _image = _write_dataset(tmp_path)
    record = json.loads((root / "training" / "train.jsonl").read_text())
    feature = ft.encode_record(
        record,
        dataset_root=root,
        processor=FakeProcessor(),
        max_seq_length=4096,
    )
    supervised = feature["labels"] != ft.IGNORE_INDEX
    first = int(torch.nonzero(supervised, as_tuple=False)[0])
    assert first > 0
    assert torch.equal(feature["labels"][first:], feature["input_ids"][first:])
    assert bool(torch.all(feature["labels"][:first] == ft.IGNORE_INDEX))
    assert not any(key.startswith("roi_") for key in feature)
    assert feature["task"] == "general_vqa"


def test_collator_right_pads_text_and_concatenates_images(tmp_path: Path) -> None:
    root, _image = _write_dataset(tmp_path)
    record = json.loads((root / "training" / "train.jsonl").read_text())
    processor = FakeProcessor()
    first = ft.encode_record(record, dataset_root=root, processor=processor, max_seq_length=4096)
    second = dict(first)
    for key in ("input_ids", "attention_mask", "labels", "mm_token_type_ids"):
        second[key] = second[key][:-5]
    collated = ft.VisualPlannerCollator(processor.tokenizer.pad_token_id)([first, second])
    assert collated["input_ids"].shape[0] == 2
    assert bool(torch.all(collated["labels"][1, -5:] == ft.IGNORE_INDEX))
    assert collated["pixel_values"].shape[0] == 4
    assert collated["image_grid_thw"].shape == (2, 3)
    assert not any(key.startswith("roi_") for key in collated)


def test_explicit_roi_is_supervised_only_as_assistant_tokens(tmp_path: Path) -> None:
    root, _image = _write_dataset(tmp_path)
    record = _record("training_images/train/image.png", explicit=True)
    feature = ft.encode_record(
        record,
        dataset_root=root,
        processor=FakeProcessor(),
        max_seq_length=4096,
    )
    assert not any(key.startswith("roi_") for key in feature)
    supervised_ids = feature["labels"][feature["labels"] != ft.IGNORE_INDEX]
    target_ids = FakeTokenizer()(_target(explicit=True)).input_ids
    assert all(token in supervised_ids.tolist() for token in target_ids)


def test_preflight_prefers_task_coverage() -> None:
    class TinyDataset:
        def __init__(self) -> None:
            self.index = type(
                "Index",
                (),
                {
                    "records": [
                        ft.IndexedRecord(0, "a", "general_vqa"),
                        ft.IndexedRecord(1, "b", "general_vqa"),
                        ft.IndexedRecord(2, "c", "change_qa"),
                    ]
                },
            )()

        def __len__(self) -> int:
            return 3

        def __getitem__(self, index: int) -> dict:
            task = self.index.records[index].task
            return {
                "input_ids": torch.ones(4, dtype=torch.long),
                "labels": torch.tensor([ft.IGNORE_INDEX, 1, 1, 1]),
                "task": task,
            }

    summary = ft._preflight(TinyDataset(), 2)
    assert summary["task_counts"] == {"change_qa": 1, "general_vqa": 1}


def test_hybrid_lora_targets_are_complete_and_language_only() -> None:
    roots = ft.locate_model_roots(FakeModel())
    targets = ft.enumerate_lora_targets(roots)
    assert len(targets) == 15  # 5+3 linear-attn, 4+3 full-attn.
    assert any("linear_attn.in_proj_qkv" in target for target in targets)
    assert any("self_attn.q_proj" in target for target in targets)
    assert all(target.startswith("model.language_model.") for target in targets)
    assert not any("visual" in target for target in targets)


def test_exact_target_regex_does_not_match_same_name_vision_module() -> None:
    roots = ft.locate_model_roots(FakeModel())
    targets = ft.enumerate_lora_targets(roots)
    pattern = "(?:" + "|".join(ft.re.escape(target) for target in targets) + ")"
    assert ft.re.fullmatch(pattern, "model.language_model.layers.1.self_attn.q_proj")
    assert not ft.re.fullmatch(pattern, "model.visual.q_proj")


def test_local_path_cannot_be_persisted_as_model_identity() -> None:
    with pytest.raises(ft.TrainingConfigurationError, match="path_independent"):
        ft._safe_model_identity("/home/user/models/Qwen3.5-9B")
    assert ft._safe_model_identity("Qwen/Qwen3.5-9B") == "Qwen/Qwen3.5-9B"


def test_latest_checkpoint_ignores_partial_directory(tmp_path: Path) -> None:
    partial = tmp_path / "checkpoint-2"
    partial.mkdir()
    (partial / "trainer_state.json").write_text("{}")
    complete = tmp_path / "checkpoint-1"
    complete.mkdir()
    for filename in (
        "adapter_config.json",
        "adapter_model.safetensors",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
    ):
        (complete / filename).write_bytes(b"x")
    assert ft._latest_checkpoint(tmp_path) == complete


def test_resume_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    (tmp_path / ft.MANIFEST_NAME).write_text(
        json.dumps({"request_identity": {"seed": 1}}), encoding="utf-8"
    )
    with pytest.raises(ft.TrainingConfigurationError, match="identity_mismatch"):
        ft.validate_resume_identity(tmp_path, {"seed": 2})


def test_completed_run_cannot_be_resumed(tmp_path: Path) -> None:
    identity = {"seed": 1}
    (tmp_path / ft.MANIFEST_NAME).write_text(
        json.dumps({"status": "completed", "request_identity": identity}),
        encoding="utf-8",
    )
    with pytest.raises(ft.TrainingConfigurationError, match="already_completed"):
        ft.validate_resume_identity(tmp_path, identity)
