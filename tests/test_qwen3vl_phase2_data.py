"""Tests for scripts/qwen3vl_phase2_data.py.

Unit tests use small temp images and a fake processor that mimics the pinned
transformers 5.14.1 Qwen3VLProcessor contract (chatml rendering with
<|vision_start|><|image_pad|><|vision_end|>, image-token expansion via the
total length delta, "assistant_masks" template mask, pixel_values of shape
(grid_h*grid_w, C*patch*patch) and image_grid_thw (1, 3)). No 8B model is
loaded anywhere.

测试使用小型临时图片与 fake processor（模拟钉死的 transformers 5.14.1
Qwen3VLProcessor 契约），不加载任何 8B 模型。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
torch = pytest.importorskip(
    "torch",
    reason="Qwen3-VL Phase2 data tests require PyTorch",
)
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "qwen3vl_phase2_data.py"

spec = importlib.util.spec_from_file_location("qwen3vl_phase2_data", SCRIPT_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["qwen3vl_phase2_data"] = mod  # dataclasses need module registration
spec.loader.exec_module(mod)

IGNORE = mod.IGNORE_INDEX
AugmentationConfig = mod.AugmentationConfig
DatasetRootConfig = mod.DatasetRootConfig
Phase2EpisodeDataset = mod.Phase2EpisodeDataset
Phase2DataCollator = mod.Phase2DataCollator
EpisodeTooLongError = mod.EpisodeTooLongError
ImagePathError = mod.ImagePathError
CollatorError = mod.CollatorError


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


def _char_id(ch: str) -> int:
    return 2000 + ord(ch)


class FakeQwenVLProcessor:
    """Mimics Qwen3VLProcessor: chatml template, placeholder expansion,
    assistant_masks, mm_token_type_ids, pixel_values/image_grid_thw."""

    def __init__(self, support_template_mask: bool = True) -> None:
        self.support_template_mask = support_template_mask
        self.seen_images: list[np.ndarray] = []

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
                ids.append(_char_id(text[i]))
                i += 1
        return ids

    def _token_starts(self, text: str, ids: list[int]) -> list[int]:
        starts: list[int] = []
        i = 0
        for _ in ids:
            starts.append(i)
            for tok in _SPECIAL_LIST:
                if text.startswith(tok, i):
                    i += len(tok)
                    break
            else:
                i += 1
        return starts

    # -- chat template -----------------------------------------------------
    def render(self, messages, add_generation_prompt: bool = False) -> str:
        out = ""
        for msg in messages:
            out += f"<|im_start|>{msg['role']}\n"
            content = msg["content"]
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

    def assistant_mask(self, text: str, ids: list[int]) -> list[int]:
        """Char-span based mask like transformers 5.14.1: assistant content
        through the <|im_end|> token; headers never marked.
        与 5.14.1 相同的基于字符跨度的 mask：assistant 内容直至 <|im_end|>。"""
        mask = [0] * len(ids)
        starts = self._token_starts(text, ids)
        pos = 0
        while True:
            start = text.find("<|im_start|>assistant\n", pos)
            if start < 0:
                break
            end = text.find("<|im_end|>", start)
            if end < 0:
                end = len(text)
            content_start = start + len("<|im_start|>assistant\n")
            content_end = end + len("<|im_end|>")
            # inclusive token range for chars [content_start, content_end)
            s_tok = max(0, np.searchsorted(starts, content_start, side="right") - 1)
            e_tok = max(0, np.searchsorted(starts, content_end, side="right") - 1)
            for t in range(s_tok, e_tok + 1):
                if t < len(mask):
                    mask[t] = 1
            pos = end + len("<|im_end|>")
        return mask

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
    ):
        text = self.render(messages, add_generation_prompt)
        if not tokenize:
            return text
        ids = self.tokenize(text)
        if return_assistant_tokens_mask:
            if not self.support_template_mask:
                raise ValueError("return_assistant_tokens_mask not supported")
            return {
                "input_ids": ids,
                "assistant_masks": self.assistant_mask(text, ids),
            }
        return {"input_ids": ids} if return_dict else ids

    # -- processor call ----------------------------------------------------
    def __call__(self, text=None, images=None, return_tensors="pt"):
        t0 = text[0]
        arr = np.asarray(images[0])
        h, w = arr.shape[:2]
        gh = max(1, (h + 27) // 28)
        gw = max(1, (w + 27) // 28)
        n = gh * gw
        ph = "<|image_pad|>"
        idx = t0.find(ph)
        assert idx >= 0, "no image placeholder in text"
        expanded = t0[:idx] + ph * n + t0[idx + len(ph):]
        ids = self.tokenize(expanded)
        mm = [1 if t == _SPECIALS["<|image_pad|>"] else 0 for t in ids]
        self.seen_images.append(arr.copy())
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            "mm_token_type_ids": torch.tensor([mm], dtype=torch.long),
            "pixel_values": torch.zeros((n, 1176), dtype=torch.float32),
            "image_grid_thw": torch.tensor([[1, gh, gw]], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def box(x1, y1, x2, y2, label="", desc="", sid=None):
    entry = {
        "xyxy_999": [x1, y1, x2, y2],
        "label": label,
        "description": desc,
        "source_object_id": sid,
    }
    if sid is None:
        entry.pop("source_object_id")
    return entry


def make_images(tmp_path):
    vroot = tmp_path / "vrsbench_root"
    groot = tmp_path / "geochat_root"
    vroot.mkdir()
    groot.mkdir()
    rng = np.random.RandomState(0)
    Image.fromarray(rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(vroot / "a.png")
    Image.fromarray(rng.randint(0, 255, (100, 80, 3), dtype=np.uint8)).save(vroot / "b.png")
    Image.fromarray(rng.randint(0, 255, (200, 150, 3), dtype=np.uint8)).save(vroot / "c.png")
    Image.fromarray(rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(groot / "r.jpg")
    Image.fromarray(rng.randint(0, 255, (100, 80, 3), dtype=np.uint8)).save(groot / "i.jpg")
    Image.fromarray(rng.randint(0, 255, (200, 150, 3), dtype=np.uint8)).save(groot / "cv.jpg")
    Image.fromarray(rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(vroot / "v.png")
    return vroot, groot


def make_episodes():
    """One grounding + paired boxed/self-attention + unboxed + geochat
    refer/identify/conversation (train) and one validation episode.
    train 含 grounding、成对有框/无框 VQA、GeoChat 各类；validation 一条。"""
    refer_turns = [
        {
            "user_text": "<image>\n[refer] where is <p>silver plane</p> ?",
            "assistant_text": "{<16><55><24><63>|<11>}",
            "input_boxes": [],
            "target_boxes": [box(160, 549, 240, 629)],
        },
        {
            "user_text": "\n[refer] give me the location of <p>2 a220 airplanes at the bottom left</p>",
            "assistant_text": "{<8><82><16><90>|<69>}",
            "input_boxes": [],
            "target_boxes": [box(80, 819, 160, 899)],
        },
    ]
    conv_turns = [
        {
            "user_text": "<image>\nHow many buildings are present in the image?",
            "assistant_text": "There are 4 buildings present in the image.",
            "input_boxes": [],
            "target_boxes": [],
        },
        {
            "user_text": "Can you count the number of cars?",
            "assistant_text": "There are 45 cars.",
            "input_boxes": [],
            "target_boxes": [],
        },
        {
            "user_text": "How many teal cars are there in the image?",
            "assistant_text": "There are 3 teal cars in the image.",
            "input_boxes": [],
            "target_boxes": [],
        },
    ]
    train = [
        {
            "schema_version": 1,
            "episode_id": "vrsbench/train/a.png/obj/0",
            "parent_episode_id": "vrsbench/train/a.png/obj/0",
            "dataset": "VRSBench", "split": "train",
            "image_source": "vrsbench", "image": "a.png",
            "task_kind": "vrsbench_grounding", "source_task": "grounding",
            "turns": [{
                "user_text": "The red car is on the road.",
                "assistant_text": "",
                "input_boxes": [],
                "target_boxes": [box(100, 100, 300, 300, "car", "The red car is on the road.", 0)],
            }],
            "augmentation_policy": {"geometry": "geometry_safe"},
            "provenance": {"source_record_id": "x", "object_id": 0, "view": "grounding"},
        },
        {
            "schema_version": 1,
            "episode_id": "vrsbench/train/b.png/qa/1/box_assisted",
            "parent_episode_id": "vrsbench/train/b.png/qa/1",
            "dataset": "VRSBench", "split": "train",
            "image_source": "vrsbench", "image": "b.png",
            "task_kind": "vqa_box_assisted", "source_task": "counting",
            "turns": [{
                "user_text": "How many small vehicles are visible?",
                "assistant_text": "1",
                "input_boxes": [box(380, 200, 450, 260, "vehicle", "The small vehicle.", 0)],
                "target_boxes": [],
            }],
            "augmentation_policy": {"geometry": "geometry_safe"},
            "provenance": {"source_record_id": "x", "question_id": 1, "view": "box_assisted"},
        },
        {
            "schema_version": 1,
            "episode_id": "vrsbench/train/b.png/qa/1/self_attention",
            "parent_episode_id": "vrsbench/train/b.png/qa/1",
            "dataset": "VRSBench", "split": "train",
            "image_source": "vrsbench", "image": "b.png",
            "task_kind": "vqa_self_attention", "source_task": "counting",
            "turns": [{
                "user_text": "How many small vehicles are visible?",
                "assistant_text": "1",
                "input_boxes": [],
                "target_boxes": [],
            }],
            "augmentation_policy": {"geometry": "geometry_safe"},
            "provenance": {"source_record_id": "x", "question_id": 1, "view": "self_attention"},
        },
        {
            "schema_version": 1,
            "episode_id": "vrsbench/train/c.png/qa/2/naturally_unboxed",
            "parent_episode_id": "vrsbench/train/c.png/qa/2",
            "dataset": "VRSBench", "split": "train",
            "image_source": "vrsbench", "image": "c.png",
            "task_kind": "vqa_naturally_unboxed", "source_task": "scene_type",
            "turns": [{
                "user_text": "What is the main structure?",
                "assistant_text": "Expressway toll station",
                "input_boxes": [],
                "target_boxes": [],
            }],
            "augmentation_policy": {"geometry": "geometry_safe"},
            "provenance": {"source_record_id": "x", "question_id": 2, "view": "naturally_unboxed"},
        },
        {
            "schema_version": 1,
            "episode_id": "geochat/train/1/r.jpg",
            "parent_episode_id": "geochat/train/1/r.jpg",
            "dataset": "GeoChat", "split": "train",
            "image_source": "geochat", "image": "r.jpg",
            "task_kind": "geochat_refer", "source_task": "refer",
            "turns": refer_turns,
            "augmentation_policy": {"geometry": "geometry_safe"},
            "provenance": {"kind": "geochat_refer", "view": "raw", "n_turns": 2, "multi_turn": True},
        },
        {
            "schema_version": 1,
            "episode_id": "geochat/train/2/i.jpg",
            "parent_episode_id": "geochat/train/2/i.jpg",
            "dataset": "GeoChat", "split": "train",
            "image_source": "geochat", "image": "i.jpg",
            "task_kind": "geochat_identify", "source_task": "identify",
            "turns": [{
                "user_text": "<image>\n[identify] the object in is",
                "assistant_text": "<p>1 dump-truck at the center</p>",
                "input_boxes": [box(350, 320, 370, 360)],
                "target_boxes": [],
            }],
            "augmentation_policy": {"geometry": "geometry_safe"},
            "provenance": {"kind": "geochat_identify", "view": "raw", "n_turns": 1, "multi_turn": False},
        },
        {
            "schema_version": 1,
            "episode_id": "geochat/train/3/cv.jpg",
            "parent_episode_id": "geochat/train/3/cv.jpg",
            "dataset": "GeoChat", "split": "train",
            "image_source": "geochat", "image": "cv.jpg",
            "task_kind": "geochat_conversation", "source_task": "conversation",
            "turns": conv_turns,
            "augmentation_policy": {"geometry": "geometry_safe"},
            "provenance": {"kind": "geochat_conversation", "view": "raw", "n_turns": 3, "multi_turn": True},
        },
        {
            "schema_version": 1,
            "episode_id": "vrsbench/train/a.png/obj/1",
            "parent_episode_id": "vrsbench/train/a.png/obj/1",
            "dataset": "VRSBench", "split": "train",
            "image_source": "vrsbench", "image": "a.png",
            "task_kind": "vrsbench_grounding", "source_task": "grounding",
            "turns": [{
                "user_text": "The small vehicle near the top-middle of the image.",
                "assistant_text": "",
                "input_boxes": [],
                "target_boxes": [box(380, 200, 450, 260, "vehicle", "top-middle", 1)],
            }],
            "augmentation_policy": {"geometry": "orientation_locked"},
            "provenance": {"source_record_id": "x", "object_id": 1, "view": "grounding"},
        },
    ]
    val = [
        {
            "schema_version": 1,
            "episode_id": "vrsbench/val/v.png/qa/1/box_assisted",
            "parent_episode_id": "vrsbench/val/v.png/qa/1",
            "dataset": "VRSBench", "split": "validation",
            "image_source": "vrsbench", "image": "v.png",
            "task_kind": "vqa_box_assisted", "source_task": "counting",
            "turns": [{
                "user_text": "How many boats?",
                "assistant_text": "1",
                "input_boxes": [box(200, 200, 400, 400, "boat", "A small boat.", 0)],
                "target_boxes": [],
            }],
            "augmentation_policy": {"geometry": "geometry_safe"},
            "provenance": {"source_record_id": "x", "question_id": 1, "view": "box_assisted"},
        },
    ]
    return train, val


@pytest.fixture()
def data(tmp_path):
    vroot, groot = make_images(tmp_path)
    train, val = make_episodes()
    ep_file = tmp_path / "train.jsonl"
    val_file = tmp_path / "validation.jsonl"
    ep_file.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in train) + "\n",
        encoding="utf-8",
    )
    val_file.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in val) + "\n",
        encoding="utf-8",
    )
    roots = DatasetRootConfig({"vrsbench": str(vroot), "geochat": str(groot)})
    return {
        "ep_file": ep_file,
        "val_file": val_file,
        "vroot": vroot,
        "groot": groot,
        "roots": roots,
        "train_episodes": train,
        "val_episodes": val,
    }


def make_dataset(data, processor=None, aug=None, max_seq_length=2048, seed="test-seed",
                 split="train", epoch=0, episode_file=None):
    return Phase2EpisodeDataset(
        episode_file or data["ep_file"],
        data["roots"],
        processor if processor is not None else FakeQwenVLProcessor(),
        aug if aug is not None else AugmentationConfig(),
        max_seq_length,
        seed,
        split=split,
        start_epoch=epoch,
    )


FORCED_AUG = AugmentationConfig(
    rotate90_prob=1.0,
    affine_rotation_prob=1.0,
    scale_prob=1.0,
    translate_prob=1.0,
    perspective_prob=1.0,
    degradation_probability=1.0,
    min_degradations=1,
    max_degradations=3,
)


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


def test_path_safety_rejects_unsafe_paths(data):
    roots = data["roots"]
    ok = roots.image_path("vrsbench", "a.png")
    assert ok.name == "a.png"
    bad_cases = [
        ("vrsbench", "/abs/a.png"),
        ("vrsbench", "C:/a.png"),
        ("vrsbench", "//server/share/a.png"),
        ("vrsbench", "a\\b.png"),
        ("vrsbench", "../a.png"),
        ("vrsbench", "a/../../b.png"),
        ("vrsbench", "./a.png"),
        ("vrsbench", ""),
        ("vrsbench", "a/./b.png"),
        ("unknown_source", "a.png"),
    ]
    for source, rel in bad_cases:
        with pytest.raises(ImagePathError) as ei:
            roots.image_path(source, rel)
        assert ei.value.code in ("unsafe_image_path", "unknown_image_source")


def test_path_safety_symlink_escape(data, tmp_path):
    outside = tmp_path / "outside.png"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(outside)
    try:
        (data["vroot"] / "evil.png").symlink_to(outside)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("symlink creation requires Windows Developer Mode or SeCreateSymbolicLinkPrivilege")
        raise
    with pytest.raises(ImagePathError) as ei:
        data["roots"].image_path("vrsbench", "evil.png")
    assert ei.value.code == "unsafe_image_path"


def test_missing_image_stable_error(data):
    # missing files surface via the dataset as image_missing
    ep = dict(data["train_episodes"][0])
    ep["image"] = "missing.png"
    ep_file = data["ep_file"].parent / "missing_case.jsonl"
    ep_file.write_text(json.dumps(ep) + "\n", encoding="utf-8")
    ds = make_dataset(data, episode_file=ep_file)
    with pytest.raises(ImagePathError) as ei:
        ds[0]
    assert ei.value.code == "image_missing"
    # the message never contains the machine absolute path
    assert str(data["vroot"]) not in str(ei.value)


# ---------------------------------------------------------------------------
# Geometry box transforms
# ---------------------------------------------------------------------------


def test_rot90_exact_box_transform(data):
    """90/180/270 box transforms match the continuous geometry convention
    exactly (corners -> AABB -> round) and stay within 1px of np.rot90's
    discrete pixel mapping.
    90/180/270 框变换与连续几何约定精确一致，并与 np.rot90 离散像素映射
    相差不超过 1px。"""
    w, h = 100, 100
    # box xyxy pixels [10, 20, 30, 40]
    analytic = {
        1: [200, 689, 400, 889],  # (x,y) -> (y, 99-x)
        2: [689, 589, 889, 789],  # (x,y) -> (99-x, 99-y)
        3: [589, 100, 789, 300],  # (x,y) -> (99-y, x)
    }
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:40, 10:30] = 255
    for k in (1, 2, 3):
        rotated = np.rot90(mask, k)
        ys, xs = np.nonzero(rotated)
        mask_aabb = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        new_w, new_h = rotated.shape[1], rotated.shape[0]
        quad = mod._quad_from_xyxy([10.0, 20.0, 30.0, 40.0])
        out = mod.transform_box_quad(quad, [("rot90", k)], w, h)
        aabb = mod._aabb(out)
        aabb = [min(max(aabb[0], 0.0), new_w), min(max(aabb[1], 0.0), new_h),
                min(max(aabb[2], 0.0), new_w), min(max(aabb[3], 0.0), new_h)]
        got = mod.pixels_to_box_999(aabb, new_w, new_h)
        assert got == analytic[k], f"k={k}: {got} != {analytic[k]}"
        # continuous AABB within 1 px of the discrete np.rot90 result
        for c, m in zip(aabb, mask_aabb):
            assert abs(c - m) <= 1.0, f"k={k}: continuous {aabb} vs discrete {mask_aabb}"


def test_affine_and_perspective_corner_transforms(data):
    """Affine/homography box transforms follow the same matrix math as the
    image warp (corners -> enclosing AABB -> clip -> round).
    仿射/单应框变换与图像 warp 使用同一矩阵（角点 -> 包围 AABB -> 裁剪 -> 取整）。"""
    w, h = 100, 80
    box_px = [10.0, 16.0, 30.0, 32.0]
    matrix = np.array([[0.9, 0.1, 3.0], [-0.1, 0.95, -2.0]], dtype=np.float64)
    quad = mod._quad_from_xyxy(box_px)
    out = mod._transform_quad_affine(quad, matrix)
    # manual same-math expectation (independent implementation in the test)
    ones = np.ones((4, 1))
    pts = np.concatenate([quad, ones], axis=1)
    expected_pts = pts @ matrix.T
    np.testing.assert_allclose(out, expected_pts)
    # homography
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64)
    dst = src + np.array([[5, 5], [-5, 5], [-5, -5], [5, -5]], np.float64)
    hom = cv2_getPerspectiveTransform(src, dst)
    out_h = mod._transform_quad_perspective(quad, hom)
    expected_h = mod_homography_points(quad, hom)
    np.testing.assert_allclose(out_h, expected_h, atol=1e-9)


def cv2_getPerspectiveTransform(src, dst):
    import cv2
    return cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))


def mod_homography_points(quad, hom):
    # manual homography application in homogeneous coordinates
    hom = hom.astype(np.float64)
    pts = np.concatenate([quad, np.ones((4, 1))], axis=1)
    out = pts @ hom.T
    return out[:, :2] / out[:, 2:]


def test_geometry_failure_falls_back_identity(data):
    """A required box failing the quality gates falls the whole episode back
    to the identity transform (image and boxes untouched).
    任一必需框未过质量门禁，整条 episode 回退 identity（图像与框不变）。"""
    cfg = AugmentationConfig(
        rotate90_prob=0.0, affine_rotation_prob=0.0, scale_prob=0.0,
        translate_prob=0.0, perspective_prob=1.0,
        perspective_frac_range=(0.49, 0.5), degradation_probability=0.0,
    )
    image = Image.fromarray(np.random.RandomState(1).randint(0, 255, (64, 64, 3), dtype=np.uint8))
    original = np.asarray(image)
    seed = None
    # full-image box: the strong perspective stretches the AABB beyond the
    # inflation gate for a deterministic set of group seeds
    for probe in range(80):
        group_seed = hashlib.sha256(f"probe-{probe}".encode()).hexdigest()
        entries_probe = [dict(box(0, 0, 999, 999))]
        _, meta = mod.apply_augmentation(image, entries_probe, "geometry_safe", cfg, group_seed)
        if meta["geometry"]["fallback_code"] is not None:
            seed = group_seed
            break
    assert seed is not None, "no fallback trigger found"
    entries = [box(0, 0, 999, 999)]
    out, meta = mod.apply_augmentation(image, entries, "geometry_safe", cfg, seed)
    assert meta["geometry"]["fallback_code"] == "geometry_fallback:box_quality"
    assert meta["geometry"]["kind"] == "identity"
    assert entries[0]["xyxy_999"] == [0, 0, 999, 999]
    np.testing.assert_array_equal(np.asarray(out), original)


# ---------------------------------------------------------------------------
# Degradations
# ---------------------------------------------------------------------------


def test_degradations_preserve_size_and_boxes(data):
    """Every degradation keeps image size and never touches boxes.
    每种退化保持图像尺寸且绝不改动框。"""
    image = Image.fromarray(np.random.RandomState(2).randint(0, 255, (80, 100, 3), dtype=np.uint8))
    entries = [box(100, 100, 400, 400, "car")]
    before = [dict(e) for e in entries]
    cfg = AugmentationConfig(
        rotate90_prob=0.0, affine_rotation_prob=0.0, scale_prob=0.0,
        translate_prob=0.0, perspective_prob=0.0, degradation_probability=1.0,
        min_degradations=3, max_degradations=3,
    )
    group_seed = "0" * 64
    out, meta = mod.apply_augmentation(image, entries, "geometry_safe", cfg, group_seed)
    assert out.size == image.size
    assert meta["degradations"]["fallback_code"] is None
    assert [e["xyxy_999"] for e in entries] == [e["xyxy_999"] for e in before]


def test_brightness_dims_and_preserves_rgb_ratio(data):
    img = np.random.RandomState(3).rand(40, 40, 3).astype(np.float32) * 0.8 + 0.1
    out = mod.apply_brightness(img, 0.7)
    assert out.mean() < img.mean()
    assert np.all(out <= img + 1e-6)
    # RGB channel ratios preserved per pixel
    mask = img[..., 2] > 0.2
    ratio_in = img[mask][:, 0] / img[mask][:, 2]
    ratio_out = out[mask][:, 0] / out[mask][:, 2]
    np.testing.assert_allclose(ratio_out, ratio_in, rtol=1e-4)


def test_contrast_keeps_per_channel_mean(data):
    img = np.random.RandomState(4).rand(50, 50, 3).astype(np.float32)
    out = mod.apply_low_contrast(img, 0.6)
    np.testing.assert_allclose(
        out.mean(axis=(0, 1)), img.mean(axis=(0, 1)), atol=1e-6
    )
    cfg = AugmentationConfig()
    assert cfg.contrast_factor_max <= 1.0


def test_blur_kernels_normalized_and_size(data):
    for k in (3, 5, 7):
        for angle in (0.0, 45.0, 90.0, 135.0, 179.9):
            kernel = mod.motion_blur_kernel(k, angle)
            assert kernel.shape == (k, k)
            assert abs(float(kernel.sum()) - 1.0) < 1e-5
    img = np.random.RandomState(5).rand(60, 60, 3).astype(np.float32)
    blurred = mod.apply_defocus_blur(img, 5, 0.8)
    assert blurred.shape == img.shape
    motioned = mod.apply_motion_blur(img, mod.motion_blur_kernel(5, 30.0))
    assert motioned.shape == img.shape


def test_noise_reproducible_and_in_range(data):
    img = np.random.RandomState(6).rand(32, 32, 3).astype(np.float32)
    seed = "a" * 64
    a = mod.apply_sensor_noise(img, 0.02, seed)
    b = mod.apply_sensor_noise(img, 0.02, seed)
    np.testing.assert_array_equal(a, b)
    c = mod.apply_sensor_noise(img, 0.02, "b" * 64)
    assert not np.array_equal(a, c)
    assert np.all(a >= 0.0) and np.all(a <= 1.0)


def test_jpeg_in_memory_keeps_rgb_and_size(data):
    img = np.random.RandomState(7).rand(64, 64, 3).astype(np.float32)
    out = mod.apply_jpeg_compression(img, 60)
    assert out.shape == img.shape
    assert out.dtype == np.float32
    assert np.all(out >= 0.0) and np.all(out <= 1.0)
    # lossy roundtrip changed values
    assert not np.allclose(out, img, atol=1e-3)


def test_vignette_center_unchanged_monotonic(data):
    img = np.ones((80, 80, 3), dtype=np.float32)
    out = mod.apply_vignette(img, 0.7, 2.0)
    # the mask center (39.5, 39.5) is exactly 1; the central pixel is ~1
    center = float(out[40, 40, 0])
    assert center > 0.999
    # corners darker than the center
    assert out[0, 0, 0] < center
    # monotonic non-decreasing along the ray from corner (0,0) to center
    samples = []
    for t in np.linspace(0, 1, 21):
        x = int(round(t * 40))
        y = int(round(t * 40))
        samples.append(float(out[y, x, 0]))
    diffs = np.diff(samples)
    assert np.all(diffs >= -1e-6)
    assert out.shape == img.shape


def test_degradation_selection_bounds_and_blur_exclusive(data):
    """1..max degradations per sample; defocus and motion blur never both.
    每次 1..配置上限个退化；失焦与运动模糊不会同时出现。"""
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    cfg = AugmentationConfig(
        rotate90_prob=0.0, affine_rotation_prob=0.0, scale_prob=0.0,
        translate_prob=0.0, perspective_prob=0.0,
        degradation_probability=1.0, min_degradations=1, max_degradations=3,
    )
    for i in range(60):
        group_seed = hashlib.sha256(f"sel-{i}".encode()).hexdigest()
        _, meta = mod.apply_augmentation(image, [], "geometry_safe", cfg, group_seed)
        selected = meta["degradations"]["selected"]
        assert 1 <= len(selected) <= 3
        names = [s["name"] for s in selected]
        assert names.count("blur") <= 1
        blur = [s for s in selected if s["name"] == "blur"]
        if blur:
            assert blur[0]["kind"] in ("defocus", "motion")


def test_geometry_and_degradation_substreams_independent(data):
    """Toggling a geometry op never changes the degradation selection.
    调整几何操作不改变同一 Episode 的退化选择。"""
    image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    seed = "c" * 64
    cfg_a = AugmentationConfig(
        rotate90_prob=0.0, affine_rotation_prob=0.0, scale_prob=0.0,
        translate_prob=0.0, perspective_prob=0.0,
        degradation_probability=1.0, min_degradations=2, max_degradations=2,
    )
    cfg_b = AugmentationConfig(
        rotate90_prob=1.0, affine_rotation_prob=1.0, scale_prob=1.0,
        translate_prob=1.0, perspective_prob=1.0,
        degradation_probability=1.0, min_degradations=2, max_degradations=2,
    )
    _, meta_a = mod.apply_augmentation(image, [], "geometry_safe", cfg_a, seed)
    _, meta_b = mod.apply_augmentation(image, [], "geometry_safe", cfg_b, seed)
    assert meta_a["degradations"] == meta_b["degradations"]
    assert meta_b["geometry"]["ops"], "geometry should have applied in cfg_b"


def test_degradation_failure_falls_back(monkeypatch, data):
    """Degradation failure falls back to the geometry-stage output.
    退化失败回退到几何阶段输出。"""
    def boom(img, quality):
        raise RuntimeError("jpeg exploded")

    monkeypatch.setattr(mod, "apply_jpeg_compression", boom)
    image = Image.fromarray(np.random.RandomState(8).randint(0, 255, (40, 40, 3), dtype=np.uint8))
    entries = [box(200, 200, 400, 400)]
    cfg = AugmentationConfig(
        rotate90_prob=0.0, affine_rotation_prob=0.0, scale_prob=0.0,
        translate_prob=0.0, perspective_prob=0.0,
        degradation_probability=1.0, min_degradations=1, max_degradations=1,
        contrast_weight=0.0, brightness_weight=0.0, vignette_weight=0.0,
        defocus_weight=0.0, motion_blur_weight=0.0, sensor_noise_weight=0.0,
        jpeg_weight=1.0,
    )
    out, meta = mod.apply_augmentation(image, entries, "geometry_safe", cfg, "d" * 64)
    assert meta["degradations"]["fallback_code"] == "degradation_fallback:jpeg"
    np.testing.assert_array_equal(np.asarray(out), np.asarray(image))
    assert entries[0]["xyxy_999"] == [200, 200, 400, 400]


# ---------------------------------------------------------------------------
# Augmentation policy rules
# ---------------------------------------------------------------------------


def test_orientation_locked_no_geometry_but_degradation(data):
    ep = data["train_episodes"][-1]  # orientation_locked grounding
    assert ep["augmentation_policy"]["geometry"] == "orientation_locked"
    ep_file = data["ep_file"].parent / "locked.jsonl"
    ep_file.write_text(json.dumps(ep) + "\n", encoding="utf-8")
    proc = FakeQwenVLProcessor()
    ds = make_dataset(data, processor=proc, aug=FORCED_AUG, episode_file=ep_file)
    feature = ds[0]
    meta = feature["augmentation"]
    assert meta["geometry"]["kind"] == "identity"
    assert meta["geometry"]["ops"] == []
    assert meta["degradations"]["selected"], "degradations still allowed"
    original = np.asarray(Image.open(data["vroot"] / "a.png"))
    recorded = proc.seen_images[0]
    assert recorded.shape == original.shape
    assert not np.array_equal(recorded, original)  # degraded


def test_validation_is_identity(data):
    proc = FakeQwenVLProcessor()
    ds = make_dataset(data, processor=proc, aug=FORCED_AUG, split="validation",
                      episode_file=data["val_file"])
    feature = ds[0]
    meta = feature["augmentation"]
    assert meta["geometry"]["kind"] == "identity"
    assert meta["degradations"]["selected"] == []
    original = np.asarray(Image.open(data["vroot"] / "v.png"))
    np.testing.assert_array_equal(proc.seen_images[0], original)


def test_paired_views_share_augmentation_and_epoch_determinism(data):
    """boxed + self-attention views get the identical augmented image in the
    same epoch; different epochs derive different seeds.
    有框/无框配对视图在同 epoch 得到相同增强图像；不同 epoch 种子不同。"""
    proc = FakeQwenVLProcessor()
    ds = make_dataset(data, processor=proc, aug=FORCED_AUG)
    # indices: boxed = 1, self_attention = 2
    boxed = ds[1]
    sa = ds[2]
    assert boxed["episode_id"].endswith("/box_assisted")
    assert sa["episode_id"].endswith("/self_attention")
    assert boxed["augmentation"] == sa["augmentation"]
    np.testing.assert_array_equal(proc.seen_images[-2], proc.seen_images[-1])
    # both views carry distinct tensors (no shared mutable objects)
    assert boxed["input_ids"] is not sa["input_ids"]


def test_reproducible_across_dataset_instances(data):
    """Rebuilding the dataset (e.g. new workers) gives identical augmentations.
    重建 dataset（如新 worker）得到相同的增强。"""
    a = make_dataset(data, aug=FORCED_AUG)
    b = make_dataset(data, aug=FORCED_AUG)
    fa, fb = a[1], b[1]
    assert fa["augmentation"] == fb["augmentation"]
    np.testing.assert_array_equal(fa["input_ids"], fb["input_ids"])


def test_augmentation_config_validation(data):
    with pytest.raises(ValueError):
        AugmentationConfig(brightness_factor_max=1.2)
    with pytest.raises(ValueError):
        AugmentationConfig(contrast_factor_max=1.1)
    with pytest.raises(ValueError):
        AugmentationConfig(max_degradations=7)
    with pytest.raises(ValueError):
        AugmentationConfig(min_degradations=3, max_degradations=1)
    with pytest.raises(ValueError):
        AugmentationConfig(defocus_kernel_sizes=(3, 4))
    with pytest.raises(ValueError):
        AugmentationConfig(rotate90_prob=1.5)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_grounding_and_refer(data):
    ep = data["train_episodes"][0]
    messages, turns = mod.render_messages(ep)
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["type"] == "image"
    user = turns[0]["user_text"]
    assert user.startswith("Locate the region described below.")
    assert "Description: The red car is on the road." in user
    assert "Return the bounding boxes as JSON in 0..999 xyxy coordinates." in user
    assert turns[0]["assistant_text"] == '{"boxes":[{"xyxy":[100,100,300,300]}]}'
    # geochat refer: old box protocol never leaks; JSON rendering instead
    ref_ep = data["train_episodes"][4]
    _, ref_turns = mod.render_messages(ref_ep)
    assert "{<" not in ref_turns[0]["assistant_text"]
    assert ref_turns[0]["assistant_text"] == '{"boxes":[{"xyxy":[160,549,240,629]}]}'
    assert "<p>" not in ref_turns[0]["user_text"]
    assert "<image>" not in ref_turns[0]["user_text"]
    assert "bottom left" in ref_turns[1]["user_text"]


def test_render_boxed_vs_unboxed_prompt_diff(data):
    boxed = data["train_episodes"][1]
    sa = data["train_episodes"][2]
    _, b_turns = mod.render_messages(boxed)
    _, s_turns = mod.render_messages(sa)
    b_user = b_turns[0]["user_text"]
    s_user = s_turns[0]["user_text"]
    assert "Question: How many small vehicles are visible?" in b_user
    assert "Available annotated regions:" in b_user
    assert "- vehicle: [380, 200, 450, 260] — The small vehicle." in b_user
    assert s_user == "Question: How many small vehicles are visible?"
    assert "Available annotated regions:" not in s_user
    assert "380" not in s_user and "450" not in s_user
    assert b_turns[0]["assistant_text"] == s_turns[0]["assistant_text"] == "1"


def test_render_identify_and_conversation(data):
    ident = data["train_episodes"][5]
    _, turns = mod.render_messages(ident)
    assert "[identify]" in turns[0]["user_text"]
    assert "Available annotated regions:" in turns[0]["user_text"]
    assert "- region: [350, 320, 370, 360]" in turns[0]["user_text"]
    assert turns[0]["assistant_text"] == "1 dump-truck at the center"  # <p> stripped
    conv = data["train_episodes"][6]
    _, c_turns = mod.render_messages(conv)
    assert len(c_turns) == 3
    assert "<image>" not in c_turns[0]["user_text"]
    assert c_turns[0]["user_text"].startswith("How many buildings")
    assert c_turns[1]["user_text"] == "Can you count the number of cars?"
    assert c_turns[2]["assistant_text"] == "There are 3 teal cars in the image."


# ---------------------------------------------------------------------------
# Labels / masks / alignment
# ---------------------------------------------------------------------------


def _label_positions(feature):
    labels = feature["labels"].tolist()
    return {i for i, v in enumerate(labels) if v != IGNORE}


def test_labels_supervise_assistant_only(data):
    proc = FakeQwenVLProcessor()
    ds = make_dataset(data, processor=proc, aug=AugmentationConfig(enabled=False))
    feature = ds[1]  # boxed VQA, single turn
    ids = feature["input_ids"].tolist()
    labels = feature["labels"].tolist()
    # assistant content tokens are supervised with their own ids
    supervised = _label_positions(feature)
    assert supervised, "no supervised tokens"
    for i in supervised:
        assert labels[i] == ids[i], f"label {i} must equal its input id"
    # user tokens (including all image/vision tokens) are ignored
    mm = feature["mm_token_type_ids"].tolist()
    for i, (lab, m, tok) in enumerate(zip(labels, mm, ids)):
        if m == 1:  # image pad tokens
            assert lab == IGNORE
        elif tok == 1003 or tok == 1004:  # vision start/end
            assert lab == IGNORE
    # the answer text tokens are all supervised
    answer_ids = proc.tokenize("1")
    assert any(
        ids[i:i + len(answer_ids)] == answer_ids and i in supervised
        for i in range(len(ids) - len(answer_ids) + 1)
    )


def test_multi_turn_all_assistant_supervised(data):
    ds = make_dataset(data, aug=AugmentationConfig(enabled=False))
    feature = ds[6]  # 3-turn conversation
    supervised = _label_positions(feature)
    ids = feature["input_ids"].tolist()
    labels = feature["labels"].tolist()
    assert len(supervised) > 0
    # every assistant text appears fully supervised
    for text in ("There are 4 buildings present in the image.",
                 "There are 45 cars.",
                 "There are 3 teal cars in the image."):
        toks = FakeQwenVLProcessor().tokenize(text)
        found = False
        for i in range(len(ids) - len(toks) + 1):
            if ids[i:i + len(toks)] == toks:
                assert all(labels[i + j] == toks[j] for j in range(len(toks)))
                found = True
        assert found, f"assistant text not found/supervised: {text}"


def test_fallback_turn_boundary_mask_matches(data):
    """The turn-boundary fallback mask marks the same assistant spans as the
    template mask (both fixed policies agree on content + end token).
    turn-boundary 回退 mask 与 template mask 标记相同的 assistant 跨度。"""
    proc = FakeQwenVLProcessor(support_template_mask=False)  # force fallback
    ds = make_dataset(data, processor=proc, aug=AugmentationConfig(enabled=False))
    feature = ds[6]  # multi-turn
    supervised = _label_positions(feature)
    assert supervised
    for i in supervised:
        assert feature["labels"].tolist()[i] == feature["input_ids"].tolist()[i]


def test_labels_aligned_across_image_sizes(data):
    """Delta alignment holds for different image token counts.
    不同图像 token 数下 delta 对齐依然正确。"""
    for idx in (1, 3, 5):  # b.png (12 img tokens), c.png (36), i.jpg (12)
        proc = FakeQwenVLProcessor()
        ds = make_dataset(data, processor=proc, aug=AugmentationConfig(enabled=False))
        feature = ds[idx]
        supervised = _label_positions(feature)
        ids = feature["input_ids"].tolist()
        labels = feature["labels"].tolist()
        assert supervised
        for i in supervised:
            assert labels[i] == ids[i]


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def test_truncation_keeps_turn_pairs_and_supervision(data):
    proc = FakeQwenVLProcessor()
    full_ds = make_dataset(data, processor=proc, aug=AugmentationConfig(enabled=False),
                           max_seq_length=100000)
    full = full_ds[6]
    n_full = int(full["input_ids"].shape[0])
    # max just below the full length forces truncation to fewer turn pairs
    ds = make_dataset(data, processor=FakeQwenVLProcessor(),
                      aug=AugmentationConfig(enabled=False), max_seq_length=n_full - 1)
    truncated = ds[6]
    n_trunc = int(truncated["input_ids"].shape[0])
    assert n_trunc < n_full
    assert n_trunc <= n_full - 1
    assert _label_positions(truncated), "truncated feature must keep supervision"
    # truncation keeps as many complete turn pairs from the start as fit;
    # here the third pair is dropped, so the first two pairs remain
    first_two = mod.render_messages(data["train_episodes"][6])[0][:4]
    first_two_ids = mod.encode_episode(
        FakeQwenVLProcessor(),
        Image.open(data["groot"] / "cv.jpg"),
        first_two,
        100000, "probe",
    )["input_ids"]
    assert torch.equal(truncated["input_ids"], first_two_ids)


def test_single_turn_too_long_raises(data):
    proc = FakeQwenVLProcessor()
    ds = make_dataset(data, processor=proc, aug=AugmentationConfig(enabled=False),
                      max_seq_length=10)
    with pytest.raises(EpisodeTooLongError):
        ds[1]
    # preflight reports it instead of crashing
    counts = ds.preflight(limit=len(ds))
    assert counts["episode_too_long"] >= 1
    assert counts["too_long"] >= 0


def test_preflight_counts(data):
    ds = make_dataset(data, aug=AugmentationConfig(enabled=False), max_seq_length=2048)
    counts = ds.preflight(limit=3)
    assert counts["checked"] == 3
    assert counts["image_errors"] == 0
    assert counts["other_errors"] == 0
    assert counts["too_long"] == 0  # all short episodes


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


def test_collator_mixed_lengths_and_visual_tokens(data):
    ds = make_dataset(data, aug=AugmentationConfig(enabled=False))
    features = [ds[1], ds[3], ds[6]]  # different seq lengths and image sizes
    collator = Phase2DataCollator()
    batch, meta = collator(features)
    max_len = max(int(f["input_ids"].shape[0]) for f in features)
    assert batch["input_ids"].shape == (3, max_len)
    assert batch["attention_mask"].shape == (3, max_len)
    assert batch["mm_token_type_ids"].shape == (3, max_len)
    assert batch["labels"].shape == (3, max_len)
    # padded labels are IGNORE
    for f, row in zip(features, batch["labels"]):
        n = int(f["labels"].shape[0])
        assert torch.equal(row[:n], f["labels"])
        assert torch.all(row[n:] == IGNORE)
    # vision tensors concatenated: per-image (grid_h*grid_w, 1176) and (1, 3)
    g_total = sum(int(f["pixel_values"].shape[0]) for f in features)
    assert batch["pixel_values"].shape == (g_total, 1176)
    assert batch["image_grid_thw"].shape == (3, 3)
    # metadata stripped from the batch, returned separately
    assert "episode_id" not in batch
    assert "augmentation" not in batch
    assert [m["episode_id"] for m in meta] == [f["episode_id"] for f in features]
    assert meta[0]["augmentation"] is not None


def test_collator_rejects_missing_keys(data):
    collator = Phase2DataCollator()
    bad = {"input_ids": torch.ones(4, dtype=torch.long)}
    with pytest.raises(CollatorError):
        collator([bad])
    with pytest.raises(CollatorError):
        collator([])


# ---------------------------------------------------------------------------
# Live contract guard (runs where transformers is installed, e.g. M3 env)
# ---------------------------------------------------------------------------


def test_live_image_processor_contract():
    """When transformers is available, verify the pinned image processor
    contract (pixel_values (G, C*patch*patch), image_grid_thw (1, 3)).
    transformers 可用时验证钉死版本的 image processor 契约。"""
    try:
        import transformers  # noqa: F401
    except ImportError:
        pytest.skip("transformers not installed in this environment")
    from transformers import Qwen2VLImageProcessor
    proc = Qwen2VLImageProcessor()
    for (w, h) in [(100, 80), (200, 150)]:
        img = Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))
        out = proc(images=[img], return_tensors="pt")
        assert set(out.keys()) == {"pixel_values", "image_grid_thw"}
        grid = out["image_grid_thw"][0].tolist()
        g = grid[1] * grid[2]
        assert out["pixel_values"].shape[0] == g
        assert out["pixel_values"].shape[1] == 1176
