"""Tests for scripts/prepare_qwen3vl_phase2_sft.py.

Unit tests use small temporary fixtures only — never the full datasets.
Covers: VRSBench block/JSONL parsing, grounding expansion, box-assisted
VQA, stratified 40% self-attention augmentation (extra copies, train only,
reproducible), naturally-unboxed VQA, GeoChat refer/identify/conversation
parsing and 0..100 -> 0..999 conversion, rejection codes with closure
counts, JSON-safety / no absolute paths, byte-stable outputs and input
checksum preservation.

测试仅使用小型临时 fixture；覆盖 VRSBench 块/JSONL 解析、grounding 展开、
有框 VQA、分层 40% self-attention 增广（额外副本、仅 train、可复现）、
天然无框 VQA、GeoChat refer/identify/conversation 解析与坐标转换、
拒绝码与闭合计数、JSON 安全与无绝对路径、字节级稳定输出与输入校验和。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "prepare_qwen3vl_phase2_sft.py"

spec = importlib.util.spec_from_file_location("prepare_qwen3vl_phase2_sft", SCRIPT_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

DEFAULT_SEED = mod.DEFAULT_SEED
ALLOWED_TASK_KINDS = mod.ALLOWED_TASK_KINDS if hasattr(mod, "ALLOWED_TASK_KINDS") else frozenset(
    {
        "vrsbench_grounding",
        "vqa_box_assisted",
        "vqa_self_attention",
        "vqa_naturally_unboxed",
        "geochat_refer",
        "geochat_identify",
        "geochat_conversation",
    }
)
REJECTION_CODES = mod.REJECTION_CODES


# ---------------------------------------------------------------------------
# Fixtures: small VRSBench phase2 files (pretty blocks and single-line JSONL)
# ---------------------------------------------------------------------------


def _obj(obj_id, cls, box_999=None, box_valid=True, sentence=None):
    obj = {
        "obj_id": obj_id,
        "obj_cls": cls,
        "referring_sentence": sentence or f"A {cls}.",
        "obj_coord": [0.1, 0.1, 0.2, 0.2],
    }
    if box_valid:
        obj["box_999"] = box_999
        obj["box_valid"] = True
    else:
        obj["box_valid"] = False
    return obj


def _vrsbench_records():
    """5 train + 2 validation images with known ground truth.
    5 张 train + 2 张 validation 图片，ground truth 已知。"""
    train = [
        {
            "id": "vrsbench/train/P1419_0005.png",
            "dataset": "VRSBench",
            "split": "train",
            "image_id": "P1419_0005.png",
            "image": "Images_train/P1419_0005.png",
            "objects": [
                _obj(0, "vehicle", [380, 200, 450, 260],
                     sentence="The small vehicle with a white color is positioned near the top-middle of the image."),
                _obj(1, "car", [10, 10, 50, 50], sentence="The red car is on the road."),
                _obj(2, "truck", box_valid=False, sentence="An invalid truck box."),
            ],
            "qa_pairs": [
                {"ques_id": 1, "question": "Where is the vehicle located?",
                 "type": "object position", "task": "object_position", "answer": "top-middle"},
                {"ques_id": 2, "question": "How many small vehicles are visible?",
                 "type": "object quantity", "task": "counting", "answer": "1"},
                {"ques_id": 3, "question": "What is the main structure in the center of the image?",
                 "type": "scene type", "task": "scene_type", "answer": "expressway-toll-station"},
            ],
            "source": {"annotation_file": "Annotations_train/P1419_0005.json"},
        },
        {
            "id": "vrsbench/train/P0001_0001.png",
            "dataset": "VRSBench",
            "split": "train",
            "image_id": "P0001_0001.png",
            "image": "Images_train/P0001_0001.png",
            "objects": [_obj(0, "car", box_valid=False)],
            "qa_pairs": [
                {"ques_id": 4, "question": "How many cars?", "type": "object quantity",
                 "task": "counting", "answer": "3"},
            ],
        },
        {
            "id": "vrsbench/train/P0002_0002.png",
            "dataset": "VRSBench",
            "split": "train",
            "image_id": "P0002_0002.png",
            "image": "Images_train/P0002_0002.png",
            "objects": [],
            "qa_pairs": [
                {"ques_id": 5, "question": "What color is the bus?", "type": "object color",
                 "task": "object_color", "answer": "yellow"},
            ],
        },
        {
            "id": "vrsbench/train/P0003_0003.png",
            "dataset": "VRSBench",
            "split": "train",
            "image_id": "P0003_0003.png",
            "image": "Images_train/P0003_0003.png",
            "objects": [_obj(0, "bus", [100, 100, 200, 200], sentence="The blue bus.")],
            "qa_pairs": [
                {"ques_id": 6, "question": "How many buses?", "type": "object quantity",
                 "task": "counting", "answer": "2"},
                {"ques_id": 7, "question": "How many buses are there?",
                 "type": "object quantity", "task": "counting", "answer": "2"},
                {"ques_id": 8, "question": "How large is the bus?", "type": "object size",
                 "task": "object_size", "answer": "medium"},
            ],
        },
        {
            "id": "vrsbench/train/P0004_0004.png",
            "dataset": "VRSBench",
            "split": "train",
            "image_id": "P0004_0004.png",
            "image": "Images_train/P0004_0004.png",
            "objects": [_obj(0, "airplane", [300, 300, 350, 360], sentence="A white airplane.")],
            "qa_pairs": [
                {"ques_id": 9, "question": "Is there a plane?", "type": "object existence",
                 "task": "object_existence", "answer": "Yes"},
                # duplicate ques_id inside the same image: ids must be
                # disambiguated (qa/9 vs qa/9.2)
                # 同一图内重复 ques_id：id 必须消歧（qa/9 与 qa/9.2）
                {"ques_id": 9, "question": "What color is the plane?", "type": "object color",
                 "task": "object_color", "answer": "white"},
            ],
        },
    ]
    val = [
        {
            "id": "vrsbench/val/P9001_0001.png",
            "dataset": "VRSBench",
            "split": "val",
            "image_id": "P9001_0001.png",
            "image": "Images_train/P9001_0001.png",
            "objects": [_obj(0, "boat", [20, 20, 60, 70], sentence="A small boat.")],
            "qa_pairs": [
                {"ques_id": 10, "question": "How many boats?", "type": "object quantity",
                 "task": "counting", "answer": "1"},
            ],
        },
        {
            "id": "vrsbench/val/P9002_0002.png",
            "dataset": "VRSBench",
            "split": "val",
            "image_id": "P9002_0002.png",
            "image": "Images_train/P9002_0002.png",
            "objects": [],
            "qa_pairs": [
                {"ques_id": 11, "question": "What scene is this?", "type": "scene type",
                 "task": "scene_type", "answer": "harbor"},
            ],
        },
    ]
    return train, val


def write_vrsbench_dir(base: Path, train_fmt: str = "pretty", val_fmt: str = "pretty") -> Path:
    """Write fixture VRSBench files (pretty blocks or single-line JSONL).
    写出 fixture VRSBench 文件（缩进块或单行 JSONL）。"""
    vdir = base / "vrsbench"
    vdir.mkdir(parents=True, exist_ok=True)
    train, val = _vrsbench_records()
    if train_fmt == "pretty":
        (vdir / "VRSBench_train.jsonl").write_text(
            "\n\n".join(json.dumps(r, indent=2, ensure_ascii=False) for r in train) + "\n",
            encoding="utf-8",
        )
    else:
        (vdir / "VRSBench_train.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n",
            encoding="utf-8",
        )
    if val_fmt == "pretty":
        (vdir / "VRSBench_val.jsonl").write_text(
            "\n\n".join(json.dumps(r, indent=2, ensure_ascii=False) for r in val) + "\n",
            encoding="utf-8",
        )
    else:
        (vdir / "VRSBench_val.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in val) + "\n",
            encoding="utf-8",
        )
    return vdir


# ---------------------------------------------------------------------------
# Fixture: small GeoChat JSON array
# ---------------------------------------------------------------------------

_GEOCHAT_RECORDS = [
    # 1: refer, single turn -> 1 target box
    {"id": "r1.jpg", "image": "imgs/r1.jpg", "conversations": [
        {"from": "human", "value": "<image>\n[refer] where is <p>silver boeing737 airplane</p> ?"},
        {"from": "gpt", "value": "{<16><55><24><63>|<11>}"},
    ]},
    # 2: refer with two target boxes
    {"id": "r2.jpg", "image": "imgs/r2.jpg", "conversations": [
        {"from": "human", "value": "<image>\n[refer] give me the location of <p>2 a220 airplanes</p>"},
        {"from": "gpt", "value": "{<8><82><16><90>|<69>}{<6><44><14><48>|<32>}"},
    ]},
    # 3: multi-turn refer (3 pairs)
    {"id": "r3.jpg", "image": "imgs/r3.jpg", "conversations": [
        {"from": "human", "value": "<image>\n[refer] where is <p>silver boeing737 airplane</p> ?"},
        {"from": "gpt", "value": "{<7><39><15><47>|<56>}"},
        {"from": "human", "value": "\n[refer] give me the location of <p>2 a220 airplanes at the bottom left</p>"},
        {"from": "gpt", "value": "{<8><82><16><90>|<69>}"},
        {"from": "human", "value": "\n[refer] where can I locate the <p>silver airplane</p> ?"},
        {"from": "gpt", "value": "{<6><44><14><48>|<32>}"},
    ]},
    # 4: identify -> 1 input box + text answer
    {"id": "i1.jpg", "image": "imgs/i1.jpg", "conversations": [
        {"from": "human", "value": "<image>\n[identify] the object in {<35><32><37><36>|<67>} is"},
        {"from": "gpt", "value": "<p>1 dump-truck at the center</p>"},
    ]},
    # 5: plain multi-turn conversation
    {"id": "c1.jpg", "image": "imgs/c1.jpg", "conversations": [
        {"from": "human", "value": "<image>\nHow many buildings are present in the image?"},
        {"from": "gpt", "value": "There are 4 buildings present in the image."},
        {"from": "human", "value": "Can you count the number of cars?"},
        {"from": "gpt", "value": "There are 45 cars."},
    ]},
    # 6: plain answer with inline boxes (grounding-style description)
    {"id": "c2.jpg", "image": "imgs/c2.jpg", "conversations": [
        {"from": "human", "value": "<image>\n[grounding] give a thorough description of what you see in this image"},
        {"from": "gpt", "value": "In the satellite image, there are <p>two gray ships</p> {<50><50><60><60>|<52>} at the bottom right."},
    ]},
    # 7: out-of-range coordinate -> rejected
    {"id": "bad1.jpg", "image": "imgs/bad1.jpg", "conversations": [
        {"from": "human", "value": "<image>\n[refer] where is <p>x</p>?"},
        {"from": "gpt", "value": "{<77><981><83><100>|<29>}"},
    ]},
    # 8: degenerate box (x1 == x2) -> rejected
    {"id": "bad2.jpg", "image": "imgs/bad2.jpg", "conversations": [
        {"from": "human", "value": "<image>\n[refer] where is <p>x</p>?"},
        {"from": "gpt", "value": "{<35><32><35><36>|<67>}"},
    ]},
    # 9: unparseable box (only three coordinates) -> rejected
    {"id": "bad3.jpg", "image": "imgs/bad3.jpg", "conversations": [
        {"from": "human", "value": "<image>\n[identify] what is {<1><2><3>|<4>}?"},
        {"from": "gpt", "value": "nothing"},
    ]},
    # 10: missing image field -> rejected
    {"id": "bad4.jpg", "conversations": [
        {"from": "human", "value": "<image>\nhello"},
        {"from": "gpt", "value": "hi"},
    ]},
    # 11: invalid role order (gpt first) -> rejected
    {"id": "bad5.jpg", "image": "imgs/bad5.jpg", "conversations": [
        {"from": "gpt", "value": "hi"},
        {"from": "human", "value": "<image>\nhello"},
    ]},
    # 12: missing turn text -> rejected
    {"id": "bad6.jpg", "image": "imgs/bad6.jpg", "conversations": [
        {"from": "human", "value": "<image>"},
        {"from": "gpt", "value": "   "},
    ]},
    # 13: image token count mismatch -> rejected
    {"id": "bad7.jpg", "image": "imgs/bad7.jpg", "conversations": [
        {"from": "human", "value": "<image>\n<image> how many?"},
        {"from": "gpt", "value": "2"},
    ]},
    # 14: mixed refer + identify -> rejected
    {"id": "bad8.jpg", "image": "imgs/bad8.jpg", "conversations": [
        {"from": "human", "value": "<image>\n[refer] where is <p>x</p>? [identify] what is this?"},
        {"from": "gpt", "value": "{<1><2><3><4>|<5>}"},
    ]},
    # 15: conversations not a list -> rejected
    {"id": "bad9.jpg", "image": "imgs/bad9.jpg", "conversations": "not-a-list"},
    # 16: non-integer numeric coordinate -> rejected
    {"id": "bad10.jpg", "image": "imgs/bad10.jpg", "conversations": [
        {"from": "human", "value": "<image>\n[refer] where is <p>x</p>?"},
        {"from": "gpt", "value": "{<1.5><2><3><4>|<5>}"},
    ]},
]


def write_geochat_file(base: Path) -> Path:
    """Write the fixture GeoChat JSON array. / 写出 fixture GeoChat JSON 数组。"""
    gdir = base / "geochat"
    gdir.mkdir(parents=True, exist_ok=True)
    path = gdir / "GeoChat_Instruct.json"
    path.write_text(
        json.dumps(_GEOCHAT_RECORDS, ensure_ascii=False), encoding="utf-8"
    )
    return path


def run_pipeline(vrsbench_dir, geochat_file, out_dir, seed=DEFAULT_SEED, ratio=0.4):
    """Run the pipeline entry point and return its exit code.
    运行流水线入口并返回退出码。"""
    return mod.main([
        "--vrsbench-dir", str(vrsbench_dir),
        "--geochat-file", str(geochat_file),
        "--output-dir", str(out_dir),
        "--seed", seed,
        "--self-attention-ratio", str(ratio),
    ])


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL output file. / 读取 JSONL 输出文件。"""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def data_dirs(tmp_path: Path):
    """Write fixtures and run the pipeline once; return all artifacts.
    写出 fixture 并运行一次流水线；返回全部产物。"""
    vrsbench_dir = write_vrsbench_dir(tmp_path)
    geochat_file = write_geochat_file(tmp_path)
    out_dir = tmp_path / "out"
    rc = run_pipeline(vrsbench_dir, geochat_file, out_dir)
    assert rc == 0
    return {
        "vrsbench_dir": vrsbench_dir,
        "geochat_file": geochat_file,
        "out_dir": out_dir,
        "train": load_jsonl(out_dir / "train.jsonl"),
        "validation": load_jsonl(out_dir / "validation.jsonl"),
        "rejected": load_jsonl(out_dir / "rejected.jsonl"),
        "manifest": json.loads((out_dir / "manifest.json").read_text(encoding="utf-8")),
    }


def by_kind(episodes: list[dict], kind: str) -> list[dict]:
    return [e for e in episodes if e["task_kind"] == kind]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_vrsbench_pretty_and_jsonl_formats_equivalent(tmp_path: Path):
    """Pretty-printed blocks and single-line JSONL must produce identical
    episodes. 缩进块与单行 JSONL 必须产出完全相同的 episodes。"""
    gc = write_geochat_file(tmp_path)
    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    rc = run_pipeline(write_vrsbench_dir(tmp_path / "a", "pretty", "pretty"), gc, out_a)
    assert rc == 0
    rc = run_pipeline(write_vrsbench_dir(tmp_path / "b", "jsonl", "jsonl"), gc, out_b)
    assert rc == 0
    for name in ("train.jsonl", "validation.jsonl", "rejected.jsonl"):
        assert sha256_file(out_a / name) == sha256_file(out_b / name)
    # manifest intentionally differs: it records the input files' actual
    # bytes, and pretty vs single-line sources have different bytes.
    # manifest 记录输入文件实际字节校验和，两种输入格式字节不同属预期。


def test_vrsbench_grounding_expansion(data_dirs):
    """One grounding episode per valid object with target boxes.
    每个合法 object 生成一条 grounding episode，含 target 框。"""
    train = data_dirs["train"]
    grounding = by_kind(train, "vrsbench_grounding")
    assert len(grounding) == 4  # A:2 valid, D:1, E:1 (B/C none; A truck invalid)
    by_obj = {e["provenance"]["object_id"]: e for e in grounding if "/P1419_0005.png/obj/" in e["episode_id"]}
    veh = by_obj[0]
    assert veh["turns"][0]["user_text"].startswith("The small vehicle")
    assert veh["turns"][0]["assistant_text"] == ""
    assert veh["turns"][0]["input_boxes"] == []
    assert veh["turns"][0]["target_boxes"] == [{
        "xyxy_999": [380, 200, 450, 260],
        "label": "vehicle",
        "description": "The small vehicle with a white color is positioned near the top-middle of the image.",
        "source_object_id": 0,
    }]
    assert veh["provenance"]["view"] == "grounding"
    assert veh["episode_id"].endswith("/obj/0")
    assert veh["augmentation_policy"]["geometry"] == "orientation_locked"  # top-middle


def test_vqa_box_assisted_all_qa_with_valid_boxes(data_dirs):
    """Every QA of an image with valid boxes gets a box_assisted view whose
    input_boxes are all valid annotation boxes of that image.
    有合法框的图片，其每个 QA 都生成 vqa_box_assisted，input_boxes 为该图
    全部合法框。"""
    train = data_dirs["train"]
    boxed = by_kind(train, "vqa_box_assisted")
    assert len(boxed) == 8  # A:3 + D:3 + E:2 (duplicate ques_id in E)
    a_ep = [e for e in boxed if "/P1419_0005.png/qa/" in e["episode_id"]]
    assert len(a_ep) == 3
    for ep in a_ep:
        assert ep["turns"][0]["input_boxes"] == [
            {"xyxy_999": [380, 200, 450, 260], "label": "vehicle",
             "description": "The small vehicle with a white color is positioned near the top-middle of the image.",
             "source_object_id": 0},
            {"xyxy_999": [10, 10, 50, 50], "label": "car",
             "description": "The red car is on the road.", "source_object_id": 1},
        ]
        assert ep["turns"][0]["target_boxes"] == []
        assert ep["parent_episode_id"] == ep["episode_id"].rsplit("/", 1)[0]
    # question/answer preserved verbatim
    qa1 = [e for e in boxed if e["episode_id"].endswith("/qa/1/box_assisted")][0]
    assert qa1["turns"][0]["user_text"] == "Where is the vehicle located?"
    assert qa1["turns"][0]["assistant_text"] == "top-middle"
    # duplicate ques_id within one image is disambiguated by occurrence
    e_ids = {
        e["episode_id"]
        for e in boxed
        if "/P0004_0004.png/qa/" in e["episode_id"]
    }
    assert e_ids == {
        "vrsbench/train/P0004_0004.png/qa/9/box_assisted",
        "vrsbench/train/P0004_0004.png/qa/9.2/box_assisted",
    }


def test_self_attention_is_extra_copy_not_replacement(data_dirs):
    """40% selection adds copies on top of the boxed views.
    40% 选择是额外副本，不替代有框版。"""
    train = data_dirs["train"]
    boxed = by_kind(train, "vqa_box_assisted")
    sa = by_kind(train, "vqa_self_attention")
    assert len(sa) == 1  # counting stratum: 3 parents -> round(1.2) = 1
    assert len(boxed) == 8  # all boxed views still present
    sa_ep = sa[0]
    assert sa_ep["episode_id"].endswith("/self_attention")
    parent = sa_ep["parent_episode_id"]
    parents = {e["parent_episode_id"] for e in boxed}
    assert parent in parents
    twin = [e for e in boxed if e["parent_episode_id"] == parent][0]
    assert twin["turns"][0]["user_text"] == sa_ep["turns"][0]["user_text"]
    assert twin["turns"][0]["assistant_text"] == sa_ep["turns"][0]["assistant_text"]
    assert sa_ep["turns"][0]["input_boxes"] == []
    assert sa_ep["turns"][0]["target_boxes"] == []
    # coordinates never in the prompt-facing text; only a summary in audit
    assert "audit" in sa_ep["provenance"]
    summary = sa_ep["provenance"]["audit"]["box_summary"]
    assert summary["count"] >= 1
    assert isinstance(summary["labels"], list)


def test_self_attention_stratified_train_only_reproducible(data_dirs, tmp_path):
    """40% is stratified by source_task, train only, reproducible with the
    same seed, and selection follows sha256(seed + parent) ordering.
    40% 按 source_task 分层、仅 train、同 seed 可复现，并按稳定 hash 排序。"""
    manifest = data_dirs["manifest"]
    strata = manifest["self_attention"]["strata"]
    assert strata["counting"] == {"parents": 3, "selected": 1, "actual_ratio": 1 / 3}
    for task in ("object_position", "scene_type", "object_size", "object_existence", "object_color"):
        assert strata[task] == {"parents": 1, "selected": 0, "actual_ratio": 0.0}
    # naturally-unboxed QA (img B/C) never enters any stratum
    assert sum(s["parents"] for s in strata.values()) == 8  # == box_assisted count
    # validation split gets no self-attention copies
    validation = data_dirs["validation"]
    assert by_kind(validation, "vqa_self_attention") == []
    # same seed -> byte-identical outputs
    out2 = tmp_path / "out2"
    rc = run_pipeline(data_dirs["vrsbench_dir"], data_dirs["geochat_file"], out2)
    assert rc == 0
    for name in ("train.jsonl", "validation.jsonl", "rejected.jsonl", "manifest.json"):
        assert sha256_file(data_dirs["out_dir"] / name) == sha256_file(out2 / name)
    # selection is a prefix of the digest-sorted parent order
    sa = by_kind(data_dirs["train"], "vqa_self_attention")
    assert len(sa) == 1
    chosen = sa[0]["parent_episode_id"]
    counting_parents = {
        "vrsbench/train/P1419_0005.png/qa/2",
        "vrsbench/train/P0003_0003.png/qa/6",
        "vrsbench/train/P0003_0003.png/qa/7",
    }
    digests = sorted(
        (hashlib.sha256((DEFAULT_SEED + p).encode("utf-8")).hexdigest(), p)
        for p in counting_parents
    )
    assert chosen == digests[0][1]
    # a different seed selects a different parent (verified values below)
    out3 = tmp_path / "out3"
    rc = run_pipeline(data_dirs["vrsbench_dir"], data_dirs["geochat_file"], out3,
                      seed="other-seed")
    assert rc == 0
    sa3 = by_kind(load_jsonl(out3 / "train.jsonl"), "vqa_self_attention")
    assert [e["parent_episode_id"] for e in sa3] != [chosen]


def test_naturally_unboxed_not_in_denominator(data_dirs):
    """Images without valid boxes produce vqa_naturally_unboxed only.
    无合法框的图片只生成 vqa_naturally_unboxed。"""
    train = data_dirs["train"]
    unboxed = by_kind(train, "vqa_naturally_unboxed")
    assert len(unboxed) == 2  # img B (qa4) + img C (qa5)
    for ep in unboxed:
        assert ep["turns"][0]["input_boxes"] == []
        assert ep["episode_id"].endswith("/naturally_unboxed")
    # their parents are not among the box-assisted parents
    parents = {e["parent_episode_id"] for e in by_kind(train, "vqa_box_assisted")}
    assert not any(e["parent_episode_id"] in parents for e in unboxed)


def test_vrsbench_invalid_boxes_audited_only(data_dirs):
    """Invalid boxes are counted, never clamped or turned into targets.
    非法框只计数审计，不修复、不生成训练目标。"""
    manifest = data_dirs["manifest"]
    assert manifest["vrsbench_invalid_objects"] == {"train": 2, "validation": 0}
    grounding = by_kind(data_dirs["train"], "vrsbench_grounding")
    assert all(b["xyxy_999"][0] < b["xyxy_999"][2] for e in grounding for b in e["turns"][0]["target_boxes"])


def test_geochat_refer_parse_and_conversion(data_dirs):
    """[refer]: multi-box parse and 0..100 -> 0..999 conversion.
    [refer]：多框解析与 0..100 -> 0..999 转换。"""
    train = data_dirs["train"]
    refer = by_kind(train, "geochat_refer")
    assert len(refer) == 3
    single = [e for e in refer if e["episode_id"].endswith("/r1.jpg")][0]
    assert single["turns"][0]["input_boxes"] == []
    assert single["turns"][0]["target_boxes"] == [{
        "xyxy_999": [160, 549, 240, 629],  # round(16*999/100) etc.
        "label": "",
        "description": "",
        "source_class_id": "11",
    }]
    # source protocol stays in provenance only
    assert all("source_class_id" in b for e in refer for t in e["turns"] for b in t["target_boxes"])
    multi = [e for e in refer if e["episode_id"].endswith("/r2.jpg")][0]
    assert [b["xyxy_999"] for b in multi["turns"][0]["target_boxes"]] == [
        [80, 819, 160, 899],
        [60, 440, 140, 480],
    ]
    assert multi["turns"][0]["assistant_text"].startswith("{<")
    assert multi["turns"][0]["user_text"].find("{<") == -1


def test_geochat_identify_parse(data_dirs):
    """[identify]: user boxes become input_boxes, description stays text.
    [identify]：user 框进 input_boxes，描述保持文本。"""
    identify = by_kind(data_dirs["train"], "geochat_identify")
    assert len(identify) == 1
    ep = identify[0]
    assert ep["turns"][0]["input_boxes"] == [{
        "xyxy_999": [350, 320, 370, 360],  # round(35*999/100)=350, 37->370
        "label": "",
        "description": "",
        "source_class_id": "67",
    }]
    assert ep["turns"][0]["target_boxes"] == []
    # box tokens removed from user text, answer text preserved
    assert "{<" not in ep["turns"][0]["user_text"]
    assert ep["turns"][0]["user_text"].startswith("<image>\n[identify] the object in is")
    assert ep["turns"][0]["assistant_text"] == "<p>1 dump-truck at the center</p>"


def test_geochat_multi_turn_preserved(data_dirs):
    """Plain multi-turn conversations keep role and text order.
    普通多轮对话保持角色与文本顺序。"""
    conv = by_kind(data_dirs["train"], "geochat_conversation")
    assert len(conv) == 2
    plain = [e for e in conv if e["episode_id"].endswith("/c1.jpg")][0]
    assert len(plain["turns"]) == 2
    assert plain["turns"][0]["user_text"] == "<image>\nHow many buildings are present in the image?"
    assert plain["turns"][0]["assistant_text"] == "There are 4 buildings present in the image."
    assert plain["turns"][1]["user_text"] == "Can you count the number of cars?"
    assert plain["turns"][1]["assistant_text"] == "There are 45 cars."
    assert plain["turns"][0]["input_boxes"] == []
    assert plain["provenance"]["multi_turn"] is True
    # multi-turn refer keeps all 3 pairs
    refer = [e for e in by_kind(data_dirs["train"], "geochat_refer")
             if e["episode_id"].endswith("/r3.jpg")][0]
    assert len(refer["turns"]) == 3
    assert "bottom left" in refer["turns"][1]["user_text"]
    assert refer["augmentation_policy"]["geometry"] == "orientation_locked"


def test_geochat_inline_boxes_structured(data_dirs):
    """Plain answers with inline boxes keep the phrase as label and strip
    the old coordinate protocol from the text.
    普通回答的内联框结构化保存，label 取紧邻短语，文本移除旧坐标协议。"""
    conv = [e for e in by_kind(data_dirs["train"], "geochat_conversation")
            if e["episode_id"].endswith("/c2.jpg")][0]
    turn = conv["turns"][0]
    assert turn["target_boxes"] == [{
        "xyxy_999": [500, 500, 599, 599],  # round(50*999/100)=500, 60->599
        "label": "two gray ships",
        "description": "two gray ships",
        "source_class_id": "52",
    }]
    assert "{<" not in turn["assistant_text"]
    assert turn["assistant_text"] == "In the satellite image, there are <p>two gray ships</p> at the bottom right."
    assert conv["augmentation_policy"]["geometry"] == "orientation_locked"  # bottom right


def test_geochat_rejections_and_closure(data_dirs):
    """Invalid records land in rejected.jsonl with stable codes and the
    manifest counts close: records == episodes + rejected.
    非法记录以稳定错误码进入 rejected.jsonl，计数闭合。"""
    rejected = data_dirs["rejected"]
    by_reason = {}
    for entry in rejected:
        assert set(entry.keys()) == {"dataset", "source_record_id", "reason"}
        assert entry["reason"] in REJECTION_CODES
        assert entry["dataset"] == "GeoChat"
        by_reason[entry["reason"]] = by_reason.get(entry["reason"], 0) + 1
    expected = {
        "out_of_range_box": 1,
        "degenerate_box": 1,
        "unparseable_box": 1,
        "non_finite_box": 1,
        "missing_image_field": 1,
        "invalid_role_order": 1,
        "missing_turn_text": 1,
        "image_token_count_mismatch": 1,
        "invalid_conversation_type": 2,  # mixed kind + non-list conversations
    }
    assert by_reason == expected
    manifest = data_dirs["manifest"]
    assert manifest["geochat_rejections"]["total"] == 10
    assert manifest["closure"]["geochat_records == geochat_episodes + geochat_rejected"] is True
    geochat_episodes = (
        manifest["dataset_counts"]["GeoChat"]["refer"]["episodes"]
        + manifest["dataset_counts"]["GeoChat"]["identify"]["episodes"]
        + manifest["dataset_counts"]["GeoChat"]["conversation"]["episodes"]
    )
    assert manifest["dataset_counts"]["GeoChat"]["records"] == geochat_episodes + 10
    assert all(manifest["closure"][k] is True for k in manifest["closure"])


def test_episode_contract_and_uniqueness(data_dirs):
    """All episodes honor the canonical schema and unique ids.
    所有 episode 符合 canonical schema 且 id 唯一。"""
    all_eps = data_dirs["train"] + data_dirs["validation"]
    ids = [e["episode_id"] for e in all_eps]
    assert len(ids) == len(set(ids))
    for ep in all_eps:
        assert ep["schema_version"] == 1
        assert ep["task_kind"] in ALLOWED_TASK_KINDS
        assert ep["split"] in ("train", "validation")
        assert ep["image_source"] in ("vrsbench", "geochat")
        assert ep["image"] and not ep["image"].startswith("/")
        assert ".." not in ep["image"].split("/")
        assert isinstance(ep["turns"], list) and ep["turns"]
        for turn in ep["turns"]:
            assert set(turn.keys()) == {"user_text", "assistant_text", "input_boxes", "target_boxes"}
            for box in turn["input_boxes"] + turn["target_boxes"]:
                xyxy = box["xyxy_999"]
                assert len(xyxy) == 4
                assert all(isinstance(v, int) for v in xyxy)
                assert xyxy[0] < xyxy[2] and xyxy[1] < xyxy[3]
                assert 0 <= xyxy[0] <= xyxy[2] <= 999 and 0 <= xyxy[1] <= xyxy[3] <= 999
        assert ep["augmentation_policy"]["geometry"] in ("orientation_locked", "geometry_safe")


def test_no_absolute_paths_and_json_safe(data_dirs, tmp_path):
    """Outputs are strictly JSON-safe with no machine absolute paths.
    输出严格 JSON-safe 且不含机器绝对路径。"""
    out_dir = data_dirs["out_dir"]
    raw = ""
    for name in ("train.jsonl", "validation.jsonl", "rejected.jsonl"):
        text = (out_dir / name).read_text(encoding="utf-8")
        raw += text
        for line in text.splitlines():
            if not line.strip():
                continue
            # NaN / Infinity are rejected at parse time
            json.loads(line, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    # the manifest is pretty-printed; validate it as one document
    manifest_text = (out_dir / "manifest.json").read_text(encoding="utf-8")
    raw += manifest_text
    json.loads(manifest_text, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    for needle in (str(tmp_path), "Users/", "C:\\", ":\\"):
        assert needle not in raw
    # every field must be a JSON-native value
    def check(value):
        if isinstance(value, dict):
            for k, v in value.items():
                assert isinstance(k, str)
                check(v)
        elif isinstance(value, list):
            for v in value:
                check(v)
        elif value is None or isinstance(value, (str, int, float, bool)):
            if isinstance(value, float):
                assert math.isfinite(value)
        else:
            raise AssertionError(f"non-JSON-safe value: {type(value)}")
    for ep in data_dirs["train"] + data_dirs["validation"]:
        check(ep)


def test_inputs_unchanged_and_checksums_recorded(data_dirs, tmp_path):
    """Input checksums before/after the run are identical and recorded in
    the manifest. 运行前后输入校验和不变，且记录进 manifest。"""
    before = {
        "vrsbench_train": sha256_file(data_dirs["vrsbench_dir"] / "VRSBench_train.jsonl"),
        "vrsbench_val": sha256_file(data_dirs["vrsbench_dir"] / "VRSBench_val.jsonl"),
        "geochat": sha256_file(data_dirs["geochat_file"]),
    }
    out2 = tmp_path / "out_again"
    rc = run_pipeline(data_dirs["vrsbench_dir"], data_dirs["geochat_file"], out2)
    assert rc == 0
    after = {
        "vrsbench_train": sha256_file(data_dirs["vrsbench_dir"] / "VRSBench_train.jsonl"),
        "vrsbench_val": sha256_file(data_dirs["vrsbench_dir"] / "VRSBench_val.jsonl"),
        "geochat": sha256_file(data_dirs["geochat_file"]),
    }
    assert before == after
    manifest = data_dirs["manifest"]
    assert manifest["inputs"]["vrsbench_train"]["sha256"] == before["vrsbench_train"]
    assert manifest["inputs"]["vrsbench_validation"]["sha256"] == before["vrsbench_val"]
    assert manifest["inputs"]["geochat"]["sha256"] == before["geochat"]
    assert len(manifest["outputs"]["train.jsonl_sha256"]) == 64


def test_augmentation_policy_rules(data_dirs):
    """Spatial types/text lock orientation; non-spatial stays safe.
    空间类型/文本锁定 orientation；非空间保持 safe。"""
    train = data_dirs["train"]
    qa1 = [e for e in train if e["episode_id"].endswith("/qa/1/box_assisted")][0]
    assert qa1["augmentation_policy"] == {
        "geometry": "orientation_locked", "reason": "source_type_spatial",
    }
    qa2 = [e for e in train if e["episode_id"].endswith("/qa/2/box_assisted")][0]
    assert qa2["augmentation_policy"]["geometry"] == "geometry_safe"
    qa3 = [e for e in train if e["episode_id"].endswith("/qa/3/box_assisted")][0]
    assert qa3["augmentation_policy"] == {
        "geometry": "orientation_locked", "reason": "spatial_language",
    }  # "center" in question
    qa9 = [e for e in train if e["episode_id"].endswith("/qa/9/box_assisted")][0]
    assert qa9["augmentation_policy"]["geometry"] == "geometry_safe"


def test_cli_smoke(tmp_path):
    """CLI: --help works and an end-to-end subprocess run succeeds.
    CLI：--help 可用且子进程端到端运行成功。"""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0
    assert "--self-attention-ratio" in proc.stdout

    vrsbench_dir = write_vrsbench_dir(tmp_path)
    geochat_file = write_geochat_file(tmp_path)
    out_dir = tmp_path / "cli_out"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT_PATH),
         "--vrsbench-dir", str(vrsbench_dir),
         "--geochat-file", str(geochat_file),
         "--output-dir", str(out_dir),
         "--seed", "cli-seed",
         "--self-attention-ratio", "0.5"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    for name in ("train.jsonl", "validation.jsonl", "rejected.jsonl", "manifest.json"):
        assert (out_dir / name).is_file()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation"]["seed"] == "cli-seed"
    assert manifest["generation"]["self_attention_ratio"] == 0.5


def test_refuses_output_into_source_dirs(tmp_path):
    """--output-dir pointing at a source data dir is refused.
    --output-dir 指向源数据目录时拒绝执行。"""
    vrsbench_dir = write_vrsbench_dir(tmp_path)
    geochat_file = write_geochat_file(tmp_path)
    rc = mod.main([
        "--vrsbench-dir", str(vrsbench_dir),
        "--geochat-file", str(geochat_file),
        "--output-dir", str(vrsbench_dir),
    ])
    assert rc == 1
    rc = mod.main([
        "--vrsbench-dir", str(vrsbench_dir),
        "--geochat-file", str(geochat_file),
        "--output-dir", str(geochat_file.parent),
    ])
    assert rc == 1
