#!/usr/bin/env python3
"""Direct model-layer probe: call M3RealModel.predict() without HTTP.

模型层直调探针：不经过 HTTP 外壳，直接构建 M3RealModel 并对合成题调
predict()，用于快速迭代（加载模型一次 ≈ 数分钟）。

Usage:
  ~/.conda/envs/m3/bin/python probe_direct.py --package ~/eval/packages/rsu-dev100-v2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# qi1 has two H100s; GPU0 is shared with others' workloads. Default this probe
# to GPU1 unless the caller overrides it.
# qi1 双卡：GPU0 常有他人任务；本探针默认使用 GPU1（可被环境变量覆盖）。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")


def main() -> int:
    eval_root = Path(__file__).resolve().parent.parent
    if str(eval_root) not in sys.path:
        sys.path.insert(0, str(eval_root))
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", default=str(eval_root / "packages" / "rsu-dev100-v2"))
    args = parser.parse_args()

    from m3_bridge.real_model import M3RealModel

    print("building M3RealModel (loads the model once) ...")
    started = time.perf_counter()
    model = M3RealModel()
    print(f"model ready in {time.perf_counter() - started:.1f}s")

    package = Path(args.package)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    assets = manifest["assets"]
    a1, a2 = assets[0], assets[1]

    from PIL import Image

    with Image.open(package / "assets" / a1["asset_id"]) as image:
        width, height = image.size

    def refs(*items):
        return [{key: item[key] for key in ("asset_id", "sha256", "mime_type")} for item in items]

    def paths(ref_list):
        return [package / "assets" / item["asset_id"] for item in ref_list]

    cases = [
        ("enum_1img", {
            "question": "Is there anything visible in the image? Answer Yes or No.",
            "images": refs(a1),
            "response_constraint": {"type": "enum", "values": ["Yes", "No"]},
            "answer_type": "short_text",
        }),
        ("single_choice_1img", {
            "question": "What is the dominant scene in this image?",
            "images": refs(a1),
            "choices": {"A": "urban area", "B": "water body", "C": "forest or vegetation", "D": "farmland"},
            "response_constraint": {"type": "single_choice", "values": ["A", "B", "C", "D"]},
            "answer_type": "short_text",
        }),
        ("integer_1img", {
            "question": "How many clearly separated large objects are visible? Reply with a number.",
            "images": refs(a1),
            "response_constraint": {"type": "integer"},
            "answer_type": "short_text",
        }),
        ("short_text_1img", {
            "question": "Briefly describe what this remote sensing image shows.",
            "images": refs(a1),
            "response_constraint": {"type": "short_text"},
            "answer_type": "short_text",
        }),
        ("bbox_1img", {
            "question": "Localize the single most salient object in the image.",
            "images": refs(a1),
            "response_constraint": {"type": "bbox", "coordinate_format": "pixel_xyxy", "length": 4},
            "answer_type": "bbox",
            "image_width": width,
            "image_height": height,
        }),
        ("enum_2img_change", {
            "question": "Do the two images show any obvious change? Answer Yes or No.",
            "images": refs(a1, a2),
            "response_constraint": {"type": "enum", "values": ["Yes", "No"]},
            "answer_type": "short_text",
        }),
    ]

    failures = 0
    for name, case in cases:
        request = {
            "protocol_version": "2.0",
            "request_id": f"direct:{name}:1",
            "item_id": f"DIRECT_{name}",
            **case,
        }
        started = time.perf_counter()
        try:
            answer = model.predict(request, paths(case["images"]))
            print(f"[OK ] {name:20s} {time.perf_counter() - started:6.1f}s answer={str(answer)[:100]}")
        except Exception as exc:  # noqa: BLE001 - probe reports stable shapes
            failures += 1
            code = getattr(exc, "code", type(exc).__name__)
            print(f"[ERR] {name:20s} {time.perf_counter() - started:6.1f}s {code}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
