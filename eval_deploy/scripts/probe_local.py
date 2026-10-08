#!/usr/bin/env python3
"""Synthetic self-check for the local participant API (no website quota).

本地合成自检：从图片包取真实 asset 与哈希，构造 5 类约束的**合成题**
（非真题），逐题调用本地 /v1/predict，校验协议格式并统计耗时。
不消耗网站评测次数；输出不含 token。

Usage:
  MODEL_API_KEY=$(cat ~/eval/secrets/model_api_key) \
    ~/.conda/envs/m3/bin/python probe_local.py \
    --origin http://127.0.0.1:9001 --package ~/eval/packages/rsu-dev100-v2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def call(origin: str, token: str, path: str, payload: dict | None = None, timeout: int = 130):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        origin.rstrip("/") + path,
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            **({"Content-Type": "application/json"} if body else {}),
        },
        method="GET" if body is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.load(exc)
        except Exception:
            return exc.code, {"error": "non_json_response"}


def build_cases(package: Path):
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    assets = manifest["assets"]
    a1, a2 = assets[0], assets[1]

    from PIL import Image

    with Image.open(package / "assets" / a1["asset_id"]) as image:
        width, height = image.size

    def ref(asset):
        return {key: asset[key] for key in ("asset_id", "sha256", "mime_type")}

    return [
        ("enum_1img", {
            "question": "Is there anything visible in the image? Answer Yes or No.",
            "images": [ref(a1)],
            "response_constraint": {"type": "enum", "values": ["Yes", "No"]},
            "answer_type": "short_text",
        }),
        ("single_choice_1img", {
            "question": "What is the dominant scene in this image?",
            "images": [ref(a1)],
            "choices": {
                "A": "urban area",
                "B": "water body",
                "C": "forest or vegetation",
                "D": "farmland",
            },
            "response_constraint": {"type": "single_choice", "values": ["A", "B", "C", "D"]},
            "answer_type": "short_text",
        }),
        ("integer_1img", {
            "question": "How many clearly separated large objects are visible? Reply with a number.",
            "images": [ref(a1)],
            "response_constraint": {"type": "integer"},
            "answer_type": "short_text",
        }),
        ("short_text_1img", {
            "question": "Briefly describe what this remote sensing image shows.",
            "images": [ref(a1)],
            "response_constraint": {"type": "short_text"},
            "answer_type": "short_text",
        }),
        ("bbox_1img", {
            "question": "Localize the single most salient object in the image.",
            "images": [ref(a1)],
            "response_constraint": {"type": "bbox", "coordinate_format": "pixel_xyxy", "length": 4},
            "answer_type": "bbox",
            "image_width": width,
            "image_height": height,
        }),
        ("enum_2img_change", {
            "question": "Do the two images show any obvious change? Answer Yes or No.",
            "images": [ref(a1), ref(a2)],
            "response_constraint": {"type": "enum", "values": ["Yes", "No"]},
            "answer_type": "short_text",
        }),
    ], width, height


def validate(case: dict, status: int, body: dict, payload: dict, width: int, height: int):
    """Hard-fail only on true protocol/format violations; content mismatches
    are surfaced as notes (they are scored by the site, not malformed).

    只对真正的协议/格式违规判失败；内容不匹配仅提示（由网站计分）。
    """
    if status != 200:
        return False, f"http={status} error={body.get('error')}"
    if body.get("request_id") != payload["request_id"] or body.get("item_id") != payload["item_id"]:
        return False, "identifier echo mismatch"
    answer = body.get("answer")
    kind = case["response_constraint"]["type"]
    if kind == "bbox":
        if not (
            isinstance(answer, list)
            and len(answer) == 4
            and all(isinstance(v, (int, float)) for v in answer)
            and answer[0] < answer[2]
            and answer[1] < answer[3]
        ):
            return False, f"bbox invalid: {answer}"
        if answer[0] < 0 or answer[1] < 0 or answer[2] > width or answer[3] > height:
            return False, f"bbox outside image: {answer}"
        return True, ""
    if not isinstance(answer, str) or not answer or len(answer) > 4096:
        return False, f"answer not a legal string: {answer!r}"
    if kind == "integer" and not answer.isdigit():
        return False, f"integer not digits-string: {answer!r}"
    if kind == "enum" and answer not in case["response_constraint"]["values"]:
        return True, f"note: enum value outside allowed list ({answer!r})"
    if kind == "single_choice" and answer not in case["choices"]:
        return True, f"note: choice letter not in choices ({answer!r})"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin", default="http://127.0.0.1:9001")
    parser.add_argument("--package", required=True)
    args = parser.parse_args()
    token = os.environ.get("MODEL_API_KEY", "")
    if not token:
        print("MODEL_API_KEY is required", file=sys.stderr)
        return 2

    cases, width, height = build_cases(Path(args.package))
    status, health = call(args.origin, token, "/healthz")
    print(f"healthz: http={status} {json.dumps(health, ensure_ascii=False)}")
    if status != 200 or health.get("status") != "ready":
        print("health check failed; aborting", file=sys.stderr)
        return 1

    results = []
    for name, case in cases:
        payload = {
            "protocol_version": "2.0",
            "request_id": f"probe:{name}:1",
            "item_id": f"PROBE_{name}",
            **case,
        }
        started = time.perf_counter()
        status, body = call(args.origin, token, "/v1/predict", payload)
        elapsed = time.perf_counter() - started
        ok, note = validate(case, status, body, payload, width, height)
        preview = str(body.get("answer"))[:90]
        results.append((name, ok, elapsed, note))
        print(f"[{'OK ' if ok else 'BAD'}] {name:20s} http={status} {elapsed:6.1f}s answer={preview} {note}")

    passed = sum(1 for _, ok, _, _ in results if ok)
    times = sorted(t for _, _, t, _ in results)
    p50 = times[len(times) // 2]
    print(
        f"passed {passed}/{len(results)} | elapsed P50={p50:.1f}s "
        f"max={times[-1]:.1f}s min={times[0]:.1f}s"
    )
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
