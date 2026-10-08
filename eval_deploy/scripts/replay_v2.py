"""Local probe: resource metrics + targeted item replay (no eval quota used).

本地探针：资源指标 + 定向题目回放（不消耗评测额度）。
Usage: M3_CONFIG=<yaml> CUDA_VISIBLE_DEVICES=1 python replay_v2.py \
         --spec <replay_items.json> --mode resource|replay|both --tag <name>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/home/tongyi/eval")
from m3_bridge.real_model import M3RealModel  # noqa: E402

PKG = Path("/home/tongyi/eval/packages/rsu-dev100-v2")
LOGDIR = Path("/home/tongyi/eval/logs")


def load_manifest() -> dict:
    manifest = json.loads((PKG / "manifest.json").read_text(encoding="utf-8"))
    return {a["asset_id"]: a for a in manifest["assets"]}


def write_run_log(tag: str, rows: list[dict]) -> None:
    out = LOGDIR / f"replay_v2_{tag}.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[replay_v2] log -> {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="/home/tongyi/eval/private/replay_items.json")
    ap.add_argument("--mode", choices=["resource", "replay", "both"], default="both")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tag = args.tag or time.strftime("%m%dT%H%M", time.localtime())
    by_id = load_manifest()
    from PIL import Image

    def ref(asset_id: str) -> dict:
        a = by_id[asset_id]
        return {"asset_id": asset_id, "sha256": a["sha256"], "mime_type": a["mime_type"]}

    def asset_of(rel: str) -> str:
        return rel.split("/", 1)[-1]

    def dims(asset_id: str) -> tuple[int, int]:
        with Image.open(PKG / "assets" / asset_id) as im:
            return im.size

    t0 = time.time()
    model = M3RealModel()
    print(f"[replay_v2] LOAD_SECONDS {round(time.time() - t0, 1)}", flush=True)

    rows: list[dict] = []

    def run_case(case: str, request: dict, image_paths: list[Path]) -> None:
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        try:
            ans = model.predict(request, image_paths)
            status = "ok"
        except Exception as exc:  # AdapterError or worse
            ans, status = None, f"error:{getattr(exc, 'code', type(exc).__name__)}"
        row = {
            "kind": "case", "case": case, "status": status,
            "latency_s": round(time.time() - t, 1),
            "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
            "answer": None if ans is None else str(ans)[:80],
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    items = spec["items"]
    if args.limit:
        items = items[: args.limit]

    if args.mode in ("resource", "both"):
        a1 = "f02c7e3d7f0c74809a45d6de06839999"  # 4097x4096
        a2 = "97558d5d445597cfd1f6b2e1c827188f"  # 7360x4912
        w1, h1 = dims(a1)
        run_case("shorttext_4097", {
            "protocol_version": "2.0", "request_id": "res-st", "item_id": "RES-ST",
            "question": "What is the dominant land use in this scene? Answer in two words.",
            "images": [ref(a1)],
            "response_constraint": {"type": "short_text"}, "answer_type": "short_text",
        }, [PKG / "assets" / a1])
        run_case("bbox_4097", {
            "protocol_version": "2.0", "request_id": "res-bb", "item_id": "RES-BB",
            "question": "Locate the most prominent large building in the scene.",
            "images": [ref(a1)],
            "response_constraint": {"type": "bbox", "coordinate_format": "pixel_xyxy", "length": 4},
            "answer_type": "bbox", "image_width": w1, "image_height": h1,
        }, [PKG / "assets" / a1])
        run_case("shorttext_7360", {
            "protocol_version": "2.0", "request_id": "res-st2", "item_id": "RES-ST2",
            "question": "How many distinct rooftop colors can be seen? Reply with a single word.",
            "images": [ref(a2)],
            "response_constraint": {"type": "short_text"}, "answer_type": "short_text",
        }, [PKG / "assets" / a2])
        change = next((it for it in items if it["kind"] == "change_enum"), None)
        if change is not None:
            imgs = [asset_of(rel) for rel in change["images"]]
            run_case("change_2img", {
                "protocol_version": "2.0", "request_id": "res-chg", "item_id": "RES-CHG",
                "question": change["question"],
                "images": [ref(a) for a in imgs],
                "response_constraint": {"type": "enum", "values": change["values"]},
                "answer_type": "enum",
            }, [PKG / "assets" / a for a in imgs])

    if args.mode in ("replay", "both"):
        for item in items:
            imgs = [asset_of(rel) for rel in item["images"]]
            request = {
                "protocol_version": "2.0", "request_id": "rep-" + item["id"],
                "item_id": item["id"], "question": item["question"],
                "images": [ref(a) for a in imgs],
            }
            if item["kind"] == "bbox":
                w, h = dims(imgs[0])
                request["response_constraint"] = {
                    "type": "bbox", "coordinate_format": "pixel_xyxy", "length": 4}
                request["answer_type"] = "bbox"
                request["image_width"] = w
                request["image_height"] = h
            else:
                request["response_constraint"] = {"type": "enum", "values": item["values"]}
                request["answer_type"] = "enum"
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            try:
                ans = model.predict(request, [PKG / "assets" / a for a in imgs])
                status = "ok"
            except Exception as exc:
                ans, status = None, f"error:{getattr(exc, 'code', type(exc).__name__)}"
            row = {
                "kind": "item", "id": item["id"], "group": item.get("group"),
                "task": item["kind"], "status": status,
                "latency_s": round(time.time() - t, 1),
                "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
                "answer": None if ans is None else str(ans)[:80],
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    write_run_log(tag, rows)


if __name__ == "__main__":
    main()
