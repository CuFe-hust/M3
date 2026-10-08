"""Measure pure-inference resource metrics (latency + peak GPU memory)."""
import json, sys, time
from pathlib import Path
import torch

sys.path.insert(0, "/home/tongyi/eval")
from m3_bridge.real_model import M3RealModel

PKG = Path("/home/tongyi/eval/packages/rsu-dev100-v2")
manifest = json.loads((PKG / "manifest.json").read_text(encoding="utf-8"))
by_id = {a["asset_id"]: a for a in manifest["assets"]}
from PIL import Image

def ref(asset_id):
    a = by_id[asset_id]
    return {"asset_id": asset_id, "sha256": a["sha256"], "mime_type": a["mime_type"]}

def dims(asset_id):
    with Image.open(PKG / "assets" / asset_id) as im:
        return im.size

A1 = "f02c7e3d7f0c74809a45d6de06839999"  # 4097x4096
A2 = "97558d5d445597cfd1f6b2e1c827188f"  # 7360x4912

t0 = time.time()
model = M3RealModel()
print("LOAD_SECONDS", round(time.time() - t0, 1), flush=True)

def run(name, asset, constraint, question, answer_type="short_text", extra=None):
    w, h = dims(asset)
    req = {"protocol_version": "2.0", "request_id": "res-" + name, "item_id": "RES-" + name,
           "question": question, "images": [ref(asset)],
           "response_constraint": constraint, "answer_type": answer_type}
    if extra:
        req.update(extra)
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    ans = model.predict(req, [PKG / "assets" / asset])
    lat = time.time() - t
    print(json.dumps({
        "case": name, "image": f"{w}x{h}", "latency_s": round(lat, 1),
        "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "answer": str(ans)[:60]}), flush=True)

run("shorttext_4097", A1, {"type": "short_text"}, "What is the dominant land use in this scene? Answer in two words.")
run("bbox_4097", A1, {"type": "bbox", "coordinate_format": "pixel_xyxy", "length": 4},
    "Locate the most prominent large building in the scene.",
    "bbox", {"image_width": dims(A1)[0], "image_height": dims(A1)[1]})
run("shorttext_7360", A2, {"type": "short_text"}, "How many distinct rooftop colors can be seen? Reply with a single word.")
