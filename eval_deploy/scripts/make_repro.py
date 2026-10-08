"""Build one bbox repro request (B_0002) for the local API. Diagnostics only."""
import json
from pathlib import Path

from PIL import Image

PKG = Path("/home/tongyi/eval/packages/rsu-dev100-v2")
spec = json.loads(Path("/home/tongyi/eval/private/replay_items.json").read_text(encoding="utf-8"))
item = next(i for i in spec["items"] if i["id"] == "RSU_B_0002")
man = json.loads((PKG / "manifest.json").read_text(encoding="utf-8"))
by_id = {a["asset_id"]: a for a in man["assets"]}
aid = item["images"][0].split("/")[-1]
asset = by_id[aid]
w, h = Image.open(PKG / "assets" / aid).size
request = {
    "protocol_version": "2.0",
    "request_id": "repro-b0002",
    "item_id": "RSU_B_REPRO",
    "question": item["question"],
    "images": [{"asset_id": aid, "sha256": asset["sha256"], "mime_type": asset["mime_type"]}],
    "response_constraint": {"type": "bbox", "coordinate_format": "pixel_xyxy", "length": 4},
    "answer_type": "bbox",
    "image_width": w,
    "image_height": h,
}
Path("/tmp/bbox_req.json").write_text(json.dumps(request), encoding="utf-8")
print("wrote /tmp/bbox_req.json", w, h)
