from __future__ import annotations

import json
from pathlib import Path

from scripts import prepare_change_qwen_sft as mod


def test_levir_prepare_keeps_pairs_ordered_and_excludes(tmp_path: Path) -> None:
    source = tmp_path / "levir.json"
    source.write_text(json.dumps({"images": [
        {"filename": "one.png", "split": "train", "captions": ["A building appeared.", "NO_CHANGE"]},
        {"filename": "two.png", "split": "val", "caption": "A road was removed."},
    ]}), encoding="utf-8")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("production prompt", encoding="utf-8")
    output = tmp_path / "out"
    assert mod.main(["--source-type", "levir_caption", "--source", str(source), "--output-dir", str(output), "--prompt-file", str(prompt), "--excluded-sample-ids", "two.png"]) == 0
    train = [json.loads(line) for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(train) == 2
    assert [image["role"] for image in train[0]["images"]] == ["raw_full_t1", "raw_full_t2"]
    assert train[1]["target"]["result"]["answer"] == "No significant semantic change detected."
    rejected = [json.loads(line) for line in (output / "rejected.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rejected == [{"source_record_id": "two.png", "reason": "excluded_parent_sample"}]
