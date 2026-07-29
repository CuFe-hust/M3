import hashlib
import json
from pathlib import Path

from PIL import Image

from spacers_agent.data_audit import inspect_dataset_root, write_dataset_audit
from spacers_agent.cli import main


def test_read_only_dataset_audit_reports_images_fields_and_missing_references(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    Image.new("RGB", (12, 8), "blue").save(root / "image.png")
    records = [
        {"id": "one", "split": "train", "image": "image.png", "question": "How many?", "answer": "1"},
        {"id": "one", "split": "test", "image": "missing.png", "caption": "caption"},
    ]
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(records), encoding="utf-8")
    before = hashlib.sha256(manifest.read_bytes()).hexdigest()

    report = inspect_dataset_root(root)
    output = tmp_path / "audit.json"
    write_dataset_audit(report, output)

    assert report.image_count == 1
    assert report.image_samples[0]["width"] == 12
    assert report.duplicate_ids == ["one"]
    assert report.missing_referenced_images == ["missing.png"]
    assert {"question", "answer", "caption"}.issubset(report.discovered_field_names)
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == before
    assert json.loads(output.read_text(encoding="utf-8"))["root"] == str(root)


def test_inspect_data_cli_writes_a_separate_report(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    Image.new("RGB", (4, 4)).save(root / "one.png")
    output = tmp_path / "audit.json"

    status = main(["inspect-data", "--root", str(root), "--output", str(output)])

    assert status == 0
    assert json.loads(output.read_text(encoding="utf-8"))["image_count"] == 1
