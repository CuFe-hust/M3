"""Local structure validation for the supported datasets.
本地数据集结构校验。
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image

from data.downloader import DEFAULT_DATA_ROOT
from data.loader import _read_json_records


# Dataset roots that map one-to-one to subdirectories of the data root.
# 与数据根目录下子目录一一对应的数据集名。
DATASET_ROOTS = ("vrsbench", "xlrs_bench", "levir_cc", "mme_real_rs")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}


class DatasetValidationError(Exception):
    """Raised when a dataset fails structure validation.
    数据集结构校验失败时抛出。
    """


def validate_dataset(name: str, data_root: str | Path) -> dict[str, Any]:
    """Validate one dataset root; raise DatasetValidationError on failure.
    校验单个数据集目录；失败时抛出 DatasetValidationError。
    """

    if name not in DATASET_ROOTS:
        raise ValueError(f"Unsupported dataset validation target: {name}")
    root = Path(data_root) / name
    errors: list[str] = []
    info: dict[str, Any] = {"dataset": name, "root": str(root)}
    if name == "vrsbench":
        _validate_vrsbench(root, errors, info)
    elif name == "xlrs_bench":
        _validate_xlrs_bench(root, errors, info)
    elif name == "levir_cc":
        _validate_levir_cc(root, errors, info)
    else:
        _validate_mme_real_rs(root, errors, info)
    if errors:
        raise DatasetValidationError(f"{name} validation failed:\n" + "\n".join(errors))
    info["ok"] = True
    return info


def validate_all(data_root: str | Path, names: Iterable[str] | None = None) -> dict[str, Any]:
    """Validate configured datasets and return aggregated per-dataset reports.
    校验配置的数据集并返回按数据集汇总的报告。
    """

    selected = tuple(names) if names is not None else DATASET_ROOTS
    reports: dict[str, Any] = {}
    for name in selected:
        try:
            reports[name] = validate_dataset(name, data_root)
        except DatasetValidationError as error:
            reports[name] = {"dataset": name, "root": str(Path(data_root) / name), "ok": False, "error": str(error)}
    reports["failed"] = [name for name in selected if not reports[name].get("ok", False)]
    reports["ok"] = not reports["failed"]
    return reports


def _validate_vrsbench(root: Path, errors: list[str], info: dict[str, Any]) -> None:
    """Validate VRSBench annotation JSONs and image directories.
    校验 VRSBench 标注 JSON 与图片目录。
    """

    annotation_counts: dict[str, int] = {}
    for filename in ("VRSBench_EVAL_Cap.json", "VRSBench_EVAL_vqa.json", "VRSBench_EVAL_referring.json"):
        path = root / filename
        if not path.is_file():
            errors.append(f"Missing VRSBench annotation: {filename}")
            continue
        try:
            records = _read_json_records(path)
            annotation_counts[filename] = len(records)
        except (OSError, ValueError) as error:
            errors.append(f"Cannot parse VRSBench annotation {filename}: {error}")
    info["annotation_counts"] = annotation_counts
    image_count = _image_count(root)
    info["image_count"] = image_count
    if image_count == 0:
        errors.append("No image files found under the vrsbench root.")
    else:
        _sample_open_images(root, errors)


def _validate_xlrs_bench(root: Path, errors: list[str], info: dict[str, Any]) -> None:
    """Validate XLRS-Bench Hugging Face dataset directories.
    校验 XLRS-Bench 的 Hugging Face 数据集目录。
    """

    required_layout = {
        "XLRS-Bench-lite": ("dataset_dict.json", "train/state.json"),
        "XLRS-Bench_caption_en": ("train/state.json",),
        "XLRS-Bench_visual_grounding_en": ("test/state.json",),
    }
    for directory, required_files in required_layout.items():
        base = root / directory
        if not base.is_dir():
            errors.append(f"Missing XLRS release directory: {directory}")
            continue
        for relative in required_files:
            if not (base / relative).is_file():
                errors.append(f"Missing XLRS file: {directory}/{relative}")
    arrow_count = _file_count(root, ".arrow")
    info["arrow_file_count"] = arrow_count
    if arrow_count == 0:
        errors.append("No .arrow data files found under the xlrs_bench root.")
    _check_xlrs_rows(root, errors, info)


def _check_xlrs_rows(root: Path, errors: list[str], info: dict[str, Any]) -> None:
    """Optionally load one row from each XLRS split when datasets is installed.
    当安装了 datasets 库时，可选地从各 XLRS 切分读取一行。
    """

    try:
        from datasets import load_from_disk
    except ImportError:
        info["row_check"] = "not_checked (datasets library not installed)"
        return
    try:
        checked: dict[str, str] = {}
        lite = root / "XLRS-Bench-lite"
        if (lite / "dataset_dict.json").is_file():
            row = next(iter(load_from_disk(str(lite))["train"]), None)
            checked["XLRS-Bench-lite"] = "ok" if row is not None else "empty"
        caption = root / "XLRS-Bench_caption_en" / "train"
        if (caption / "state.json").is_file():
            row = next(iter(load_from_disk(str(caption))), None)
            checked["XLRS-Bench_caption_en"] = "ok" if row is not None else "empty"
        grounding = root / "XLRS-Bench_visual_grounding_en" / "test"
        if (grounding / "state.json").is_file():
            row = next(iter(load_from_disk(str(grounding))), None)
            checked["XLRS-Bench_visual_grounding_en"] = "ok" if row is not None else "empty"
        info["row_check"] = checked
    except Exception as error:
        errors.append(f"XLRS row load failed: {error}")


def _validate_levir_cc(root: Path, errors: list[str], info: dict[str, Any]) -> None:
    """Validate LEVIR-CC annotation and A/B image pairs.
    校验 LEVIR-CC 标注与前后时相图片对。
    """

    annotation = root / "Levir-CC-dataset" / "LevirCCcaptions.json"
    if not annotation.is_file():
        errors.append("Missing LEVIR-CC annotation: Levir-CC-dataset/LevirCCcaptions.json")
    else:
        try:
            records = _read_json_records(annotation)
            info["annotation_count"] = len(records)
        except (OSError, ValueError) as error:
            errors.append(f"Cannot parse LEVIR-CC annotation: {error}")
    images_root = root / "Levir-CC-dataset" / "images"
    image_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        for side in ("A", "B"):
            directory = images_root / split / side
            count = _image_count(directory)
            image_counts[f"{split}/{side}"] = count
            if count == 0:
                errors.append(f"LEVIR-CC images directory is empty: {directory.relative_to(root)}")
    info["image_counts"] = image_counts
    if any(image_counts.values()):
        _sample_open_images(images_root, errors)


def _validate_mme_real_rs(root: Path, errors: list[str], info: dict[str, Any]) -> None:
    """Validate MME-RealWorld annotation and Remote Sensing images.
    校验 MME-RealWorld 标注与 Remote Sensing 图片。
    """

    annotation = root / "MME_RealWorld.json"
    if not annotation.is_file():
        errors.append("Missing MME-RealWorld annotation: MME_RealWorld.json")
    else:
        try:
            records = _read_json_records(annotation)
            info["annotation_count"] = len(records)
            rs_records = [
                record
                for record in records
                if "remote sensing" in str(record.get("Subtask", record.get("subtask", ""))).lower()
                or "remote sensing" in str(record.get("Question_id", record.get("question_id", ""))).lower()
            ]
            info["remote_sensing_record_count"] = len(rs_records)
            if not rs_records:
                errors.append("No Remote Sensing records found in MME_RealWorld.json.")
        except (OSError, ValueError) as error:
            errors.append(f"Cannot parse MME-RealWorld annotation: {error}")
    image_count = _image_count(root)
    info["image_count"] = image_count
    if image_count == 0:
        errors.append("No image files found under the mme_real_rs root.")
    else:
        _sample_open_images(root, errors)


def _image_count(path: Path) -> int:
    """Count supported image files under a directory.
    统计目录下支持的图片文件数量。
    """

    return _file_count(path, tuple(sorted(IMAGE_SUFFIXES)))


def _file_count(path: Path, suffixes: tuple[str, ...] | str) -> int:
    """Count files with matching suffixes under a directory.
    统计目录下后缀匹配的文件数量。
    """

    if not path.is_dir():
        return 0
    normalized = {suffix.lower() for suffix in suffixes} if isinstance(suffixes, tuple) else {suffixes.lower()}
    return sum(1 for child in path.rglob("*") if child.is_file() and child.suffix.lower() in normalized)


def _sample_open_images(root: Path, errors: list[str], limit: int = 3) -> None:
    """Open a few images to verify they are decodable.
    抽样打开少量图片以验证其可解码。
    """

    opened = 0
    for path in sorted(child for child in root.rglob("*") if child.suffix.lower() in IMAGE_SUFFIXES):
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:
            errors.append(f"Cannot open image {path}: {error}")
        opened += 1
        if opened >= limit:
            break


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate local dataset structures.")
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT, help="Local data root (default: /data).")
    parser.add_argument("--datasets", nargs="+", choices=DATASET_ROOTS, default=DATASET_ROOTS)
    args = parser.parse_args(argv)
    reports = validate_all(args.root, args.datasets)
    for name in args.datasets:
        report = reports[name]
        print(f"{name}: {'OK' if report.get('ok', False) else 'FAILED'}")
        if not report.get("ok", False):
            print(report.get("error", ""))
    return 0 if reports["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
