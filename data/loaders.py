"""Official dataset download helpers and baseline sample adapters.
官方数据集下载工具与基线样本适配器。
"""

from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from PIL import Image

from data.schema import CanonicalSample


DATASET_REPOS = {
    "vrsbench": "xiang709/VRSBench",
    "mme_real_rs": "yifanzhang114/MME-RealWorld",
    "xlrs_caption_en": "initiacms/XLRS-Bench_caption_en",
    "xlrs_grounding_en": "initiacms/XLRS-Bench_visual_grounding_en",
    "xlrs_vqa_lite": "initiacms/XLRS-Bench-lite",
    "levir_cc": "lcybuaa/LEVIR-CC",
}

DATASET_SPLITS = {
    "xlrs_caption_en": "train",
    "xlrs_grounding_en": "test",
    "xlrs_vqa_lite": "train",
}


def download_datasets(names: Iterable[str], data_root: Path) -> dict[str, Path]:
    """Download only official dataset releases into an external data root.
    仅将官方数据集发布版下载到仓库外部的数据根目录。
    """

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError("Install requirements.txt before downloading datasets.") from error

    downloaded: dict[str, Path] = {}
    for name in names:
        if name not in DATASET_REPOS:
            raise ValueError(f"Unsupported dataset download target: {name}")
        target = data_root / name
        snapshot_download(repo_id=DATASET_REPOS[name], repo_type="dataset", local_dir=target)
        if name in {"vrsbench", "levir_cc"}:
            _extract_archives(target)
        downloaded[name] = target
    return downloaded


def load_samples(dataset_name: str, data_root: Path) -> Iterator[CanonicalSample]:
    """Load one named evaluation stream in its official evaluation scope.
    按官方评测范围加载一个具名评测样本流。
    """

    if dataset_name == "vrsbench_caption":
        yield from _load_vrsbench(data_root / "vrsbench", "caption")
    elif dataset_name == "vrsbench_vqa":
        yield from _load_vrsbench(data_root / "vrsbench", "vqa")
    elif dataset_name == "vrsbench_grounding":
        yield from _load_vrsbench(data_root / "vrsbench", "grounding")
    elif dataset_name == "mme_real_rs":
        yield from _load_mme_real_rs(data_root / "mme_real_rs")
    elif dataset_name in DATASET_SPLITS:
        yield from _load_hf_dataset(dataset_name, _local_dataset_root(data_root, dataset_name))
    elif dataset_name == "levir_cc":
        yield from _load_levir_cc(data_root / "levir_cc")
    else:
        raise ValueError(f"Unsupported dataset evaluation target: {dataset_name}")


def _local_dataset_root(data_root: Path, dataset_name: str) -> Path:
    """Resolve official XLRS releases downloaded under their published names."""

    candidates = {
        "xlrs_vqa_lite": (data_root / "xlrs_bench" / "XLRS-Bench-lite",),
        "xlrs_caption_en": (data_root / "xlrs_bench" / "XLRS-Bench_caption_en",),
        "xlrs_grounding_en": (data_root / "xlrs_bench" / "XLRS-Bench_visual_grounding_en",),
    }.get(dataset_name, ())
    for candidate in (*candidates, data_root / dataset_name):
        if candidate.exists():
            return candidate
    return data_root / dataset_name


def _extract_archives(root: Path) -> None:
    for archive in root.rglob("*.zip"):
        destination = archive.with_suffix("")
        if destination.exists():
            continue
        with zipfile.ZipFile(archive) as zip_file:
            zip_file.extractall(destination)


def _load_vrsbench(root: Path, task_type: str) -> Iterator[CanonicalSample]:
    annotation = _find_vrs_annotation(root, task_type)
    records = _read_json_records(annotation)
    image_index = _image_index(root)
    for record_number, record in enumerate(records):
        image = _resolve_named_image(
            record,
            root,
            image_index,
            ("image", "Image", "image_path", "img_path", "img", "image_name", "file_name", "filename", "image_id"),
        )
        base_id = _record_id(record, f"vrs-{task_type}-{record_number}")
        if task_type == "caption":
            caption = _first_text(record, ("caption", "text", "answer", "description", "ground_truth"))
            if caption:
                yield CanonicalSample(
                    id=base_id,
                    task_type="caption",
                    images=[image],
                    prompt="Describe this remote sensing image in detail.",
                    answers=[caption],
                    meta={"source": "VRSBench", "annotation": str(annotation)},
                )
            continue
        if task_type == "vqa":
            pairs = record.get("qa_pairs") or record.get("qas") or [record]
            for pair_number, pair in enumerate(pairs):
                question = _first_text(pair, ("question", "ques", "text"))
                answer = _first_text(pair, ("answer", "ans", "label", "ground_truth"))
                question_id = _first_value(pair, ("question_id", "ques_id", "id", "ID", "uid"))
                if question and answer:
                    yield CanonicalSample(
                        id=str(question_id) if question_id is not None else f"{base_id}-qa-{pair_number}",
                        task_type="vqa",
                        images=[image],
                        prompt=f"Answer the question using only the image. {question}",
                        answers=[answer],
                        meta={"source": "VRSBench", "question": question},
                    )
            continue
        objects = record.get("objects") or record.get("refs") or [record]
        for object_number, obj in enumerate(objects):
            referring = _first_text(obj, ("ref", "referring", "question", "text"))
            box = _first_value(obj, ("bbox", "box", "boxes", "polygon", "ground_truth", "obj_corner"))
            if referring and box is not None:
                yield CanonicalSample(
                    id=_record_id(obj, f"{base_id}-ref-{object_number}"),
                    task_type="grounding",
                    images=[image],
                    prompt=(
                        "Locate the referred object. Return only its bounding box as "
                        "[x1, y1, x2, y2] with coordinates normalized from 0 to 100. "
                        f"Referring expression: {referring}"
                    ),
                    boxes=[_normalize_box(box)],
                    meta={"source": "VRSBench", "referring_expression": referring},
                )


def _load_mme_real_rs(root: Path) -> Iterator[CanonicalSample]:
    annotation = _find_file(root, "MME_RealWorld.json")
    image_index = _image_index(root)
    for record_number, record in enumerate(_read_json_records(annotation)):
        subtask = str(record.get("Subtask", record.get("subtask", ""))).lower().replace("_", " ")
        question_id = str(record.get("Question_id", record.get("question_id", ""))).lower()
        if "remote sensing" not in subtask and "remote sensing" not in question_id:
            continue
        choices = record.get("Answer choices", record.get("answer_choices", []))
        question = _first_text(record, ("Text", "text", "question"))
        if not question or not choices:
            raise ValueError(f"Invalid MME-RealWorld RS record at index {record_number}.")
        prompt = _multiple_choice_prompt(question, choices, allow_multiple=False, option_count=5)
        image = _resolve_image(record, root, image_index)
        yield CanonicalSample(
            id=_record_id(record, f"mme-rs-{record_number}"),
            task_type="vqa",
            images=[image],
            prompt=prompt,
            answers=[str(record.get("Ground truth", record.get("ground_truth", "")))],
            choices=[str(choice) for choice in choices],
            meta={"source": "MME-RealWorld", "record": record},
        )


def _load_hf_dataset(dataset_name: str, local_root: Path) -> Iterator[CanonicalSample]:
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as error:
        raise RuntimeError("Install requirements.txt before loading Hugging Face datasets.") from error

    if local_root.exists() and (local_root / "dataset_dict.json").exists():
        dataset = load_from_disk(local_root)[DATASET_SPLITS[dataset_name]]
    elif (local_root / DATASET_SPLITS[dataset_name] / "state.json").exists():
        dataset = load_from_disk(local_root / DATASET_SPLITS[dataset_name])
    else:
        dataset = load_dataset(DATASET_REPOS[dataset_name], split=DATASET_SPLITS[dataset_name])
    for row_number, row in enumerate(dataset):
        images = _hf_images(row)
        sample_id = _record_id(row, f"{dataset_name}-{row_number}")
        if dataset_name == "xlrs_caption_en":
            question = _first_text(row, ("question",))
            answer_value = _first_value(row, ("caption", "text", "answer", "description"))
            answers = [str(answer) for answer in answer_value] if isinstance(answer_value, list) else [str(answer_value or "")]
            if not question or not answers[0]:
                raise ValueError(f"XLRS caption row {row_number} has no caption field: {row.keys()}")
            yield CanonicalSample(
                id=sample_id,
                task_type="caption",
                images=images,
                prompt=question,
                answers=answers,
                meta={"source": "XLRS-Bench full English caption release", "release_split": "train"},
            )
            continue
        if dataset_name == "xlrs_grounding_en":
            question = _first_text(row, ("question",))
            box = _first_value(row, ("bbox", "box", "boxes", "polygon", "answer"))
            if not question or box is None:
                raise ValueError(f"XLRS grounding row {row_number} is missing text or box: {row.keys()}")
            yield CanonicalSample(
                id=sample_id,
                task_type="grounding",
                images=images,
                prompt=question,
                boxes=[_normalize_box(box)],
                meta={
                    "source": "XLRS-Bench full English grounding release",
                    "release_split": "test",
                    "image_width": float(row["image_width"]),
                    "image_height": float(row["image_height"]),
                },
            )
            continue
        question = _first_text(row, ("question", "text", "query"))
        choices = _choices(row)
        if not question or not isinstance(choices, list):
            raise ValueError(f"XLRS Lite row {row_number} is missing VQA fields: {row.keys()}")
        multi_answer = "overall land use" in str(row).lower()
        yield CanonicalSample(
            id=sample_id,
            task_type="vqa",
            images=images,
            prompt=_multiple_choice_prompt(question, choices, multi_answer, option_count=4),
            answers=[str(_first_value(row, ("answer", "label", "ground_truth")) or "")],
            choices=[str(choice) for choice in choices],
            meta={"source": "XLRS-Bench Lite VQA release", "release_split": "train"},
        )


def _load_levir_cc(root: Path) -> Iterator[CanonicalSample]:
    annotation = _find_file(root, "LevirCCcaptions.json")
    dataset_root = annotation.parent
    image_index = _image_index(dataset_root)
    records = _read_json_records(annotation)
    for record_number, record in enumerate(records):
        split = str(record.get("split", record.get("Split", "test"))).lower()
        if split not in {"test", "testing"}:
            continue
        image_a, image_b = _levir_image_pair(record, dataset_root, image_index)
        captions = _first_value(record, ("captions", "caption", "sentences", "description"))
        if captions is None:
            raise ValueError(f"No LEVIR-CC captions found at index {record_number}.")
        answers = _caption_texts(captions)
        if not all((image_a, image_b, answers[0])):
            raise ValueError(f"Invalid LEVIR-CC test record at index {record_number}.")
        yield CanonicalSample(
            id=_record_id(record, f"levir-cc-{record_number}"),
            task_type="change_caption",
            images=[image_a, image_b],
            prompt=(
                "The first image is before and the second image is after. "
                "Describe only the visible land-cover changes between the two remote sensing images."
            ),
            answers=answers,
            meta={"source": "LEVIR-CC", "split": "test"},
        )


def _find_vrs_annotation(root: Path, task_type: str) -> Path:
    candidates = list(root.rglob("*.json"))
    tokens = {"caption": ("caption", "cap"), "vqa": ("vqa", "qa"), "grounding": ("ref", "ground")}[task_type]
    preferred = [
        path for path in candidates
        if any(token in path.name.lower() for token in tokens)
        and any(token in path.name.lower() for token in ("eval", "val", "test", "validation"))
    ]
    if not preferred:
        preferred = [path for path in candidates if any(token in path.name.lower() for token in tokens)]
    if len(preferred) != 1:
        paths = ", ".join(str(path.relative_to(root)) for path in preferred[:10])
        raise FileNotFoundError(
            f"Could not uniquely select VRSBench {task_type} validation annotation. Candidates: {paths}"
        )
    return preferred[0]


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "annotations", "samples", "items", "images"):
            if isinstance(payload.get(key), list):
                return [item for item in payload[key] if isinstance(item, dict)]
    raise ValueError(f"Unsupported annotation JSON structure: {path}")


def _find_file(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {filename} under {root}, found {len(matches)}.")
    return matches[0]


def _image_index(root: Path) -> dict[str, Path]:
    return {path.name: path for path in root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}}


def _resolve_image(record: dict[str, Any], root: Path, image_index: dict[str, Path]) -> Image.Image:
    value = _first_value(
        record,
        ("image", "Image", "image_path", "img_path", "img", "image_name", "file_name", "filename", "image_id"),
    )
    if value is None:
        raise ValueError(f"No image field found in record keys: {record.keys()}")
    return _open_image(value, root, image_index)


def _resolve_named_image(
    record: dict[str, Any], root: Path, image_index: dict[str, Path], keys: tuple[str, ...]
) -> Image.Image:
    value = _first_value(record, keys)
    if value is None:
        raise ValueError(f"No image field found for {keys} in record keys: {record.keys()}")
    return _open_image(value, root, image_index)


def _open_image(value: Any, root: Path, image_index: dict[str, Path]) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    value_path = Path(str(value))
    candidates = [root / value_path, image_index.get(value_path.name)]
    for candidate in candidates:
        if candidate and candidate.exists():
            with Image.open(candidate) as image:
                return image.convert("RGB")
    raise FileNotFoundError(f"Could not resolve image path {value!r} under {root}")


def _hf_images(row: dict[str, Any]) -> list[Any]:
    values = []
    for key in ("image", "Image", "image_a", "image_b", "before", "after", "images"):
        value = row.get(key)
        if value is None:
            continue
        values.extend(value if isinstance(value, list) else [value])
    if not values:
        raise ValueError(f"No image field found in Hugging Face row keys: {row.keys()}")
    return values


def _choices(row: dict[str, Any]) -> list[Any] | None:
    value = _first_value(row, ("choices", "options", "answer_choices", "multi-choice options"))
    if isinstance(value, list):
        return value
    option_values = [row[key] for key in ("A", "B", "C", "D", "E") if row.get(key) not in (None, "")]
    return option_values or None


def _levir_image_pair(record: dict[str, Any], root: Path, image_index: dict[str, Path]) -> tuple[Image.Image, Image.Image]:
    first = _first_value(record, ("image_A", "A", "image1", "before"))
    second = _first_value(record, ("image_B", "B", "image2", "after"))
    if first is not None and second is not None:
        return _open_image(first, root, image_index), _open_image(second, root, image_index)
    filepath = _first_value(record, ("filepath", "file_name", "filename", "image", "image_path"))
    if filepath is None:
        raise ValueError(f"No LEVIR-CC image-pair field found in record keys: {record.keys()}")
    filename = _first_value(record, ("filename", "file_name"))
    if filename is not None and str(filepath) in {"train", "val", "validation", "test"}:
        image_a_path = root / "images" / str(filepath) / "A" / str(filename)
        image_b_path = root / "images" / str(filepath) / "B" / str(filename)
        if image_a_path.exists() and image_b_path.exists():
            return _open_image(image_a_path, root, image_index), _open_image(image_b_path, root, image_index)
    first_path = str(filepath)
    second_path = first_path.replace("/A/", "/B/").replace("\\A\\", "\\B\\")
    if second_path == first_path:
        raise ValueError(f"Cannot derive LEVIR-CC post-change image from {first_path!r}")
    return _open_image(first_path, root, image_index), _open_image(second_path, root, image_index)


def _caption_texts(value: Any) -> list[str]:
    values = value if isinstance(value, list) else [value]
    texts = []
    for item in values:
        if isinstance(item, dict):
            text = _first_text(item, ("raw", "caption", "text", "sentence"))
        else:
            text = str(item).strip()
        if text:
            texts.append(text)
    if not texts:
        raise ValueError("No non-empty reference caption was found.")
    return texts


def _record_id(record: dict[str, Any], fallback: str) -> str:
    value = _first_value(record, ("id", "ID", "Question_id", "question_id", "ques_id", "uid", "image_id", "imgid"))
    return str(value) if value is not None else fallback


def _first_value(record: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return None


def _first_text(record: dict[str, Any], names: tuple[str, ...]) -> str | None:
    value = _first_value(record, names)
    return str(value).strip() if value is not None else None


def _normalize_box(value: Any) -> list[float]:
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    numbers = [float(number) for number in re.findall(r"-?\d+(?:\.\d+)?", str(value))]
    if len(numbers) == 4:
        box = numbers
    elif len(numbers) == 8:
        xs = numbers[0::2]
        ys = numbers[1::2]
        box = [min(xs), min(ys), max(xs), max(ys)]
    else:
        raise ValueError(f"Expected 4 or 8 bounding-box values, got {value!r}")
    if max(abs(number) for number in box) <= 1:
        return [number * 100 for number in box]
    return box


def _multiple_choice_prompt(question: str, choices: list[Any], allow_multiple: bool, option_count: int) -> str:
    letters = "ABCDE"[:option_count]
    rendered_choices = "\n".join(str(choice) for choice in choices)
    suffix = (
        "There may be more than one correct option. Respond only with the correct letters separated by spaces."
        if allow_multiple
        else f"Respond only with one letter ({', '.join(letters)})."
    )
    return f"{question}\nThe choices are listed below:\n{rendered_choices}\n{suffix}"
