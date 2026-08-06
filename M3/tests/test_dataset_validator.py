import json

from PIL import Image

from data.validator import DatasetValidationError, validate_all, validate_dataset


def _save_image(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), "blue").save(path)


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_hf_split(dataset, split) -> None:
    """Create a minimal loadable Hugging Face dataset split directory.
    创建可被 datasets 库加载的最小 HF 数据集切分目录。
    """

    split_dir = dataset / split
    split_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
    except ImportError:
        pa = None
    if pa is None:
        (split_dir / "data-00000-of-00001.arrow").write_bytes(b"arrow")
    else:
        table = pa.table({"image": ["a.png"], "question": ["Q?"]})
        with pa.OSFile(str(split_dir / "data-00000-of-00001.arrow"), "wb") as sink:
            with pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
    _write_json(
        split_dir / "dataset_info.json",
        {
            "builder_name": "parquet",
            "config_name": "default",
            "dataset_name": dataset.name,
            "features": {
                "image": {"dtype": "string", "_type": "Value"},
                "question": {"dtype": "string", "_type": "Value"},
            },
            "num_examples": 1,
            "splits": {
                split: {"name": split, "num_bytes": 10, "num_examples": 1, "dataset_name": dataset.name}
            },
            "version": {"version_str": "0.0.0", "major": 0, "minor": 0, "patch": 0},
        },
    )
    _write_json(
        split_dir / "state.json",
        {
            "_data_files": [{"filename": "data-00000-of-00001.arrow"}],
            "_fingerprint": "test-fingerprint",
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": split,
        },
    )


def _make_vrsbench(root) -> None:
    dataset = root / "vrsbench"
    _save_image(dataset / "image.png")
    for filename in ("VRSBench_EVAL_Cap.json", "VRSBench_EVAL_vqa.json", "VRSBench_EVAL_referring.json"):
        _write_json(dataset / filename, [{"image": "image.png", "id": filename}])


def _make_levir_cc(root) -> None:
    dataset = root / "levir_cc" / "Levir-CC-dataset"
    for split in ("train", "val", "test"):
        for side in ("A", "B"):
            _save_image(dataset / "images" / split / side / f"{side}.png")
    _write_json(dataset / "LevirCCcaptions.json", [{"split": "test", "captions": ["caption"]}])


def _make_mme_real_rs(root) -> None:
    dataset = root / "mme_real_rs"
    _save_image(dataset / "remote_sensing" / "image.png")
    _write_json(
        dataset / "MME_RealWorld.json",
        [{"Question_id": "Remote Sensing", "Subtask": "Remote Sensing", "Text": "Q?", "Answer choices": ["A"]}],
    )


def _make_xlrs_bench(root) -> None:
    dataset = root / "xlrs_bench"
    lite = dataset / "XLRS-Bench-lite"
    _write_hf_split(lite, "train")
    _write_json(lite / "dataset_dict.json", {"splits": ["train"]})
    _write_hf_split(dataset / "XLRS-Bench_caption_en", "train")
    _write_hf_split(dataset / "XLRS-Bench_visual_grounding_en", "test")


def test_vrsbench_valid_structure_passes(tmp_path) -> None:
    _make_vrsbench(tmp_path)

    report = validate_dataset("vrsbench", tmp_path)

    assert report["ok"] is True
    assert len(report["annotation_counts"]) == 3
    assert report["image_count"] == 1


def test_vrsbench_missing_annotation_fails(tmp_path) -> None:
    _make_vrsbench(tmp_path)
    (tmp_path / "vrsbench" / "VRSBench_EVAL_Cap.json").unlink()

    try:
        validate_dataset("vrsbench", tmp_path)
    except DatasetValidationError as error:
        assert "VRSBench_EVAL_Cap.json" in str(error)
    else:
        raise AssertionError("Expected DatasetValidationError.")


def test_levir_cc_valid_structure_passes(tmp_path) -> None:
    _make_levir_cc(tmp_path)

    report = validate_dataset("levir_cc", tmp_path)

    assert report["ok"] is True
    assert report["annotation_count"] == 1
    assert report["image_counts"]["test/A"] == 1


def test_levir_cc_missing_image_side_fails(tmp_path) -> None:
    _make_levir_cc(tmp_path)
    missing_side = tmp_path / "levir_cc" / "Levir-CC-dataset" / "images" / "test" / "B"
    (missing_side / "B.png").unlink()
    missing_side.rmdir()

    try:
        validate_dataset("levir_cc", tmp_path)
    except DatasetValidationError as error:
        assert "test/B" in str(error)
    else:
        raise AssertionError("Expected DatasetValidationError.")


def test_mme_real_rs_valid_structure_passes(tmp_path) -> None:
    _make_mme_real_rs(tmp_path)

    report = validate_dataset("mme_real_rs", tmp_path)

    assert report["ok"] is True
    assert report["remote_sensing_record_count"] == 1


def test_mme_real_rs_without_remote_sensing_records_fails(tmp_path) -> None:
    dataset = tmp_path / "mme_real_rs"
    _save_image(dataset / "remote_sensing" / "image.png")
    _write_json(dataset / "MME_RealWorld.json", [{"Question_id": "Traffic", "Text": "Q?", "Answer choices": ["A"]}])

    try:
        validate_dataset("mme_real_rs", tmp_path)
    except DatasetValidationError as error:
        assert "No Remote Sensing records" in str(error)
    else:
        raise AssertionError("Expected DatasetValidationError.")


def test_xlrs_bench_valid_structure_passes(tmp_path) -> None:
    _make_xlrs_bench(tmp_path)

    report = validate_dataset("xlrs_bench", tmp_path)

    assert report["ok"] is True
    assert report["arrow_file_count"] == 3
    if isinstance(report["row_check"], dict):
        assert set(report["row_check"].values()) == {"ok"}
    else:
        assert report["row_check"].startswith("not_checked")


def test_xlrs_bench_missing_release_fails(tmp_path) -> None:
    _make_xlrs_bench(tmp_path)
    (tmp_path / "xlrs_bench" / "XLRS-Bench_caption_en").rename(tmp_path / "xlrs_bench" / "XLRS-Bench_caption_zh")

    try:
        validate_dataset("xlrs_bench", tmp_path)
    except DatasetValidationError as error:
        assert "XLRS-Bench_caption_en" in str(error)
    else:
        raise AssertionError("Expected DatasetValidationError.")


def test_validate_all_aggregates_failures(tmp_path) -> None:
    _make_vrsbench(tmp_path)

    reports = validate_all(tmp_path)

    assert reports["ok"] is False
    assert set(reports["failed"]) == {"xlrs_bench", "levir_cc", "mme_real_rs"}
    assert reports["vrsbench"]["ok"] is True
