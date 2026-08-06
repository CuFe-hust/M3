import json

from PIL import Image

from data.loader import DATASET_TARGETS, load_dataset, load_samples


def _save_image(path, size=(8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "red").save(path)


def _make_vrsbench(root) -> None:
    dataset = root / "vrsbench"
    dataset.mkdir(parents=True)
    _save_image(dataset / "P0003_0002.png")
    annotation = [
        {
            "image_id": "P0003_0002.png",
            "question": f"What is visible in image {index}?",
            "ground_truth": f"answer-{index}",
            "question_id": index,
        }
        for index in range(3)
    ]
    (dataset / "VRSBench_EVAL_vqa.json").write_text(json.dumps(annotation), encoding="utf-8")


def _make_levir_cc(root) -> None:
    dataset = root / "levir_cc" / "Levir-CC-dataset"
    _save_image(dataset / "images" / "test" / "A" / "a.png")
    _save_image(dataset / "images" / "test" / "B" / "b.png")
    annotation = [
        {
            "split": "train",
            "image_A": "images/train/A/a.png",
            "image_B": "images/train/B/b.png",
            "captions": ["train caption"],
        },
        {
            "split": "test",
            "image_A": "images/test/A/a.png",
            "image_B": "images/test/B/b.png",
            "captions": [{"raw": "a new building appears ."}],
        },
    ]
    (dataset / "LevirCCcaptions.json").write_text(json.dumps(annotation), encoding="utf-8")


def _make_mme_real_rs(root) -> None:
    dataset = root / "mme_real_rs"
    _save_image(dataset / "mme.png")
    annotation = [
        {
            "Question_id": "Remote Sensing",
            "Subtask": "Remote Sensing",
            "Text": "How many planes are visible?",
            "Answer choices": ["A", "B", "C", "D", "E"],
            "Ground truth": "C",
            "image": "mme.png",
        },
        {
            "Question_id": "Traffic",
            "Subtask": "Traffic",
            "Text": "How many cars?",
            "Answer choices": ["A", "B", "C", "D", "E"],
            "Ground truth": "A",
            "image": "mme.png",
        },
    ]
    (dataset / "MME_RealWorld.json").write_text(json.dumps(annotation), encoding="utf-8")


def test_load_dataset_streams_vrsbench_vqa_with_limit(tmp_path) -> None:
    _make_vrsbench(tmp_path)

    samples = list(load_dataset("vrsbench_vqa", tmp_path, limit=2))

    assert len(samples) == 2
    assert [sample.id for sample in samples] == ["0", "1"]
    assert all(sample.task_type == "vqa" for sample in samples)
    assert all(isinstance(sample.images[0], Image.Image) for sample in samples)


def test_load_dataset_levir_cc_yields_one_sample_with_two_images(tmp_path) -> None:
    _make_levir_cc(tmp_path)

    samples = list(load_dataset("levir_cc", tmp_path, limit=5))

    assert len(samples) == 1
    assert samples[0].task_type == "change_caption"
    assert len(samples[0].images) == 2
    assert samples[0].answers == ["a new building appears ."]


def test_load_dataset_mme_filters_out_non_remote_sensing_records(tmp_path) -> None:
    _make_mme_real_rs(tmp_path)

    samples = list(load_dataset("mme_real_rs", tmp_path))

    assert len(samples) == 1
    assert samples[0].meta["source"] == "MME-RealWorld"
    assert "Remote Sensing" in samples[0].prompt or "Remote Sensing" in str(samples[0].meta)


def test_load_samples_is_compatible_alias(tmp_path) -> None:
    _make_vrsbench(tmp_path)

    assert load_samples is load_dataset
    assert len(list(load_samples("vrsbench_vqa", tmp_path, limit=1))) == 1


def test_load_dataset_rejects_unknown_target(tmp_path) -> None:
    try:
        load_dataset("not_a_dataset", tmp_path)
    except ValueError as error:
        assert "not_a_dataset" in str(error)
    else:
        raise AssertionError("Expected ValueError for unknown dataset target.")


def test_load_dataset_rejects_invalid_limit(tmp_path) -> None:
    _make_vrsbench(tmp_path)

    try:
        load_dataset("vrsbench_vqa", tmp_path, limit=0)
    except ValueError as error:
        assert "limit" in str(error)
    else:
        raise AssertionError("Expected ValueError for non-positive limit.")


def test_load_dataset_default_root_reads_environment(monkeypatch, tmp_path) -> None:
    _make_vrsbench(tmp_path)
    monkeypatch.setenv("DATASET_ROOT", str(tmp_path))

    samples = list(load_dataset("vrsbench_vqa", limit=1))

    assert len(samples) == 1


def test_dataset_targets_cover_all_evaluation_names() -> None:
    assert set(DATASET_TARGETS) == {
        "vrsbench_caption",
        "vrsbench_vqa",
        "vrsbench_grounding",
        "mme_real_rs",
        "xlrs_caption_en",
        "xlrs_grounding_en",
        "xlrs_vqa_lite",
        "levir_cc",
    }
