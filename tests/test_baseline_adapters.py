import json

from PIL import Image

from data.loaders import _caption_texts, _load_vrsbench, _normalize_box
from eval.metrics import evaluate_records
from data.schema import CanonicalSample
from models.qwen3vl import _choice_letter, _extract_boxes, _grounding_postprocess, _message_content, _official_pixel_boxes, _task_max_new_tokens


def test_vrs_vqa_adapter_expands_question_answer_pairs(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(image_path)
    annotation = [
        {
            "id": "image-1",
            "image": "image.png",
            "qa_pairs": [{"ques_id": "q-1", "question": "How many ships?", "answer": "2"}],
        }
    ]
    (tmp_path / "test_vqa.json").write_text(json.dumps(annotation), encoding="utf-8")

    samples = list(_load_vrsbench(tmp_path, "vqa"))

    assert len(samples) == 1
    assert samples[0].id == "q-1"
    assert samples[0].answers == ["2"]


def test_vrs_vqa_adapter_supports_official_eval_fields(tmp_path) -> None:
    image_path = tmp_path / "P0003_0002.png"
    Image.new("RGB", (8, 8)).save(image_path)
    annotation = [
        {
            "image_id": "P0003_0002.png",
            "question": "What is visible?",
            "ground_truth": "a harbor",
            "question_id": 7,
        }
    ]
    (tmp_path / "VRSBench_EVAL_vqa.json").write_text(json.dumps(annotation), encoding="utf-8")

    samples = list(_load_vrsbench(tmp_path, "vqa"))

    assert len(samples) == 1
    assert samples[0].id == "7"
    assert samples[0].answers == ["a harbor"]
    assert samples[0].images[0].size == (8, 8)


def test_baseline_parses_choice_and_grounding_box() -> None:
    assert _choice_letter("The answer is (C).") == "C"
    assert _extract_boxes("[1, 2, 3, 4]") == [[1.0, 2.0, 3.0, 4.0]]
    vrs_caption = CanonicalSample(id="vrs", task_type="caption", images=["image"], prompt="caption", meta={"source": "VRSBench"})
    xlrs_caption = CanonicalSample(id="xlrs", task_type="caption", images=["image"], prompt="caption", meta={"source": "XLRS-Bench full English caption release"})
    assert _task_max_new_tokens(vrs_caption) == 512
    assert _task_max_new_tokens(xlrs_caption) == 768
    assert _normalize_box([1, 2, 5, 6]) == [1.0, 2.0, 5.0, 6.0]
    assert _official_pixel_boxes([], xlrs_caption) == []
    xlrs_grounding = CanonicalSample(
        id="grounding", task_type="grounding", images=["image"], prompt="official pixel prompt",
        meta={"source": "XLRS-Bench full English grounding release", "image_width": 4000, "image_height": 2000},
    )
    boxes, status, source, conversion = _grounding_postprocess([[100.0, 200.0, 500.0, 500.0]], xlrs_grounding)
    assert boxes == [[10.0, 20.0, 50.0, 50.0]]
    assert status == "converted_model_native"
    assert source == "qwen3vl_normalized_0_1000"
    assert conversion == "value * 100 / 1000"
    assert _official_pixel_boxes([[400.0, 400.0, 500.0, 500.0]], xlrs_grounding) == [[1600.0, 800.0, 2000.0, 1000.0]]
    assert _message_content(xlrs_grounding)[-1]["text"] == "official pixel prompt"
    assert _caption_texts([{"raw": " a new building appears ."}]) == ["a new building appears ."]


def test_exact_match_metric_uses_canonical_answer_field() -> None:
    records = [
        {
            "sample": {"id": "q-1", "task_type": "vqa", "answers": ["B"]},
            "prediction": {"id": "q-1", "task_type": "vqa", "text": "B", "answer": "B"},
        }
    ]

    metrics = evaluate_records(records)

    assert metrics["score"] == 1.0
