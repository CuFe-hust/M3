import json

from PIL import Image

from data.loaders import _caption_texts, _load_vrsbench, _normalize_box
from eval.metrics import evaluate_records
from models.qwen3vl import _choice_letter, _extract_boxes


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


def test_vrs_official_evaluation_fields_are_supported(tmp_path) -> None:
    image_path = tmp_path / "P0003_0002.png"
    Image.new("RGB", (8, 8)).save(image_path)
    annotations = {
        "VRSBench_EVAL_Cap.json": [
            {
                "image_id": image_path.name,
                "ground_truth": "Several yellow buses are parked beside a road.",
                "question": "Describe the image in detail",
                "question_id": 0,
                "type": "caption",
            }
        ],
        "VRSBench_EVAL_vqa.json": [
            {
                "image_id": image_path.name,
                "question": "What color are the large vehicles?",
                "ground_truth": "Yellow",
                "question_id": 1,
                "type": "object color",
            }
        ],
        "VRSBench_EVAL_referring.json": [
            {
                "image_id": image_path.name,
                "question": "The large yellow vehicle closest to the green area.",
                "ground_truth": "{<25><40><33><60>}",
                "question_id": 2,
                "type": "ref",
            }
        ],
    }
    for filename, records in annotations.items():
        (tmp_path / filename).write_text(json.dumps(records), encoding="utf-8")

    caption_sample = next(_load_vrsbench(tmp_path, "caption"))
    vqa_sample = next(_load_vrsbench(tmp_path, "vqa"))
    grounding_sample = next(_load_vrsbench(tmp_path, "grounding"))

    assert caption_sample.id == "0"
    assert caption_sample.answers == ["Several yellow buses are parked beside a road."]
    assert vqa_sample.id == "1"
    assert vqa_sample.answers == ["Yellow"]
    assert grounding_sample.id == "2"
    assert grounding_sample.boxes == [[25.0, 40.0, 33.0, 60.0]]

    caption_sample.images[0].close()
    vqa_sample.images[0].close()
    grounding_sample.images[0].close()


def test_baseline_parses_choice_and_grounding_box() -> None:
    assert _choice_letter("The answer is (C).") == "C"
    assert _extract_boxes("[1, 2, 3, 4]") == [[1.0, 2.0, 3.0, 4.0]]
    assert _normalize_box([1, 2, 5, 6]) == [1.0, 2.0, 5.0, 6.0]
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
