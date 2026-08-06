from __future__ import annotations

import json
from dataclasses import replace

import pytest

from m3rs_eval.contracts import PredictionRecord, RequestRecord
from m3rs_eval.datasets.levir_cc import classify_no_change, evaluate_levir_alignment
from m3rs_eval.datasets.mme_rs import evaluate_mme_alignment, extract_mme_answer
from m3rs_eval.datasets.vrsbench import (
    evaluate_vrs_alignment,
    inclusive_hbb_iou,
    normalize_vqa_answer,
)
from m3rs_eval.datasets.xlrs_bench import (
    XLRS_L3_TASKS,
    evaluate_xlrs_alignment,
    extract_choice_set,
)
from m3rs_eval.evaluation import (
    EvaluationError,
    MetricContext,
    OfficialMetricScore,
    align_predictions,
    ingest_official_scorer_output,
    merge_metric_records,
)


def context(benchmark_version: str = "fixture-v1") -> MetricContext:
    return MetricContext(
        run_id="run-task-5",
        recorded_at="2026-08-05T12:00:00+08:00",
        protocol_id="official_full_v1",
        benchmark_version=benchmark_version,
        source_log_path="logs/fixture-scorer.json",
    )


def request(
    sample_id: str,
    *,
    dataset: str,
    task: str,
    expected_output: str = "choice",
    language: str | None = None,
    variant: str | None = None,
    choices: tuple[str, ...] | None = None,
) -> RequestRecord:
    payload = {
        "sample_id": sample_id,
        "dataset": dataset,
        "benchmark_version": "fixture-v1",
        "split": "test",
        "task": task,
        "images": ["fixture.png"],
        "prompt": "Answer.",
        "expected_output": expected_output,
        "request_hash": f"hash-{sample_id}",
    }
    if language:
        payload["language"] = language
    if variant:
        payload["variant"] = variant
    if choices is not None:
        payload["choices"] = list(choices)
    return RequestRecord.from_dict(payload)


def text_prediction(sample_id: str, value: str) -> PredictionRecord:
    return PredictionRecord.from_dict(
        {"sample_id": sample_id, "status": "ok", "prediction": value}
    )


def box_prediction(sample_id: str, box: list[float]) -> PredictionRecord:
    return PredictionRecord.from_dict(
        {"sample_id": sample_id, "status": "ok", "boxes": [box]}
    )


def metric_map(records):
    return {record.metric_id: record for record in records}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("A", "A"),
        ("Answer: B.", "A"),
        ("(C)", "C"),
        ("option d", ""),
        ("E. none", "E"),
        ("Because the runway is gray", "B"),
        ("A building", "A"),
        ("A and A", "A"),
        ("A or B", "A"),
        ("this response has more than ten lowercase words and contains no uppercase option label", ""),
    ],
)
def test_mme_answer_extraction_matches_official_compatibility_parser(raw, expected):
    assert extract_mme_answer(raw) == expected


def test_mme_answer_extraction_falls_back_to_choice_text_and_removes_official_prefixes():
    fixture_choices = ("A. gray runway", "B. blue river")
    upstream_index_choices = ("(A) gray runway", "(B) blue river")

    assert extract_mme_answer("gray runway", fixture_choices) == "."
    assert extract_mme_answer("gray runway", upstream_index_choices) == "A"
    assert extract_mme_answer("The answer is B", fixture_choices) == "B"


def test_mme_fixture_style_choice_fallback_is_invalid_and_cannot_score_correct(registry):
    item = request(
        "mme-fallback",
        dataset="mme_rs",
        task="color",
        choices=("A. gray runway", "B. blue river"),
    )
    records = metric_map(
        evaluate_mme_alignment(
            align_predictions([item], [text_prediction(item.sample_id, "gray runway")]),
            {item.sample_id: {"sample_id": item.sample_id, "answer": "A"}},
            registry,
            context(),
        )
    )

    assert records["mme_rs.acc.color"].value_canonical == 0.0
    assert records["mme_rs.invalid_parse_rate"].value_canonical == 1.0
    assert records["mme_rs.acc.color"].n_failures == 1


def test_mme_one_character_matching_choice_is_invalid_instead_of_crashing(registry):
    assert extract_mme_answer("x", ("x",)) == ""

    item = request(
        "mme-short-choice",
        dataset="mme_rs",
        task="color",
        choices=("x",),
    )
    records = metric_map(
        evaluate_mme_alignment(
            align_predictions([item], [text_prediction(item.sample_id, "x")]),
            {item.sample_id: {"sample_id": item.sample_id, "answer": "A"}},
            registry,
            context(),
        )
    )

    assert records["mme_rs.acc.color"].value_canonical == 0.0
    assert records["mme_rs.acc.color"].n_failures == 1
    assert records["mme_rs.invalid_parse_rate"].value_canonical == 1.0


def test_mme_sample_weighted_avg_and_equal_task_avg_c_do_not_alias(registry):
    tasks = ["color", "count", "count", "position", "position", "position"]
    answers = ["A", "A", "B", "B", "B", "B"]
    outputs = ["A", "A", "A", "A", "not parseable", "A"]
    requests = [
        request(f"mme-{index}", dataset="mme_rs", task=task)
        for index, task in enumerate(tasks)
    ]
    alignment = align_predictions(
        requests,
        [text_prediction(item.sample_id, output) for item, output in zip(requests, outputs)],
    )
    references = {
        item.sample_id: {"sample_id": item.sample_id, "answer": answer}
        for item, answer in zip(requests, answers)
    }

    records = metric_map(evaluate_mme_alignment(alignment, references, registry, context()))

    assert records["mme_rs.acc.color"].value_canonical == 1.0
    assert records["mme_rs.acc.count"].value_canonical == 0.5
    assert records["mme_rs.acc.position"].value_canonical == 0.0
    assert records["mme_rs.avg"].value_canonical == pytest.approx(2 / 6)
    assert records["mme_rs.avg_c"].value_canonical == pytest.approx(0.5)
    assert records["mme_rs.invalid_parse_rate"].value_canonical == pytest.approx(1 / 6)
    assert records["mme_rs.avg"].ci95_low is not None


def test_vrs_inclusive_pixel_iou_and_grounding_thresholds(registry):
    assert inclusive_hbb_iou((0, 0, 9, 9), (0, 0, 5, 9)) == pytest.approx(0.6)
    predicted = [[0, 0, 9, 9], [0, 0, 5, 9], [0, 0, 3, 9], [20, 20, 29, 29]]
    slices = ["unique", "non_unique", "unique", "non_unique"]
    requests = [
        request(f"vrs-g-{index}", dataset="vrsbench", task="grounding", expected_output="boxes")
        for index in range(4)
    ]
    alignment = align_predictions(
        requests,
        [box_prediction(item.sample_id, box) for item, box in zip(requests, predicted)],
    )
    references = {
        item.sample_id: {
            "sample_id": item.sample_id,
            "boxes": [[0, 0, 9, 9]],
            "grounding_slice": slice_name,
        }
        for item, slice_name in zip(requests, slices)
    }

    records = metric_map(evaluate_vrs_alignment(alignment, references, registry, context()))

    assert records["vrs.grounding.hbb.all.acc_0_5"].value_canonical == 0.5
    assert records["vrs.grounding.hbb.all.acc_0_7"].value_canonical == 0.25
    assert records["vrs.grounding.hbb.unique.acc_0_5"].value_canonical == 0.5
    assert records["vrs.grounding.hbb.non_unique.acc_0_5"].value_canonical == 0.5


@pytest.mark.parametrize(
    "box",
    [
        (0, 0, float("nan"), 1),
        (0, 0, float("inf"), 1),
        (-1, 0, 1, 1),
        (0, 0, 101, 1),
        (2, 0, 1, 1),
        (0, 0, 1),
    ],
)
def test_vrs_hbb_rejects_nonfinite_inverted_or_out_of_normalized_domain(box):
    with pytest.raises(Exception, match="HBB|coordinates|four"):
        inclusive_hbb_iou(box, (0, 0, 1, 1))


def test_vrs_normalized_vqa_is_deterministic_and_supplemental(registry):
    assert normalize_vqa_answer(" The AIRFIELD. ") == "airfield"
    requests = [request("vrs-vqa-1", dataset="vrsbench", task="vqa")]
    alignment = align_predictions(requests, [text_prediction("vrs-vqa-1", "The airfield.")])
    references = {
        "vrs-vqa-1": {
            "sample_id": "vrs-vqa-1",
            "answer": "airfield",
            "vqa_category": "scene",
        }
    }

    records = metric_map(evaluate_vrs_alignment(alignment, references, registry, context()))

    assert records["vrs.vqa.acc.all"].value_canonical == 1.0
    assert records["vrs.vqa.acc.scene"].value_canonical == 1.0
    assert records["vrs.vqa.acc.all"].provenance == "supplemental"


def test_xlrs_choice_set_and_full_lite_aggregations_remain_distinct(registry):
    assert extract_choice_set("Answer: C, A") == frozenset({"A", "C"})
    assert extract_choice_set("A or B") == frozenset({"A", "B"})

    requests = []
    predictions = []
    references = {}
    l2_tasks = (
        "counting",
        "scene_classification",
        "object_spatial_relationship",
        "object_properties",
        "complex_reasoning",
        "planning",
        "spatiotemporal_reasoning",
        "anomaly_reasoning",
    )
    for index, l2 in enumerate(l2_tasks):
        item = request(
            f"full-en-{index}", dataset="xlrs_bench", task="vqa", language="en", variant="full"
        )
        requests.append(item)
        predictions.append(text_prediction(item.sample_id, "A" if index < 4 else "B"))
        references[item.sample_id] = {"sample_id": item.sample_id, "answer": "A", "l2": l2}

    for index, l3 in enumerate(XLRS_L3_TASKS):
        count = 2 if index == 0 else 1
        for occurrence in range(count):
            item = request(
                f"lite-en-{index}-{occurrence}",
                dataset="xlrs_bench",
                task="vqa",
                language="en",
                variant="lite",
            )
            requests.append(item)
            correct = index == 0
            predictions.append(text_prediction(item.sample_id, "A" if correct else "B"))
            references[item.sample_id] = {"sample_id": item.sample_id, "answer": "A", "l3": l3}

    records = metric_map(
        evaluate_xlrs_alignment(
            align_predictions(requests, predictions), references, registry, context()
        )
    )

    assert records["xlrs.vqa.en.paper_avg_l2"].value_canonical == 0.5
    assert records["xlrs.vqa.en.lite.micro_acc"].value_canonical == pytest.approx(2 / 14)
    assert records["xlrs.vqa.en.lite.macro_l3_acc"].value_canonical == pytest.approx(1 / 13)
    assert records["xlrs.vqa.en.paper_avg_l2"].benchmark_version == "xlrs-bench:full:en"
    assert records["xlrs.vqa.en.lite.micro_acc"].benchmark_version == "xlrs-bench:lite:en"


def test_xlrs_lite_only_rows_never_populate_paper_l3_ids(registry):
    lite_en = request(
        "lite-en-object", dataset="xlrs_bench", task="vqa", language="en", variant="lite"
    )
    lite_zh = request(
        "lite-zh-object", dataset="xlrs_bench", task="vqa", language="zh", variant="lite"
    )
    alignment = align_predictions(
        [lite_en, lite_zh],
        [text_prediction(lite_en.sample_id, "A"), text_prediction(lite_zh.sample_id, "A")],
    )
    references = {
        lite_en.sample_id: {
            "sample_id": lite_en.sample_id,
            "answer": "A",
            "l3": XLRS_L3_TASKS[0],
        },
        lite_zh.sample_id: {
            "sample_id": lite_zh.sample_id,
            "answer": "A",
            "l3": XLRS_L3_TASKS[0],
        },
    }

    records = metric_map(evaluate_xlrs_alignment(alignment, references, registry, context()))
    paper_l3 = [
        record
        for metric_id, record in records.items()
        if metric_id.startswith(("xlrs.vqa.en.l3.", "xlrs.vqa.zh.l3."))
    ]

    assert len(paper_l3) == 26
    assert all(record.availability == "missing" and record.n_samples == 0 for record in paper_l3)
    assert all(record.benchmark_version.startswith("xlrs-bench:full:") for record in paper_l3)
    assert records["xlrs.vqa.en.lite.micro_acc"].value_canonical == 1.0
    assert records["xlrs.vqa.en.lite.micro_acc"].benchmark_version == "xlrs-bench:lite:en"


def test_xlrs_full_rows_populate_en_and_zh_paper_l3_and_emit_all_dimensions(registry):
    full_en = request(
        "full-en-paper", dataset="xlrs_bench", task="vqa", language="en", variant="full"
    )
    full_zh = request(
        "full-zh-paper", dataset="xlrs_bench", task="vqa", language="zh", variant="full"
    )
    references = {
        full_en.sample_id: {
            "sample_id": full_en.sample_id,
            "answer": "A",
            "l2": "counting",
            "l3": XLRS_L3_TASKS[0],
        },
        full_zh.sample_id: {
            "sample_id": full_zh.sample_id,
            "answer": "A",
            "l2": "counting",
            "l3": XLRS_L3_TASKS[1],
        },
    }
    records = metric_map(
        evaluate_xlrs_alignment(
            align_predictions(
                [full_en, full_zh],
                [text_prediction(full_en.sample_id, "A"), text_prediction(full_zh.sample_id, "B")],
            ),
            references,
            registry,
            context(),
        )
    )
    paper_l3_ids = [
        metric_id
        for metric_id in records
        if metric_id.startswith(("xlrs.vqa.en.l3.", "xlrs.vqa.zh.l3."))
    ]

    assert len(paper_l3_ids) == 26
    assert records[f"xlrs.vqa.en.l3.{XLRS_L3_TASKS[0]}.acc"].value_canonical == 1.0
    assert records[f"xlrs.vqa.zh.l3.{XLRS_L3_TASKS[1]}.acc"].value_canonical == 0.0
    assert records[f"xlrs.vqa.en.l3.{XLRS_L3_TASKS[1]}.acc"].availability == "missing"
    assert records[f"xlrs.vqa.en.l3.{XLRS_L3_TASKS[0]}.acc"].benchmark_version == "xlrs-bench:full:en"
    assert records[f"xlrs.vqa.zh.l3.{XLRS_L3_TASKS[1]}.acc"].benchmark_version == "xlrs-bench:full:zh"
    assert records["xlrs.vqa.en.lite.micro_acc"].availability == "missing"
    assert records["xlrs.vqa.en.lite.micro_acc"].benchmark_version == "xlrs-bench:lite:en"


@pytest.mark.parametrize(
    "caption",
    [
        "there is no difference.",
        "the two scenes seem identical",
        "the scene is the same as before",
        "no change has occurred",
        "almost nothing has changed",
    ],
)
def test_levir_five_official_no_change_templates_are_compatible(caption):
    assert classify_no_change(caption)


def test_levir_ingests_caption_scores_and_marks_no_change_cider_not_applicable(registry):
    requests = [
        request("levir-change", dataset="levir_cc", task="caption", expected_output="caption"),
        request("levir-no-change", dataset="levir_cc", task="caption", expected_output="caption"),
    ]
    alignment = align_predictions(
        requests,
        [
            text_prediction("levir-change", "A building appeared."),
            text_prediction("levir-no-change", "There is no difference."),
        ],
    )
    references = {
        "levir-change": {"sample_id": "levir-change", "change": "change"},
        "levir-no-change": {"sample_id": "levir-no-change", "change": "no-change"},
    }
    official = [
        OfficialMetricScore("levir.caption.all.bleu_4", 0.75, 2, 0, "fixture-caption-v1"),
        OfficialMetricScore("levir.caption.change.cider_d", 1.2, 1, 0, "fixture-caption-v1"),
    ]

    records = metric_map(
        evaluate_levir_alignment(alignment, references, registry, context(), official)
    )

    assert records["levir.caption.all.bleu_4"].value_canonical == 0.75
    assert records["levir.caption.all.bleu_4"].provenance == "official"
    assert records["levir.caption.no_change.bleu_4"].availability == "missing"
    assert records["levir.caption.change.rouge_l"].availability == "missing"
    assert records["levir.discrimination.all.accuracy"].value_canonical == 1.0
    no_change_cider = records["levir.caption.no_change.cider_d"]
    assert no_change_cider.value_canonical is None
    assert no_change_cider.availability == "not_applicable"
    assert no_change_cider.ci95_low is None


def test_official_output_ingestion_fails_closed_and_has_no_synthetic_ci(tmp_path, registry):
    valid = tmp_path / "scores.json"
    valid.write_text(
        json.dumps(
            {
                "scorer_version": "official-fixture-v1",
                "metrics": [
                    {
                        "metric_id": "vrs.caption.bleu_4",
                        "value_canonical": 0.4,
                        "n_samples": 2,
                        "n_failures": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    scores = ingest_official_scorer_output(valid, registry)
    assert scores == [OfficialMetricScore("vrs.caption.bleu_4", 0.4, 2, 0, "official-fixture-v1")]

    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"metrics": [{"metric_id": "vrs.caption.bleu_4"}]}', encoding="utf-8")
    with pytest.raises(EvaluationError, match="malformed official scorer output"):
        ingest_official_scorer_output(malformed, registry)


def test_supplemental_records_cannot_overwrite_official_records(registry):
    official_score = OfficialMetricScore("vrs.caption.bleu_4", 0.4, 2, 0, "official-v1")
    official = evaluate_vrs_alignment(
        align_predictions([], []), {}, registry, context(), [official_score]
    )
    supplemental = [replace(official[0], provenance="supplemental", value_canonical=0.9)]

    with pytest.raises(EvaluationError, match="overwrite official"):
        merge_metric_records(official, supplemental)
