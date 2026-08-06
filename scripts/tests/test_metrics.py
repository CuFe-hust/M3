"""Hand-calculable verification of the self-implemented metric package."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from metrics import bleu, cider, em, iou, levir, meteor, mme, rouge_l  # noqa: E402


# ---------------------------------------------------------------- BLEU
def test_bleu_identical_sentence_is_one():
    scores = bleu.corpus_bleu(["the cat sits on the mat"], [["the cat sits on the mat"]])
    assert scores[1] == pytest.approx(1.0)
    assert scores[4] == pytest.approx(1.0)


def test_bleu_brevity_penalty_reduces_score():
    scores = bleu.corpus_bleu(["cat"], [["the cat sits"]])
    assert scores[1] < 1.0


def test_bleu_modified_precision_clips_overcounted_ngrams():
    # candidate repeats "the" four times; reference has one "the"
    scores = bleu.corpus_bleu(["the the the the"], [["the cat"]])
    assert scores[1] == pytest.approx(1 / 4)


# ---------------------------------------------------------------- ROUGE-L
def test_rouge_l_exact_match_is_one():
    assert rouge_l.rouge_l_fmeasure("the cat sits", "the cat sits") == pytest.approx(1.0)


def test_rouge_l_lcs_counts():
    lcs, ref_len, cand_len = rouge_l.rouge_l_stats("the cat", "a cat sits")
    assert lcs == 1  # "cat"
    assert ref_len == 3
    assert cand_len == 2


def test_rouge_l_disjoint_is_zero():
    assert rouge_l.rouge_l_fmeasure("abc", "xyz") == 0.0


# ---------------------------------------------------------------- METEOR
def test_meteor_exact_match_near_one():
    # Paper formula applies the fragmentation penalty 0.5*(chunks/matches)^3
    # even to exact matches; long sentences drive it towards 1.0.
    assert meteor.meteor_score("the cat sits on the mat", "the cat sits on the mat") == pytest.approx(1.0, abs=0.01)


def test_meteor_partial_match_positive():
    score = meteor.meteor_score("the dog runs", "the cat sits")
    assert 0.0 < score < 1.0


def test_meteor_no_match_zero():
    assert meteor.meteor_score("abc", "xyz") == 0.0


# ---------------------------------------------------------------- CIDEr
def test_cider_identical_is_one():
    # plain CIDEr (no Gaussian weighting): identical texts score 1.0
    assert cider.corpus_cider(
        ["the cat sits on the mat"], [["the cat sits on the mat"]], use_cider_d=False
    ) == pytest.approx(1.0)


def test_cider_d_identical_is_one():
    # CIDEr-D applies the sigma=6 Gaussian kernel per n: identical texts
    # score mean(1, exp(-1/72), exp(-4/72), exp(-9/72)) = 0.95367
    assert cider.corpus_cider(
        ["the cat sits on the mat"], [["the cat sits on the mat"]], use_cider_d=True
    ) == pytest.approx(0.95367, abs=1e-4)


def test_cider_short_sentence_missing_high_order_grams():
    # 2-token sentences have no 3-4 grams: with CIDEr-D weighting
    # mean(1, exp(-1/72), 0, 0) = 0.49655
    assert cider.corpus_cider(["the cat"], [["the cat"]]) == pytest.approx(0.49655, abs=1e-4)


def test_cider_disjoint_is_zero():
    assert cider.corpus_cider(["abc"], [["xyz"]]) == pytest.approx(0.0)


# ---------------------------------------------------------------- IoU
def test_iou_identical_boxes_is_one():
    results = iou.slice_metrics(
        [("s1", [0.0, 0.0, 1.0, 1.0])],
        {"s1": [[0.0, 0.0, 1.0, 1.0]]},
    )
    assert results["all"]["acc_0_5"] == 1.0
    assert results["all"]["acc_0_7"] == 1.0
    assert results["all"]["mean_iou"] == pytest.approx(1.0)


def test_iou_half_overlap_below_threshold():
    results = iou.slice_metrics(
        [("s1", [0.0, 0.0, 1.0, 1.0])],
        {"s1": [[0.5, 0.0, 1.5, 1.0]]},  # IoU = 1/3 < 0.5
    )
    assert results["all"]["acc_0_5"] == 0.0
    assert results["all"]["acc_0_7"] == 0.0
    assert results["all"]["mean_iou"] == pytest.approx(1 / 3)


def test_iou_above_05_below_07():
    # prediction [0,0,10,10], reference [0,0,10,4]: overlap area 40,
    # union 100 -> IoU = 0.4; instead use [0,0,10,6] -> overlap 60, union 100
    results = iou.slice_metrics(
        [("s1", [0.0, 0.0, 10.0, 10.0])],
        {"s1": [[0.0, 0.0, 10.0, 6.0]]},  # IoU = 0.6
    )
    assert results["all"]["acc_0_5"] == 1.0
    assert results["all"]["acc_0_7"] == 0.0
    assert results["all"]["mean_iou"] == pytest.approx(0.6)


def test_iou_slices_respect_sample_ids():
    results = iou.slice_metrics(
        [("s1", [0.0, 0.0, 1.0, 1.0]), ("s2", [0.0, 0.0, 1.0, 1.0])],
        {"s1": [[0.0, 0.0, 1.0, 1.0]], "s2": [[0.0, 0.0, 0.1, 0.1]]},
        slices={"unique": {"s1"}, "non_unique": {"s2"}, "all": {"s1", "s2"}},
    )
    assert results["unique"]["acc_0_5"] == 1.0
    assert results["non_unique"]["acc_0_5"] == 0.0
    assert results["all"]["acc_0_5"] == 0.5


# ---------------------------------------------------------------- EM
def test_choice_match_letter_and_full_answer():
    assert em.choice_match("B", "B")
    assert em.choice_match("b. airport", "B")
    assert not em.choice_match("A", "B")


def test_exact_match_normalizes():
    assert em.exact_match("  The Cat! ", "the cat")
    assert not em.exact_match("cat", "dog")


# ---------------------------------------------------------------- MME
def test_mme_parse_choices():
    assert mme.parse_choice("A") == "A"
    assert mme.parse_choice("The answer is B.") == "B"
    assert mme.parse_choice("C. airport") == "C"
    assert mme.parse_choice("no idea") is None


def test_mme_task_metrics_and_aggregate():
    rows = [
        ("color", "A", "A"),
        ("color", "B", "A"),        # wrong
        ("count", "C", "C"),
        ("count", "garbage", "C"),  # invalid parse
        ("position", "B", "B"),
        ("position", "E", "B"),     # wrong, choice E
    ]
    results = mme.task_metrics(rows)
    assert results["color"]["accuracy"] == 0.5
    assert results["count"]["invalid_parse"] == 0.5
    assert results["position"]["choice_e"] == 0.5
    agg = mme.aggregate(results)
    assert agg["avg"] == pytest.approx((1 + 1 + 1) / 6)  # 3 correct of 6
    assert agg["avg_c"] == pytest.approx((0.5 + 0.5 + 0.5) / 3)


# ---------------------------------------------------------------- LEVIR
def test_levir_templates():
    assert levir.classify_no_change("No change")
    assert levir.classify_no_change("there is no change between images")
    assert levir.classify_no_change("unchanged")
    assert levir.classify_no_change("The scene is same")
    assert not levir.classify_no_change("A building appeared")


def test_levir_discrimination_metrics():
    rows = [
        ("A building appeared", False),  # correct change
        ("No change", True),             # correct no-change
        ("No change", False),            # wrong (missed change)
    ]
    result = levir.discrimination_metrics(rows)
    assert result["all"] == pytest.approx(2 / 3)
    assert result["change"] == pytest.approx(0.5)
    assert result["no_change"] == 1.0


# ---------------------------------------------------------------- score_all end-to-end
def test_score_all_levir_end_to_end(tmp_path):
    from score_all import main

    references = tmp_path / "references.jsonl"
    references.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"sample_id": "1", "caption": "A building appeared.", "change": "change"},
                {"sample_id": "2", "caption": "No visible change.", "change": "no-change"},
                {"sample_id": "3", "caption": "A road is built.", "change": "change"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"sample_id": "1", "status": "ok", "prediction": "A building appeared."},
                {"sample_id": "2", "status": "ok", "prediction": "no change"},
                {"sample_id": "3", "status": "error", "error_code": "inference_error", "error": "boom"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    requests = tmp_path / "requests.jsonl"
    requests.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"sample_id": "1", "task": "caption", "expected_output": "caption"},
                {"sample_id": "2", "task": "caption", "expected_output": "caption"},
                {"sample_id": "3", "task": "caption", "expected_output": "caption"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "score.json"
    assert main(["--dataset", "levir_cc", "--requests", str(requests),
                 "--references", str(references),
                 "--predictions", str(predictions), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["scorer_version"]
    ids = {row["metric_id"] for row in payload["metrics"]}
    assert "levir.caption.all.bleu_4" in ids
    assert "levir.caption.change.bleu_4" in ids
    assert "levir.caption.no_change.cider_d" in ids
    # short captions (< 4 tokens) legitimately have no 4-grams: BLEU-1 must be positive
    all_bleu4 = next(row for row in payload["metrics"] if row["metric_id"] == "levir.caption.all.bleu_4")
    assert all_bleu4["n_samples"] == 2  # only valid (status=ok) predictions
    assert all_bleu4["n_failures"] == 1  # error prediction counts as failure
    all_bleu1 = next(row for row in payload["metrics"] if row["metric_id"] == "levir.caption.all.bleu_1")
    assert all_bleu1["value_canonical"] > 0.0
