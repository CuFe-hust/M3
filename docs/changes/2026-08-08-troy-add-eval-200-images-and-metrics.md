# Modification Note: Add 200-Image Eval Subset and Caption/VQA Metrics - 2026-08-08 10:28:26 CST

## Modification Time

2026-08-08 10:28:26 CST

## Modifier

tRoy <791056216@qq.com>

## Modification Goal

Make the Qwen3-VL merger-LoRA evaluation script evaluate the first 200 test
images (with their caption and VQA annotations) by default, and report caption
metrics (BLEU-1..4, METEOR, ROUGE-L, CIDEr) plus VQA overall accuracy and
per-question-type accuracy.
让 Qwen3-VL merger-LoRA 评测脚本默认评估测试集前 200 张图片（含对应 caption
与 VQA 标注），并报告 caption 指标（BLEU-1..4、METEOR、ROUGE-L、CIDEr）以及
VQA 整体准确率与按问题类型准确率。

## Modified Files

- `scripts/evaluate_qwen3vl_merger_lora.py`
- `tests/test_evaluate_qwen3vl_merger_lora.py`
- `README.md`
- `DETAILS.md`
- `requirements.txt`

## Core Changes

- Added `--max-images` (default 200; 0 disables the cap). The test image order
  is defined by `VRSBench_test_caption.jsonl`; the first N unique images are
  selected and used for both caption and VQA records.
  新增 `--max-images`（默认 200，0 表示不设上限）。测试图片顺序由
  `VRSBench_test_caption.jsonl` 定义，取前 N 张不重复图片，并同时用于 caption
  与 VQA 记录。
- Caption metrics: BLEU-1..4 / METEOR / ROUGE-L / CIDEr via `pycocoevalcap`.
  METEOR uses the official Java scorer when `java` is on PATH, otherwise falls
  back to nltk `meteor_score` with WordNet data.
  caption 指标：通过 `pycocoevalcap` 计算 BLEU-1..4 / METEOR / ROUGE-L /
  CIDEr。METEOR 在 PATH 中有 `java` 时使用官方 Java 评分器，否则回退到带
  WordNet 数据的 nltk `meteor_score`。
- VQA metrics: existing overall exact match is kept as `exact_match` and
  duplicated as `all_accuracy`; new `accuracy_by_type` groups exact matches by
  the record `task` field (object_classification / scene_classification /
  object_existence).
  VQA 指标：保留原 `exact_match` 并新增 `all_accuracy`；新增
  `accuracy_by_type`，按记录 `task` 字段（object_classification /
  scene_classification / object_existence）分组统计精确匹配准确率。
- Per-sample meta now records `question_type` and `original_type` from the
  annotation source.
  样本 meta 现在记录标注中的 `question_type` 与 `original_type`。
- Added `nltk>=3.9` to `requirements.txt` as the Python METEOR fallback.
  在 `requirements.txt` 中新增 `nltk>=3.9` 作为 METEOR 的 Python 兜底实现。

## Whether the Canonical Sample Format Was Changed

No. The script still writes the canonical `{"sample", "prediction"}` JSONL;
only `sample.meta` gains two descriptive fields.

## Whether the Model Interface Was Changed

No.

## Whether the Configuration Was Changed

Only the standalone evaluation script CLI gained `--max-images`; no repository
configuration file changed.

## Whether Evaluation Was Affected

Yes, within the standalone merger-LoRA evaluation script: the default test
scope changed from the full test set to the first 200 images, and the summary
now includes caption metrics and VQA per-type accuracy. Existing `exact_match`
semantics are unchanged; historical full-set results remain valid.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

Yes. `tests/test_evaluate_qwen3vl_merger_lora.py` adds coverage for image
selection, task-record filtering, question-type metadata, VQA per-type
accuracy, and caption metric computation (skipped when `pycocoevalcap` is not
installed).

## Whether .gitignore Was Updated

No.

## Validation Method

- `/opt/miniconda3/envs/m3/bin/python -m pytest -q
  tests/test_evaluate_qwen3vl_merger_lora.py` passed (9 tests).
- `python -m compileall -q scripts/evaluate_qwen3vl_merger_lora.py` passed.
- Local smoke of `pycocoevalcap` BLEU/ROUGE-L/CIDEr and the Java METEOR scorer
  passed on fabricated references.
- Real remote evaluation with the first 200 test images was completed on
  2026-08-08 (0 failures; caption BLEU-1..4 0.3429/0.2010/0.1220/0.0772,
  METEOR 0.2636, ROUGE-L 0.2819, CIDEr 0.2435; VQA all accuracy 0.7368 with
  object_classification 0.5829, scene_classification 0.5676,
  object_existence 0.9014). The remote METEOR run used nltk 3.10.2 with
  WordNet data because the node has no Java.

## Risks and Follow-up TODOs

- METEOR on Java-less hosts requires `pip install nltk` plus
  `nltk.download('wordnet')`; the script fails with a clear message when
  neither Java nor nltk is available.
- The 200-image subset changes comparability with any previous full-test
  results; treat 200-image metrics as a separate evaluation scope.
