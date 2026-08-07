# Modification Note: Translate VRSBench Train/Val Annotations into Chinese - 2026-08-06 15:31:55 CST

## Modification Time

2026-08-06 15:31:55 CST

## Modifier

tRoy <791056216@qq.com>

## Modification Goal

Provide a simple, resumable script that translates the train and val caption
and VQA annotations into simplified Chinese using the user's Bailian
(DashScope) API. Train captions are translated from the cleaned file only;
caption `instruction` is replaced with one fixed Chinese instruction without an
API call; the test split is not translated.

## Modified Files

- `scripts/translate_vrsbench_zh.py` (new)
- `README.md`
- `DETAILS.md`
- `docs/changes/2026-08-06-agent-translate-vrsbench-zh.md` (this note)

## Core Changes

- Added `scripts/translate_vrsbench_zh.py`, a stdlib-only script that calls the
  DashScope OpenAI-compatible chat completions endpoint
  (`https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`,
  default model `qwen-turbo`).
- The API key is read only from the `DASHSCOPE_API_KEY` environment variable;
  the script fails visibly when it is missing and never writes the key to disk.
- Source/output mapping:
  - `VRSBench_train_caption_cleaned.jsonl` ->
    `VRSBench_train_caption_cleaned_zh.jsonl` (train uses the cleaned file)
  - `VRSBench_val_caption.jsonl` -> `VRSBench_val_caption_zh.jsonl`
  - `VRSBench_train_vqa.jsonl` -> `VRSBench_train_vqa_zh.jsonl`
  - `VRSBench_val_vqa.jsonl` -> `VRSBench_val_vqa_zh.jsonl`
- Caption records replace `instruction` with the fixed Chinese string
  `请描述这张图片的内容。` directly; VQA records keep their original structure
  and translate only `question` and `answer`.
- Unique caption/question/answer strings are collected across all four files,
  translated in batches (default 20, `--batch-size`), and cached in
  `vrsbench_zh_translations.json`; the cache is saved after every batch, so an
  interrupted run resumes by translating only missing strings.
- Outputs are rebuilt from the source rows plus the cache, written atomically,
  and a `vrsbench_zh_translation_meta.json` records the model, base URL, prompt
  version, batch size, cache path, fixed instruction, and row counts (no API
  key value).
- Real data contains 41,911 unique strings to translate (18,233 train cleaned
  captions, 2,027 val captions, 19,767 train VQA strings, 3,045 val VQA
  strings), about 2,096 batches at the default batch size.

## Whether the Canonical Sample Format Was Changed

No. The generated `*_zh.jsonl` files are derived annotation artifacts, not
canonical samples.

## Whether the Model Interface Was Changed

No.

## Whether the Configuration Was Changed

No.

## Whether Evaluation Was Affected

No. The test split is not translated, and no evaluation metric, split, or
reference-answer reader was changed.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

No. The change adds a standalone data-translation script; no repository
interface, adapter, router, metric, or configuration parsing logic was
involved. Existing tests are unaffected.

## Whether .gitignore Was Updated

No. Generated files already live under `datasets/`, which is ignored.

## Validation Method

- `python3 -m compileall -q scripts/translate_vrsbench_zh.py` passed.
- `python3 scripts/translate_vrsbench_zh.py --help` passed.
- Offline integration test with a monkeypatched API transport on a temporary
  fixture passed: four outputs written with correct row counts, caption
  `instruction` replaced, VQA `question`/`answer` translated, all other fields
  preserved, cache persisted, second run made zero API calls, and the missing
  `DASHSCOPE_API_KEY` guard failed visibly.
- Counted unique strings on the real files (41,911) and verified the script
  lists the four expected source files.
- No live DashScope call was made: the local environment has no API key, cloud
  calls require explicit user authorization, and network access is restricted.
- Full pytest could not be run because pytest is not installed in the local
  environment (`No module named pytest`).

## Risks and Follow-up TODOs

- Translation quality depends on the chosen model and prompt; a sample review
  of the generated Chinese files is recommended before training use.
- The translation cache maps English strings to Chinese deterministically, but
  a rerun with a different model or prompt version reuses old cache entries
  unless the cache file is removed; delete the cache when changing the prompt
  version.
- The script is sequential by default; if the ~2,100 calls take too long,
  raising `--batch-size` or adding a concurrency option is a small follow-up.
- Live execution requires the user to export `DASHSCOPE_API_KEY` and run the
  script; generated `*_zh.jsonl` files are git-ignored under `datasets/`.
