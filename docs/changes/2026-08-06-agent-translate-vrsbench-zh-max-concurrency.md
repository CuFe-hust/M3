# Modification Note: Raise Translation Concurrency to 2000 - 2026-08-06 15:59:11 CST

## Modification Time

2026-08-06 15:59:11 CST

## Modifier

tRoy <791056216@qq.com>

## Modification Goal

Run the VRSBench Chinese translation at a much higher concurrency, close to the
official DeepSeek `deepseek-v4-flash` account-level concurrency ceiling, while
keeping the existing shared train/val translation cache behavior unchanged.

## Modified Files

- `scripts/translate_vrsbench_zh.py` (cache persistence and SSL handling)
- `.env` (`DEEPSEEK_MAX_CONCURRENCY=2000`)
- `.env.example` (`DEEPSEEK_MAX_CONCURRENCY=2000`)
- `README.md` (concurrency description)
- `DETAILS.md` (concurrency description)
- `docs/changes/2026-08-06-agent-translate-vrsbench-zh-max-concurrency.md` (this note)

## Core Changes

- Confirmed from DeepSeek's official Rate Limit & Isolation page that
  `deepseek-v4-flash` has an account-level concurrency ceiling of 2500
  (concurrency is counted per account, not per API key, and excess requests
  return HTTP 429). `DEEPSEEK_MAX_CONCURRENCY` is set to 2000, keeping 20%
  headroom as requested; the code default remains 4 for environments without
  the variable.
- Coalesced translation-cache disk writes: instead of saving the full JSON
  cache after every completed batch (a bottleneck at high concurrency), the
  cache is now persisted after every 25 completed batches and once more on
  exit. The cache write is atomic (temp file + `os.replace`) and still resumes
  interrupted runs.
- Reused a single SSL context for all HTTP requests instead of rebuilding it
  per batch, reducing per-request overhead when thousands of requests are
  in flight.

## Whether the Canonical Sample Format Was Changed

No. The generated `*_zh.jsonl` files are derived annotation artifacts, not
canonical samples.

## Whether the Model Interface Was Changed

No.

## Whether the Configuration Was Changed

Only the standalone script's `.env`/`.env.example` value
`DEEPSEEK_MAX_CONCURRENCY` changed from 4 to 2000; no repository runtime
configuration semantics changed.

## Whether Evaluation Was Affected

No. The test split is not translated, and no evaluation metric, split, or
reference-answer reader was changed.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

No. The change touches a standalone data-translation script and documentation;
pytest is not installed locally. Validation was done with `compileall`, an
offline monkeypatched integration test, and a live DeepSeek run (see below).

## Whether .gitignore Was Updated

No. `.env` and generated files under `datasets/` are already ignored.

## Validation Method

- `python3 -m compileall -q scripts/translate_vrsbench_zh.py` passed.
- Offline test with a monkeypatched translator passed: 4 batches of 20 strings
  completed with correct cache persistence, a second run made zero API calls
  (resume), and concurrency parsing validated.
- Live DeepSeek run completed with `DEEPSEEK_MAX_CONCURRENCY=2000`: 40,591 new
  strings translated (41,911 total unique strings in the cache, resuming from
  1,320 already cached), and all four outputs were written:
  `VRSBench_train_caption_cleaned_zh.jsonl` (18,237 rows),
  `VRSBench_val_caption_zh.jsonl` (2,027 rows),
  `VRSBench_train_vqa_zh.jsonl` (31,054 rows),
  `VRSBench_val_vqa_zh.jsonl` (3,416 rows).
- Post-run field-level verification passed: row counts match the sources, every
  non-translated field is byte-identical, caption `instruction` is replaced
  with `请描述这张图片的内容。`, every question and translated answer contains
  Chinese, and the cache/meta files are valid. A small number of answers remain
  intentionally unchanged because they are numbers, literal text written on the
  ground in the image (for example `GRANGER`, `SCHOOL`, `JEWELL`), or a
  pre-existing garbage answer in the original data (`Тщ`).

## Risks and Follow-up TODOs

- The official ceiling of 2500 is account-level; if the account's actual plan
  limit is lower, HTTP 429 responses may appear. In that case lower
  `DEEPSEEK_MAX_CONCURRENCY` or raise `--max-retries`.
- The translation cache is model/prompt dependent; delete
  `vrsbench_zh_translations.json` when changing the model or prompt version.
- The original VRSBench data contains one malformed answer (`Тщ`); it is kept
  as-is in the translated output and should be handled later during data
  quality cleanup if needed.
