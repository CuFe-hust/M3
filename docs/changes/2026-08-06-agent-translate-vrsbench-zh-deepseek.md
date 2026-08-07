# Modification Note: Switch Translation Backend from DashScope to DeepSeek - 2026-08-06 15:40:14 CST

## Modification Time

2026-08-06 15:40:14 CST

## Modifier

tRoy <791056216@qq.com>

## Modification Goal

Change the VRSBench Chinese translation script to call the DeepSeek API instead
of the DashScope (Bailian) API, reusing the repository's existing
`DEEPSEEK_*` environment-variable conventions.

## Modified Files

- `scripts/translate_vrsbench_zh.py` (backend switch)
- `.env.example`
- `README.md`
- `DETAILS.md`
- `docs/changes/2026-08-06-agent-translate-vrsbench-zh-deepseek.md` (this note)

## Core Changes

- Default endpoint changed to DeepSeek's OpenAI-compatible chat completions URL
  (`https://api.deepseek.com/chat/completions`), and the default model is
  `deepseek-v4-flash`, matching `.env.example`.
- Environment variables renamed from `DASHSCOPE_*` to the repository's
  existing DeepSeek convention: `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL`,
  `DEEPSEEK_BASE_URL`, and `DEEPSEEK_MAX_CONCURRENCY`; CLI flags
  (`--model`, `--base-url`, `--workers`) still take precedence.
- Removed the DashScope-specific `response_format: json_object` request field
  so the script works with more DeepSeek model variants; the existing robust
  JSON-object extractor still parses the response.
- Removed the DashScope entries from `.env.example` and added
  `DEEPSEEK_MAX_CONCURRENCY=4`; concurrency default stays 4 and remains
  configurable.
- No change to batching, caching, resume, caption instruction replacement, or
  output naming; the metadata JSON still records model, base URL, and
  concurrency without any API key value.

## Whether the Canonical Sample Format Was Changed

No. The generated `*_zh.jsonl` files are derived annotation artifacts, not
canonical samples.

## Whether the Model Interface Was Changed

No.

## Whether the Configuration Was Changed

The standalone script reads `DEEPSEEK_*` instead of `DASHSCOPE_*` from
`.env`/environment; no repository runtime configuration semantics changed.

## Whether Evaluation Was Affected

No. The test split is not translated, and no evaluation metric, split, or
reference-answer reader was changed.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

No. The change touches a standalone data-translation script and documentation;
no repository interface, adapter, router, metric, or configuration parsing
logic was involved. Existing tests are unaffected.

## Whether .gitignore Was Updated

No. `.env` and generated files under `datasets/` are already ignored.

## Validation Method

- `python3 -m compileall -q scripts/translate_vrsbench_zh.py` passed.
- `python3 scripts/translate_vrsbench_zh.py --help` passed.
- Offline integration test with a monkeypatched API transport passed using the
  `DEEPSEEK_*` environment variables: `.env` values used, process environment
  overrides `.env`, CLI flags override both, concurrent batches complete with
  correct outputs, the second run makes zero API calls (cache resume), and a
  missing `DEEPSEEK_API_KEY` fails visibly.
- No live DeepSeek call was made; cloud calls require explicit user
  authorization, and network access is restricted.
- Full pytest could not be run because pytest is not installed in the local
  environment (`No module named pytest`).

## Risks and Follow-up TODOs

- The default model `deepseek-v4-flash` must be available on the user's
  DeepSeek account; if not, set `DEEPSEEK_MODEL` in `.env` (for example
  `deepseek-chat`).
- DeepSeek limits are typically expressed as RPM/TPM; if HTTP 429 errors occur,
  lower `DEEPSEEK_MAX_CONCURRENCY` or raise `--max-retries`.
- The translation cache is model/prompt dependent; delete
  `vrsbench_zh_translations.json` when switching models or prompts.
