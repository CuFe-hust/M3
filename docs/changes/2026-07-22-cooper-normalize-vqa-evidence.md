# Modification Note: Normalize VQA Evidence and Repair Local JSON - 2026-07-22 17:12:38 +08:00

## Modification Time

2026-07-22 17:12:38 +08:00

## Modifier

Cooper (`crj31415926@gmail.com`)

## Modification Goal

Resolve general structured-output failures observed in the first fresh Spark evidence run without discarding boxes, weakening validation, or introducing sample-specific answers.

## Modified Files

- `spacers_agent/schemas.py` and `spacers_agent/vqa_geometry.py`
- `spacers_agent/clients/qwen_transformers.py` and `spacers_agent/cli.py`
- `prompts/general_vqa_v2.md` and `prompts/spatial_v2.md`
- `tests/test_multiagent_vqa_pipeline.py` and `tests/test_vrsbench_vqa_geometry.py`
- `README.md` and `DETAILS.md`

## Core Changes

- Normalized adjacent two-value corner arrays into one `[x1,y1,x2,y2]` box before strict validation.
- Reclassified a lone two-value box as a point, combined simultaneous box/point corner pairs, clamped only boundary coordinates to `0..999`, and recorded every transformation.
- Added one versioned text-only JSON repair attempt to the local Transformers client. The repair call receives no image and preserves both raw responses, validation errors, timing, and cumulative token usage.
- Limited ordinary General VQA to four representative evidence regions while spatial extrema still request all relevant candidates.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

Yes, compatibly. Common Qwen corner-pair output is normalized into the existing strict evidence contract. The checkpoint, processor, tokenizer, and weight loading are unchanged.

## Whether the Configuration Was Changed

No configuration key or default was changed.

## Whether Evaluation Was Affected

No metric, split, reference answer, answer normalization, or DeepSeek payload changed. The failed 7/20 Spark run is not a valid comparison result and will not be reused.

## Whether Deployment Was Affected

No. The change applies to the existing in-process Transformers path and does not introduce vLLM.

## Whether pytest Was Updated

Yes. Tests cover corner-pair normalization, single-point preservation, repair without an image, and repair artifact metadata.

## Whether .gitignore Was Updated

No. The existing ignore rules cover all run artifacts and local configuration.

## Validation Method

- `python -m pytest -q` — 99 passed.
- `python -m compileall -q spacers_agent` — passed.
- `git diff --check` — passed.

## Risks and Follow-up TODOs

- Format repair cannot recover visual facts absent from a truncated raw response; prompt evidence limits reduce that risk.
- Any second parse/schema failure remains a visible sample failure and is not silently skipped.
- A second fresh Spark run with a new run ID is required; the first failed run must remain separate.
