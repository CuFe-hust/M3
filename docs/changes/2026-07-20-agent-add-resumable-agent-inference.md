# Modification Note: Add Resumable Baseline and Agent Inference - 2026-07-20 13:07:48 +08:00

## Modification Time

2026-07-20 13:07:48 +08:00

## Modifier

roxanne517

## Modification Goal

Make long full-dataset runs recoverable after terminal, network, or process interruption and
provide a repository command for repeating the same evaluation through a fixed LangGraph
workflow without changing the Qwen3-VL model.

## Modified Files

- `agents/__init__.py`
- `agents/langgraph_qwen.py`
- `main.py`
- `requirements.txt`
- `tests/test_resumable_inference.py`
- `README.md`
- `DETAILS.md`
- `docs/changes/2026-07-20-agent-add-resumable-agent-inference.md`

## Core Changes

- Added `agent-infer` with the fixed nodes `read_sample`, `call_qwen`,
  `validate_prediction`, and `save_result`.
- Added mutually exclusive `--resume` and `--overwrite` behavior to baseline and Agent runs.
- Flushes every successful prediction immediately, records per-sample exceptions separately,
  continues after model-level sample failures, and periodically writes atomic metadata.
- Keeps baseline and Agent result files separate and regenerates the MME official-format file
  from the complete saved result so resumed runs do not lose earlier records.
- Records model time and peak allocated CUDA memory without changing prediction content or
  evaluation metrics.

## Whether the Canonical Sample Format Was Changed

No.

## Whether the Model Interface Was Changed

No. The Agent calls the existing `Qwen3VLBaseline.predict` interface.

## Whether the Configuration Was Changed

No configuration key was added or changed. `langgraph==1.2.5` was added as a runtime dependency.

## Whether Evaluation Was Affected

Inference persistence and failure reporting changed, but metric calculations, dataset splits,
reference answers, prompts, and raw model predictions did not change. Baseline and Agent files
remain separate so their outputs can be compared directly.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

Yes. Tests cover resume skipping, visible per-sample failure recording with continuation, and
the fixed LangGraph save node.

## Whether .gitignore Was Updated

No additional update was required for this change. Existing rules already exclude outputs,
logs, local configurations, and caches; `.pytest_cache/` is covered by the companion VRSBench
compatibility change.

## Validation Method

- Ran `python -m pytest -q --basetemp D:\\model_go\\pytest-temp-resumable-20260720-unique1`;
  all 9 tests passed.
- Final compile, CLI-help, diff, and repository-status checks are required after documentation
  is complete.

## Risks and Follow-up TODOs

- A process killed between writing a result line and the next metadata checkpoint can leave
  metadata behind the JSONL file; `--resume` treats the JSONL IDs as authoritative and rewrites
  metadata during the resumed run.
- Loader errors raised before a canonical sample is produced remain fatal because there is no
  reliable sample ID to record. Run `inspect` on a small scope before each full dataset.
- Full-dataset baseline and Agent runs still require server-side validation and are not claimed
  complete by this code change.
