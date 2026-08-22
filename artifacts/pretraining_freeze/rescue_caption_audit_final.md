# Rescue caption audit — final implementation pass

Date: 2026-08-21

## Scope

This audit records the deterministic implementation and offline regression checks
for the final rescue-caption sanitation change. The requested 11-sample
production rerun was not executed in this workspace because the pinned sample
list and production inference artifact/runner were not available here.

## Deterministic checks

| Check | Result |
|---|---|
| Internal candidate / expert identifiers trigger fallback | PASS |
| `candidate`, `segmenter_`, `proposal`, `expert` are forbidden | PASS |
| `tree`, `rangeland`, `developed_space` trigger fallback | PASS |
| Clean rescue caption is preserved | PASS |
| Fallback count uses confirmed candidates | PASS |
| Fallback location uses candidate edge flags | PASS |
| Fallback performs no additional Qwen call | PASS |
| Binary verdict, boxes, and selected candidates are unchanged by sanitation | PASS |

## Production sample audit

- 11 rescue-trigger sample rerun: NOT RUN
- `final_source = building_rescue` forbidden-term violations: NOT MEASURED
- runtime failures: NOT MEASURED

The production gate must be rerun with the pinned 100-sample configuration
before issuing `FREEZE_CHANGE_INFERENCE`.
