# Fix shifted vehicle evidence deduplication

Modified: 2026-08-04 00:05 +08:00  
Modifier: Cooper (`crj31415926@gmail.com`)

## Scope

- Treat generic labels containing `vehicle`, including `isolated_vehicle`, `target vehicle`, and `reference-vehicle`, as a generic vehicle role that can match an explicit small/large vehicle class.
- Adjust the shifted-small-box duplicate guard from 0.80 smaller-box coverage / 0.25 normalized centre distance to 0.45 / 0.40.
- Preserve the IoU 0.70 fast path and explicit small-vs-large class conflict protection.
- Prefer the candidate box when it is at least 15% tighter than an otherwise duplicate existing box.

## Evidence

The change is based on direct inspection of VRSBench report items 8, 9, and 16. Each item contained one physical vehicle represented by an independently generated first-pass box and candidate-review box. Their measured IoU values were 0.267, 0.320, and 0.509; the old guards retained all three duplicate pairs. The new conjunction matches these shifted boxes while an adjacent-vehicle regression fixture with only 0.30 smaller-box coverage remains distinct.

## Compatibility and risk

Canonical samples, evaluation metrics, model loading, prompts, and final answer rules are unchanged. The behavioral risk is over-merging two unusually overlapping adjacent vehicles. Requiring both at least 0.45 smaller-box coverage and at most 0.40 normalized centre distance, while keeping conflicting explicit classes distinct, bounds that risk.

## Validation

- `python -m pytest tests/agents/spatial/test_evidence_merge.py -q`: 16 passed.
- `python -m pytest -q`: 393 passed.
- Both commands ran in the Spark `Cooper_for_qwen9b` environment from an isolated temporary Git worktree based on `d0c59df`.
