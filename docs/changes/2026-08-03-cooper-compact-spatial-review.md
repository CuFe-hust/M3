# Compact spatial candidate review for Qwen3.5-9B

## Scope

- Add an optional per-call `max_tokens` override to both structured Qwen clients.
- Configure spatial candidate review independently with `models.qwen.spatial_review_max_tokens`, defaulting to 128.
- Replace the active spatial review response with a compact box-only contract: `boxes[[label,x1,y1,x2,y2]]` plus `complete`.
- Deterministically clamp only the common normalized-coordinate drift values `-1` and `1000` to `0` and `999`.
- Recover Qwen's observed missing-inner-bracket candidate sequence locally from complete five-field tuples, recording the recovery marker and avoiding a second model repair call.
- Add conservative first-pass review skipping for extreme vehicle questions only when at least two classed boxes exist, the selected class matches the answer, and the selected centre lies within 40 normalized units of the queried image edge.
- Keep arrangement review unconditional and preserve the existing valid single-target grid skip.

## Compatibility

The final `ExpertResult`, canonical predictions, dataset adapters, evaluation rules, model loading, and weight paths are unchanged. Superseded v2/v3 prompts remain for reproducibility; active review prompts are v4/v5. Candidate review no longer spends tokens reproducing answer text, prose evidence, confidence, points, geometry, or workflow status.

## Verification

- Full suite: 388 passed on Spark in `Cooper_for_qwen9b`.
- VRSBench first-20 run: `qwen35-torch212-compact-recovery-first20-v3-20260803`.
