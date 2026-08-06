# Modification Note: Preserve VRSBench Quantity Targets - 2026-08-04

## Modification Goal

Prevent non-vehicle VRSBench quantity questions from being silently converted to the generic `vehicle` target.

## Changes

- `vrsbench_count_target()` now returns a fixed target only for explicit `vehicle`, `small vehicle`, and `large vehicle` questions.
- `CountingAgent` uses the existing target parser for every other VRSBench quantity noun.
- The selector routes those generic VRSBench quantity targets through a compatible YOLO detector when available, otherwise `qwen_point`; the vehicle-only proposal backend remains vehicle-only.

## Compatibility

Explicit vehicle questions retain their existing deterministic ontology and VRSBench proposal/localizer route. Canonical samples, splits, metrics, and model-loading interfaces are unchanged.

## Validation

Focused VRSBench target-selection and geometry tests cover explicit vehicles, non-vehicle nouns, generic backend routing, and the parser fallback.
