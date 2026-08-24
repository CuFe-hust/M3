# Model-agnostic multimodal SFT interface

The generic post-training path separates three independent concerns:

1. `DataProfile` describes the task and canonical episode contract.
2. `MultimodalModelAdapter` owns model loading, processor encoding, forward
   keys, structure discovery, and model-specific export.
3. `TuningPolicy` selects semantic roles; it never names a model's internal
   projection or merger paths.

The generic core lives under `training/multimodal_sft/`. Built-in adapters are
`qwen3_vl` and `qwen3_5`; `hf_generic_multimodal` is explicit opt-in only and
rejects models until it can prove a safe structure.

Use `--model-adapter auto` only when the model identity matches a registered
adapter. Unknown models fail with `UNSUPPORTED_MODEL_ADAPTER`; they are not
guessed into the closest family.

The legacy Qwen-named scripts remain compatibility entry points. They are not
the extension seam for new families.
