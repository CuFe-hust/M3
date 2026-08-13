# Modification Note: Phase 2 Qwen3-VL checkpoint exporter - 2026-08-08

## Modification Time

2026-08-08 (task docs/train/04_EXPORT_QWEN3VL_PHASE2_CHECKPOINT.md, round 4)

## Modifier

troy (AI coding agent, per user instruction "执行，无需请求允许")

## Modification Goal

Implement the independent Phase 2 composite checkpoint exporter
(scripts/export_qwen3vl_phase2_checkpoint.py) that turns the round-3
composite training checkpoint (base + LLM LoRA adapter + fully trained
main/DeepStack merger states) into a complete deployable Qwen3-VL
checkpoint loadable by `AutoModelForImageTextToText.from_pretrained()`.
The exporter only restores, merges, saves and validates the model; it never
reads training data, never trains and never re-interprets data configs.
实现第四轮独立导出器：把第三轮复合训练 checkpoint 导出为完整可部署
checkpoint；只恢复/合并/保存/验证，不训练、不读训练集。

## Modified Files

- `architecture/allowed_python_files.txt` (allowlist; independent commit,
  user pre-authorized with the task instruction): added
  `scripts/export_qwen3vl_phase2_checkpoint.py` and
  `tests/test_export_qwen3vl_phase2_checkpoint.py`.
- `scripts/export_qwen3vl_phase2_checkpoint.py` (new): the exporter.
- `tests/test_export_qwen3vl_phase2_checkpoint.py` (new): 23 tests, fake
  Qwen tree + fake processor + real peft, all CPU/offline.
- `DETAILS.md`: added the script to `scripts/` responsibilities (section 85
  area, after the finetune script paragraph).

## Core Changes

- Fixed order (doc 04 section 5): validate training manifest → load base
  (AutoConfig model_type check first) → enumerate expected merger keys from
  the live base model → three-way strict merger load → attach LLM LoRA via
  PEFT → verify adapter identity/targets → `merge_and_unload()` → audit
  final model → save model+processor (`safe_serialization=True`) → copy
  auxiliary configs (never overwrite) → offline reload validation →
  optional `--verify-forward` → export manifest → atomic publish.
- Pre-weight gates: checkpoint layout completeness, manifest schema
  version, manifest safety scan (secret-like keys + absolute paths under
  path-like keys; `data_sampling.group_key` is not a credential), adapter +
  merger sha256 vs manifest, base logical identity/revision fingerprint vs
  manifest, processor identity (checkpoint vs base vs manifest).
- Merger strict load: three-way comparison of base-model enumeration,
  manifest parameter table and safetensors content; missing/unexpected
  keys, shape mismatch, count mismatch and non-float dtypes fail; explicit
  float conversions to the target dtype are recorded in the export
  manifest.
- LoRA: `PeftModel.from_pretrained`; adapter `base_model_name_or_path` vs
  manifest `model_id_as_given`; peft_type must be LORA; target module set
  must equal the manifest set exactly; every adapter tensor consumed (peft
  on-disk `lora_A.weight` vs in-memory `lora_A.default.weight` normalized);
  no visual/merger LoRA targets; after merge no `lora_*`/`base_model.`
  keys and no lora-named modules; merger tensors byte-equal to the trained
  values.
- Reload validation from the temp dir with `local_files_only=True`:
  model_type, config identity vs base, deepstack count, no LoRA residue,
  minimal image+text chat-template render, weight shards, all file
  checksums. Optional forward check reuses the reloaded model (one copy,
  no double load).
- Atomicity: `output_path` must not exist; everything happens in
  `<name>.export-tmp-<pid>-<n>` next to the final path; success publishes
  with `os.replace`; any failure or `KeyboardInterrupt` (exit 130) cleans
  the temp dir and never creates the final directory; cleanup never touches
  user-given paths.
- Public stderr prints only the stable stage + exception type; the export
  manifest never records secrets, machine absolute paths as logical
  identity, or raw exception dumps.

## Verification (M3 conda env: torch 2.13.0 / transformers 5.14.1 / peft 0.20.0)

- `pytest tests/test_export_qwen3vl_phase2_checkpoint.py`: 23 passed
  (import/`--help` weight-free; missing files / checksum mismatch fail
  before any weight load; output-exists refusal; merger missing/unexpected/
  shape/unsafe-dtype failures; fixed merger-before-LoRA order; LoRA target
  and adapter base identity mismatch; base fingerprint mismatch; no LoRA
  residue after merge; merger values preserved; processor + aux files;
  export manifest checksums truthful; reload failure and interrupt leave no
  final output and no temp leak; default `local_files_only=True`; public
  error carries no simulated secret or absolute path; identity fingerprints
  match the producer module).
- `pytest tests/test_finetune_qwen3vl_phase2.py tests/test_qwen3vl_phase2_data.py
  tests/test_prepare_qwen3vl_phase2_sft.py`: 71 passed (no regression).
- Architecture tests: 41 passed; 1 pre-existing failure
  (test_every_existing_python_file_matches_the_whitelist) caused by the
  user's own uncommitted files (scripts/qwen3vl_lora_cli.py,
  scripts/qwen3vl_lora_remote.py, scripts/prepare_vrsbench_phase2.py,
  tests/test_qwen3vl_lora_cli.py, tests/test_qwen3vl_lora_remote.py) which
  are not in the allowlist yet — not part of this task (same as round 3).
- `python -m compileall -q scripts/export_qwen3vl_phase2_checkpoint.py`:
  OK. `git diff --check`: OK.

## Not Done / Risks

- Real Qwen3-VL-8B export, real offline reload and the real
  `--verify-forward` gate were NOT run: no real Phase 2 composite
  checkpoint exists yet (round-3 training was never run on real hardware;
  only the old merger-LoRA checkpoints exist under outputs/, which lack the
  phase2_training_manifest.json / merger_model.safetensors contract), and
  the base weights directory is not populated. The exporter is validated
  against the producer's exact manifest/checkpoint contract (built with
  scripts/finetune_qwen3vl_phase2.py's own helpers in tests) and with a
  real peft 0.20.0 seam; the real 8B run must be executed separately on
  resource-equipped hardware and reported honestly.
- `_publish` uses `os.replace` on the temp directory; verified on macOS,
  Windows behavior follows the standard library (same-filesystem rename).
