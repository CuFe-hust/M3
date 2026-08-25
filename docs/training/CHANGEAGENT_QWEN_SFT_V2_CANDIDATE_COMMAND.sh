#!/usr/bin/env bash
set -euo pipefail

cd /home/user/cooper/M3

PY=/home/user/miniconda3/envs/Cooper_for_qwen9b/bin/python
CORPUS=/home/user/cooper/posttrain_formal_prep/sft_corpus/v2_agent_result_no_evidence
LEVIROOT=/home/user/cooper/datasets/levir_mci_changehead/source/LEVIR-MCI-dataset
RUN=/home/user/cooper/posttrain_runs/changeagent_qwen35_sft_v2_$(date +%Y%m%d_%H%M%S)

test "$(git branch --show-current)" = change_agent
git merge-base --is-ancestor a917c935369ab01b85aeb0cb2944377d80704834 HEAD
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/change_agent)"
test -z "$(git status --porcelain)"
echo '626958265486660783e7caab336a91113057c59d979b3f57b2e2e0e46f3a18a7  /home/user/cooper/posttrain_formal_prep/sft_corpus/v2_agent_result_no_evidence/train.jsonl' | sha256sum -c -
echo 'd709c8e01f4ced952aaf198efb0652bd72014749913bb1f0fff4f13925428297  /home/user/cooper/posttrain_formal_prep/sft_corpus/v2_agent_result_no_evidence/validation.jsonl' | sha256sum -c -
echo 'b3e891d18a6d564c843af38aac29f5a05f58f947bae38477b240d02d36786587  /home/user/cooper/posttrain_formal_prep/sft_corpus/v2_agent_result_no_evidence/manifest.json' | sha256sum -c -
test ! -e "$RUN"
mkdir -p "$RUN"

{
  date --iso-8601=seconds
  git rev-parse HEAD
  git rev-parse 'HEAD^{tree}'
  git status --short --branch
  "$PY" --version
  "$PY" -m pip freeze
  sha256sum "$CORPUS/train.jsonl" "$CORPUS/validation.jsonl" "$CORPUS/manifest.json" prompts/change_dual_path_v9.md
  nvidia-smi
  free -h
} > "$RUN/run_identity.txt" 2>&1

"$PY" scripts/finetune_multimodal_sft.py \
  --model-id /home/user/models/Qwen3.5-9B \
  --model-adapter qwen3_5 \
  --data-profile change_agent \
  --train-file "$CORPUS/train.jsonl" \
  --validation-manifest "$CORPUS/validation.jsonl" \
  --data-manifest "$CORPUS/manifest.json" \
  --image-root "levir=$LEVIROOT" \
  --prompt-ref change_dual_path_v9 \
  --tuning-policy lora_plus_projector \
  --lora-rank 64 --lora-alpha 128 --lora-dropout 0.05 \
  --lora-lr 1e-4 --connector-lr 1e-5 \
  --weight-decay 0.01 --warmup-ratio 0.03 --max-grad-norm 1.0 \
  --dtype bfloat16 --device cuda:0 \
  --batch-size 1 --gradient-accumulation 8 --max-seq-length 4096 \
  --epochs 1 --save-steps 1000 --logging-steps 10 --save-total-limit 4 \
  --seed 1234 --local-files-only --output-dir "$RUN" \
  2>&1 | tee "$RUN/console.log"
