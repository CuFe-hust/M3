# Experiment Record: Qwen3-VL-8B Merger-Layer LoRA SFT Smoke Test - 2026-08-07

## Time

2026-08-07 14:20-14:30 CST (remote server local time)

## Dataset

Minimal smoke SFT set built from the remote VRSBench JSONL annotations under
`~/M3/data/vrsbench`:

- Train: 2 records (1 English caption from `VRSBench_train_caption_cleaned.jsonl`,
  1 VQA from `VRSBench_train_vqa.jsonl`)
- Validation: 1 record (1 English caption from `VRSBench_val_caption.jsonl`)
- SFT files: `~/M3/outputs/smoke/qwen3vl8b_merger_lora_smoke/data/`

冒烟用最小 SFT 集：2 条训练记录（1 caption + 1 VQA）、1 条验证记录。

## Model

`Qwen3-VL-8B-Instruct` checkpoint hosted at
`~/M3/models/qwen3_vl_8b/weights/` on the remote server
`100.88.222.9:1522` (host `qi2`, NVIDIA RTX 4090 48GB). Pure LoRA is attached
to the eight `nn.Linear` layers inside the four `Qwen3VLVisionPatchMerger`
modules (`visual.merger` + `visual.deepstack_merger_list.0/1/2`).

## Configuration File

No repository config file was used. The run was launched directly with CLI
arguments of `scripts/finetune_qwen3vl_merger_lora.py` (equivalent to the
entry script, with smoke-oriented overrides).

## Run Command

```bash
cd ~/M3
nohup /home/lijia/miniconda3/envs/m3/bin/python scripts/finetune_qwen3vl_merger_lora.py \
  --model_id /home/lijia/M3/models/qwen3_vl_8b/weights \
  --local_files_only True \
  --train_file outputs/smoke/qwen3vl8b_merger_lora_smoke/data/vrsbench_sft_train.json \
  --eval_file outputs/smoke/qwen3vl8b_merger_lora_smoke/data/vrsbench_sft_val.json \
  --image_folder /home/lijia/M3/data/vrsbench \
  --image_min_pixels 262144 --image_max_pixels 524288 --max_seq_length 2048 \
  --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 --lora_bias none \
  --freeze_merger_base True \
  --output_dir outputs/smoke/qwen3vl8b_merger_lora_smoke/run \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 --num_train_epochs 1 --max_steps 2 \
  --learning_rate 1e-4 --weight_decay 0.0 --warmup_ratio 0.0 \
  --lr_scheduler_type cosine --bf16 True --fp16 False \
  --gradient_checkpointing True \
  --logging_steps 1 --save_strategy steps --save_steps 1 --save_total_limit 1 \
  --eval_strategy steps --eval_steps 1 --report_to none \
  --remove_unused_columns False --dataloader_num_workers 0 \
  > outputs/smoke/qwen3vl8b_merger_lora_smoke/run/train_smoke.log 2>&1
```

Environment: Python 3.11.15, torch 2.13.0+cu130, transformers 5.14.1,
peft 0.18.1 (conda env `m3`).

## Metric Results

Smoke metrics only; they do not represent model quality:

| Step | Train loss | Grad norm | LR | Eval loss |
| --- | --- | --- | --- | --- |
| 1 | 3.428 | 0.414 | 1e-4 | 3.274 |
| 2 | 9.934 | 2.023 | 5e-5 | 3.280 |

- `global_step`: 2
- Final `train_loss`: 6.681; `train_runtime`: 2.166 s
- Trainable parameters: 2,293,760 / 8,769,417,456 (0.0262%), i.e. LoRA on the
  eight merger linear layers

## Resource Consumption

- GPU: one NVIDIA RTX 4090 (48GB), bf16, gradient checkpointing enabled,
  batch size 1, max pixels 524288
- The process returned GPU memory to 15 MiB after exit
- Adapter artifact: `adapter_model.safetensors` ~9.2 MB
- Checkpoint: `checkpoint-2/` (adapter + optimizer + scheduler + RNG state)

## Conclusion

The Qwen3-VL-8B merger-layer LoRA SFT script completes an end-to-end smoke run
on the remote 4090 node: model loading from local weights, processor/tokenizer
initialization, LoRA attachment to exactly eight merger linear layers, frozen
base weights, two train steps, two eval steps, checkpoint saving, and final
adapter saving all succeeded. This validates the pipeline; the loss values are
meaningless for quality conclusions.

## Reproducibility Statement

The smoke run was executed once on `100.88.222.9:1522` (`qi2`) with the
commands above and full logs at
`~/M3/outputs/smoke/qwen3vl8b_merger_lora_smoke/run/train_smoke.log` on the
remote server. No source file on the remote or local repository was modified
for this run. No real domestic AI chip validation was performed; only this
4090-node smoke test was completed.
