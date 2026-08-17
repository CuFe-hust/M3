# 99 — Final Live Gate（Spark 真机验收）Runbook

## 状态：BLOCKED — ENVIRONMENT_BLOCKER（2026-08-09 复核）

最终 live gate 需要 NVIDIA Spark 目标机 + 本地 Qwen checkpoint，当前环境
**不具备**执行条件：

| 前置条件 | 当前状态 |
|---|---|
| NVIDIA GPU | 本机有（nvidia-smi 正常，CUDA 13.3），但这是开发机而非 Spark 目标机 |
| Qwen3-VL-4B-Instruct checkpoint | **缺失**：`~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/` 只有空 `refs/`，无 snapshots/权重 |
| 配置 | `configs/*.yaml` 均为空骨架（默认 `model=qwen3-vl-4b-instruct`、`allow_download=False`，绝不自动下载） |
| 真实数据集（LEVIR-CC/VRSBench/MME-RealWorld/XLRS） | 本地 `dataset/` 不存在 |
| Spark 目标机可达性 | 未提供连接/授权配置 |
| live 授权 | 无 live_qwen/live_deepseek 授权（AGENTS.md：live 测试需显式授权） |

**本 gate 有意推迟**（11J 验收说明）；不得以 fake/mocked 结果冒充真机通过。
权重与数据集就位且获得 Spark 目标机授权后，按下方步骤执行。

## 前置条件（就位后）

1. 本地 Qwen3-VL-4B-Instruct checkpoint（或 Spark 机上等效权重），显式
   `cache_model_id` 逻辑标识；`allow_download` 保持 false（绝不联网下载）。
2. Spark 目标机可访问（显式授权），本机经配置指向该机。
3. 至少一个真实数据集切片（LEVIR-CC 或 VRSBench 小切片）；如需下载，
   只能经显式 `download-data`（唯一自动下载路径），且需用户明确授权。
4. DeepSeek judge 仅在有显式授权 api key 时启用。

## 执行清单

```bash
# 0. 环境与权重自检
python main.py health qwen            # 元数据（不加载）
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 1. 模型加载与单次请求（health live / smoke-qwen）
python main.py health qwen --live     # 恰好一次探测；验证一次加载/GPU 显存
python main.py smoke-qwen --image <real_rs_image> --question "..."

# 2. 手动 ask（VQA / 变化）
python main.py ask --images-dir <dir> --question "Is there a road?"          # auto → VQA
python main.py ask --images-dir <pair_dir> --task change_caption             # 变化任务（t1/t2）

# 3. HTTP 服务
python main.py serve --host 127.0.0.1 --port 8000 &
curl -s localhost:8000/health
curl -s -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"image_dir": "<dir>", "question": "q", "task": "general_vqa"}'

# 4. 计数（真实遥感大图 → 验证切片/平铺）
python main.py count-image --image <large_rs_image> --question "How many vehicles?" \
    --render --evaluate --run-id live-count

# 5. 数据集小切片（每个可用内建数据集各一小片）
python main.py run-dataset --dataset VRSBench --root <data> --split val \
    --task general_vqa,caption --max-samples 8 --run-id live-vrs
python main.py run-dataset --dataset LEVIR-CC --root <data> --split val \
    --task change_caption,change_qa --max-samples 8 --run-id live-levir
# （MME-RealWorld/XLRS-Bench 同理；auto-task 切片：--auto-task --max-samples 8）

# 6. resume（run_request 权威）
python main.py resume-run --run-id live-vrs

# 7. 离线评估（零 Qwen）
python main.py evaluate-run --run-id live-vrs --deepseek        # 仅显式授权 key 时
python main.py judge-vqa-run --run-id live-vrs                  # 同上
python main.py summarize-evaluations --run-id live-vrs

# 8. 报告生成（run-dataset 已自动持久化 bundle）
ls runs/live-vrs/report/           # report.html/json、samples.csv/jsonl、metadata.json、deepseek_audit.jsonl

# 9. 验证单模型加载/GPU 显存
nvidia-smi --query-gpu=memory.used --format=csv   # 多次请求前后对比：一次加载、复用
```

## 通过判据（与任务包一致）

- `health qwen --live` / `smoke-qwen` / 手动 ask（VQA+变化）/ HTTP health+ask
  全部成功；
- `count-image` 在大图（超过 max_pixels_without_tiling）上完成切片计数；
- 每个可用内建数据集小切片运行成功（auto-task 切片走 v3 VisualTaskPlanner）；
- resume 从 run_request 权威重建成功；
- evaluate-run/judge-vqa-run（授权时）/summarize 零 Qwen 成功；
- Reporting 产物完整生成；
- 全程单次模型加载（显存曲线验证）、无隐式网络、无密钥产物；
- 全部产物路径/身份契约与离线门一致。

## 验收输出

```text
FINAL_LIVE_GATE_REPORT
CORE_AGENT_PARITY=PASS|FAIL
DATASET_PARITY=PASS|FAIL
MANUAL_ASK_PARITY=PASS|FAIL
HTTP_SERVICE_PARITY=PASS|FAIL
OPERATIONS_CLI_PARITY=PASS|FAIL
COUNT_IMAGE_PARITY=PASS|FAIL
OFFLINE_EVALUATION_PARITY=PASS|FAIL
STANDARD_EVALUATOR_PARITY=PASS|FAIL
REPORT_EXPORT_PARITY=PASS|FAIL
DATASET_UTILITY_PARITY=PASS|FAIL
LEVIR_CALIBRATION_PARITY=PASS|FAIL
WINDOWS_OFFLINE_PARITY=PASS|FAIL
FULL_FUNCTIONAL_PARITY=PASS
READY_FOR_FINAL_LIVE_GATE
```

当前阻塞态：`FINAL_LIVE_GATE=BLOCKED(ENVIRONMENT_BLOCKER)`——权重/数据集/
Spark 目标机就位后按本 runbook 执行，届时以真实结果更新。
