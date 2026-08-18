# M3

面向太空智算的多模态遥感大模型应用探索。

M3 当前采用分层的多模态遥感 Agent 架构，围绕 **统一样本契约、任务解析与确定性路由、领域 Agent、可恢复运行、确定性评测、可选 Judge 和只读报告** 组织代码。主流程默认面向本地 Qwen 多模态模型，支持数据集批量评测、手动问答、计数、变化理解、空间关系、Grounding、Caption、VQA，以及运行后的评测与报告导出。

仓库主线 `main` 是新的长期架构。迁移期名称 `new_structure` 仅用于历史文档；旧
`try_yolo` 分支仅作为迁移和行为对齐参考，不再作为运行时依赖。

---

## 1. 主要能力

当前公开任务包括：

| Task | 说明 | 主要 Agent |
|---|---|---|
| `counting` | 通用目标计数 | CountingAgent |
| `fine_grained_counting` | 细粒度目标计数 | CountingAgent |
| `change_caption` | 双时相变化描述 | ChangeAgent |
| `change_qa` | 双时相变化问答 | ChangeAgent |
| `grounding` | 文本目标定位 | GroundingAgent |
| `spatial_relation` | 空间关系理解 | GeneralVQAAgent |
| `scene_classification` | 遥感场景分类 | GeneralVQAAgent |
| `general_vqa` | 通用遥感 VQA | GeneralVQAAgent |
| `caption` | 遥感图像描述 | CaptionAgent |
| `multiple_choice_vqa` | 多选 VQA | GeneralVQAAgent |

内建数据集适配器：

- **VRSBench**
- **LEVIR-CC**
- **MME-RealWorld**
- **XLRS-Bench**
- **XLRS-Bench-lite**

另外支持显式 manifest 驱动的 `SampleDraft` 读取路径，用于没有标准逐样本 task 字段的数据。

---

## 2. 当前架构

主链路：

```text
Dataset / Local Images
        |
        v
data/
  Adapter -> UnifiedSample
          -> SampleDraft
                |
                v
        VisualTaskPlanner (images + raw question)
                |
                v
        materialize UnifiedSample + deterministic TaskRouter
                |
                v
routing/
  deterministic TaskRouter
                |
                v
agents/
  Counting / Change / Grounding / Caption / General VQA
                |
                v
workflows/
  SampleRunner -> DatasetRunner
                |
                v
evaluation/
  deterministic metrics + optional DeepSeek Judge
                |
                v
persisted run artifacts
                |
                v
reporting/
  read-only report / audit / export
                |
                v
application/
  composition root and use cases
                |
                v
main.py
  sole public CLI surface
```

几个最重要的边界：

- `data.schema.UnifiedSample` 是内部统一样本契约。
- 新鲜推理统一先调用一次 `VisualTaskPlanner`：第一次 user content 只有按序图像与原始问题，输出 task 与可选视觉辅助计划。
- `UnifiedSample` 在规划后物化；`TaskRouter` 只回答“已知 task 交给哪个 Agent”，不读 question、不调用模型。
- Router 是同步、确定性、无模型调用的。
- Agent 依赖模型协议，不自行创建具体 Qwen 客户端。
- `application/` 是唯一 composition root。
- `reporting/` 只读取持久化结果，不重新推理。
- `evaluation` 中的 Judge 永远不能覆盖确定性指标。
- 新架构不依赖旧 `spacers_agent/` 和 `eval/`。

更完整的内部说明见 [`DETAILS.md`](DETAILS.md)，编码规则见 [`AGENTS.md`](AGENTS.md)。

---

## 3. 环境要求

当前项目配置以 `pyproject.toml` 为准。

最低要求：

```text
Python >= 3.11
```

基础依赖：

- Pydantic 2
- Pillow
- PyYAML
- typing-extensions

推荐先建立独立 Python 环境，再安装项目：

```bash
python -m pip install -U pip
python -m pip install -e .
```

开发和测试：

```bash
python -m pip install -e ".[dev]"
```

变化检测相关可选依赖：

```bash
python -m pip install -e ".[change]"
```

Change V3 的 SegFormer auxiliary path 使用独立重依赖 extra；默认仍保持轻量、纯数学
测试和 core import 不需要它：

```bash
python -m pip install -e ".[dev,change]"          # offline/core tests
python -m pip install -e ".[change-semantic]"    # explicit SegFormer runtime
```

迁移/部分离线工具需要 NumPy 时：

```bash
python -m pip install -e ".[migration]"
```

> CUDA、PyTorch、Transformers、ONNX Runtime、YOLO detector runtime 等与具体模型/硬件相关的依赖需要根据部署环境单独准备。仓库不会在普通 import 或离线测试时强制加载全部模型和 detector 运行时。

---

## 4. 本地 Qwen 模型

主流程模型通过：

```python
models.entry.create_model(name, ...)
```

统一构造。

当前模型入口：

```text
qwen_transformers
qwen3_vl_baseline
qwen3_5_transformers
```

Agent/Workflow 不直接 import 具体 Qwen 实现；具体模型只在 `application` composition root 选择和创建。

### 本地 checkpoint

默认 Qwen 配置使用逻辑模型名。实际部署时可以通过 YAML 或环境变量指定本地 checkpoint。

例如：

```yaml
models:
  qwen:
    model: /path/to/Qwen3-VL-4B-Instruct
    cache_model_id: qwen3-vl-4b-instruct-local
    allow_download: false
    device_map: auto
    dtype: auto
```

如果 `model` 是本地绝对路径，必须同时提供与机器路径无关的：

```text
cache_model_id
```

它用于 request hash、cache identity 和 trace，避免把本机 checkpoint 绝对路径变成逻辑模型身份。

也可以使用：

```bash
export QWEN_MODEL=/path/to/Qwen3-VL-4B-Instruct
```

覆盖模型路径。

默认：

```text
allow_download = false
```

因此普通运行不会把“本地模型缺失”自动转换为 Hugging Face 下载。

### 本地专家模型

仓库声明三份本地专家模型资产：

```text
models/segformer_mitb2_isaid/model.safetensors
models/segformer_mitb2_oem/model.safetensors
models/yolo_obb/yolov5m_obb_csl_dotav20.onnx
```

大权重通过 Git LFS 或本地外部存储管理；Git 对象不得直接包含大 binary。工作树中的
LFS 文件可以是已 hydrated binary，部署也可以提供 catalog 指向的本地资产。代码会在
加载前区分文件缺失、Git LFS pointer 和 SHA256 不匹配。小型 `config.json`、`classes.json`、
`metrics.json` 可以版本化。资产摘要和逻辑 ID 见
[`models/MODELS.md`](models/MODELS.md)。

SegFormer 可选依赖：

```bash
python -m pip install -e ".[segformer]"
```

Change V3 消融配置位于 `configs/change_ablations/`。这些文件是可直接传给
`--config` 的 partial YAML，覆盖 legacy、registration、三路融合与多尺度特征；
旧的 low+semantic、low+feature 配置仍保留作为对照。Change V3 配置示例位于
`configs/change_v3.example.yaml`；这些配置不扩展 evaluation public contract。

调用层：

```python
from models.segformer_transformers import SegFormerRuntime

runtime = SegFormerRuntime(settings.models.segformer_isaid)
result = runtime.predict(image)
```

该 runtime 封装本地加载、processor、device/dtype、预处理、logits 上采样、
argmax 和类别映射。iSAID 必须读取经训练 mask 验证的 `classes.json`；OEM
源资产只有 `LABEL_0..8` 占位标签，代码不会猜测另一套类别顺序。

---

## 5. 配置

应用配置由 `application.settings.AppSettings` 管理。

主要配置组：

```text
models
counting
runs
router
paths
backend
agents
```

运行时加载顺序：

```text
built-in defaults
    -> optional YAML
    -> supported environment overrides
```

公共入口支持：

```bash
python main.py --config /path/to/local.yaml <command> ...
```

不传 `--config` 时使用代码中的默认设置。

当前支持的普通环境变量覆盖包括：

```text
QWEN_MODEL
SEGFORMER_ISAID_MODEL
SEGFORMER_OEM_MODEL
DEEPSEEK_BASE_URL
DEEPSEEK_MODEL
DATASET_ROOT
OUTPUT_ROOT
```

DeepSeek API key 的**值**不进入 AppSettings；配置只声明环境变量名，实际 secret 由 composition root 在需要 Judge 时读取并注入。

---

## 6. 数据准备

### 6.1 查看内建数据集

```bash
python main.py list-datasets
```

### 6.2 显式下载官方数据

项目默认不允许 loader/adapter 隐式联网。

需要下载数据时使用明确命令：

```bash
python main.py download-data \
  --root /data/m3 \
  --datasets vrsbench
```

可以一次指定多个 dataset key：

```bash
python main.py download-data \
  --root /data/m3 \
  --datasets vrsbench levir_cc
```

下载逻辑位于：

```text
data/downloader.py
```

下载是显式行为，普通 `run-dataset` 不会因为数据不存在而偷偷联网。

### 6.3 数据集只读审计

在正式运行之前可以检查数据根目录：

```bash
python main.py inspect-data \
  --root /data/m3/VRSBench \
  --scan-mode quick
```

完整扫描：

```bash
python main.py inspect-data \
  --root /data/m3/VRSBench \
  --scan-mode full \
  --output outputs/vrsbench-audit.json
```

Adapter 对源数据保持只读。

---

## 7. 统一样本契约

跨 Adapter、Workflow、Router、Agent 和 Evaluation 的内部样本使用：

```text
data.schema.UnifiedSample
```

核心字段：

```text
sample_id
dataset
split
task
images
question
ground_truth
metadata
normalization
```

图片路径使用 dataset-root-relative 表示，不把机器绝对路径作为样本身份。

双时相任务：

```text
change_caption
change_qa
```

使用有序角色：

```text
t1 -> t2 -> context...
```

其他任务使用：

```text
image -> context...
```

对于没有明确逐样本 task 的数据，先生成：

```text
SampleDraft
```

再通过：

```text
VisualTaskPlanner -> materialize_sample -> UnifiedSample
```

旧 resolver/联合规划器已从当前 runtime 删除；历史 run 只通过 reporting 的只读
兼容 seam 审计，不参与新鲜推理。

---

## 8. VisualTaskPlanner 与 Router

每条新鲜样本（包括手动 ask 的显式/auto task，以及 dataset 的
explicit/default/auto 模式）都经过：

```text
normalized image previews + raw question
    -> one VisualTaskPlanner Qwen call
    -> materialize/rebuild UnifiedSample
    -> deterministic TaskRouter
```

规划输出版本为 `visual-task-plan-v5`。显式 CLI/dataset task 只作审计，不发送给
第一次规划调用，也不覆盖规划结果。规划预览最长边为 1080；显式区域输出严格整数
`0..999` `xyxy`，运行时按最长边向上量化到 1024 整数倍，再直接截断越界的理想正方形。
最终裁片可以是长方形；direct、VQA evidence 与 Grounding 共享同一个实际裁片。

`SampleDraft` 路径也由同一规划调用物化，不再单独走文本任务解析路径。

历史任务解析规则仅可在迁移文档中审计，不是新鲜运行契约。

TaskRouter 本身：

```text
no question reading
no model call
no unknown-task guessing
```

---

## 9. 快速开始

### 9.1 查看所有命令

```bash
python main.py --help
```

查看某个命令：

```bash
python main.py run-dataset --help
python main.py ask --help
python main.py count-image --help
```

### 9.2 本地 HTTP 服务

无子命令时默认启动：

```bash
python main.py
```

等价于：

```bash
python main.py serve --host 127.0.0.1 --port 8000
```

如果使用自定义配置：

```bash
python main.py --config /path/to/local.yaml serve
```

当前 HTTP surface：

```text
GET  /health
POST /ask
```

服务进程只组装一次 Runtime；请求 handler 不重复加载 Qwen。

默认监听 `127.0.0.1`。不要在没有额外安全措施时暴露到不受信任网络。

---

## 10. 手动 Ask

对本地图片目录执行一次任务：

```bash
python main.py --config /path/to/local.yaml ask \
  --images-dir /data/question-001 \
  --question "图中有多少架飞机？" \
  --task counting
```

自动判断任务：

```bash
python main.py --config /path/to/local.yaml ask \
  --images-dir /data/question-002 \
  --question "图中主要是什么场景？" \
  --task auto
```

Caption 可以使用空问题：

```bash
python main.py --config /path/to/local.yaml ask \
  --images-dir /data/question-003 \
  --task caption
```

输出到文件：

```bash
python main.py --config /path/to/local.yaml ask \
  --images-dir /data/question-003 \
  --task caption \
  --output outputs/manual-answer.json
```

手动 `ask` 是单请求路径，不等同于完整 DatasetRunner benchmark：它不会自动生成完整数据集评测和报告流程。

---

## 11. 数据集运行

基本形式：

```bash
python main.py --config /path/to/local.yaml run-dataset \
  --dataset VRSBench \
  --root /data/VRSBench \
  --split validation
```

如果没有 `--task`，默认使用 Adapter 声明的支持任务集合。

只运行某个任务：

```bash
python main.py --config /path/to/local.yaml run-dataset \
  --dataset VRSBench \
  --root /data/VRSBench \
  --split validation \
  --task general_vqa
```

多个任务使用逗号分隔：

```bash
python main.py --config /path/to/local.yaml run-dataset \
  --dataset VRSBench \
  --root /data/VRSBench \
  --split validation \
  --task general_vqa,caption,grounding
```

限制样本数做 smoke test：

```bash
python main.py --config /path/to/local.yaml run-dataset \
  --dataset VRSBench \
  --root /data/VRSBench \
  --split validation \
  --task general_vqa \
  --limit 20
```

指定 run id：

```bash
python main.py --config /path/to/local.yaml run-dataset \
  --dataset LEVIR-CC \
  --root /data/LEVIR-CC \
  --split test \
  --task change_caption \
  --run-id levir-change-caption-v1
```

### VRSBench 三任务系统测试

官方 validation 发布按本项目约定作为 `val`/test 输入时，可直接运行：

```bash
bash scripts/run_vrsbench_system_test.sh
```

脚本通过公开 `run-dataset` 输入入口和 `VRSBenchAdapter` 依次执行
`caption,grounding,general_vqa`，并在 `outputs/runs/<run_id>/` 生成：

```text
predictions.jsonl
command_result.json
report/report.json
report/samples.jsonl
report/report.html
```

可用环境变量 `PYTHON`、`M3_CONFIG`、`VRSBENCH_ROOT`、`VRSBENCH_RUN_ID`、
`VRSBENCH_LIMIT`、`VRSBENCH_SAMPLE_CONCURRENCY`、`VRSBENCH_SHARD_INDEX` 与
`VRSBENCH_SHARD_COUNT` 覆盖运行参数。HTML 样本页展示 grounding 的原图叠加、
顶层模块执行路径和已持久化的模型 raw/parsed 输出。

---

## 12. Auto-task Dataset Mode

对于显式使用 `SampleDraft` 的数据路径：

```bash
python main.py --config /path/to/local.yaml run-dataset \
  --dataset <dataset-name> \
  --root /data/<dataset> \
  --split test \
  --auto-task
```

三种模式语义不同：

```text
no --task
    -> adapter default tasks

--task task1,task2
    -> explicit tasks

--auto-task
    -> per-sample VisualTaskPlanner
```

无论是哪种模式，新鲜样本都会调用一次视觉规划器；`--task` 只保留为审计输入。

---

## 13. Sharding 与并发

分片：

```bash
python main.py run-dataset \
  --dataset VRSBench \
  --root /data/VRSBench \
  --split validation \
  --task general_vqa \
  --shard-index 0 \
  --shard-count 4
```

`--num-shards` 是 `--shard-count` 的别名。

当前分片使用稳定 SHA-256 逻辑，不使用 Python 随机 hash。

单进程 asyncio 并发：

```bash
python main.py run-dataset \
  --dataset VRSBench \
  --root /data/VRSBench \
  --split validation \
  --task general_vqa \
  --sample-concurrency 4
```

当前 artifact JSONL 层只承诺**同一 Python 进程内**并发写入安全，不宣称多个独立进程可以同时追加同一个 run。

---

## 14. Resume

首次运行：

```bash
python main.py --config /path/to/local.yaml run-dataset \
  --dataset VRSBench \
  --root /data/VRSBench \
  --split validation \
  --task general_vqa \
  --run-id vrsbench-vqa-v1
```

恢复：

```bash
python main.py --config /path/to/local.yaml resume-run \
  --run-id vrsbench-vqa-v1
```

也可以通过 `run-dataset --resume` 进入同一恢复语义。

Resume 的具体原始调用由：

```text
runs/<run_id>/run_request.json
```

持久化。

系统不会因为当前 YAML、CLI 默认值发生变化就静默改变原运行的 task mode、dataset root、judge policy、sample selection 等关键行为。

---

## 15. 确定性评测

`run-dataset` 默认：

```text
evaluate = true
judge_policy = none
```

即默认做支持的确定性评测，但**不会默认调用 DeepSeek**。

当前主要 deterministic metric family：

### Counting

```text
predicted_count
gold_count
exact_match
absolute_error
relative_error
smooth_error_score
```

### General VQA

```text
exact_match
```

### Grounding

```text
IoU
IoU@0.5
```

当前内建 Grounding deterministic path 对坐标契约严格 fail-closed；不能把未知坐标系、source-pixel 坐标或 polygon 静默当成统一 xyxy。

### Caption

逐样本保存：

```text
candidate
references
```

语料级 BLEU / METEOR / ROUGE / CIDEr 由 aggregate/标准 evaluator 路径负责，不把逐样本记录伪装成完整 corpus metric。
当前运行未配置经批准的 CHAIR2 scorer；系统会在 `command_result.json` 与
HTML caption metrics 区域明确标记 `CHAIR2` 未计算，不会伪造分数。METEOR
需要 Java；缺少 Java 时只将 METEOR 标为未计算，BLEU/ROUGE_L/CIDEr 仍独立
报告。

---

## 16. DeepSeek Judge

Judge 是可选审计层，不取代 deterministic metrics。

启用 Judge 前在运行环境设置：

```bash
export DEEPSEEK_API_KEY=...
```

数据集运行：

```bash
python main.py --config /path/to/local.yaml run-dataset \
  --dataset VRSBench \
  --root /data/VRSBench \
  --split validation \
  --task general_vqa \
  --judge-policy all
```

策略：

```text
none
errors-only
all
```

可以设置确定性 Judge 抽样率：

```bash
python main.py run-dataset \
  --dataset VRSBench \
  --root /data/VRSBench \
  --split validation \
  --task general_vqa \
  --judge-policy all \
  --judge-sample-rate 0.1
```

Judge 结果与 deterministic metrics 并列记录：

```text
deterministic metrics
+
judge_status / judge_parsed / judge_inconsistency
```

Judge 不能覆盖 deterministic exact-match、IoU 或 counting error。

---

## 17. 运行后评测

对已有 run 做离线评测补全：

```bash
python main.py --config /path/to/local.yaml evaluate-run \
  --run-id vrsbench-vqa-v1
```

仅补缺失项：

```bash
python main.py --config /path/to/local.yaml evaluate-run \
  --run-id vrsbench-vqa-v1 \
  --only-missing
```

在已有 run 上显式增加 DeepSeek Judge：

```bash
python main.py --config /path/to/local.yaml evaluate-run \
  --run-id vrsbench-vqa-v1 \
  --deepseek
```

VQA 专用 Judge pass：

```bash
python main.py --config /path/to/local.yaml judge-vqa-run \
  --run-id vrsbench-vqa-v1
```

这些命令基于已经持久化的 prediction/result 工作，不通过重新调用 Qwen 生成第二份预测来“补评测”。

---

## 18. 外部标准评测

团队或官方 evaluator 使用独立 seam：

```bash
python main.py standard-evaluate \
  --result /path/to/canonical-result.jsonl \
  --tool-dir /path/to/eval_standard
```

可指定 evaluator Python：

```bash
python main.py standard-evaluate \
  --result /path/to/canonical-result.jsonl \
  --tool-dir /path/to/eval_standard \
  --python /path/to/python
```

外部标准指标存放在独立：

```text
external_standard
```

命名空间，不与内部 deterministic metric 名称混写。

VRSBench 数据集特定 official seam 位于：

```text
evaluation/datasets/vrsbench.py
```

---

## 19. 单图计数

运行：

```bash
python main.py --config /path/to/local.yaml count-image \
  --image /data/demo.png \
  --question "How many buildings are visible?"
```

评测并渲染：

```bash
python main.py --config /path/to/local.yaml count-image \
  --image /data/demo.png \
  --question "How many buildings are visible?" \
  --run-id demo-count \
  --evaluate \
  --render
```

可选：

```text
--target-spec
--resume
--force
--no-seam-verify
--max-qwen-calls
--max-deepseek-calls
```

Count-image 会冻结影响行为的调用参数，使 resume/force 不被新的 CLI 默认值或 config 漂移悄悄改变。

---

## 20. Counting Pipeline

CountingAgent 当前包含独立的：

```text
target parsing
backend planning
tile / point pipeline
geometry
evidence normalization
seam handling
backend execution
runtime fallback
```

主结果：

```text
counting_result.json
```

`final_count` 与 accepted evidence 保持一致。

当前显式 backend kind：

```text
qwen_point
quantity_proposal
semantic_segmentation
yolo_obb
```

`auto` 模式只按显式 capability 采用固定顺序：

```text
Detection > Semantic Segmentation > QuantityProposal > QwenPoint
```

同类 expert 才按 catalog priority 排序。模型暂时不可用或运行失败时，executor 按计划的
完整 fallback chain 继续；QwenPoint 失败是 terminal，不会伪造结果。合法零计数不会自动
触发 fallback，只有显式 zero-review policy 可以复核。

SegFormer 只在 verified class map 和 target-specific `connected_components` policy 同时
成立时成为候选。它输出 semantic region 而不是 instance mask，相接对象可能合并成一个
component 并造成 undercount。OEM 当前没有 verified class map，因此默认不注册。

Catalog 已明确声明 composite capability：`vehicle` 的链是 Detection → SegFormer →
QuantityProposal → QwenPoint，`aircraft` 是 Detection → SegFormer → QwenPoint。Semantic
backend 对每个 model label 分别做 connected components，不会先合并不同类别 mask。

小目标 minimum scan depth、empty-tile review、optional upscale 和 ambiguous seam visual
review 都由 target/catalog hints 与显式 settings 驱动，不依赖 dataset 名。新增同类 expert
主要修改 catalog、资产和 composition settings，不要求修改 `CountingAgent`。

目标来源与核验规则：

```text
VisualTaskPlanner v5 count_target
    -> normalization.count_target_hint（确定性 verifier）
    -> legacy metadata count_target_hint（兼容 verifier）
    -> deterministic CountTargetResolver
```

无 plan 的显式 structured normalization hint 允许 direct 执行，trace 标记
`normalization_explicit_hint`；legacy metadata direct 仅为历史兼容，标记
`legacy_direct_hint`。无 plan 且无 hint 时稳定失败。无效 hint 显式失败，不静默吞掉。

---

## 21. YOLO OBB Counting

Python settings 只定义通用 schema，默认 `enabled=false`、`detectors=[]`，不内置具体
checkpoint inventory。`ExpertCatalog` 声明专家能力与逻辑身份；部署配置（当前本地样例为
`configs/local.yaml`）声明是否启用、物理权重路径、provider/device 与阈值。加载该配置后会
注册 `detector_obb_csl_001`；不加载部署配置时不注册 YOLO。权重与 ONNX Runtime 仍由本地
环境准备且不自动下载，模型加载保持惰性。

部署启用方式：

```yaml
backend:
  yolo:
    enabled: true
    detectors:
      - name: detector_obb_csl_001
        weights: models/yolo_obb/yolov5m_obb_csl_dotav20.onnx
        # 其余 identity/runtime 字段见 configs/local.yaml
```

设计边界：

- detector 权重由本地环境准备；
- 相对权重路径在 composition root 按 `project_root` 解析；外部绝对部署路径原样保留；
- 不自动下载权重；
- detector profile/权重 hash/task/class map 需要一致；
- CUDA/CPU fallback 行为由 detector settings 声明；
- detector unavailable/runtime error 由 `counting.fallback_on_backend_*` 通用策略控制；
- zero detection 可由 `counting.verify_empty_detection` 触发 ordered chain 的下一位专家复核；
- `quantity_proposal` 不被当作 YOLO detector；
- YOLO 输出最终仍转换进统一 CountingResult/evidence 契约。

SegFormer 的 catalog entry 冻结 logical model id、SHA 与 verified labels；部署配置可通过
`models.segformer_experts.<backend>.model_path` 指向外部挂载目录。普通 wheel 只包含
catalog、class/config/preprocessor metadata 和 prompts，不包含大模型权重。
installed-wheel runtime 在任意工作目录会依次尝试显式 prompt root、项目 prompt root，
最后使用 wheel 内 packaged prompts；无效显式 override 会稳定失败而不会静默 fallback。

当前 `pyproject.toml` 声明了 `yolo` / `yolo-onnx` extras。二者按目标 runtime
择一安装；不要同时无条件安装 CPU 与 GPU ONNX Runtime。CUDA provider、驱动
和 ONNX Runtime 版本仍应以目标部署机器的已验证环境为准。

---

## 22. ChangeAgent

变化任务：

```text
change_caption
change_qa
```

输入是有序 T1/T2 图对。

Change V3 的主链路是：

```text
validation
-> global registration + quality gate
-> radiometric harmonization
-> deterministic multi-source perception
-> proposals
-> evidence-driven Qwen
```

其中 registration 只负责保守的全局几何配准，harmonization 只负责辐射一致化，
二者是分离的模块。SegFormer（可选）在一次推理中同时提供 semantic probabilities
和 intermediate/pyramid features；deterministic perception 将 low-level、feature
residual 与 semantic difference 融合为候选 proposal。Qwen 使用 raw full T1/T2
与 proposal-local evidence 确认和解释变化，派生图与 mask 只是辅助证据，不能替代原图。

当前 learned change interface 默认关闭，仓库没有提供任何后训练、ChangeHead 或
Qwen adapter 训练实现。

无效输入、配准质量不足和不可比较的 pair 会尽可能在模型调用前以稳定状态失败或
按配置受控回退；registration 失败不会被静默当作成功。

NumPy/OpenCV 相关能力属于：

```bash
python -m pip install -e ".[change]"
```

可选依赖。

另有离线 LEVIR harmonization 评测脚本：

```text
scripts/evaluate_levir_harmonization.py
```

用于独立评估图像协调/校准表现，不调用 Qwen/DeepSeek 主推理链路。

另有 VRSBench-counting 全流程 Counting Agent 评测脚本：

```text
scripts/evaluate_vrsbench_counting.py
```

在远端 GPU 上以真实 Qwen + 完整后端注册表 + 回退逐样本运行 CountingAgent，
并把每个最终答案的来源（YOLO / qwen_point 回退等）写入结果 JSONL。

---

## 23. 报告

Reporting 是只读层。

它从：

```text
predictions.jsonl
+
sample/status/trace/result/evaluation artifacts
```

构建当前 run 的：

```text
Report
ReportSample
TaskSummary
```

不会：

- 调 Qwen；
- 调 Agent；
	- 重跑任何规划器或任务解析器；
- 修改 prediction；
- 为了报告重新计算另一套 prediction。

标准报告 bundle：

```text
outputs/runs/<run_id>/report/
├── report.html
├── report.json
├── samples.csv
├── samples.jsonl
├── metadata.json
├── deepseek_audit.jsonl
└── external_standard.json   # optional
```

HTML 完全离线，不依赖 CDN。

CSV 使用 `utf-8-sig`，方便 Windows Excel。

### Report V2 audit dashboard

`reporting.builder` projects persisted artifacts into typed, stable view
models. The report exposes run metadata, deterministic result quality,
latency percentiles, task summaries, Counting expert candidate/attempt/
fallback history, stable failure codes, and bounded task-specific details.
It never embeds a raw trace or re-runs inference/evaluation.

`persist_report_bundle(..., max_visual_samples=200)` materializes the most
useful visual samples first (failed, partial, deterministic incorrect,
fallback, then warnings). Original previews are bounded WEBP files and
overlays are PNG files under `report/assets/`. Prediction, rejected, ground
truth, unresolved, and reviewer geometry use stable semantic colors and
1–2 px outline-only rendering. Missing, unsafe, unbound, or dimension-mismatched
geometry is reported explicitly and is never guessed.

The HTML has Overview, Tasks, Expert Routing, Samples, Failures, and Runtime
sections plus local search/filter controls. It uses no network resources,
external scripts/styles/fonts, or inline image payloads. Host paths, dataset
roots, raw exception messages, and credentials are excluded from every text
artifact in the bundle.

---

## 24. 运行产物

典型 run：

```text
outputs/runs/<run_id>/
├── manifest.json
├── config.snapshot.json
├── run_request.json
├── prompts.snapshot/
├── events.jsonl
├── predictions.jsonl
├── report/
└── tasks/
    └── <run_task>/
        ├── dataset_probe.json
        ├── dataset_summary.json
        └── samples/
            └── <storage_key>/
                ├── sample.json
                ├── status.json
                ├── routing_decision.json
                ├── agent_result.json
                │   or counting_result.json
                ├── vqa_evaluation.json
                │   or counting_evaluation.json
                │   or grounding_evaluation.json
                │   or caption_evaluation.json
                └── agent_trace.json
```

不同 task 不一定都有逐样本 deterministic evaluation；系统不会为缺少定义的任务伪造指标。

---

## 25. 三种 Task 身份

运行产物中需要区分：

### Resolved task

VisualTaskPlanner/UnifiedSample 的 canonical task：

```text
sample.json.task
agent_trace.resolved_task
```

### Execution task

实际 attempt 执行 task：

```text
status.json.task
agent_trace.execution_task
evaluation semantics
```

### Run task

DatasetRunner namespace：

```text
predictions.jsonl.run_task
tasks/<run_task>/
```

当低置信度 candidate fallback 成功时，这三个值可能不完全相同。

Resume 和 Evaluation 不能把 resolved task 与 execution task 混为一谈。

---

## 26. Report / Artifact Path Safety

`status.json.result_path` 是 sample-relative 的纯文件名，例如：

```text
agent_result.json
counting_result.json
```

不会持久化为：

```text
C:\...
/home/...
../...
```

`predictions.jsonl.result_path` 是 run-relative 索引/展示路径。

Reporting 使用冻结的 `(run_task, sample_id)` 推导真实 sample directory，不把任意 result path 当作文件读取权限。

---

## 27. 运维命令

### 创建 run

```bash
python main.py run-init --run-id local-smoke
```

创建 run 本身不调用模型。

### Qwen readiness

```bash
python main.py health qwen
```

显式 live probe：

```bash
python main.py --config /path/to/local.yaml health qwen --live
```

### DeepSeek readiness

```bash
python main.py health deepseek
```

显式 live probe：

```bash
python main.py health deepseek --live
```

### Direct Qwen smoke

```bash
python main.py --config /path/to/local.yaml smoke-qwen \
  --image /data/test.png \
  --question "Describe this image."
```

---

## 28. Counting 可视化与评测汇总

已经存在 CountingResult 时：

```bash
python main.py render-count \
  --image /data/demo.png \
  --result /path/to/counting_result.json \
  --output outputs/counting-overlay.png
```

汇总一个 run：

```bash
python main.py summarize-evaluations \
  --run-id <run-id>
```

或汇总显式 EvaluationRecord JSONL：

```bash
python main.py summarize-evaluations \
  --input /path/to/evaluations.jsonl \
  --output outputs/evaluation-summary.json
```

以上命令不重新运行主模型。

---

## 29. MME-RealWorld 官方提交

Reporting exporter 支持基于原始 MME 记录构造官方提交：

- 原始记录只读；
- 按 question id 写入 prediction；
- 只替换官方 `Output`；
- 其他字段保持。

相关逻辑位于：

```text
reporting/exporters.py
```

---

## 30. 默认离线与安全约束

默认行为：

```text
no model auto-download
no dataset auto-download
no DeepSeek call
no cloud API
```

只有显式能力可能联网，例如：

```text
download-data
health --live
DeepSeek Judge
显式允许 Qwen download
```

Secret value 不应进入：

```text
config snapshot
manifest
run request
trace
public error
report metadata
DeepSeek audit
```

不要把 DeepSeek key、Authorization header、本机 credential 或 Base64 image 写入配置、日志、测试 fixture 或文档。

---

## 31. 开发与测试

运行全部测试：

```bash
python -m pytest
```

架构测试：

```bash
python -m pytest tests/architecture
```

领域测试按目录运行，例如：

```bash
python -m pytest tests/agents/counting
python -m pytest tests/workflows
python -m pytest tests/evaluation
python -m pytest tests/reporting
```

迁移 parity：

```bash
python -m pytest tests/parity
```

Live tests 使用 pytest marker 单独区分：

```text
live_qwen
live_deepseek
live_dataset
```

默认开发验证不应因为缺少真实模型、API key 或真实数据集而偷偷联网。

---

## 32. 架构保护

项目有机器可检查的架构控制文件：

```text
architecture/allowed_python_files.txt
architecture/implementation_status.json
architecture/import_rules.json
architecture/ALLOWLIST_CHANGE_POLICY.md
```

重要规则：

- Python 文件路径需要在 allowlist 中；
- 普通任务不直接扩白名单；
- `spacers_agent/**` 和 `eval/**` 永久禁止重新出现；
- `main.py` 只 import `application`；
- Router 不 import models；
- Agent/Workflow 只依赖模型协议；
- 具体模型实现只由 `application` 选择；
- `__init__.py` 不承担业务副作用。

---

## 33. 迁移与行为基线

新架构与旧 `try_yolo` 不是通过普通源码 diff 维护一致性。

迁移参考：

```text
try_yolo@ec962eb87c3ad0b8c1502efcbd08db0daec48868
```

基线材料：

```text
docs/migration/
tests/fixtures/migration/
tests/parity/
```

迁移目标是保持需要保持的**可观察行为**，而不是复制旧包结构。

例如旧：

```text
spacers_agent/
eval/
```

已经被新的：

```text
agents/
routing/
workflows/
evaluation/
reporting/
application/
```

取代。

---

## 34. 文档

### 普通使用者

当前文件：

```text
README.md
```

### 编码代理

必须先读：

```text
AGENTS.md
DETAILS.md
```

### 架构设计

```text
docs/architecture/
```

### 迁移与 parity

```text
docs/migration/
```

README 只维护用户需要的当前用法，不再记录 Task 00/11A/11G.5 等迁移流水账。

---

## 35. 当前边界

当前架构已经完成主要离线实现与迁移收口，但实际运行能力仍取决于本地环境：

- Qwen checkpoint 是否存在；
- PyTorch/CUDA 是否与机器匹配；
- VRSBench / MME / XLRS / LEVIR 数据是否已准备；
- DeepSeek API key 是否在明确需要 Judge 时提供；
- YOLO/ONNX detector runtime 与权重是否按目标设备准备；
- Spark/4090/其他部署机器上的真实资源是否完成 live 验证。

因此：

```text
offline tests passed
```

不等于：

```text
every live model / dataset / GPU / deployment gate passed
```

真实实验结果应结合对应运行配置、run artifacts 和报告记录。
