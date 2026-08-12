# 14A2 — Agent Integration, Artifacts, Budget, and Resume

> Execute only after 14A1 is complete and its policy gates are approved.
> 仅在 14A1 完成且其策略门禁已获批准后执行。

## 1. Session context and preflight

必读：根 `AGENTS.md`、当前 `DETAILS.md`、14A 索引、14A1 交接、14B/14C 全文，以及
当前 `agents/base.py`、GeneralVQA/Grounding/Counting agents、SampleRunner、
ArtifactWriter、RunStore、CallBudget、DatasetRunner 和对应测试。

```bash
git status --short
git rev-parse HEAD
git branch --show-current
```

确认前包测试已通过、所有策略值已由用户批准且已进入 typed config/policy；缺一项即停止。

## 2. Outcome and decomposition

```text
C6  reusable counting evidence seam + Grounding evidence seam
C7  feature-flagged protocol-owner/SampleRunner integration (default off)
C8  artifacts, trace, cache, budget, run_request, resume fidelity
```

本包可以让新路径在显式 feature flag 下端到端运行，但不得默认开启，不做真实模型
校准，不改变现有 deterministic metrics。

## 3. C6 — Domain evidence seams

### 3.1 Counting evidence

先检查现有 `agents/counting/evidence.py`、executor 和 agent；优先复用稳定函数，不平行
实现。只有现有职责不足时才在已批准路径中提取明确 service seam：

```text
question / approved target hint / image / AgentContext
  -> CountingResult
```

- `CountingAgent` 仍是 public counting task 的协议 owner；
- VQA internal-counting 只消费 `CountingResult` 作为证据，最终仍由 GeneralVQAAgent
  输出合法 VQA/choices 结果；
- VQA evaluation 仍是 VQA，不写 counting evaluation；
- target hint 优先级、backend selection、fallback、budget、cache 和 artifact root 复用
  counting 唯一实现；
- 禁止调用 `CountingAgent.run()` 后改 sample.task 或伪造 persisted sample；
- 如果确需新增未批准的 `agents/counting/service.py`，停止申请独立 allowlist 变更。

### 3.2 Grounding evidence

实现 `agents/grounding/evidence.py`，只遵守 14C：

```text
planned ROI + requested leaves
  -> each ROI one full YOLO inference
  -> filter requested labels
  -> one final Grounding Qwen call:
       YOLO-hit leaf: choose existing box_id only
       missing leaf: may emit ROI-local [0,1] xyxy
  -> deterministic validation and whole-image 0..999 conversion
```

- Grounding 不 import `agents/general_vqa/evidence`，不使用 SegFormer，不看 overlay；
- 与 VQA 读取同一 catalog version/YOLO label mapping 和 detection protocol；
- final Qwen 看 clean ROI + candidate text，不看置信度或带框图；
- 未请求标签丢弃；YOLO-hit 类别禁止自由框覆盖/微调；未知 box_id、越权自由框、
  非法坐标稳定拒绝/丢弃；清理后无合法框显式失败；
- ROI-local `[0,1]` 只有在最终确定性后处理后才转现有整图
  `normalized_0_999_top_left` Grounding contract；不得猜坐标系。

```bash
pytest -q \
  tests/agents/counting/test_agent.py \
  tests/agents/counting/test_executor.py \
  tests/agents/counting/test_target_parser.py \
  tests/agents/grounding/test_evidence.py \
  tests/agents/grounding/test_agent.py \
  tests/evaluation/test_counting_metrics.py \
  tests/evaluation/test_vqa_metrics.py \
  tests/evaluation/test_grounding_metrics.py
```

## 4. C7 — Disabled feature integration

### 4.1 Typed configuration

在 `AppSettings` 增加严格 `visual_planning` 配置组，至少持久化：

```text
enabled = false
prompt/catalog version
max_rois
halo_ratio
planner failure/low-confidence policy
per-label detector/segmenter policies
cross-ROI dedup policy
max detections/context policy
ROI partial-failure policy
```

默认必须关闭。未冻结的策略不得藏在代码常量中。配置变更同步测试和 `DETAILS.md`；
本包不改全局默认开启值。

### 4.2 Runtime sequence

```text
UnifiedSample ready
  -> if enabled: VisualPlanner (one planned call)
  -> persist validated visual plan
  -> TaskRouter chooses protocol owner from original sample.task
  -> protocol owner consumes typed plan / injected evidence services
  -> optional VQA/Grounding/Counting evidence preparation
  -> exactly one final protocol-owner Qwen call when that path requires it
  -> evaluation dispatch from original task
```

通过 `AgentContext` 传入轻量协议/service/typed plan 引用；不得放完整 AppSettings、
PromptCatalog、API key、权重、PIL、Base64 或完整 mask。

### 4.3 Protocol owners

`GeneralVQAAgent`：

- `direct_vqa`：clean full image/ROI -> one final Qwen；
- `object_evidence_vqa`：调用 `agents/general_vqa/evidence/`，按 14B 组装 clean ROI、
  optional mask overlay、YOLO text evidence，再 one final Qwen；
- internal counting 若获 compatibility 批准，只把 CountingResult 作为最终 VQA evidence；
- multiple-choice 继续执行现有 choice constraint postprocess。

`GroundingAgent`：只用 C6 Grounding evidence seam，遵守 box_id/free-box 权限和现有
geometry/evaluation contract。

`CountingAgent`：public counting 继续输出 `CountingResult`/`counting_result.json`；
plan 只可提供已校验 target/ROI hint，不能选择 backend。

`CaptionAgent`/`ChangeAgent`：本包最多接收/记录 family validation；不得借此重写 change
SegFormer 或双时相角色逻辑。

### 4.4 Compatibility gate

接入前必须由用户批准 external task -> allowed internal path 矩阵。不兼容组合只能按
冻结策略 fallback/fail，绝不能改 sample.task。尤其不得自行决定：

```text
caption/change with counting or VQA evidence
grounding with VQA/counting finalization
counting with VQA finalization
VQA with change workflow
```

Feature flag 关闭时，现有调用次数、结果、trace 和 artifact 必须逐项保持不变。

```bash
pytest -q \
  tests/workflows/test_sample_runner.py \
  tests/agents/general_vqa/test_agent.py \
  tests/agents/grounding/test_agent.py \
  tests/agents/counting/test_agent.py \
  tests/agents/change/test_agent.py \
  tests/agents/caption/test_agent.py \
  tests/evaluation/test_vqa_metrics.py \
  tests/evaluation/test_grounding_metrics.py \
  tests/evaluation/test_counting_metrics.py
```

## 5. C8 — Persistence and resume fidelity

### 5.1 Artifact ownership

在本阶段冻结最终 basename/目录结构；以下只是语义占位，未批准前不得直接采用：

```text
visual plan JSON
VQA evidence JSON
clean ROI images
optional SegFormer overlay images
Grounding candidate JSON
```

- JSON 由现有原子写入原语发布；图片也使用 temp + replace；
- 所有 result/status/additional-result filename 为安全 basename；run index 只保存安全相对路径；
- visual plan 保存 validated schema，不保存 raw Qwen body；
- evidence 保存稳定状态、catalog version、逻辑模型身份、ROI/local-global geometry；
- confidence 可以存在私有 evidence artifact 用于确定性后处理，但不得进入 final Qwen、
  公共 trace/报告或最终 Grounding/VQA result；
- 不保存 secret、Base64、绝对路径、raw exception、完整 tensor/mask 数组。

### 5.2 Budget/cache identity

单样本共享同一 budget，至少区分：optional TaskResolver、one VisualPlanner、optional
counting calls、detector/segmenter（按现有 budget 类型扩展）、one final Qwen、optional
Judge。fallback 不得创建新 budget。

Planner/final-answer request hash 覆盖所有语义输入，包括 logical model identity、
revision、generation、prompt/schema/catalog version、messages、image digest、client version。
不得减少 hash 输入制造 cache hit。

### 5.3 Resume

```text
succeeded
  -> no planner, detector, segmenter, or final-Qwen rerun

succeeded + only missing/corrupt evaluation, judge, or report
  -> repair only that post-inference stage

partial/failed/running/pending/missing or invalid execution state
  -> follow existing explicit rerun contract
```

- visual plan/evidence/ROI artifact 损坏处理必须有专门测试；
- fresh run 的实际选项进入权威 `run_request.json`/冻结 snapshot；
- resume 不得用当前 prompt/config/CLI default 猜原调用；新请求冲突稳定拒绝；
- actual execution task 继续决定指标族，不以 canonical plan family 覆盖；
- predictions index append-only，历史不覆盖，summary 闭合。

```bash
pytest -q \
  tests/workflows/test_artifact_writer.py \
  tests/workflows/test_run_store.py \
  tests/workflows/test_sample_runner.py \
  tests/workflows/test_dataset_runner.py \
  tests/workflows/test_call_budget.py \
  tests/models/test_response_cache.py \
  tests/contracts/test_artifact_contract.py \
  tests/integration/test_dataset_runner_resume.py
```

## 6. Final checks and handoff

```bash
pytest -q tests/agents/general_vqa tests/agents/grounding tests/agents/counting tests/workflows
pytest -q \
  tests/architecture/test_import_boundaries.py \
  tests/architecture/test_init_side_effects.py \
  tests/architecture/test_package_discovery.py
git diff --check
git status --short
```

验收：flag off 完全保持旧行为；flag on 的路径可离线 fake-client 集成；所有样本终态和
summary 闭合；resume 不重复成功推理；evaluation family 仍由外部 task 决定；VQA
evidence 没有移出 GeneralVQAAgent 包。交接时列出尚未做的真实 assembly、wheel、live
calibration 和默认值切换。
