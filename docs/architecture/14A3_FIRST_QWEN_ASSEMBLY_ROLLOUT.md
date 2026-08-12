# 14A3 — Application Assembly, Validation, Calibration, and Rollout

> Execute only after 14A2 is complete with the feature flag still disabled by default.
> 仅在 14A2 完成且 feature flag 仍默认关闭后执行。

## 1. Session context and preflight

必读：根 `AGENTS.md`、当前 `DETAILS.md`、14A 索引、前三包交接、14B/14C、application
bootstrap/settings/prompts/runtime、CLI/HTTP、reporting、packaging/config 和相关测试。

```bash
git status --short
git rev-parse HEAD
git branch --show-current
```

确认 14A2 的 disabled-path 集成、artifact/resume/budget 测试真实通过。若前置仅声明
通过但没有可验证记录，先复验，不得直接 rollout。

## 2. Outcome and decomposition

```text
C9   application assembly + settings/prompts/package/public-entry parity
C10  complete offline integration gate
C11  explicitly authorized live calibration + staged default rollout
```

C9/C10 默认离线。C11 只有用户明确授权真实模型、真实数据和必要网络后才能执行。

## 3. C9 — Single composition root

`application/bootstrap.py` 是唯一真实组装位置：

- Qwen client 每次 runtime assembly 创建一次并共享给 Planner/final protocol owner；
- 唯一 YOLO store/client 创建一次，Counting/VQA/Grounding 都通过协议消费；
- SegFormer clients 只按 verified catalog capability 惰性组装，VQA 使用，Grounding 不用；
- 注入 VisualPlanner、GeneralVQA evidence executor、Grounding evidence seam 和现有
  counting evidence seam；
- Agent/Workflow 不 import `models.entry` 或具体模型实现，不读取 API key。

配置与 prompt：

- `application/settings.py` 严格解析完整 visual-planning policy，`extra="forbid"` 不弱化；
- 默认 `enabled=false`，离线下载开关仍为 false；本阶段不切默认；
- `application/prompts.py` 绑定 planner/final prompt version；run 创建时 snapshot；
- 本地模型物理路径与 `cache_model_id`/logical identity 继续分离；secret value 不进入
  settings/snapshot/artifact。

打包与公共入口：

- wheel 包含 planner prompt、shared evidence catalog 及必要非 Python 资产；
- `main.py` 只委托 application commands；不新增第二套 CLI；
- `ask`、`run-dataset`、`resume-run`、HTTP `/ask` 复用同一 Runtime；
- HTTP 不按请求重建模型，不暴露 raw exception/secret/绝对路径；
- reporting 只读已持久化 plan/evidence，可展示 execution family 和稳定 fallback code，
  不重推理、不重跑 detector、不修改执行产物、不重新计算更好指标。

实现事实同步 `DETAILS.md`；只有公开安装/数据准备/命令变化才更新 README。配置示例
只记录真实支持字段。不得修改 migration Golden fixtures 迎合新路径。

```bash
pytest -q \
  tests/test_main.py \
  tests/workflows/test_sample_runner.py \
  tests/workflows/test_dataset_runner.py \
  tests/architecture/test_package_discovery.py \
  tests/architecture/test_init_side_effects.py
```

还必须构建 wheel，在隔离环境检查 prompt/catalog 被包含，并证明 `import models`、
`import agents`、`import application.settings` 不加载 torch/transformers/权重。

## 4. C10 — Offline integration gate

先运行聚焦集合：

```bash
pytest -q \
  tests/agents \
  tests/workflows \
  tests/models \
  tests/evaluation \
  tests/contracts \
  tests/integration \
  tests/architecture
```

然后运行仓库完整离线 pytest 和静态检查：

```bash
pytest -q
python -m compileall -q application data models agents routing workflows evaluation reporting main.py
git diff --check
git status --short
```

必须用固定 fake-client/in-memory 或小型静态 fixture 覆盖：

- flag off 与 baseline 的调用次数、artifact、状态完全一致；
- VQA direct/object-evidence、1/2/3 ROI、YOLO hit/empty/error、SegFormer mask/empty、
  全部 visual fallback；
- Grounding candidate selection、missing leaf free box、越权/未知 box_id 拒绝；
- VQA internal counting 的协议/评测身份不变；
- planner invalid/low-confidence/budget/decode failure 的冻结 fallback；
- artifact 损坏与 resume 不重复推理；
- multiple-choice、Grounding、Counting、Caption、Change 原有结果契约；
- summary 闭合、failed/skipped 不过滤、Judge 不覆盖 deterministic metrics；
- trace/path/secret/Base64/absolute-path safety；
- Windows/POSIX 稳定序列化、基础 import 无副作用。

无法运行的命令必须报告原命令、原因、替代检查和剩余风险；不得写“all tests passed”
除非确实运行完整集合。

## 5. C11 — Authorized calibration only

未获得用户对联网、真实权重和真实数据的明确授权时，到 C10 即停止。

授权后，在不改变 GT/split/样本纳入规则的前提下比较：

```text
baseline vs planned answer accuracy
planner schema-valid rate
full-image fallback rate / ROI miss audit
per-leaf YOLO and SegFormer hit/missing/error rate
Grounding candidate/free-box rates
Qwen calls per sample
latency and peak memory
failed/partial/skipped distribution
resume/cache hit behavior
```

所有被选择样本进入统计；不得过滤失败样本提高结果。阈值、NMS、跨 ROI 去重、mask
样式、context 上限等校准值必须进入 typed config、测试和可审计记录，不硬编码在 Agent。

不得在 calibration 中更换主 Qwen、processor/tokenizer/checkpoint 身份；如确需更换，
另开模型身份任务并重新建立可比基线。

## 6. Staged rollout

每一步是独立配置/default-change 任务，保留回退开关与固定样本 A/B 记录：

1. VQA `direct_vqa` planning only；
2. VQA `object_evidence_vqa` YOLO path；
3. VQA SegFormer missing-leaf fallback；
4. Grounding candidate-evidence path；
5. VQA internal counting evidence；
6. public counting plan hints；
7. caption/change family validation；
8. global default enable。

只有前一步验收满足批准阈值且没有 resume/evaluation 回归，才能进入下一步。全局默认
开启必须是独立 commit，不得夹在 C9 实现或 calibration 调参中。

## 7. Final acceptance

- plan schema/catalog/prompt 版本一致且可从 run snapshot 审计；
- VQA evidence 实现只位于 `agents/general_vqa/evidence/`，无 `agents/object_evidence/`；
- Grounding 与 VQA 共用底层协议/catalog，但不共享最终 evidence workflow/Prompt；
- YOLO/SegFormer/Qwen 单次 composition、惰性加载、默认离线；
- external task、答案协议、deterministic evaluation family、GT/split 不变；
- VQA final Qwen 不见 confidence，SegFormer mask 不转框/计数；
- Grounding box_id/free-box 权限和 whole-image 坐标转换正确；
- artifact、cache、budget、resume、report 可复现且路径/敏感信息安全；
- flag off 保持旧行为；每一 rollout stage 有真实测试和 A/B 记录。

最终汇报必须列出实际修改、文件、测试结果、未运行项、对所有核心契约的影响、已知
风险和当前 rollout stage，不得把“离线测试通过”等同于真实模型质量已验证。
