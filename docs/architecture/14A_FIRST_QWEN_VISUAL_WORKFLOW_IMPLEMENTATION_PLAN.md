# 14A — First-Qwen Visual Workflow Implementation Index

> Audience: coding agents implementing one bounded package per clean session.
> 面向对象：每次在全新会话中只实施一个有边界任务包的 coding agent。
> Status: implementation index only; it does not describe completed production behavior.
> 状态：仅实施索引；不表示相关生产行为已经存在。

## 1. How to use this index

原 14A 长计划已拆成四个按顺序执行的任务包。每次新会话只把**一个任务包**交给
coding agent；任务包已经包含该阶段所需的上下文、修改边界、测试、停止条件与交接
要求，不需要把原长计划全文重新注入上下文。

| Order | Task package | Main outcome |
|---|---|---|
| 1 | [`14A0_FIRST_QWEN_FOUNDATION_CONTRACTS.md`](./14A0_FIRST_QWEN_FOUNDATION_CONTRACTS.md) | 独立白名单与职责边界审批，不实施业务 |
| 2 | [`14A1_FIRST_QWEN_PLANNER_VQA_EVIDENCE.md`](./14A1_FIRST_QWEN_PLANNER_VQA_EVIDENCE.md) | 严格契约/目录、小模型协议、ROI、隔离 Planner、VQA 证据 |
| 3 | [`14A2_FIRST_QWEN_AGENT_INTEGRATION_RESUME.md`](./14A2_FIRST_QWEN_AGENT_INTEGRATION_RESUME.md) | Counting/Grounding seam、禁用式 Agent 集成、artifact/resume/budget |
| 4 | [`14A3_FIRST_QWEN_ASSEMBLY_ROLLOUT.md`](./14A3_FIRST_QWEN_ASSEMBLY_ROLLOUT.md) | composition、配置/打包、离线门禁、授权校准与分阶段 rollout |

必须按顺序执行。后一个任务包不得替前一个任务包补做未通过的门禁，也不得因为相关
代码已经“顺手写好”而跳过前置验收。

## 2. Source-of-truth precedence

每个阶段都必须先读根 `AGENTS.md`、当前 `DETAILS.md`、对应任务包和相关生产代码/
测试。设计冲突按以下顺序处理：

1. 当前用户对该阶段的明确要求；
2. [`14B_VQA_OBJECT_EVIDENCE_SUBWORKFLOW.md`](./14B_VQA_OBJECT_EVIDENCE_SUBWORKFLOW.md)
   覆盖 VQA 的 ROI、YOLO/SegFormer、回退和最终输入早期设想；
3. [`14C_GROUNDING_AGENT_SUBWORKFLOW.md`](./14C_GROUNDING_AGENT_SUBWORKFLOW.md)
   覆盖 Grounding 的候选、回退、最终 Qwen 和坐标早期设想；
4. [`14_FIRST_QWEN_VISUAL_WORKFLOW_PLAN.md`](./14_FIRST_QWEN_VISUAL_WORKFLOW_PLAN.md)
   提供整体产品目标；
5. 本索引与四个实施包提供实施顺序和门禁。

若文档与当前 HEAD 不一致，先以机器白名单、实现状态、测试和生产代码确认事实；
不得静默选择一方继续实现。

## 3. Objective and invariant data flow

```text
UnifiedSample（保留数据集 task、答案协议和评测身份）
  -> optional first-Qwen visual plan
  -> deterministic TaskRouter selects protocol owner from sample.task
  -> protocol-owner-specific evidence preparation
  -> protocol owner / final Qwen returns existing result contract
  -> existing deterministic evaluation family
```

必须始终保持：

```text
sample.task = 外部任务身份、答案协议、评测身份
visual_plan.execution_family = 内部完成路径，不改写 sample.task
```

## 4. Frozen ownership decisions

1. `TaskRouter` 保持同步、确定性、无模型调用，只读取已知 task。
2. `TaskResolver` 只解析缺少外部 task 的 `SampleDraft`；VisualPlanner 处理已经物化
   的 `UnifiedSample`，两者不得互相调用或覆盖。
3. dataset adapter 不调用 VisualPlanner、Agent 或具体模型。
4. 第一次 Qwen 不选择 detector/segmenter backend、checkpoint、processor 或 device。
5. 具体模型只在 `application` composition root 创建一次，领域层依赖协议。
6. 外部答案格式、Ground Truth 与确定性评测族不因内部执行路径改变。
7. VQA 只保留 14B 冻结的 `direct_vqa` 与 `object_evidence_vqa` 两个一级子工作流。
8. **VQA 视觉证据归属 `agents/general_vqa/evidence/`；允许在该现有 Agent 包下创建
   子文件夹，但不得创建 `agents/object_evidence/` 或第二个通用证据 Agent。**
9. Grounding 证据归属 `agents/grounding/`，不得 import VQA evidence 子包；两者只
   共享任务无关的类别目录、模型协议和图像裁切原语。
10. Counting 继续使用 `agents/counting/` 的唯一实现和 `CountingResult`；VQA 只能
    消费 counting evidence，不得调用 `CountingAgent.run()` 后伪造 task。
11. VQA ROI 使用 14B 冻结的 `[0,1]` top-left `xyxy`；现有最终 Grounding/
    `VisualEvidence` 的整图 `0..999` 坐标是另一个明确转换后的契约。
12. VQA 最终 Qwen 不看检测置信度；SegFormer 掩膜不转实例框、不做精确计数。
13. Grounding 不使用 SegFormer；YOLO 命中类别只能由最终 Qwen 选择 `box_id`，
    只有缺失类别允许 Qwen 自由补框。
14. Judge 永远不覆盖 deterministic metrics。

## 5. Target package layout

以下路径只是目标职责模型；Python 路径必须先经过任务包 1 的独立 allowlist 审批：

```text
workflows/visual_planner.py
    first-Qwen request、严格校验、cache/budget seam

agents/evidence_catalog.py
agents/evidence_catalog.json
    VQA/Grounding 共用的版本化组合类别和模型标签映射；不执行模型

agents/general_vqa/evidence/
    schema.py       VQA evidence typed contracts
    geometry.py     VQA ROI/local-global geometry
    rendering.py    clean ROI / SegFormer overlay assembly
    executor.py     VQA YOLO -> SegFormer -> final-visual-fallback 状态机

agents/grounding/evidence.py
    Grounding 专属 YOLO 候选筛选、Qwen 权限校验和坐标后处理
```

共享图片读取/EXIF/RGB/ROI 裁切继续复用 `models/images.py`。如果当前 HEAD 已存在
`crop_image_region(...)`，后续阶段先验证并复用，不得复制第二个裁切器。

## 6. Global stop conditions

任一 coding agent 遇到以下情况必须停止并报告：

- 所需 Python 或测试路径未获 allowlist 批准；
- 需要修改 deterministic metric、GT 解释、split、样本纳入规则或 Golden fixture；
- 需要让 Router 读取 question/图片或调用模型；
- 需要复制 YOLO/SegFormer loader，或在 Agent 内选择具体模型；
- 需要改变主 Qwen/checkpoint/processor/tokenizer 加载语义；
- 当前任务包列出的用户决策门禁尚未冻结；
- resume 无法从持久化调用身份确定性重建；
- 需要新增未批准依赖或默认联网；
- 工作区已有修改与本任务目标文件冲突。

## 7. Handoff contract

每个任务包结束必须报告：task id、修改内容与原因、文件、实际测试及结果、未运行项、
对 UnifiedSample/task-routing/model/evaluation/report/CLI/resume 的影响、已知风险、
下一门禁。不得在单个任务包完成后声称整条 first-Qwen pipeline 已完成。
