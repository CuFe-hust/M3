# 14A0 — Architecture and Allowlist Approval

> Execute only this package in the current coding-agent session.
> 当前 coding-agent 会话只执行本任务包。
> Prerequisite: none; this is the first package.
> 前置：无；这是第一个任务包。

## 1. Session context

开始前依次阅读：

```text
AGENTS.md
DETAILS.md
architecture/ALLOWLIST_CHANGE_POLICY.md
architecture/allowed_python_files.txt
architecture/implementation_status.json
architecture/import_rules.json
docs/architecture/14A_FIRST_QWEN_VISUAL_WORKFLOW_IMPLEMENTATION_PLAN.md
docs/architecture/14B_VQA_OBJECT_EVIDENCE_SUBWORKFLOW.md
docs/architecture/14C_GROUNDING_AGENT_SUBWORKFLOW.md
```

执行并记录：

```bash
git status --short
git rev-parse HEAD
git branch --show-current
```

不得覆盖用户已有修改。默认离线；本包不加载权重、不调用云 API、不下载资产。

## 2. Outcome

本包只执行：

```text
C0  architecture allowlist approval only
```

这是独立架构任务，不实施任何业务行为，不创建生产/测试 Python 文件，不加载模型。
用户把本文件作为明确实施任务交给 coding agent，才构成对第 3 节所列架构文件修改
范围的授权。C0 完成后结束会话；C1 从下一个独立文件/全新会话开始。

## 3. C0 — Architecture allowlist gate

### 3.1 Proposed Python paths

只审批职责明确且后续任务确实需要的路径：

```text
workflows/visual_planner.py

agents/evidence_catalog.py

agents/general_vqa/evidence/__init__.py
agents/general_vqa/evidence/schema.py
agents/general_vqa/evidence/geometry.py
agents/general_vqa/evidence/rendering.py
agents/general_vqa/evidence/executor.py

agents/grounding/evidence.py

tests/workflows/test_visual_planner.py
tests/agents/test_evidence_catalog.py
tests/agents/general_vqa/evidence/__init__.py
tests/agents/general_vqa/evidence/test_schema.py
tests/agents/general_vqa/evidence/test_geometry.py
tests/agents/general_vqa/evidence/test_rendering.py
tests/agents/general_vqa/evidence/test_executor.py
tests/agents/grounding/test_evidence.py
```

非 Python 资产不进入 Python allowlist：

```text
prompts/first_qwen_visual_plan_v1.md
agents/evidence_catalog.json
```

### 3.2 Ownership rationale

- `workflows/task_resolver.py` 不看图片且只解析外部 task，不能容纳 VisualPlanner。
- VQA 的 YOLO/SegFormer 筛选、逐叶子类别回退、掩膜图和最终 Qwen 输入属于
  `GeneralVQAAgent` 的内部领域行为，因此必须位于 `agents/general_vqa/evidence/`。
- 禁止创建 `agents/object_evidence/`；它会错误暗示 VQA 与 Grounding 共享同一最终
  evidence workflow，并诱发跨 Agent Prompt/状态机耦合。
- `agents/evidence_catalog.py` 只维护 VQA/Grounding 共同读取的版本化类别事实，
  不调用模型、不生成证据，因此不属于 VQA 专属 evidence 子包。
- Grounding 的候选权限和最终框后处理与 VQA 不同，放在 `agents/grounding/evidence.py`。
- 现有 counting loader/backend 保持唯一实现，不移动、不复制。

### 3.3 Allowed changes

C0 只允许修改：

```text
architecture/allowed_python_files.txt
architecture/implementation_status.json
architecture/import_rules.json（仅确有 path rule 需要时）
docs/architecture/（仅记录本架构审批事实）
```

新增未来路径在 `implementation_status.json` 标为 pending；不得提前创建空壳文件。
不得放宽顶层 import DAG。

### 3.4 C0 verification and stop

```bash
pytest -q \
  tests/architecture/test_allowed_python_files.py \
  tests/architecture/test_implementation_status.py \
  tests/architecture/test_import_boundaries.py \
  tests/architecture/test_init_side_effects.py \
  tests/architecture/test_package_discovery.py \
  tests/architecture/test_no_new_to_legacy_imports.py
git diff --check
git status --short
```

提交/汇报 C0 后结束本任务。不得创建 pending 路径对应的空壳，不得开始 C1。

## 4. Acceptance and handoff

- allowlist、implementation status 和 import rules 三者一致；
- 所有新增路径均有明确职责，VQA evidence 位于 GeneralVQAAgent 子包；
- 没有 `agents/object_evidence/`，没有第二套 YOLO/SegFormer loader；
- 没有生产、测试或 asset 实现，没有 runtime/model/evaluation/resume 行为变化；
- 架构测试与 `git diff --check` 真实通过；
- 汇报批准/pending 路径清单，供全新会话执行 14A1。
