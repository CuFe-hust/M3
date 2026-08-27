# VRSBench caption 样本统一 question 实施计划

## 1. 目标

将 VRSBench `caption` 样本的 `UnifiedSample.question` 长期固定为：

```text
Describe the image in detail.
```

该行为由 VRSBench adapter 在样本构造边界统一提供，使 registry、dataset runtime、caption 评测脚本及其他复用 `VRSBenchAdapter.iter_samples(...)` 的入口得到同一输入；不依赖源数据行是否包含 `question`，也不允许源行中的其他 question 覆盖该固定值。

## 2. 当前事实与影响

- 当前实现位于 `data/adapters/vrsbench/adapter.py::_caption_sample`，在 `UnifiedSample` 和 `stable_sample_id(...)` 两处都硬编码 `question=""`。
- XLRS-Bench caption 当前从源行读取非空 `question`；本任务只统一最终评测问句语义，不修改 XLRS adapter。
- VRSBench caption 的原始行已经完整保存在 `ground_truth.raw["source_row"]`，固定规范化后的 question 不会丢失源字段审计信息。
- `stable_sample_id(...)` 的哈希输入包含 question。按统一内部样本身份契约，新固定问句必须同时传给 `UnifiedSample` 和 `stable_sample_id(...)`，因此没有安全 source ID 的 VRSBench caption 样本 ID 将发生一次确定性变化。
- 样本 ID 变化意味着旧 caption run 与新行为不是同一组可 resume 样本；不增加旧 ID 映射，不重写既有 run/artifact，也不让 resume 猜测新旧身份。

## 3. 实施步骤

### 3.1 在 adapter 边界定义唯一固定值

修改 `data/adapters/vrsbench/adapter.py`：

1. 增加职责明确的模块常量，例如 `CAPTION_QUESTION = "Describe the image in detail."`。
2. `_caption_sample(...)` 构造 `UnifiedSample` 时使用该常量作为 `question`。
3. 同一次构造中的 `stable_sample_id(...)` 也使用该常量，保证持久化样本内容与逻辑身份一致。
4. 不把 caption 的 `question` 加入 `_REQUIRED_FIELD_GROUPS`，以保证缺少源 question 的既有兼容数据仍可加载，并统一得到固定问句。
5. 不修改 caption Ground Truth、references、split、样本选择、任务类型和评测指标。

### 3.2 补充 adapter 契约测试

修改 `tests/data/test_vrsbench_adapter.py`：

1. 将现有 caption 断言从空字符串更新为精确固定字符串（包括句末句点）。
2. 测试源行带官方无句点形式 `"Describe the image in detail"` 时，输出仍为带句点的固定值。
3. 测试源行缺少 `question` 时仍输出同一固定值，锁定“所有入口生效、与源行可选字段无关”的长期行为。
4. 增加或扩展 sample ID 稳定性断言：相同输入重复加载 ID 相同，并确认 ID 使用规范化后的固定 question 计算，防止未来只改 `UnifiedSample.question` 而遗漏身份哈希。

### 3.3 补充官方格式集成覆盖

修改 `tests/fixtures/vrsbench/caption_only/VRSBench_EVAL_Cap.json`，使 fixture 包含官方 `question` 字段；同时在 `tests/data/test_adapter_integration.py::test_vrsbench_caption_only_probe_and_iterate` 中断言输出为精确固定问句。

保留单元测试中的“缺少 question”场景，分别覆盖官方格式和向后兼容格式。若 `tests/fixtures/vrsbench/full/VRSBench_EVAL_Cap.json` 与 caption-only fixture 表达同一官方结构，则同步补充该字段，避免两个官方 fixture 漂移。

### 3.4 更新当前行为文档

修改 `DETAILS.md` 的 VRSBench adapter/数据集契约相关段落，记录：

- VRSBench caption 的 canonical question 是精确字符串 `Describe the image in detail.`；
- 该值由 adapter 固定，不由源行覆盖；
- 源行仍保存在 `GroundTruth.raw` 供审计；
- 新行为会改变基于 question 哈希生成的 caption sample ID，旧 run 不做隐式 resume 迁移。

无需修改 `docs/migration/`：本任务是当前架构的长期输入规范化，不是对 `try_yolo` Golden parity 的修补；除非实施时发现现有 migration fixture 直接锁定了 VRSBench caption 空 question。

## 4. 验证计划

按以下顺序执行离线验证：

```bash
pytest -q tests/data/test_vrsbench_adapter.py
pytest -q tests/data/test_adapter_integration.py -k vrsbench
pytest -q tests/data
```

再执行静态检查，确认生产代码中不再存在 VRSBench caption 的空 question 构造，并确认未误改 XLRS 行为：

```bash
rg -n 'question=""|CAPTION_QUESTION|Describe the image in detail' data/adapters/vrsbench tests/data DETAILS.md
pytest -q tests/data/test_xlrs_adapter.py
```

本改动不改变 package、import DAG 或 `__init__.py`，因此架构测试不是最低必要门禁；若实施中出现超出上述范围的文件布局/import 修改，则补跑仓库规定的五项架构测试。

## 5. 验收标准

- 每个经 `VRSBenchAdapter` 生成的 `caption` `UnifiedSample` 都具有精确 question：`Describe the image in detail.`。
- 源行 question 缺失、无句点或内容不同，均不能改变 canonical question。
- `stable_sample_id(...)` 与 `UnifiedSample.question` 使用同一固定值，重复加载结果确定。
- VRSBench caption 的 Ground Truth、参考 captions、图像路径/角色、split 和评测逻辑保持不变。
- XLRS-Bench caption 继续读取并校验自己的行内 question。
- 既有 run/artifact 不被修改；旧 VRSBench caption run 不被静默映射到新 sample ID。
- 上述测试实际通过，未运行或失败的验证在交付时如实报告。

## 6. 明确不在范围内

- 不修改 `UnifiedSample` schema 或全局 caption Agent prompt。
- 不修改 VisualTaskPlanner、TaskRouter、SampleRunner、DatasetRunner 或 caption metric。
- 不回填或重写既有训练数据、Golden fixtures、run directories、predictions 或 reports。
- 不修改 XLRS-Bench、LEVIR-CC 或其他数据集的 question 规则。
- 不新增依赖，不联网，不下载数据或模型。
