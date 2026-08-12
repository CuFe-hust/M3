# 14C — Grounding Agent 子工作流冻结方案

> Status: frozen design decision; no production implementation yet.
> 状态：设计已冻结；尚未进入生产实现。

## 1. 文档目的

本文用于团队同步第一次 Qwen 规划之后的 Grounding Agent 子工作流，冻结当前已经
确认的 ROI、类别筛选、YOLO 候选生成、逐类别缺失回退、最终 Qwen 候选选择以及
坐标后处理规则。

本文是
[`14_FIRST_QWEN_VISUAL_WORKFLOW_PLAN.md`](./14_FIRST_QWEN_VISUAL_WORKFLOW_PLAN.md)
的 Grounding 下层决策，并复用
[`14B_VQA_OBJECT_EVIDENCE_SUBWORKFLOW.md`](./14B_VQA_OBJECT_EVIDENCE_SUBWORKFLOW.md)
中已经冻结的共享物体证据基础规则。若早期文档关于 Grounding 的编号标注图、
SegFormer 回退、全部候选直接作为答案或坐标形态与本文冲突，以本文为准。

本文不表示相关生产代码已经存在。实施仍须遵守仓库白名单、包边界、模型单次组装、
artifact、resume、评测与路径安全契约。

实现归属已进一步冻结：Grounding 证据逻辑保留在 `agents/grounding/`；不得 import
`agents/general_vqa/evidence/`，也不得创建 `agents/object_evidence/`。Grounding 与
VQA 只共享版本化类别目录、模型无关检测协议和图片裁切原语，不共享最终证据状态机、
Prompt、SegFormer 回退或结果后处理。

## 2. 适用范围

本文只定义 public task 为 `grounding` 时，由 `GroundingAgent` 执行的内部路径。
当前已知该任务主要来自 VRSBench，但实现不得：

- 在 Agent 中直接读取或解析 VRSBench 原始 JSON；
- 绕过 `data.schema.UnifiedSample`；
- 根据 dataset 名称建立第二套 Grounding 输出协议；
- 改写 `UnifiedSample.task`、Ground Truth 或确定性评测身份。

Grounding 的最终结果仍由现有 Grounding 协议和评测族消费。本文只改变未来的内部
证据准备设计，不改变当前生产事实，因此在实现和验证完成前不写入 `DETAILS.md`。

## 3. 冻结主流程

Grounding 只使用一层模型回退：YOLO 未提供目标类别候选时，由最终 Qwen 直接补框。
不经过 SegFormer。

```text
第一次 Qwen 规划
  -> 组合类别与语义注意 ROI
  -> 版本化类别目录展开叶子类别
  -> 对 ROI 执行一次 YOLO 完整推理
  -> 按请求叶子类别筛选 YOLO 输出
  -> 标记每个叶子类别为“有 YOLO 候选”或“缺失”
  -> 最终 Grounding Qwen 只调用一次
       - 有 YOLO 候选的类别：只选择 box_id
       - 缺失类别：直接输出 ROI 归一化框
  -> 确定性合并与坐标后处理
  -> 生成现有 Grounding 结果
```

回退只针对缺失叶子类别。某个类别缺失时，其他类别已经生成的 YOLO 候选继续保留，
不会整题重跑，也不会丢弃成功证据。

## 4. 与 VQA 共用的基础规则

Grounding 与 VQA 共用同一个版本化封闭类别目录：

```text
组合类别
  -> 叶子类别集合
  -> 叶子类别到 YOLO 输出标签的映射
```

组合类别必须保持真实语义关系，例如 `ship` 不得被错误归入 `vehicle`。第一次 Qwen
只输出目录内组合类别，工作流机械展开并筛选 YOLO 实际输出，不在 Grounding Agent
中复制或维护另一份映射。

以下规则也与 14B 保持一致：

- 第一次 Qwen 的类别计划去重后最多三个组合类别；
- YOLO 输入只有图片，不是类别条件推理；
- 每个 ROI 只执行一次 YOLO 完整推理，再按请求叶子类别筛选；
- 未请求的 YOLO 输出全部丢弃；
- 只有实际筛选到至少一个框才算该叶子类别命中；
- 标签受支持但实际筛选为空仍然属于缺失；
- YOLO 内部置信度用于逐类别阈值、NMS、几何去重、冲突裁决和候选排序；
- Qwen 不看到 YOLO 置信度；
- 同一目标发生类别冲突时，由 YOLO 证据层保留内部置信度更高的类别；
- 具体阈值由后续模型训练、验证和人工校准决定，不由 Qwen 决定。

Grounding 不复用 VQA 的 SegFormer 回退层和最终回答 Prompt。不同 Agent 的最终 Qwen
输入与输出契约保持独立。

## 5. ROI 与图像输入

Grounding 复用 14B 的 ROI 几何与分辨率规则：

- 没有明确、可靠的空间约束时使用全图；
- 只有“左上角”等确定性位置要求或可靠语义区域才建议局部 ROI；
- 语义区域难以确定时回退全图；
- ROI 从应用 EXIF 方向后的原始分辨率图片裁切；
- 局部 ROI 映射到原图后，默认向四边扩张自身宽高的 10%，再裁剪到原图范围；
- 传给 Qwen 的图像超过 1080p 才等比例缩小，小图不放大；
- 传给 YOLO 的是原始分辨率 ROI，由模型接口自行预处理并保留逆变换。

Grounding 的常规路径按单 ROI 设计。当前数据中基本不存在 Grounding 多 ROI 场景，
因此本文不为多 ROI 候选联合选择增加专门分支；相关行为留待真实样本证明需要后再
独立冻结。

## 6. YOLO 候选生成

### 6.1 完整推理后筛选

YOLO 不接收目标类别，只接收 ROI 图片：

```text
ROI 图片
  -> YOLO 全标签推理
  -> 类别目录映射
  -> 只保留请求叶子类别的检测
```

请求一个、两个或三个组合类别不会增加同一 ROI 的 YOLO 调用次数。

### 6.2 候选记录

每条提供给最终 Qwen 的候选标注至少包含：

```text
box_id
具体叶子类别
roi_id
相对于 ROI 的归一化 xyxy
```

其中坐标契约为：

```text
[x_min, y_min, x_max, y_max]
每个值位于 [0, 1]
原点位于 ROI 左上角
```

模型接口必须先消除 YOLO resize / letterbox 的影响，再相对于实际 ROI 归一化。
不得把模型预处理画布坐标直接暴露给最终 Qwen。

### 6.3 不生成标注图

Grounding 的最终 Qwen 只接收干净 ROI 和候选标注文本，不生成或输入带框标注图。
候选文本通过 `box_id` 关联类别与归一化坐标。Qwen 根据原始问题、干净 ROI 和候选
文本进行语义筛选。

## 7. 逐类别回退规则

每个请求叶子类别只有两种下游状态：

| YOLO 实际筛选结果 | 最终 Qwen 权限 |
|---|---|
| 至少一个候选框 | 只能选择已有 `box_id` |
| 没有候选框、标签不支持或 YOLO 不可用 | 可以直接生成 ROI 归一化框 |

关键约束：

```text
只要某类别存在 YOLO 候选，Qwen 就不再负责为该类别标注；
Qwen 不得重画、微调或替换该类别的 YOLO 框。
```

如果 Qwen 认为某个已命中类别的候选都不完全符合问题中的具体指代，它仍必须从
现有候选中选择最符合者，而不能为该类别切换为自由标注。

## 8. 最终 Grounding Qwen

### 8.1 单次调用

最终 Grounding Qwen 始终只调用一次。它在同一个结构化响应中完成两类工作：

1. 对 YOLO 已命中的类别选择候选 `box_id`；
2. 对 YOLO 缺失的类别直接输出相对于 ROI 的归一化框。

这两项是互斥权限，不允许 Qwen 为已命中类别同时返回候选选择和自由框。

### 8.2 输入

最终 Qwen 至少接收：

```text
原始问题
外部 Grounding 答案约束
干净 ROI 图像
roi_id 与 ROI 几何信息
YOLO 已命中类别的候选标注文本
允许 Qwen 直接标注的缺失叶子类别
```

YOLO 候选文本只包含 `box_id`、具体类别、`roi_id` 和 ROI 归一化坐标，不包含检测
置信度。

### 8.3 逻辑输出

建议的逻辑结构为：

```json
{
  "selected_box_ids": ["roi-1-box-2"],
  "fallback_boxes": [
    {
      "leaf_category": "requested_leaf",
      "roi_id": "roi-1",
      "bbox": [0.12, 0.20, 0.46, 0.71]
    }
  ]
}
```

字段使用列表，统一支持单框和多框 Grounding：

- 问题指向单个目标时选择或生成一个框；
- 问题明确指向多个目标时可以选择或生成多个框；
- 对存在 YOLO 候选的相关类别，至少选择一个候选；
- `fallback_boxes` 只能引用调用前已经标记为缺失的叶子类别。

最终 Qwen 不生成自由文本坐标答案，不重写所选 `box_id` 的坐标。

### 8.4 非法输出

当前阶段不为极低概率的结构化输出异常增加 Qwen 重试：

- 无法解析的结构化响应按失败处理；
- 未知 `box_id` 直接丢弃；
- 非法、退化或无法裁剪到 `[0,1]` 的自由框直接丢弃；
- Qwen 为 YOLO 已命中类别生成的自由框直接丢弃；
- Qwen 为非缺失类别或目录外类别生成的自由框直接丢弃；
- 若清理后没有任何合法框，则 Grounding 执行失败，不伪造结果。

稳定错误码、公共 trace 形态和是否复用通用结构化解析器由实现阶段收口。

## 9. 确定性后处理

最终后处理不再调用模型，只机械执行：

1. 用 `selected_box_ids` 查回原始 YOLO 候选框；
2. 接收并校验仅属于缺失类别的 Qwen 自由框；
3. 合并合法候选选择和合法自由框；
4. 对最终框执行确定性几何去重；
5. 将 ROI 归一化坐标映射到应用方向后的原图全局坐标；
6. 按现有 Grounding 输出契约完成最终坐标适配与序列化。

内部两路框统一使用 ROI `[0,1]` 坐标，只在最终后处理阶段转为全图尺度。当前仓库
内建 Grounding deterministic metric 要求兼容：

```text
normalized_0_999_top_left
4-value xyxy
```

因此未来实现必须通过显式、可测试的坐标转换生成当前协议所需结果，不能把 ROI
归一化坐标直接当成最终答案，也不能未经转换把原图像素坐标送入现有评测。

若未来 VRSBench official evaluator 使用另一坐标协议，应通过独立、显式的适配层
处理，不得静默改变现有 deterministic metric 的解释。

## 10. 模型与架构边界

所有模型实现只对外暴露模型无关的单一接口。Grounding Agent 依赖模型协议，不依赖
具体 YOLO/Qwen 类、checkpoint、processor、模型路径或 device。

模型必须在 `application` composition root 中创建并复用。不得：

- 在 Grounding Agent 内调用 `models.entry.create_model(...)`；
- 为 Grounding 复制第二套 YOLO loader；
- 为每个类别、样本或请求重复加载同一模型；
- 从旧包或旧分支 import 实现；
- 让 Grounding Agent 修改数据集循环、评测或 resume 语义。

共享物体检测 seam 应同时服务 Counting、VQA 与 Grounding，但各 Agent 保留自己的
领域输出和最终 Qwen Prompt。

## 11. 已冻结决策清单

- Grounding 当前主要面向 VRSBench，但不写 dataset-specific 解析分支；
- Grounding 采用 `YOLO -> 最终 Qwen` 的单层回退，不使用 SegFormer；
- 只回退 YOLO 缺失的叶子类别，成功类别证据继续保留；
- Grounding 与 VQA 共用版本化类别目录和 YOLO 基础规则；
- YOLO 每个 ROI 完整推理一次，再筛选请求叶子类别；
- YOLO 输入只有图片，未请求输出全部丢弃；
- 最终 Qwen 只调用一次；
- YOLO 已命中类别由 Qwen 选择 `box_id`，禁止重新标注或修改框；
- 即使候选都不完全匹配，Qwen 也从现有候选中选择最符合者；
- YOLO 缺失类别才允许 Qwen 直接生成框；
- Grounding 最终 Qwen 只看干净 ROI，不看带框标注图；
- Qwen 不看到 YOLO 置信度；
- YOLO 与 Qwen 自由框都使用相对于 ROI 的 `[0,1]` 归一化 `xyxy`；
- Qwen 输出采用列表，统一支持单框和多框；
- 坐标转换和最终结果格式化由确定性后处理完成；
- 当前不为低概率结构化输出异常增加 Qwen 重试；
- 清理非法输出后没有合法框时显式失败，不伪造定位结果。

## 12. 实现时必须验证的性质

后续实现至少应通过测试证明：

- Grounding 和 VQA 读取同一类别目录版本与 YOLO 标签映射；
- 同一 ROI 的 YOLO 调用次数不随请求类别数量增加；
- 未请求标签不会进入 Qwen 输入或最终结果；
- 标签受支持但实际筛选为空时，仅对应缺失叶子类别获得 Qwen 自由标注权限；
- 已有 YOLO 候选的类别不能被 Qwen 自由框覆盖、微调或替换；
- 未知 `box_id`、越权自由框和非法坐标稳定失败或被确定性丢弃；
- YOLO/Qwen 两路 ROI 归一化坐标具有同一语义；
- resize / letterbox 逆变换、ROI 偏移与全图坐标转换正确；
- 单框和多框输出均能稳定格式化；
- 置信度不会进入最终 Qwen、公共 trace 或 Grounding 结果；
- 最终 Qwen 调用次数保持为一次；
- 最终结果继续满足现有 Grounding geometry 和 evaluation 契约；
- 失败样本不会被过滤、伪装为成功或从数据集 summary 中遗漏；
- 不改变 `UnifiedSample`、TaskRouter、Ground Truth、Judge、report、CLI 或 resume
  语义。

## 13. 明确延期、不得自行猜测的内容

以下内容不阻碍主流程冻结，但应在实现或模型校准阶段单独收口：

1. 组合类别、叶子类别和 YOLO 标签映射的最终目录内容；
2. 目录外或部分非法类别计划的容错策略；
3. 每个叶子类别的 YOLO 阈值以及 NMS、去重和冲突阈值；
4. 候选框过多时的上下文容量、排序与确定性截断上限；
5. Grounding 多 ROI 的真实需求和跨 ROI 联合选择规则；
6. 最终 Grounding Qwen Prompt、严格响应 Schema 与通用解析器接入方式；
7. 结构化输出失败时的稳定错误码和 trace 内容；
8. 第一次规划与最终 Qwen 的预算、缓存身份和通用调用失败重试；
9. 中间证据、最终结果 artifact 的文件名、原子写入与 resume 校验；
10. VRSBench 当前源标注到现有 Grounding geometry contract 的明确适配位置；
11. 共享检测 seam 的最终 Python 路径与独立 allowlist 架构审批。

coding agent 不得因本文冻结了主流程而擅自修改 Python 白名单、公共 task、评测定义、
Golden fixtures 或 resume 契约。

## 14. 非目标

本文不定义：

- VQA、Counting、Change、Caption 或通用兜底 Agent 的最终 Qwen 输入；
- SegFormer 在 VQA 或 Change 中的行为；
- 具体模型权重、训练流程或精度指标；
- 新 public task、新 Agent 或新评测指标；
- Ground Truth、dataset split 或 official evaluator 的新解释；
- ROI 裁切工具的具体实现。
