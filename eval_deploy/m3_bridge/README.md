# m3_bridge — 桥接契约与偏离说明（挑战杯 XH-202603 客观分评测）

> 2026-10-06 首版 ｜ 2026-10-08 更新（任务级 planner 跳过策略、max_pixels 压缩、诊断日志）
> 部署位置：评测主机`~/eval/m3_bridge/`

## 干什么

把官方示例壳（`participant_api_starter`, Protocol 2.0）的一题请求映射到
本队 M3 系统：`SampleRunner.run_one()` 直连调用，输出再规范化回协议答案。

## 关键设计决定（如实记录）

1. **刻意绕过 v5 VisualTaskPlanner（`BRIDGE_TASK_MODE=constraint`）**。
   理由：Protocol 2.0 的 `response_constraint.type` 本身就是任务声明
   （bbox→grounding、single_choice→multiple_choice_vqa、其余→general_vqa/
   change_qa），再调一次 planner 只会增加 ~40% 延迟与失败面。M3 源码明示
   直连工具可省略 plan（`workflows/sample_runner.py`："Direct tools may omit
   them for an explicitly selected sample task; such calls never plan."）。
2. **不修改 M3 仓库任何代码**：桥接完全在仓库外，以库方式消费
   `application` / `data` 契约；仅 `concurrent` 使用官方壳的协议层。
3. **坐标系**：M3 内部 = 整数 0..999 xyxy（`agents.schema.VisualEvidence`）；
   协议要求原图像素。换算与 `models/images.py`（`x*W//999`、
   `(x*W+998)//999`）整数互逆，并强制 `x1<x2, y1<y2`。
4. **单选择题**：传给 M3 的 choices 使用位置字母（A..Z），模型答案按位置
   再映射回网站选项标识；`normalization.choices` 必须提供（缺失会触发
   `multiple_choice_sample_without_choices`）。
5. **日志纪律**：只记 item_id / 约束类型 / task / agent / 耗时 / 答案截断
   预览；绝不记录完整题干、Authorization 或令牌（`AGENTS.md §20` 与官方
   README 同款要求）。运行产物落在 `~/eval/runs/`，不外发、不打包。

## 2026-10-08 更新（本轮偏离与理由，如实记录）

1. **任务级 planner 跳过策略**（`real_model.py::_infer`）：
   - `enum + 双图`（区域选择题）：实测 planner 可能把区域题判为
     `change_caption`，ChangeAgent 输出整句 → 被判"非九宫格规范标签"而失分。
     约束本身已声明任务，改为直接走 change_qa 直连。
   - `bbox`：实测 planner 的 grounding 证据链**从未通过 schema**
     （`VisualTaskPlanError: VISUAL_TASK_PLAN_FAILED:SCHEMA_INVALID`，
     与对象类别（如 building）无证据凭证相关），每次白耗 ~9s 后仍回退直连；
     改为直接跳过 planner（与计数题既有策略同类）。证据链修复留待模型组
     （类别映射），修复后可通过删除该策略恢复。
2. **视觉 token 压缩**（`m3_eval.yaml`，配置级，不涉代码）：
   `models.qwen.max_pixels: 8388608`（16.7M → 8.4M，视觉 token 约减半）。
   实测（4096²，H100）：峰值 reserved 25.4 → 20.9 GiB、alloc 23.0 → 20.7 GiB；
   短答 6.6s、bbox 19.7s。4.2M 档实测收益递减（再省 ~1 GiB）未采用。
   另有 `m3_eval_4m.yaml` 备查。
3. **planner 异常诊断**：planner 回退日志从类型名升级为
   `类型: 消息(≤300字符)`，便于定位且不含完整题干。
4. **结果缓存**：M3 对相同请求有结果缓存（`~/eval/runs/service/cache`）。
   评测前清空以确保每题真实推理与真实时延；探针（replay_v2.py）同此纪律。

## 错误码

| 情况 | HTTP | code |
|---|---|---|
| 桥接 110s 硬超时 | 504 | `inference_timeout` |
| M3 无执行结果 / 无法规范化 | 502 | `unanswerable` |
| 找不到定位框 | 502 | `no_localization` |
| M3 运行异常 | 500 | `inference_failed`（沿用官方壳） |

> 注：超时只能释放 HTTP 连接；M3 底层生成不可取消（生成锁仍在），靠
> max_tokens=1024 与直连路径把概率压到极低。

## 原始响应救援（salvage）

2026-10-06 实测发现：模型（planner LoRA）对 grounding 题会在 JSON 中输出
额外字段（`count`、`evidence_items[].source`、`evidence_items[].visual_embedding`），
被 M3 的严格 `AgentResult`（extra=forbid）拒收 → 整题无执行结果。修复为双保险：

1. bbox 提示词加固：明确"只输出 agent_name/answer/boxes/evidence_items/status"；
2. **salvage**：当 run_one 无执行结果时，从 Agent 目录的 `raw_response.txt`
   （模型原始输出，含重试尝试）中用最后一个可解析的 JSON 抢救出 answer/boxes/
   evidence 框。内容即模型真实输出，仅按协议做规范化，不属于伪造答案；
   日志会记录 `salvage=raw_response` 便于统计。

## 复跑手册（简版）

```bash
# 服务：tmux new -s m3api → bash ~/eval/scripts/start_api.sh
# 隧道：tmux new -s cftun → bash ~/eval/scripts/start_tunnel.sh
# 自检：~/.conda/envs/m3/bin/python ~/eval/scripts/probe_local.py --package ~/eval/packages/rsu-dev100-v2
# 换正式包：解压到 ~/eval/packages/<new-id>/，重启 start_api.sh 时用 EVAL_PACKAGE=<new>
```
