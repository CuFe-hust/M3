# eval_deploy — Protocol 2.0 评测 API 桥接与部署（挑战杯 XH-202603）

> 本目录把 M3 系统接入官方评测协议（participant API Protocol 2.0），并记录 2026-10-06 → 10-08
> 的免训练优化（开发集成绩 **16.4 → 21.2 / 40**，零训练、零 M3 代码改动）。
> 本目录为仓库外的**消费层**：M3 主仓库代码不在此分支修改，桥接以库方式调用
> `application` / `data` 契约。

## 架构

```
评测网站 (HTTPS)
  → cloudflared 隧道 → 127.0.0.1:9001 participant_api_starter/server.py（官方协议壳 + 最小补丁）
      → m3_bridge（真实推理桥接，常驻单事件循环，模型仅加载一次）
          → M3 SampleRunner.run_one()（直连路径；按策略跳过/回退 planner）
```

## 目录

| 路径 | 说明 |
|---|---|
| `m3_bridge/` | 桥接核心：`real_model.py`（Runtime 装配/推理/救援/策略）、`protocol_map.py`（题干合成/答案规范化/坐标换算）、`errors.py`；`README.md` 为**逐条偏离与理由**（如实记录） |
| `m3_bridge/m3_eval*.yaml` | M3 运行配置：`m3_eval.yaml` = 在役 8M 像素档；`m3_eval_4m.yaml` / `m3_eval_16m.yaml` 为实测备查变体 |
| `scripts/` | 部署与验证：`run_api.py`、`start_api.sh`（环境变量与启动）、`start_tunnel.sh`、`probe_local.py`/`probe_direct.py`（本地探针，0 额度）、`measure_resources.py`（峰值显存/时延）、`replay_v2.py`（历史题目定向回放）、`make_repro.py` |
| `starter_patches/` | 对官方示例壳 `participant_api_starter` 的最小补丁副本（参考：[gitee 官方仓库]） |

## 部署速览

```bash
# 1) 环境：解压图片包与 manifest 到 packages/<dataset_id>/（结构：manifest.json + assets/）
# 2) 密钥：umask 077；token_urlsafe(48) → secrets/model_api_key
# 3) 启动（tmux 内；unset 代理由脚本处理；GPU 由 CUDA_VISIBLE_DEVICES 指定）
BRIDGE_TASK_MODE=planner bash scripts/start_api.sh
# 4) 隧道：bash scripts/start_tunnel.sh（cloudflared quick tunnel，URL 随重启变化）
# 5) 自检：GET /healthz（带 Bearer）→ {"status":"ready",...}
```

关键环境变量：`M3_ROOT` / `M3_CONFIG` / `OUTPUT_ROOT` / `EVAL_PACKAGE` / `BRIDGE_TASK_MODE`
/ `BRIDGE_TIMEOUT_SECONDS`（默认 110）/ `BRIDGE_BBOX_PICK`（默认 largest_area）。

## 配置变体（视觉 token 预算，实测 H100）

| 变体 | `max_pixels` | 峰值 reserved | 短答延迟（4096²） | 说明 |
|---|---|---|---|---|
| 8M（在役） | 8,388,608 | **20.9 GiB** | 6.6 s | 16.7M→8M 压缩：−4.5 GiB、延迟 −40%，精度以 full 对打验证 |
| 4M | 4,194,304 | 19.8 GiB | 5.3 s | 边际收益递减（再省 ~1 GiB），未采用 |
| 16M（原版） | 不设（模型默认） | 25.4 GiB | ~9–11 s | 归因对照基线 |

## 优化记录（2026-10-06 → 10-08，全部免训练）

| 日期 | 优化 | 动机（实测证据） | 效果 |
|---|---|---|---|
| 10-06 | 破损 JSON 数字挖掘救援 | 模型把角点写成 `(x,y),(x2,y2)` 致整题丢失 | 救回题目 IoU 0.88 |
| 10-06 | 严格 Schema 拒收时 salvage + 空框升级重试 | 额外字段包装键被内部 Schema 拒收 | 定位类失败率显著下降 |
| 10-06 | 九宫格标签规范映射 | 区域题整句英文被判畸形 | 畸形格式题消除 |
| 10-06 | `max_tokens` 1024 → 768 | 大图生成顶到上限致 110s 超时 | 超时题恢复（75.7s 作答） |
| 10-06 | `expandable_segments` + 定期重启 | 长跑显存膨胀至 42 GB | 回落 ~19.5 GB |
| 10-08 | 区域题反 “bottom” 锚定提示词 | 23 题中 bottom 答 14 次仅 3 对 | 变化维度 2.4 → **5.2** |
| 10-08 | bbox 全范围 + 多实例消歧提示词 | 区域型目标只框局部、最大/唯一定位错实例 | 定位维度 3.2 → **4.0** |
| 10-08 | 单选逐项独立评估提示词 | 选项 E 出现 7 次、7 次全错 | 超高分辨维度 4.0 → **5.2** |
| 10-08 | 任务级 planner 跳过策略（enum 双图 / bbox） | planner 对区域题误判 change_caption 致整句畸形；bbox 证据链必 `SCHEMA_INVALID` 白耗 ~9s | 畸形 3→0；省 ~9s/题 |
| 10-08 | `max_pixels` 8M 视觉 token 压缩 | 单题 prompt 可达 1.6 万视觉 token，prefill 主导时延与峰值 | P50 21.8 → **12.1 s**；reserved −4.5 GiB |
| 10-09 | 枚举题格式纠正重试（change_qa） | 变化题随机输出整句（0–2 题/轮，任何提示词版本都会中）→ 判畸形 0 分 | 必然 0 分转为合法作答；畸形标记消除 |
| 10-09 | 噪声标定（同配置复跑 full） | 确定性假设修正：同配置两轮 5/100 题翻转（C 2 / D 3） | A/B 判定阈值定为 **≥2 题（0.8 分）** |

**综合结果（开发集 100 题）**：总分 16.4 → **21.2 / 40**；计分 100/100、零畸形、零失败；
P50 延迟 21.8 → 12.1 s；峰值 reserved 25.4 → 20.9 GiB。

## 诚实性说明

- 桥接对模型输出的**任何救援/规范化都只做协议格式层**，绝不生成内容；无法合法化时返回稳定错误码（502/504），不伪造答案。
- planner 跳过策略的理由与恢复方式见 `m3_bridge/README.md`（修复后删除策略即可恢复 planner 证据链）。
- 日志纪律：只记 `item_id / 约束类型 / 耗时 / 答案截断预览`，不记题干与令牌；运行产物与密钥**不入库**。
