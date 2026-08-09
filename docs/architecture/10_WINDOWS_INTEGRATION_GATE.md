# Task 10 — Windows Integration Gate

HEAD: `00076b0722efd3b32884931deecfde53b6ee840e`（09.5 hardening 修复提交）
Date: 2026-08-08

## Environment

- OS: Windows 11（10.0.26200，SP0）
- Python: 3.13.12（本机默认解释器；m3 conda env 为 3.11，用于真机运行时）
- Shell: Git Bash（`C:\Program Files\Git\bin\bash.exe`）
- Repo path: `C:\Users\TZDEZACR\Desktop\spacers-agent\new_structure`
- Repo path 含空格/CJK：否（含连字符；空格/CJK 场景以显式 tmp 根覆盖，见下）

## Results

| Check | Result |
|---|---|
| compileall（8 包 + tests + main.py） | PASS |
| architecture | PASS（40） |
| contracts | PASS（136） |
| workflows | PASS（209） |
| evaluation | PASS（79） |
| reporting | PASS（17） |
| application | PASS（32） |
| integration | PASS（41） |
| main CLI（tests/test_main.py） | PASS（11） |
| full pytest | PASS（**1428 passed**，0 warning） |
| git diff --check | PASS |

## Path coverage

- **spaces + CJK**：真实运行场景——dataset root `tmp/数据 集/遥感 data`、output
  root `tmp/输出 runs 目录`（均含空格与中文），fake Qwen 全链路
  （Runtime → manifest adapter → DatasetRunner → SampleRunner → evaluation →
  reporting）通过：run 目录在 CJK 根下创建成功、predictions.jsonl 与报告
  JSON 全部正斜杠序列化（`\` 零出现）。
- **run_id 安全**：CJK/空格 run_id（`路径测试 run`）被 `_validate_run_id`
  稳定拒绝（safe plain identifier）；合法 `path-test-run` 通过。
- **Windows drive path**：settings snapshot 测试覆盖
  `C:\Users\me\runs` / `D:\data` → 正斜杠保留（host-path-preserving）。
- **UNC / traversal 拒绝**：reporting escape 回归测试覆盖
  `../outside/...`、`C:/outside/...`、`\\server\share\...`、
  `foo/../../outside/...` —— 样本目录一律由 (run_task, sample_id) 身份推导，
  run 外哨兵内容绝不进入报告，恶意 result_path 展示降级为 None。

## CLI smoke（离线，无真实模型）

| 场景 | 结果 |
|---|---|
| `main.py --help` | exit 0 |
| `run-dataset`（缺 task/auto-task） | exit 2，稳定错误 JSON |
| `--task` + `--auto-task` 互斥 | exit 2 |
| **`--resume` 无 `--run-id`** | **exit 2**，`{"status":"failed","error":"--resume requires --run-id"}`，先于运行时/模型初始化 |
| fresh 唯一 run_id / duplicate 拒绝 / resume 校验 | 由 tests/test_main + tests/application（fake runtime）覆盖：fresh 无 run-id 两次 → 两个唯一 run；explicit duplicate → FileExistsError；resume 缺 run-id/缺 run/数据集或 split 不匹配 → 稳定失败 |
| result path 正斜杠序列化 | 契约测试 + 路径场景双重覆盖 |

## Final

WINDOWS_INTEGRATION_READY


# Final Parity Section — 11J Full Functional Parity Gate (Offline)

HEAD: `72d810a`（11I2 提交，Final Parity 门前的最后功能提交）
Date: 2026-08-09（final parity 验证在最终 parity HEAD 上重复执行）

## Environment

- 与上节相同（Windows 11 + Git Bash + Python 3.13）；路径含空格/CJK 场景
  以显式 tmp 根覆盖（parity fixtures 与既有测试均含空格/CJK 用例）。

## Command surface（16 个公开命令全部可用，`--help` exit 0）

`serve`（隐式默认）/ `ask` / `run-dataset` / `run-init` / `health` /
`list-datasets` / `smoke-qwen` / `count-image` / `resume-run` /
`evaluate-run` / `judge-vqa-run` / `standard-evaluate` / `inspect-data` /
`render-count` / `summarize-evaluations` / `download-data`

## Offline parity verification（fake/mocked，零网络零真模型）

| 路径 | 方式 | 结果 |
|---|---|---|
| ask 单图/变化（auto 规则） | fake Qwen（Runtime.ask） | PASS |
| HTTP /health + /ask（1 MiB 限制/错误契约） | fake Qwen + 临时端口 | PASS |
| count-image（fresh 唯一/显式重复拒绝/resume 零 Qwen/force） | fake counting 管线 | PASS |
| run-dataset fresh/resume（run_request 权威、root canonicalize） | fake Qwen + manifest 数据集 | PASS |
| run-init | RunStore 真实 | PASS |
| inspect-data（quick/full） | 真实只读 | PASS |
| render-count / summarize（run+file 模式） | 真实产物 | PASS |
| evaluate-run（含 counting DeepSeek judge） | fake judge | PASS |
| judge-vqa-run（skip/force） | fake judge | PASS |
| standard-evaluate（external_standard 命名空间 + 报告 bundle） | fake 工具脚本 | PASS |
| download-data（6 个官方目标、zip-slip 拒绝） | mocked hub | PASS |
| LEVIR harmonization evaluator（合成图/标签） | 真实 PairHarmonizer | PASS |
| 报告导出（samples.jsonl/deepseek_audit/元数据/MME 官方） | 真实产物 | PASS |

## Parity fixtures（tests/fixtures/parity/）

12 个稳定 golden JSON 已锁定（ask_single/http_health/http_ask/dataset_fresh/
dataset_resume/count_image_summary/run_init_manifest/evaluate_run/
summarize_file/standard_evaluate/download_data/levir_summary），时间戳、
绝对路径、request/run id 全部剥离；9 个比对测试在最终 parity HEAD 上全绿。

## Results（最终 parity HEAD）

| Check | Result |
|---|---|
| compileall（9 包 + tests + main.py） | PASS |
| architecture | PASS（40） |
| contracts | PASS（136） |
| parity（含 11J fixtures 比对） | PASS |
| workflows | PASS（212） |
| evaluation | PASS（79） |
| reporting | PASS（24） |
| application | PASS（139+） |
| integration | PASS（41） |
| main CLI | PASS（37） |
| full pytest | PASS（1585 passed） |
| wheel 构建（离线 pip wheel --no-deps） | PASS（133 个 .py 全含，17 个命令模块 + downloader/loader/vrsbench/standard） |
| git diff --check | PASS |

## Final

FULL_FUNCTIONAL_PARITY=PASS
FINAL_LIVE_GATE=PENDING
