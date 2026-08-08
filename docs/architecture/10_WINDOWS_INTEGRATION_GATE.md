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
