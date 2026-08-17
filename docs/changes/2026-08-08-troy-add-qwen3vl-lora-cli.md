# Modification Note: Add Qwen3-VL LoRA Inference CLI and Remote 1522 Launcher - 2026-08-08 15:21:59 CST

## Modification Time

2026-08-08 15:21:59 CST (initial)
2026-08-08 15:29:51 CST (follow-up: interactive image path)
2026-08-08 15:37:31 CST (follow-up: local image upload by the SSH launcher)
2026-08-08 15:49:07 CST (follow-up: one persistent SSH session + server mode)

## Modifier

tRoy <791056216@qq.com>

## Modification Goal

Provide a standalone CLI that calls the Qwen3-VL-8B model fine-tuned with
merger-LoRA weights, runnable on the remote 1522 GPU node
(`100.88.222.9:1522`, host `qi2`) either directly or through an SSH launcher.
该修改提供一个独立 CLI，用于调用 merger-LoRA 微调后的 Qwen3-VL-8B 模型；
既可在远端 1522 GPU 节点（`100.88.222.9:1522`，主机名 `qi2`）直接运行，
也可通过 SSH 启动脚本远程运行。

## Modified Files

- `scripts/qwen3vl_lora_cli.py` (new): single-shot / interactive LoRA inference CLI.
- `scripts/run_qwen3vl_lora_remote.sh` (new): SSH launcher for the remote 1522 node.
- `scripts/qwen3vl_lora_remote.py` (new): local persistent SSH client (stdlib only).
- `tests/test_qwen3vl_lora_cli.py` (new): unit tests for CLI helpers.
- `tests/test_qwen3vl_lora_remote.py` (new): unit tests for the SSH client.
- `README.md`: added the LoRA inference CLI section under merger-LoRA training.
- `DETAILS.md`: added the two scripts to `scripts/` responsibilities.

## Core Changes

- The CLI loads one base Qwen3-VL checkpoint, attaches a LoRA adapter through
  `PeftModel`, freezes all parameters, and runs greedy inference for either a
  single `--image` + `--prompt` or an `--interactive` stdin prompt loop.
  Model and adapter paths come from CLI arguments or `MODEL_ID` /
  `ADAPTER_PATH` environment variables; `--local-files-only` prevents network
  fallback on the offline remote node.
  CLI 加载基础 Qwen3-VL 权重并通过 `PeftModel` 挂载 LoRA 适配器，冻结全部
  参数后做贪心推理；支持单张 `--image` + `--prompt`，或 `--interactive`
  从标准输入循环读取提示词。模型与适配器路径来自 CLI 参数或 `MODEL_ID` /
  `ADAPTER_PATH` 环境变量；`--local-files-only` 防止离线远端节点走网络回退。
- In the follow-up, `--image` became optional: one-shot mode still requires
  it, while interactive mode asks for the image path at the prompt when it is
  omitted and supports `!image <path>` to switch the current image during the
  session.
  后续调整中 `--image` 改为可选：单次模式仍必须提供；交互模式未传时会在
  提示处询问图片路径，并支持在会话内用 `!image <path>` 切换当前图片。
- In the latest follow-up, the SSH launcher treats `--image` as a LOCAL path:
  it uploads the image to `REMOTE_STAGING_DIR` (default
  `/tmp/lora-cli-uploads`) with `scp`, then runs the remote CLI with the
  uploaded path and the question. `--interactive` now reads the local image
  path and question in a loop and performs one remote inference per question.
  `SSHPASS` + `sshpass -e` reuses the password for every `ssh`/`scp`;
  `DRY_RUN=1` prints commands without executing them.
  最新调整中，SSH 启动脚本把 `--image` 视为本地路径：先用 `scp` 上传到
  `REMOTE_STAGING_DIR`（默认 `/tmp/lora-cli-uploads`），再把上传后的路径和
  问题交给远端 CLI 解析。`--interactive` 改为在本地循环读取图片路径与问题，
  每次上传图片后调用一次远端推理。`SSHPASS` + `sshpass -e` 让每次
  `ssh`/`scp` 复用密码；`DRY_RUN=1` 只打印命令不实际执行。
- In the latest follow-up, the architecture changed to ONE persistent SSH
  session and ONE model load. `scripts/qwen3vl_lora_cli.py --server` loads the
  model once and serves line-delimited JSON commands; the new local client
  `scripts/qwen3vl_lora_remote.py` sends each local image as base64 plus the
  question over the same stdin stream, prints the JSON answer, and keeps the
  session open for the next round. The bash launcher now just forwards
  arguments to the Python client.
  最新调整改为“一次持久 SSH 会话 + 一次模型加载”：`scripts/qwen3vl_lora_cli.py
  --server` 只加载一次模型并提供行式 JSON 指令服务；新增本地客户端
  `scripts/qwen3vl_lora_remote.py` 把每张本地图片以 base64 连同问题通过同一
  stdin 流发送，打印 JSON 回答后保持会话等待下一轮。bash 启动脚本现在只把
  参数转发给 Python 客户端。
- The SSH launcher defaults to `REMOTE_USER=lijia`, `REMOTE_HOST=100.88.222.9`,
  `REMOTE_PORT=1522`, `REMOTE_REPO=~/M3`, and the remote m3 conda Python. It
  passes `--local-files-only` automatically and adds `-t` when `--interactive`
  is present. All connection details remain environment-overridable; no
  credentials are stored in the file.
  SSH 启动脚本默认使用 `REMOTE_USER=lijia`、`REMOTE_HOST=100.88.222.9`、
  `REMOTE_PORT=1522`、`REMOTE_REPO=~/M3` 与远端 m3 conda Python。脚本自动
  追加 `--local-files-only`，检测到 `--interactive` 时自动加 `-t`。连接参数
  均可通过环境变量覆盖；文件不保存任何凭据。

## Whether the Canonical Sample Format Was Changed

No. The CLI builds chat messages directly and does not touch
`data/schema.py`, dataset adapters, or evaluation readers.
否。CLI 直接构造聊天消息，不涉及 `data/schema.py`、数据集适配器或评测读取。

## Whether the Model Interface Was Changed

No. The script is a standalone maintenance CLI; `models/entry.py`, existing
wrappers, and weight-loading logic are unchanged.
否。脚本是独立维护工具；`models/entry.py`、现有封装与权重加载逻辑均未改动。

## Whether the Configuration Was Changed

No existing configuration field changed. New CLI arguments are local to the
new script; environment variables `MODEL_ID` and `ADAPTER_PATH` follow the
existing training-entry convention.
否。现有配置字段未改动；新 CLI 参数只属于新脚本，`MODEL_ID` 与
`ADAPTER_PATH` 环境变量沿用现有训练入口的约定。

## Whether Evaluation Was Affected

No metric, split, reference-answer reading, or result post-processing rule was
changed. The CLI is not part of `eval/` and does not alter VRSBench evaluation.
否。评测指标、数据集划分、标准答案读取与后处理规则均未改动；CLI 不属于
`eval/`，不影响 VRSBench 评测。

## Whether Deployment Was Affected

No deployment export or hardware path changed. The remote launcher only
executes the existing local-Transformers inference path on the 4090 node.
否。部署导出与硬件路径未改动；远端启动脚本只在 4090 节点执行既有的本地
Transformers 推理路径。

## Whether pytest Was Updated

Yes. Added `tests/test_qwen3vl_lora_cli.py` covering parser defaults, argument
validation, dtype resolution, image-path resolution, message building, greedy
`infer_one` with fake model/processor, and atomic JSON writing.
是。新增 `tests/test_qwen3vl_lora_cli.py`，覆盖解析器默认值、参数校验、dtype
解析、图片路径解析、消息构造、使用假模型/处理器的贪心 `infer_one` 与原子
JSON 写入。

## Whether .gitignore Was Updated

No. The change introduces only `.py` and `.sh` source files; no new output,
checkpoint, cache, or dataset file type was generated.
否。本次仅新增 `.py` 与 `.sh` 源文件，未产生新的输出、权重、缓存或数据集文件
类型。

## Remote Sync

2026-08-08 transferred `scripts/qwen3vl_lora_cli.py` to
`lijia@100.88.222.9:1522:/home/lijia/M3/scripts/` with `scp` and verified the
remote file hash equals the local hash
(`b08a493e8516cadba8e1caf7ab7ab94b`); the remote `--help` runs with the m3
conda Python. Other local files were not copied because they are not needed to
run the CLI.
2026-08-08 已用 `scp` 将 `scripts/qwen3vl_lora_cli.py` 同步到
`lijia@100.88.222.9:1522:/home/lijia/M3/scripts/`，最近一次同步后远端文件
哈希与本地一致（`76723aa68bc3c98558101b3f6df026ce`），远端 m3 conda Python
可正常执行 `--help` 且包含 `--server`。其余本地文件未同步，因为运行 CLI
不需要。

## Validation Method

Local environment `/opt/miniconda3/envs/m3/bin/python`:
本地环境 `/opt/miniconda3/envs/m3/bin/python`：

```bash
/opt/miniconda3/envs/m3/bin/python -m compileall -q scripts/qwen3vl_lora_cli.py tests/test_qwen3vl_lora_cli.py
bash -n scripts/run_qwen3vl_lora_remote.sh
/opt/miniconda3/envs/m3/bin/python -m pytest -q tests/test_qwen3vl_lora_cli.py
```

Result: syntax checks passed; `18 passed` for the two new test files; the full
suite passed with `535 passed` in `/opt/miniconda3/envs/m3` after installing
`opencv-python`.
结果：语法检查通过；两个新测试文件共 `18 passed`；在 `/opt/miniconda3/envs/m3`
安装 `opencv-python` 后全量测试 `535 passed`。

The SSH launcher was additionally verified with `bash -n` and `DRY_RUN=1`
for both one-shot and interactive modes (single persistent ssh command
printing), and the remote script was re-verified with `--help | grep --server`.
启动脚本另以 `bash -n` 与 `DRY_RUN=1` 对单次和交互两种模式做了持久会话命令
拼装验证（只打印一条 ssh 命令），并在远端用 `--help | grep --server` 复核。

## Risks and Follow-up TODOs

- No live remote connection, model loading, or GPU inference was performed in
  this modification; the SSH port default `1522` follows the user's explicit
  correction and should be verified with one remote smoke run.
  本次修改未执行真实远端连接、模型加载或 GPU 推理；SSH 端口默认值 `1522`
  按用户明确更正设置，需用一次远端冒烟运行验证。
- The remote repository must contain `scripts/qwen3vl_lora_cli.py`; sync the
  branch before running the launcher. The image path must already exist on the
  remote node.
  运行启动脚本前需先同步分支，使远端仓库包含
  `scripts/qwen3vl_lora_cli.py`；图片路径也必须已存在于远端节点。
