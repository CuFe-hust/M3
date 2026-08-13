# Modification Note: Qwen3-VL LoRA Remote 1522 Password Prompt and Loaded-Dialog Flow - 2026-08-08 15:56:21 CST

## Modification Time

2026-08-08 15:56:21 CST

## Modifier

tRoy <791056216@qq.com>

## Modification Goal

Change the remote 1522 LoRA inference launcher so the user experience matches:
script start -> prompt for the SSH password once -> connect -> the remote node
loads the LoRA model -> the CLI dialog opens only after loading succeeds ->
enter a local image path and a question -> the image and text are sent to the
remote for inference -> the answer is printed locally -> wait for the next
round.
将远端 1522 LoRA 推理启动脚本改成如下流程：脚本启动 -> 提示输入一次 SSH
密码 -> 连接成功 -> 远端加载带 LoRA 权重的模型 -> 加载成功后弹出 CLI 对话框
-> 输入本地图片路径和文本 -> 图片与文本传到远端推理 -> 本地打印返回结果 ->
等待下一次输入。

## Modified Files

- `scripts/qwen3vl_lora_remote.py`: interactive password prompt, wait for the
  remote loaded marker, and stderr forwarding.
- `scripts/run_qwen3vl_lora_remote.sh`: updated header comments only.
- `tests/test_qwen3vl_lora_remote.py`: tests for the password prompt and the
  ready-marker wait.
- `README.md`: updated the LoRA inference CLI usage section.
- `DETAILS.md`: updated the two script responsibility entries.

## Core Changes

- When `SSHPASS` is unset, the local client now prompts for the SSH password
  once on a TTY (`getpass`) and supplies it to the single `ssh` session through
  `sshpass -e`; the password is not stored in any file and is not echoed.
  `SSHPASS` still overrides the prompt, and an empty `SSHPASS` can be used with
  key-based authentication.
  未设置 `SSHPASS` 时，本地客户端现在会在 TTY 上通过 `getpass` 提示输入一次
  SSH 密码，并用 `sshpass -e` 提供给唯一一条 `ssh` 会话；密码不落盘、不回显。
  `SSHPASS` 仍可覆盖提示；使用 SSH 密钥时可把 `SSHPASS` 设为空值。
- The client captures the remote stderr with a reader thread, prints remote
  loading progress locally, and waits for the remote `Model loaded. Ready.`
  marker before opening the CLI dialog (or before sending a one-shot command).
  If the remote process exits before the marker appears (for example a wrong
  password), the client reports the last stderr lines and exits.
  客户端用读线程接管远端 stderr，把远端加载进度打印到本地，并等待远端
  `Model loaded. Ready.` 标记后才弹出 CLI 对话框（单次模式则在标记后发送
  指令）。若远端在标记出现前退出（例如密码错误），客户端会显示最近的 stderr
  内容并退出。
- Interactive mode now prints `Model loaded. CLI dialog ready.` plus usage
  instructions, then loops: local image path + question -> base64 image and
  text over the same persistent SSH stream -> remote inference -> printed
  answer -> next input. `exit`/`quit` ends the session.
  交互模式现在先打印 `Model loaded. CLI dialog ready.` 和使用说明，然后循环：
  本地图片路径 + 问题 -> 通过同一条持久 SSH 流发送 base64 图片和文本 ->
  远端推理 -> 打印回答 -> 等待下一次输入。`exit`/`quit` 结束会话。
- `sshpass` availability is checked before connecting when a password is used.
  使用密码连接前会先检查本机是否安装 `sshpass`。

## Whether the Canonical Sample Format Was Changed

No. The change only affects the local SSH client and its documentation; no
sample schema, dataset adapter, or evaluation reader was touched.
否。本次只改动本地 SSH 客户端及其文档，不涉及样本格式、数据集适配器或评测
读取逻辑。

## Whether the Model Interface Was Changed

No. `scripts/qwen3vl_lora_cli.py` on the remote is unchanged, as are
`models/entry.py`, wrappers, and weight-loading logic. The ready marker it
already emits (`Model loaded. Ready.` on stderr) is used as the handshake.
否。远端 `scripts/qwen3vl_lora_cli.py` 未改动，`models/entry.py`、封装与权重
加载逻辑均未改动；客户端使用其原有的 stderr 标记
`Model loaded. Ready.` 作为握手信号。

## Whether the Configuration Was Changed

No existing configuration field changed. New behavior uses the existing
`SSHPASS` convention; no new config field was added.
否。现有配置字段未改动；新行为沿用现有 `SSHPASS` 约定，未新增配置字段。

## Whether Evaluation Was Affected

No metric, split, reference-answer reading, or result post-processing rule was
changed. The launcher is not part of `eval/`.
否。评测指标、数据集划分、标准答案读取与后处理规则均未改动；启动脚本不属于
`eval/`。

## Whether Deployment Was Affected

No deployment export or hardware path changed. The launcher still runs the
existing local-Transformers LoRA inference path on the remote 4090 node.
否。部署导出与硬件路径未改动；启动脚本仍在远端 4090 节点执行既有本地
Transformers LoRA 推理路径。

## Whether pytest Was Updated

Yes. Added tests in `tests/test_qwen3vl_lora_remote.py` for the explicit
password parameter, the TTY password prompt, the non-TTY rejection, and the
ready-marker wait (success and remote-exit failure cases).
是。在 `tests/test_qwen3vl_lora_remote.py` 中新增显式密码参数、TTY 密码提示、
非 TTY 拒绝，以及等待加载完成标记（成功与远端提前退出两种情形）的测试。

## Whether .gitignore Was Updated

No. No new output, checkpoint, cache, or dataset file type was introduced.
否。未引入新的输出、权重、缓存或数据集文件类型。

## Validation Method

```bash
/opt/miniconda3/envs/m3/bin/python -m compileall -q scripts/qwen3vl_lora_remote.py
bash -n scripts/run_qwen3vl_lora_remote.sh
/opt/miniconda3/envs/m3/bin/python -m pytest -q tests/test_qwen3vl_lora_remote.py
DRY_RUN=1 bash scripts/run_qwen3vl_lora_remote.sh --interactive
```

Syntax check, shell syntax check, the remote-client unit tests, and the
`DRY_RUN=1` command-printing path were run locally.
本地已执行语法检查、shell 语法检查、远端客户端单元测试与 `DRY_RUN=1` 命令
打印验证。

## Risks and Follow-up TODOs

- No real remote connection, password entry, model loading, or GPU inference
  was performed in this modification. A live smoke run on
  `100.88.222.9:1522` is still required to validate the full dialog flow.
  本次修改未执行真实远端连接、密码输入、模型加载或 GPU 推理；仍需在
  `100.88.222.9:1522` 上做一次真实冒烟运行验证完整对话框流程。
- The local machine must have `sshpass` installed when a password is used
  (`brew install sshpass`); with SSH keys, run with `SSHPASS=` set to an empty
  value to skip the prompt.
  使用密码连接时本机需安装 `sshpass`（`brew install sshpass`）；使用 SSH
  密钥时把 `SSHPASS` 设为空值即可跳过提示。
- The remote repository must already contain the synced
  `scripts/qwen3vl_lora_cli.py` with the `--server` ready marker; the previous
  modification note recorded that sync.
  远端仓库需已同步包含带 `--server` 就绪标记的
  `scripts/qwen3vl_lora_cli.py`；此前修改记录已记载该同步。
