# Modification Note: Merge and Shuffle VRSBench + LEVIR-CC Fine-tuning Datasets - 2026-08-03 16:48:00 +0800

> Update 2026-08-03 16:59 +0800: after full verification, the legacy
> `vrsbench/` and `levir-cc/` folders were deleted with the owner's explicit
> confirmation; a post-deletion integrity check confirmed `merged/` is fully
> intact (all 64262/7953/7960 image references and both reference files
> resolve, all image directory counts match).

## Modification Time

2026-08-03 16:48:00 +0800（旧版删除：2026-08-03 16:59 +0800）

## Modifier

tRoy <791056216@qq.com>

## Modification Goal

Build a single merged, shuffled fine-tuning dataset from the existing VRSBench
and LEVIR-CC ShareGPT datasets so that one LLaMA-Factory `--media_dir`
(`data/微调数据集/merged/`) resolves every image during mixed InternVL LoRA
training. The merge was built as a new copy; the original `vrsbench/` and
`levir-cc/` folders were left untouched pending owner confirmation.

## Modified Files

- `data/微调数据集/merged/`（新增目录；旧版 `vrsbench/`、`levir-cc/` 校验后已删除）
  - `train.json`（60262 条）、`val.json`（7453 条）、`test.json`（7460 条）
  - `train/`、`val/`、`test/` 下 `vrsbench/` 与 `levir-cc/{A,B}/` 图像（APFS clone）
  - `levir_cc_val_references.json`、`levir_cc_test_references.json`
  - `merge_manifest.json`、`vrsbench_split_manifest.json`
  - `prepare_vrsbench_10k.py`、`prepare_levir_cc_sharegpt.py`
- `DETAILS.md`（第 15 节改写为合并版布局说明）

## Core Changes

- 将 vrsbench 与 levir-cc 的记录按 split 合并为统一 `train/val/test.json`，
  记录内 `images` 路径改写为相对 `merged/` 的形式
  （`train/vrsbench/xxx.png`、`train/levir-cc/A/xxx.png`），
  与图像新位置一致。
- 用固定种子 `20260803`（记录于 `merge_manifest.json`）在各 split 内部打乱，
  两个数据集的记录交叉排列（train 交叉切换 7485 次）。
- 图像以 APFS clonefile 复制（`cp -c`），不额外占用磁盘块；文件数与字节数
  与源目录逐一相等。
- 无任何样本跨 split 移动；记录内容除 `images` 路径外逐字段不变；LEVIR-CC
  评测 references 的五条参考答案逐字不变。
- 构建与校验阶段未改动原 `vrsbench/`、`levir-cc/` 目录；校验全部通过后，
  经数据负责人明确确认，于 2026-08-03 16:59 删除了这两个旧目录。

## Whether the Canonical Sample Format Was Changed

No. `data/schema.py` 的 `CanonicalSample` / `CanonicalPrediction` 未改动；
合并记录的字段结构与原 ShareGPT 记录完全一致。

## Whether the Model Interface Was Changed

No.

## Whether the Configuration Was Changed

No repository configuration was changed. LLaMA-Factory training should use
`--media_dir data/微调数据集/merged`（或服务器上的对应路径）。

## Whether Evaluation Was Affected

No metric, split membership, or reference-answer content was changed. LEVIR-CC
val/test 参考文件内容与旧版逐字一致（仅路径形式变化），历史评测口径保持可比。
test 集与 references 仍仅用于最终评测，不得用于训练或 Prompt 开发。

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

No code changed (dataset files only), so no pytest update is required.
Verification was done by read-only scripts as described below.

## Whether .gitignore Was Updated

No new file types were introduced; `data/微调数据集/` and `*.tar` are already
ignored (see `docs/changes/2026-08-03-troy-levir-cc-relative-image-paths.md`).

## Validation Method

- 多重集合等价：按 split 比较（旧 vrsbench + 旧 levir-cc）与合并版记录
  （忽略 images 路径、按内容哈希计数），三者完全一致（60262/7453/7460）。
- 路径解析：合并版全部 64262+7953+7960 条图像引用及 references 的 400 条
  引用均在 `merged/` 下存在；`<image>` 占位符数量与图像数一致。
- split 字段一致性：合并版中 levir-cc 记录的 `split` 字段与所在文件一致。
- 图像完整性：9 个图像目录逐一比对文件数与总字节数全部相等；每个 split
  随机抽样 60 张图像做 sha256 比对全部一致。
- 打乱交叉性：统计相邻记录数据集来源切换次数（train=7485，val=908，
  test=932），确认充分交叉。
- 来源文件：`vrsbench_split_manifest.json` 与两个生成脚本与原件 `cmp`
  逐字节一致。

## Risks and Follow-up TODOs

- 旧版 `data/微调数据集/vrsbench/` 与 `levir-cc/` 已于 2026-08-03 16:59
  删除，`merged/` 为微调数据的唯一磁盘副本；如需重建，须按生成脚本在
  制备服务器上重新生成后再做合并。
- `data/finetone_dataset.tar`（旧布局传输包）已与当前布局不符，如需传输
  合并版须重新打包。
- `merge_manifest.json` 记录了打乱种子与来源，重建合并版时应保持相同种子
  以保证可比性。
- 本机未安装 LLaMA-Factory，未做 `llamafactory-cli` 端到端加载验证。
