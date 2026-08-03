# Modification Note: Rewrite LEVIR-CC Fine-tuning Image Paths to Relative Form - 2026-08-03 16:34:10 +0800

## Modification Time

2026-08-03 16:34:10 +0800

## Modifier

tRoy <791056216@qq.com>

## Modification Goal

Make the LEVIR-CC fine-tuning JSON files usable with LLaMA-Factory on any training
machine. Their `images` fields contained absolute paths from the preparation server
(`/home/user/下载/datasets/微调数据集/levir-cc/...`), which fail to load once the
dataset is moved. Paths were rewritten to the same relative form used by the
vrsbench fine-tuning dataset, i.e. relative to each dataset's own root directory.

## Modified Files

- `data/微调数据集/levir-cc/train/train.json`（4000 条记录，8000 条路径）
- `data/微调数据集/levir-cc/val/val.json`（500 条记录，1000 条路径）
- `data/微调数据集/levir-cc/test/test.json`（500 条记录，1000 条路径）
- `data/微调数据集/levir-cc/val/val_references.json`（100 条记录，200 条路径）
- `data/微调数据集/levir-cc/test/test_references.json`（100 条记录，200 条路径）
- `.gitignore`（新增忽略 `data/微调数据集/`、`*.tar` 与 `*.tgz`）
- `DETAILS.md`（新增微调数据集路径约定说明）

## Core Changes

- 将上述 5 个 JSON 中 `images` 字段的服务器绝对路径统一改写为相对于
  `data/微调数据集/levir-cc/` 目录的相对路径，例如
  `/home/user/下载/datasets/微调数据集/levir-cc/train/A/train_005870.png`
  → `train/A/train_005870.png`，与 vrsbench 数据集 `images/train/xxx.png`
  的相对形式一致。共改写 10400 条路径，前缀逐条严格校验，不匹配即中止。
- 记录内容（对话、回答、id、pair_id、caption_id、split、changeflag）、记录数、
  JSON 书写风格（`ensure_ascii=False`、`indent=2`、末尾换行）均保持不变；
  图像文件本身未改动。
- 训练时 LLaMA-Factory 需将 `--media_dir` 指向各自的 dataset 根目录
  （vrsbench 目录 / levir-cc 目录）；混合训练时建议将两个数据集放在同一
  media 根目录下并相应组织相对路径。
- `.gitignore` 补充忽略微调数据集目录与 tar 传输包，防止大数据集误提交。

## Whether the Canonical Sample Format Was Changed

No. `data/schema.py` 的 `CanonicalSample` / `CanonicalPrediction` 未改动。

## Whether the Model Interface Was Changed

No.

## Whether the Configuration Was Changed

No repository configuration was changed. Fine-tuning launch commands need
`--media_dir` pointed at the dataset root so the relative image paths resolve.

## Whether Evaluation Was Affected

No metric, split, or reference-answer logic was changed. The val/test reference
files keep identical content apart from the path form, so evaluation results
remain comparable. Note: LEVIR-CC test/val must still not be used for training
or prompt development.

## Whether Deployment Was Affected

No.

## Whether pytest Was Updated

No code changed (dataset JSON files only), so no pytest update is required.
Post-change validation was done by a read-only script checking record counts,
path form, path existence relative to the levir-cc root, and `<image>` tag
counts (all OK, 10400/10400 paths resolve).

## Whether .gitignore Was Updated

Yes. Added `data/微调数据集/`, `*.tar`, and `*.tgz` under a new fine-tuning
section; `models/InternVL3_5-8B/` and `*.safetensors` were already present
(added by a parallel change recorded in
`docs/changes/2026-08-03-troy-gitignore-model-download.md`).

## Validation Method

- 改写脚本对每条路径做前缀断言，发现非预期前缀立即中止（未触发）。
- 改写后复验：5 个文件记录数不变（4000/500/500/100/100），无绝对路径残留，
  全部 10400 条相对路径在 `levir-cc/` 下存在，`<image>` 占位符数量与图像数
  一致（双图双占位符）。
- `data/finetone_dataset.tar` 为数据集传输副本，本次修改后需重新打包才能
  携带新路径（见风险）。

## Risks and Follow-up TODOs

- `data/finetone_dataset.tar`（4.1G 传输包）内的 LEVIR-CC JSON 仍是旧绝对路径，
  传输到训练服务器前必须重新打包。
- `prepare_levir_cc_sharegpt.py` 的默认参数仍为
  `--image-path-mode absolute` 和服务器默认路径；若在服务器上重新生成数据集，
  需显式传 `--image-path-mode relative`，否则会再次产出绝对路径。
- vrsbench 与 levir-cc 的相对路径分别以各自目录为基准；LLaMA-Factory 的
  `--media_dir` 是全局参数，混合训练时需把两个数据集统一到同一 media 根目录。
- 本次为数据文件修改，未运行 `llamafactory-cli` 端到端加载验证（本机未安装
  LLaMA-Factory）。
