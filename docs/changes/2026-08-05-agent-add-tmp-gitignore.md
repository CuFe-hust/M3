# Modification Note: Add tmp/ to .gitignore - 2026-08-05 11:07

## Modification Time

2026-08-05 11:07:00 +0800 (local)

## Modifier

Cooper (crj31415926@gmail.com)

## Modification Goal

将 `tmp/` 加入 `.gitignore`，避免本地临时测试脚本、输出图像与 HTML 报告被误提交到远程 GitHub 仓库。

## Modified Files

- `.gitignore`

## Core Changes

- 在 `.gitignore` 末尾新增 `tmp/` 条目（含注释 `# Temporary local scratch space (test scripts, outputs, venv) — never commit`），使仓库根目录下 `tmp/` 内的临时测试脚本、批量输出、HTML 报告等一律不被 Git 跟踪。
- 未改动任何代码、配置、接口、评估逻辑或样本格式。

## Whether the Canonical Sample Format Was Changed

否。

## Whether the Model Interface Was Changed

否。

## Whether the Configuration Was Changed

否（`.gitignore` 属于仓库忽略规则，不改变任何运行配置）。

## Whether Evaluation Was Affected

否。

## Whether Deployment Was Affected

否。

## Whether pytest Was Updated

否（无代码行为变更，无需新增测试；已用 `git check-ignore -v tmp/` 验证忽略规则生效）。

## Whether .gitignore Was Updated

是（本次修改本身即为 `.gitignore` 更新）。

## Whether DETAILS.md Was Updated

否（不涉及项目结构、接口或约定变更；`tmp/` 为临时目录，不进版本库）。

## Validation Method

- `git check-ignore -v tmp/` 输出 `.gitignore:35:tmp/	tmp/`，确认忽略规则生效。
- `git status --short` 仅显示 `.gitignore` 被修改，`tmp/` 目录内容未被跟踪。

## Risks and Follow-up TODOs

- 无。`tmp/` 为本地临时工作区，任何放置其中的脚本与产物均不会被提交；若后续需要共享其中的脚本，应迁移到仓库正式目录并补充文档与测试。
