# 修改说明：清理空占位目录-2026-07-10

## 修改目标

清理当前仓库中仅由 `.gitkeep` 占位的空目录，并补充项目 AI 协作说明文档。

## 修改文件

- 新增 `AGENTS.md`
- 新增 `CLAUDE.md`
- 新增 `DETAILS.md`
- 删除 `config/.gitkeep`
- 删除 `docs/.gitkeep`
- 删除 `main/agent/.gitkeep`
- 删除 `main/model/.gitkeep`
- 删除 `main/pytest/.gitkeep`
- 删除 `route/.gitkeep`
- 新增 `docs/changes/2026-07-10-agent-clean-empty-placeholders.md`

## 核心改动

移除只包含 `.gitkeep` 的空目录占位文件，使仓库不再提前保留没有实际代码或文档内容的目录结构。

## 是否改变统一样本格式

否。

## 是否改变模型接口

否。

## 是否改变配置

否。

## 是否影响评测

否。

## 是否影响部署

否。

## 是否更新 pytest

否。本次未修改代码行为、数据接口或评测逻辑。

## 是否更新 .gitignore

否。本次未新增输出、缓存、权重、日志或本地配置文件类型。

## 验证方式

- 确认待清理目录在远端 `main` 中仅包含 `.gitkeep`。
- 使用 `git status --short` 检查暂存前变更范围。

## 风险和后续 TODO

- 本次只清理空占位目录，不涉及训练、推理、评测或部署验证。
