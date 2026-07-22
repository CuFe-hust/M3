# Qwen3-VL-4B 遥感多 Agent 系统：Phase 4 点式计数状态

## Phase 4 status

- Completed locally: sequential owner-core point counting, `CountTargetSpec`, request-hash resume,
  tile checkpoints, strict tile/target validation, recursive parent replacement, and local boundary
  conflict candidate generation with explicit union-find merges.
- Validation: local Mock integration and geometry tests only; no server, SSH tunnel, Qwen, DeepSeek,
  cloud API, dataset download, or model weight download was used.
- Deferred: target-spec LLM parsing, live seam verification, missing-point review, dataset Adapter
  integration, and end-to-end live smoke testing all require further explicit authorization.

## Phase 5 status

- Completed locally: deterministic task routing, optional unknown-task text routing, bounded Qwen/
  DeepSeek call budgets, a point-derived Counting Expert presentation layer, and versioned prompt
  assets captured by `run-init`.
- Validation: route and budget behavior uses only injected Mock clients; no visual critic, DeepSeek
  judge, Qwen, server, or cloud call was made.
- Deferred: live target parsing, change/grounding/spatial/general-VQA expert execution, visual critic,
  seam verification, missing-point review, and DeepSeek structured judging remain opt-in future work.

## Phase 6 status

- Completed locally: deterministic count metrics, compact text-only judge payloads, scoped judge
  Schema, conflict flags, DeepSeek JSON-mode client, bounded retries, one repair, cache, and safe
  artifacts for raw/parsed/validation/latency/token data.
- API setup: copy `.env.example` to ignored `.env`, then replace only
  `DEEPSEEK_API_KEY=replace-with-deepseek-key`; never commit `.env` or put the key in YAML/code.
- Validation: all DeepSeek tests inject local fake completions. No DeepSeek API or cloud call was made.

## Phase 7 status

- Completed locally: owner-core/point audit overlays, deterministic evaluation summaries, local CLI
  commands, runbook, and tests covering these outputs.
- Acceptance boundary: geometry, Mock integration, resume, JSON recovery, budgets, and non-visual
  Judge constraints are tested. Four real dataset adapters, live endpoints, end-to-end samples, and
  benchmark/ablation outcomes remain unvalidated and must not be claimed as complete.

# Historical Phase 0 audit status

## 审计范围

- 审计时间：2026-07-22 02:00:43 +08:00
- 工作分支：`docs/phase0-audit`
- 审计方式：仅本地只读检查；未连接服务器、未访问云端 API、未下载数据集或模型权重。
- 当前阶段：Phase 3 已完成（统一 Schema、纯图像几何与只读数据审计）。Phase 4 及之后的点式计数、聚合、评估和数据适配工作尚未开始。

## 当前仓库状态

仓库是一个可在 Colab 执行的 Qwen3-VL-4B 零样本基线，而非现成的多 Agent 系统。当前入口为 `main.py`，使用 JSON 配置，直接通过 Transformers 加载 `Qwen/Qwen3-VL-4B-Instruct`；已有下载、检查、推理和评测子命令。

已跟踪的核心代码目录为 `data/`、`models/`、`eval/` 和 `tests/`。`config/` 中只有 `baseline.example.json`；没有 `pyproject.toml`、`setup.py`、`src/` 包目录、Pydantic Settings、运行 manifest、请求缓存、结构化日志、恢复检查点或独立 Prompt 文件。

Git 工作树在审计前为空。为满足仓库的隔离开发要求，已仅在本地创建 `docs/phase0-audit` 分支；未推送远端。

## 可复用模块

| 现有位置 | 可复用内容 | 后续使用边界 |
| --- | --- | --- |
| `data/schema.py` | `CanonicalSample`、`CanonicalPrediction` 及序列化方法 | 现为 dataclass，任务类型不含 `count`；如迁移到 Pydantic v2，需保持既有基线字段和持久化 JSONL 兼容。 |
| `data/loaders.py` | VRSBench、MME-RealWorld、XLRS、LEVIR-CC 基线适配器和图像解析辅助函数 | 目前包含候选字段/文件名推断；新 Adapter 必须先以真实本地布局验证，不可沿用猜测作为事实。 |
| `models/qwen3vl.py` | 统一样本到统一预测的本地 Transformers 推理路径 | 必须保留，不能被后续 vLLM 客户端替换或破坏。 |
| `eval/metrics.py` | 已有确定性文本/框指标和可选 DeepSeek 代理指标 | 不改动现有指标、分割或答案读取；新的计数指标应独立加入，DeepSeek 仅作文本与结构化证据判断。 |
| `main.py` | 基线 CLI、配置读取和 JSONL 记录 | 新 CLI 应避免破坏现有 `python main.py --config ...` 入口。 |
| `tests/` | Schema 与基线 Adapter 的 pytest 覆盖 | 可作为新增几何、配置、manifest、CLI 测试的风格参考。 |

## 真实数据布局审计

本地仓库根目录不存在 `datasets/`，也没有 LEVIR-CC、MME-RealWorld、VRSBench 或 XLRS-Bench-lite 的数据副本。只读扫描仅发现源码中的预期目录名和下载仓库标识，未发现 manifest、split 文件、图像文件、问题/答案字段或 bbox/point 标注文件。

第 03 节指定的 `dataset/` 根目录同样不存在；`dataset/LEVIR-CC`、`dataset/MME-RealWorld`、`dataset/VRSBench` 和 `dataset/XLRS-Bench-lite` 的只读路径检查均为不存在。因此，`inspect-data --root ./dataset` 目前不能对真实数据执行，且不得据该目录名推断 Adapter 字段映射。

因此，以下事项均为“未观察到”，不是“该数据集不具备”：

- LEVIR-CC 的标注文件与 A/B 图像相对路径；
- MME-RealWorld 的 Remote Sensing 过滤字段、题目字段和标准答案字段；
- VRSBench 的 validation manifest、QA 字段及 grounding 标注结构；
- XLRS-Bench-lite 的本地 Arrow/Parquet/JSON 布局、split 名称和数字答案表达方式；
- 可作为第一批点式计数验证样本的真实目标类别和 count 真值。

在用户提供这些本地只读数据路径，或明确授权本地数据准备后，必须先实现并运行 `inspect-data`，再决定 Adapter 字段映射。不能仅凭当前 `data/loaders.py` 的启发式字段名确认真实布局。

## 本地环境

- Python：3.13.12（满足实施指令的 Python 3.11+ 要求）。
- Windows Python Launcher 中未发现 `py -3.11` 解释器；第 02 节规定的 `py -3.11 -m venv .venv` 命令当前不能执行。
- 已安装：Pydantic 2.12.5、`openai`、`httpx`、`PyYAML`、Pillow。
- 本地 `m3` Conda 环境已升级为 Python 3.11.15，并安装 `pydantic-settings`、`openai` 与 `pytest-asyncio`。`requirements.txt` 和 `pyproject.toml` 已声明这些依赖；配置读取现使用 Pydantic Settings 且不将 API-key 值写入模型或产物。
- 未执行模型加载、数据集加载、网络健康检查或 DeepSeek 调用。

## 云端环境待确认项

实施指令预设 vLLM OpenAI-compatible Qwen 服务和 DeepSeek API；但当前用户约束是“所有代码只在本地完成，不连服务器、不使用云端，除非代码完成且明确要求”。因此，以下均被延后，不能在当前阶段验证：

- `http://127.0.0.1:8000/v1` 是否有本地 vLLM 服务、服务模型名及其 JSON 输出兼容性；
- 远端 Linux GPU、vLLM 版本、显存、并发及图像限制；
- DeepSeek 凭据、网络连通性、模型可用性和响应格式；
- 模型、数据集和第三方运行时是否已完整安装。

第 02 节的服务器审计、vLLM 安装和锁定、模型 SHA256 清单、`vllm serve`、服务器/隧道健康检查以及 OOM 调优均需要后续明确授权，且只能在实际服务器环境执行。

后续实现可提供不联网的 Mock 客户端和纯程序几何测试；真实 Qwen/DeepSeek smoke test 必须等用户完成代码后明确授权。

## 可复用 / 需新增 / 冲突 / 风险

### 可复用

现有 canonical JSONL 记录、统一样本/预测语义、基线本地模型包装、基线 CLI 和 pytest 基础可被保留为兼容层。

### 需新增

Phase 1 已新增 `spacers_agent/` 包、`python -m spacers_agent.cli`、Pydantic v2 配置模型、`.env.example`、机器可读错误码、结构化 JSONL 事件、run manifest、配置与 Prompt 快照以及 CLI 测试。后续 Phase 2--10 还需要纯程序切片几何、Mock/VLLM 客户端、点式计数、边界复核、恢复、数据检查和报告功能。

Phase 2 已新增 `VisionLanguageClient` 异步协议、`QwenVLLMClient`、`MockVisionClient`、Base64 data URI、请求哈希缓存、有限重试、Markdown JSON fence 清理、Pydantic 校验、一次 JSON 修复、raw/parsed/validation artifact 落盘和 token/latency 记录。所有验证均使用注入 Mock；未连接 Qwen、DeepSeek、SSH 隧道或服务器。

Phase 3 已新增统一 Pydantic 样本/切片/点/计数 Schema、EXIF/RGB 图像规范化、非重叠 owner core + halo、缩放与 `0..999` 坐标换算、严格所有权判定和 `inspect-data` 只读 CLI。验证使用临时 fixture；真实 `dataset/` 根目录仍不存在，因此没有猜测任何 Adapter 字段映射。

### 已识别冲突

1. 实施指令固定云端 vLLM/DeepSeek 路径；用户明确禁止当前连接服务器和云端。用户约束优先，当前只能做离线实现与 Mock 验证。
2. 实施指令要求 Pydantic v2 schema 和新包 CLI；基线目前使用 dataclass、单文件 `main.py` 与 JSON 配置。迁移必须增量进行，且保留已有 CLI、基线配置和 JSONL 兼容性。
3. 实施指令以点式 `count` 为核心任务；现有 `VALID_TASK_TYPES` 不含 `count`，虽 `CanonicalPrediction` 已有 `count` 字段。增加样本任务类型会改变规范样本格式，需以兼容方式设计并补齐测试、`DETAILS.md` 和变更记录。
4. 实施指令要求先按真实布局实现 Adapter；现有 XLRS 加载器在本地数据不存在时会调用 `datasets.load_dataset`，而 `download` 子命令也会调用 Hugging Face。这些联网路径在当前约束下不能运行，且新实现不得把网络回退作为默认行为。
5. 实施指令要求在每个 Phase 完成后停下并汇报；因此本次不会跨入 Phase 1。

### 风险

- 当前 `data/loaders.py` 的字段和文件选择依赖启发式匹配；未验证真实数据时不宜扩展为计数真值读取逻辑。
- 当前基线的可选 DeepSeek 代理使用同步 `urllib`；新评估器要求异步、缓存、结构化证据与视觉核验限制，必须作为新路径实现，不能改变现有代理指标的定义。
- 原有 `.gitignore` 已忽略 `datasets/`、`outputs/` 和部分本地 JSON 配置；Phase 1 引入 `.env`、运行缓存或新本地配置时必须补充规则，且不得提交 API key、Base64、数据、权重或日志。
- 没有本地图像和计数真值，无法验证 Prompt、切片策略、点归属、去重或计数指标。

## 文件变更计划

| 阶段 | 最小预期修改 | 阶段门 |
| --- | --- | --- |
| Phase 1 | 已新增包/CLI、Pydantic v2 settings 模型、运行存储、`.env.example`、Prompt 快照与测试；保留基线入口 | 配置、manifest、CLI help 测试已通过；Pydantic Settings 已在本地 `m3` 环境验证。 |
| Phase 2 | 已新增 async Qwen 协议/实现、Mock、JSON 修复、缓存与安全 artifact 持久化 | Mock 客户端、重试、修复、缓存、Base64 脱敏测试通过；所有 live smoke 仍待用户明确授权。 |
| Phase 3 | 已新增统一 Schema、纯程序图像、owner core/halo、坐标、所有权与只读数据审计 | 半开区间、EXIF、缩放、边界、随机覆盖、Schema 非法值与审计 fixture 测试通过。 |
| Phase 4--5 | 实现单 tile 点计数、全局聚合、边界冲突和 checkpoint | 使用 Mock/fixture 验证 `final_count` 只来源于接受点。 |
| Phase 6 | 实现不联网可测的评估记录拼装；DeepSeek 调用保持显式可选 | 确定性计数指标与 judge scope 测试通过。 |
| Phase 7--10 | 在真实本地数据审计后逐个 Adapter、其他 Agent、消融、恢复与报告 | 每阶段单独审计、测试和汇报，不跨阶段实施。 |

## 进入 Phase 1 前的未决项

1. 确认在保留既有基线接口的前提下，可以新增 `count` 任务类型与兼容的 Pydantic v2 schema。
2. 提供或指定可只读访问的本地数据根目录，供后续 `inspect-data` 审计；否则 Adapter 实现只能停留在无真实布局的 fixture 层。
3. 确认代码完成前的验证范围限于离线单元测试、Mock 客户端和静态检查；真实 vLLM/DeepSeek health 与 smoke 需要后续明确授权。

## 本阶段验证

- 已读取：`AGENTS.md`、`DETAILS.md`、`README.md`、核心源码、配置、测试、requirements、`.gitignore` 与历史变更记录。
- 已输出：根目录深度 3 的只读目录树，并搜索 Python 打包、配置、测试、脚本、API client、Adapter、日志、评测与缓存相关实现。
- 已执行：本地 Python/依赖可用性探测、只读数据目录扫描、`git diff --check` 与 `python -m pytest -q`。
- pytest 结果：`python -m pytest -q` 通过（5 passed）。直接执行 `pytest -q` 在收集阶段报 `ModuleNotFoundError: data`；这是当前测试依赖以模块方式从仓库根目录启动的基线可复现性风险，Phase 1 应在不破坏既有入口的前提下解决或明确标准测试命令。
