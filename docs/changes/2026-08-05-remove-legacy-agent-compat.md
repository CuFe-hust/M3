# 2026-08-05 Remove legacy Agent compatibility

## Scope / 范围

The runtime now uses only `TaskRouter` decisions and concrete registered Agents. The retired
`spacers_agent.workflow`, `spacers_agent.counting`, and `spacers_agent.experts` modules were
deleted. Runtime name aliases, list-based routing input, and deprecated wrapper classes were also
removed.

运行时现在只使用 `TaskRouter` 决策和已注册的具体 Agent。已删除退役的
`spacers_agent.workflow`、`spacers_agent.counting` 和 `spacers_agent.experts` 模块，同时移除了
运行时名称别名、基于列表的路由输入以及已废弃的包装类。

## Artifact contract / 产物契约

Non-counting execution writes `AgentResult` to `agent_result.json`, with a required canonical
`agent_name`. Counting-only execution continues to write `CountingResult` to
`counting_result.json`. Existing evidence validation and VRSBench geometry repair remain intact.

非计数执行将带有规范 `agent_name` 的 `AgentResult` 写入 `agent_result.json`。仅计数执行继续将
`CountingResult` 写入 `counting_result.json`。既有证据校验和 VRSBench 几何修复保持不变。
