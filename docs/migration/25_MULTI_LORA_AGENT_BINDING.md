# Doc 25 多 LoRA 运行身份迁移说明

本变更不修改 task、Agent 协议、数据选择、Ground Truth、确定性指标或 Judge 语义，但会改变
启用 LoRA 后的模型输出分布，因此历史结果与新结果只有在 base/adapter 逻辑身份及 revision
一致时才可直接比较。

新运行冻结：

```text
base logical model id/revision
adapter logical id/adapter_model.safetensors SHA-256
planner/counting/change/grounding/general_vqa/caption binding
client version
```

物理 checkpoint/adapter 路径不属于可比身份，也不进入 `run_request.json`、trace、prediction
或 report。`config.snapshot.json` 仍按既有复现契约保存配置中的主机路径。

Doc 25 之前没有 `run_request.qwen_runtime_identity` 的运行明确解释为 legacy base-only；resume
不得把它猜成当前配置的 adapter。需要重新推理且当前 runtime 不是 base-only 时稳定拒绝。
已经 succeeded 且产物有效的样本仍按既有规则零模型复用/补评测，不因当前 binding 改变而生成
第二份预测。

`tests/fixtures/migration/` 的既有 Golden fixture 未修改。Parity 测试单独断言新增的逻辑
binding 元数据，再对原功能字段使用既有 fixture 比较。
