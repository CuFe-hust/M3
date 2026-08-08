"""Centralized artifact persistence for dataset workflows.

数据集工作流的集中产物持久化。集中拥有所有运行产物文件名与原子写入
行为；纯写入器，不做任何业务判断（不计算指标、不读取模型响应目录推断
状态）。所有 JSON 写入都经过统一原子原语：临时文件写完再替换，任何
中途失败都不会暴露半个 JSON。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.base import AgentExecution, _validate_plain_basename
from data.adapters.base import AdapterProbe
from data.schema import UnifiedSample
from workflows.events import _atomic_replace, _path_lock, _reject_secrets
from workflows.schema import DatasetRunSummary, SampleRunStatus

# Owned artifact filenames. / 集中拥有的产物文件名。
SAMPLE_FILENAME = "sample.json"
STATUS_FILENAME = "status.json"
ROUTING_DECISION_FILENAME = "routing_decision.json"
AGENT_RESULT_FILENAME = "agent_result.json"
COUNTING_RESULT_FILENAME = "counting_result.json"
AGENT_TRACE_FILENAME = "agent_trace.json"
PREDICTIONS_FILENAME = "predictions.jsonl"
DATASET_SUMMARY_FILENAME = "dataset_summary.json"
DATASET_PROBE_FILENAME = "dataset_probe.json"


def atomic_write_json(path: Path, value: Any) -> None:
    """Publish a JSON artifact only after its temporary file is complete.
    仅在临时文件完整写入后发布 JSON 产物。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _atomic_replace(temporary, path)


def atomic_append_jsonl(path: Path, value: Any) -> None:
    """Append one JSON line atomically via a temporary file under a per-path
    lock. Safe for concurrent writers within one Python process; cross-process
    concurrent append is not supported by the current workflow layer. An
    interrupted write never exposes a half-written line.
    在按路径锁内通过临时文件原子追加一行 JSON。单进程内并发写入安全；
    当前工作流层不支持跨进程并发追加。中断写入不会暴露半行。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
    with _path_lock(path):
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(existing + line, encoding="utf-8")
        _atomic_replace(temporary, path)


class ArtifactWriter:
    """Write declared runtime artifacts without making business decisions.
    写入已声明的运行时产物，不执行任何业务判断。"""

    def write_sample(self, sample_dir: Path, sample: UnifiedSample) -> None:
        """Persist one canonical sample. / 持久化一条统一样本。"""

        atomic_write_json(sample_dir / SAMPLE_FILENAME, sample.model_dump(mode="json"))

    def write_running_status(self, sample_dir: Path, status: SampleRunStatus) -> None:
        """Persist the running status. / 持久化运行中状态。"""

        atomic_write_json(sample_dir / STATUS_FILENAME, status.model_dump(mode="json"))

    def write_routing(self, sample_dir: Path, routing: object) -> None:
        """Persist a routing decision. / 持久化路由决策。"""

        atomic_write_json(sample_dir / ROUTING_DECISION_FILENAME, _json_value(routing))

    def write_execution(self, sample_dir: Path, execution: AgentExecution) -> Path:
        """Persist an Agent payload under its declared result filename, plus
        every additional result under its own filename.
        使用 Agent 声明的结果文件名持久化载荷，并为每个附加结果写入独立
        文件。"""

        result_path = sample_dir / execution.result_filename
        atomic_write_json(result_path, _json_value(execution.payload))
        for filename, payload in execution.additional_results.items():
            atomic_write_json(sample_dir / filename, _json_value(payload))
        return result_path

    def write_evaluation(self, sample_dir: Path, evaluation: object, *, filename: str) -> Path:
        """Persist an evaluation under a caller-declared plain basename; any
        path-like filename is rejected before I/O.
        使用调用方声明的纯 basename 持久化评测结果；任何类路径文件名在
        I/O 前被拒绝。"""

        _validate_evaluation_filename(filename)
        path = sample_dir / filename
        atomic_write_json(path, _json_value(evaluation))
        return path

    def write_trace(self, sample_dir: Path, trace: dict[str, Any]) -> None:
        """Persist an auditable execution trace. / 持久化可审计执行轨迹。"""

        atomic_write_json(sample_dir / AGENT_TRACE_FILENAME, trace)

    def write_final_status(self, sample_dir: Path, status: SampleRunStatus) -> None:
        """Persist the final sample status. / 持久化最终样本状态。"""

        atomic_write_json(sample_dir / STATUS_FILENAME, status.model_dump(mode="json"))

    def append_prediction(
        self,
        run_dir: Path,
        *,
        sample_id: str,
        task: str,
        status: SampleRunStatus,
    ) -> None:
        """Append one stable prediction index row. / 追加一条稳定预测索引记录。"""

        value = {
            "sample_id": sample_id,
            "task": task,
            "status": status.state,
            "result_path": str(status.result_path) if status.result_path else None,
        }
        atomic_append_jsonl(run_dir / PREDICTIONS_FILENAME, value)

    def write_summary(self, run_dir: Path, summary: DatasetRunSummary) -> None:
        """Persist the dataset summary. / 持久化数据集汇总。"""

        atomic_write_json(run_dir / DATASET_SUMMARY_FILENAME, summary.model_dump(mode="json"))

    def write_dataset_probe(self, run_dir: Path, probe: AdapterProbe) -> Path:
        """Persist the dataset layout probe as its own artifact without ever
        touching manifest.json, which must stay parseable by the RunManifest
        schema. The payload is JSON-safe and secret-scanned.
        将数据集布局 probe 单独持久化为独立产物，绝不触碰必须保持 RunManifest
        schema 可解析的 manifest.json。载荷 JSON 安全且经过敏感扫描。"""

        payload: dict[str, Any] = {
            "dataset": probe.dataset,
            "version": probe.version,
            "sample_file": probe.sample_file.as_posix(),
            "observed_fields": list(probe.observed_fields),
            "sample_count": probe.sample_count,
        }
        if probe.task is not None:
            payload["task"] = probe.task
        if probe.available_tasks:
            payload["available_tasks"] = list(probe.available_tasks)
        _reject_secrets(payload, "dataset probe")
        path = run_dir / DATASET_PROBE_FILENAME
        atomic_write_json(path, payload)
        return path


def _json_value(value: object) -> Any:
    """Convert supported schema objects to JSON-compatible values.
    将受支持的 Schema 对象转换为 JSON 兼容值。"""

    model_dump = getattr(value, "model_dump", None)
    return model_dump(mode="json") if callable(model_dump) else value


def _validate_evaluation_filename(filename: str) -> str:
    """Reuse the AgentExecution basename contract (POSIX + Windows semantics)
    and additionally reject control characters, so an evaluation filename can
    never escape the sample directory. 复用 AgentExecution 的 basename 契约
    （POSIX + Windows 语义）并额外拒绝控制字符，使评测文件名绝不逃逸样本
    目录。"""
    _validate_plain_basename(filename, "evaluation filename")
    if any(character in filename for character in ("\x00", "\n", "\r")):
        raise ValueError("evaluation filename contains control characters")
    return filename
