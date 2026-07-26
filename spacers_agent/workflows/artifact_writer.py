"""Artifact persistence primitives for the new runtime.
新运行时的产物持久化原语。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spacers_agent.agents.base import AgentExecution
from spacers_agent.schemas import DatasetRunSummary, SampleRunStatus, UnifiedSample


def atomic_write_json(path: Path, value: Any) -> None:
    """Publish a JSON artifact only after its temporary file is complete.
    仅在临时文件完整写入后发布 JSON 产物。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class ArtifactWriter:
    """Write declared runtime artifacts without making business decisions.
    写入已声明的运行时产物，不执行任何业务判断。
    """

    def write_sample(self, sample_dir: Path, sample: UnifiedSample) -> None:
        """Persist one canonical sample. / 持久化一条统一样本。"""

        atomic_write_json(sample_dir / "sample.json", sample.model_dump(mode="json"))

    def write_running_status(self, sample_dir: Path, status: SampleRunStatus) -> None:
        """Persist the running status. / 持久化运行中状态。"""

        atomic_write_json(sample_dir / "status.json", status.model_dump(mode="json"))

    def write_routing(self, sample_dir: Path, routing: object) -> None:
        """Persist a routing decision. / 持久化路由决策。"""

        atomic_write_json(sample_dir / "routing_decision.json", _json_value(routing))

    def write_execution(self, sample_dir: Path, execution: AgentExecution) -> Path:
        """Persist an Agent payload under its declared result filename.
        使用 Agent 声明的结果文件名持久化载荷。
        """

        result_path = sample_dir / execution.result_filename
        atomic_write_json(result_path, _json_value(execution.payload))
        for filename, payload in execution.additional_results.items():
            atomic_write_json(sample_dir / filename, _json_value(payload))
        return result_path

    def write_evaluation(self, sample_dir: Path, evaluation: object, *, filename: str) -> Path:
        """Persist an evaluation under a caller-declared filename.
        使用调用方声明的文件名持久化评测结果。
        """

        path = sample_dir / filename
        atomic_write_json(path, _json_value(evaluation))
        return path

    def write_trace(self, sample_dir: Path, trace: dict[str, Any]) -> None:
        """Persist an auditable execution trace. / 持久化可审计执行轨迹。"""

        atomic_write_json(sample_dir / "agent_trace.json", trace)

    def write_final_status(self, sample_dir: Path, status: SampleRunStatus) -> None:
        """Persist the final sample status. / 持久化最终样本状态。"""

        atomic_write_json(sample_dir / "status.json", status.model_dump(mode="json"))

    def append_prediction(
        self,
        run_dir: Path,
        *,
        sample_id: str,
        task: str,
        status: SampleRunStatus,
    ) -> None:
        """Append one legacy-compatible prediction index row.
        追加一条兼容旧格式的预测索引记录。
        """

        path = run_dir / "predictions.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {
            "sample_id": sample_id,
            "task": task,
            "status": status.state,
            "result_path": str(status.result_path) if status.result_path else None,
        }
        with path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")

    def write_summary(self, run_dir: Path, summary: DatasetRunSummary) -> None:
        """Persist the dataset summary. / 持久化数据集汇总。"""

        atomic_write_json(run_dir / "dataset_summary.json", summary.model_dump(mode="json"))


def _json_value(value: object) -> Any:
    """Convert supported schema objects to JSON-compatible values.
    将受支持的 Schema 对象转换为 JSON 兼容值。
    """

    model_dump = getattr(value, "model_dump", None)
    return model_dump(mode="json") if callable(model_dump) else value
