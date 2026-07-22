from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from spacers_agent.cli import build_parser
from spacers_agent.counting import PointCountingOrchestrator
from spacers_agent.dataset_adapters import DatasetProbeError, get_adapter
from spacers_agent.schemas import CountTargetSpec, TileCountResponse
from spacers_agent.settings import CountingSettings, QwenSettings


class LowConfidenceClient:
    """Return one low-confidence point without networking. / 在不联网时返回一个低置信度点。"""

    async def complete_json(self, *, messages: list[dict], response_model: type[TileCountResponse], request_meta: object) -> TileCountResponse:
        """Return a schema-valid tile response. / 返回符合 Schema 的 tile 响应。"""

        return response_model.model_validate({"target": "building", "tile_id": "r000_c000", "reported_count": 1, "points": [{"local_id": "p1", "x": 500, "y": 500, "confidence": 0.1, "short_evidence": "roof"}]})


def test_new_cli_contracts_are_parseable() -> None:
    """Expose all runnable command names before live use. / 在实际在线使用前暴露所有可运行命令名。"""

    parser = build_parser()
    assert parser.parse_args(["list-datasets"]).command == "list-datasets"
    assert parser.parse_args(["health", "qwen", "--live"]).command == "health"
    assert parser.parse_args(["smoke-qwen", "--image", "a.png", "--question", "describe"]).command == "smoke-qwen"
    assert parser.parse_args(["count-image", "--image", "a.png", "--question", "count", "--run-id", "r"]).command == "count-image"
    count = parser.parse_args(["count-image", "--image", "a.png", "--question", "count", "--target-spec", "target.json", "--resume", "--force", "--no-seam-verify", "--max-qwen-calls", "3", "--max-deepseek-calls", "0"])
    assert count.target_spec == Path("target.json") and count.force and count.no_seam_verify
    assert parser.parse_args(["run-dataset", "--dataset", "XLRS-Bench-lite", "--root", "data", "--split", "test", "--task", "counting", "--run-id", "r"]).command == "run-dataset"
    dataset = parser.parse_args(["run-dataset", "--dataset", "XLRS-Bench-lite", "--root", "data", "--split", "test", "--max-samples", "0", "--shard-index", "1", "--num-shards", "2", "--sample-concurrency", "1"])
    assert dataset.limit == 0 and dataset.shard_count == 2 and dataset.judge_policy == "all"
    assert parser.parse_args(["resume-run", "--run-id", "r"]).command == "resume-run"
    assert parser.parse_args(["evaluate-run", "--run-id", "r", "--deepseek"]).command == "evaluate-run"
    assert parser.parse_args(["judge-vqa-run", "--run-id", "r"]).command == "judge-vqa-run"


def test_manifest_adapter_probes_explicit_mapping_and_rejects_guessing(tmp_path: Path) -> None:
    """Require a declared mapping and surface observed fields on mismatch. / 要求声明映射并在不匹配时显示观察字段。"""

    adapter = get_adapter("XLRS-Bench-lite")
    with pytest.raises(DatasetProbeError):
        adapter.probe(tmp_path)
    Image.new("RGB", (4, 4)).save(tmp_path / "sample.png")
    (tmp_path / "samples.json").write_text(json.dumps([{"uid": "one", "partition": "test", "kind": "counting", "query": "count buildings", "imgs": ["sample.png"], "n": 2}]), encoding="utf-8")
    (tmp_path / "spacers_adapter.json").write_text(json.dumps({"dataset": "XLRS-Bench-lite", "version": "1", "samples_file": "samples.json", "fields": {"id": "uid", "split": "partition", "task": "kind", "question": "query", "images": "imgs", "count": "n"}}), encoding="utf-8")
    probe = adapter.probe(tmp_path)
    assert probe.observed_fields == ("imgs", "kind", "n", "partition", "query", "uid")
    sample = next(adapter.iter_samples(tmp_path, "test", "counting"))
    assert sample.ground_truth is not None and sample.ground_truth.count == 2


@pytest.mark.asyncio
async def test_minimum_confidence_rejects_point_from_final_count(tmp_path: Path) -> None:
    """Keep low-confidence model points out of point-derived totals. / 使低置信度模型点不进入点导出的总数。"""

    result = await PointCountingOrchestrator(LowConfidenceClient(), counting=CountingSettings(tile_core_size=32, halo_size=0, model_max_side=32, min_confidence=0.2), qwen=QwenSettings(model="mock"), system_prompt="count", run_dir=tmp_path).count_image(Image.new("RGB", (16, 16)), sample_id="sample", question="count", target=CountTargetSpec(canonical_label="building", inclusion_rule="count", exclusion_rule="none"))
    assert result.final_count == 0
    assert result.global_points[0].rejection_reason == "LOW_CONFIDENCE"
