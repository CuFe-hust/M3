from __future__ import annotations

import json
import shutil
import struct
import zlib
from dataclasses import replace
from pathlib import Path

import pytest

from m3rs_eval.config import DatasetConfig
from m3rs_eval.contracts import ContractError
from m3rs_eval.datasets import DatasetError, create_adapters


def read_raw_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def create_adapter(dataset: str, fixture_config, protocol, registry):
    return next(
        adapter
        for adapter in create_adapters(fixture_config, protocol, registry)
        if adapter.dataset == dataset
    )


def config_with_dataset_root(fixture_config, dataset: str, root: Path, profile: str = "fixture"):
    datasets = dict(fixture_config.datasets)
    datasets[dataset] = DatasetConfig(root=root, asset_root=root.parent, profile=profile)
    return replace(fixture_config, datasets=datasets)


def copied_dataset_root(fixture_config, dataset: str, destination: Path) -> Path:
    source_assets = fixture_config.datasets[dataset].asset_root
    asset_root = destination / "assets"
    shutil.copytree(source_assets, asset_root)
    return asset_root / dataset


@pytest.mark.parametrize("dataset", ["levir_cc", "vrsbench", "xlrs_bench", "mme_rs"])
def test_fixture_materialization_has_disjoint_requests_and_references(
    dataset, fixture_config, protocol, registry, tmp_path
):
    """Fails if a materializer emits hidden reference data in requests."""
    result = create_adapter(dataset, fixture_config, protocol, registry).materialize(
        mode="smoke", limit=2, destination=tmp_path
    )

    request_rows = read_raw_jsonl(result.requests_path)
    reference_rows = read_raw_jsonl(result.references_path)

    expected_counts = {"levir_cc": 2, "vrsbench": 6, "xlrs_bench": 10, "mme_rs": 6}
    assert len(request_rows) == len(reference_rows) == expected_counts[dataset]
    assert "answer" not in request_rows[0]
    assert reference_rows[0]["sample_id"] == request_rows[0]["sample_id"]
    assert result.expected_samples == expected_counts[dataset]
    assert result.coverage["profile"] == "fixture"


def test_levir_requires_two_ordered_images(levir_adapter, tmp_path):
    """Fails if temporal A/B ordering is lost in LEVIR requests."""
    row = read_raw_jsonl(levir_adapter.materialize("smoke", 1, tmp_path).requests_path)[0]

    assert len(row["images"]) == 2
    assert row["images"][0].endswith("/A/test_000001.png")
    assert row["images"][1].endswith("/B/test_000001.png")


def test_vrs_materializes_test_caption_grounding_and_vqa_separately(vrsbench_adapter, tmp_path):
    """Fails if VRS task boundaries or its official test split are pooled."""
    result = vrsbench_adapter.materialize("full", None, tmp_path)
    rows = read_raw_jsonl(result.requests_path)

    assert {row["task"] for row in rows} == {"caption", "grounding", "vqa"}
    assert {row["split"] for row in rows} == {"test"}
    assert result.task_counts == {"caption": 2, "grounding": 2, "vqa": 2}
    references = read_raw_jsonl(result.references_path)
    grounding = next(row for row in references if row["sample_id"].endswith("grounding-000001"))
    vqa = next(row for row in references if row["sample_id"].endswith("vqa-000001"))
    assert grounding["grounding_slice"] == "unique"
    assert vqa["vqa_category"] == "scene"


def test_xlrs_preserves_explicit_variant_language_and_unavailable_full_vqa(
    xlrs_bench_adapter, tmp_path
):
    """Fails if Full/Lite or en/zh is aliased, especially unavailable Full VQA."""
    result = xlrs_bench_adapter.materialize("full", None, tmp_path)
    rows = read_raw_jsonl(result.requests_path)

    assert {(row["variant"], row["language"]) for row in rows} == {
        ("full", "en"),
        ("full", "zh"),
        ("lite", "en"),
        ("lite", "zh"),
    }
    assert not any(row["variant"] == "full" and row["task"] == "vqa" for row in rows)
    assert result.coverage["unavailable_scopes"] == [
        "variant=full|language=en|task=vqa",
        "variant=full|language=zh|task=vqa",
    ]
    references = read_raw_jsonl(result.references_path)
    lite_vqa = next(row for row in references if row["sample_id"] == "xlrs_bench:lite:en:vqa:lite-vqa-en-000001")
    assert lite_vqa["l3"] == "overall_counting"


def test_mme_allows_only_remote_sensing_and_separates_three_tasks(mme_rs_adapter, tmp_path):
    """Fails if non-RS records enter MME formal materialization or tasks are pooled."""
    result = mme_rs_adapter.materialize("full", None, tmp_path)
    rows = read_raw_jsonl(result.requests_path)

    assert {row["task"] for row in rows} == {"color", "count", "position"}
    assert result.task_counts == {"color": 2, "count": 2, "position": 2}
    assert result.coverage["domain"] == "Remote_Sensing"


@pytest.mark.parametrize(
    ("dataset", "expected_scopes"),
    [
        ("levir_cc", {"split=test|task=caption"}),
        ("vrsbench", {"task=caption", "task=grounding", "task=vqa"}),
        (
            "xlrs_bench",
            {
                "variant=full|language=en|task=caption",
                "variant=full|language=en|task=grounding",
                "variant=full|language=zh|task=caption",
                "variant=full|language=zh|task=grounding",
                "variant=lite|language=en|task=caption",
                "variant=lite|language=en|task=grounding",
                "variant=lite|language=en|task=vqa",
                "variant=lite|language=zh|task=caption",
                "variant=lite|language=zh|task=grounding",
                "variant=lite|language=zh|task=vqa",
            },
        ),
        ("mme_rs", {"task=color", "task=count", "task=position"}),
    ],
)
def test_smoke_limit_selects_first_n_from_every_formal_scope(
    dataset, expected_scopes, fixture_config, protocol, registry, tmp_path
):
    """Fails if smoke limiting truncates parser order instead of formal reporting scopes."""
    result = create_adapter(dataset, fixture_config, protocol, registry).materialize(
        "smoke", 1, tmp_path
    )

    assert set(result.coverage["scopes"]) == expected_scopes
    assert all(counts["selected"] == 1 for counts in result.coverage["scopes"].values())


@pytest.mark.parametrize("dataset", ["levir_cc", "vrsbench", "xlrs_bench", "mme_rs"])
def test_full_mode_rejects_a_limit(dataset, fixture_config, protocol, registry, tmp_path):
    """Fails if a truncated run can be labeled as formal full evaluation."""
    adapter = create_adapter(dataset, fixture_config, protocol, registry)

    with pytest.raises(DatasetError, match="full mode does not permit a limit"):
        adapter.materialize("full", 1, tmp_path)


def test_manifest_hash_is_stable_for_the_same_fixture_materialization(levir_adapter, tmp_path):
    """Fails if deterministic inputs produce different request manifests."""
    first = levir_adapter.materialize("smoke", 2, tmp_path / "first")
    second = levir_adapter.materialize("smoke", 2, tmp_path / "second")

    assert first.manifest_hash == second.manifest_hash


def test_manifest_hash_is_portable_when_the_same_vrs_fixture_is_relocated(
    fixture_config, protocol, registry, tmp_path
):
    """Fails if a manifest contains machine-specific absolute image paths."""
    first_root = copied_dataset_root(fixture_config, "vrsbench", tmp_path / "first")
    second_root = copied_dataset_root(fixture_config, "vrsbench", tmp_path / "second")
    first = create_adapter(
        "vrsbench", config_with_dataset_root(fixture_config, "vrsbench", first_root), protocol, registry
    ).materialize("full", None, tmp_path / "first-out")
    second = create_adapter(
        "vrsbench", config_with_dataset_root(fixture_config, "vrsbench", second_root), protocol, registry
    ).materialize("full", None, tmp_path / "second-out")

    assert first.manifest_hash == second.manifest_hash


def test_manifest_hash_changes_when_an_emitted_request_prompt_changes(
    fixture_config, protocol, registry, tmp_path
):
    """Fails if request contract changes are absent from the manifest evidence."""
    root = copied_dataset_root(fixture_config, "xlrs_bench", tmp_path)
    adapter = create_adapter(
        "xlrs_bench", config_with_dataset_root(fixture_config, "xlrs_bench", root), protocol, registry
    )
    first = adapter.materialize("full", None, tmp_path / "first")
    tasks = root / "tasks.jsonl"
    tasks.write_text(
        tasks.read_text(encoding="utf-8").replace("Describe the image.", "Describe the changed image.", 1),
        encoding="utf-8",
    )
    second = adapter.materialize("full", None, tmp_path / "second")

    assert first.manifest_hash != second.manifest_hash


def test_manifest_hash_changes_when_levir_image_order_changes(fixture_config, protocol, registry, tmp_path):
    """Fails if manifest canonicalization sorts ordered temporal image pairs."""
    root = copied_dataset_root(fixture_config, "levir_cc", tmp_path)
    adapter = create_adapter(
        "levir_cc", config_with_dataset_root(fixture_config, "levir_cc", root), protocol, registry
    )
    first = adapter.materialize("full", None, tmp_path / "first")
    annotations = root / "annotations.json"
    annotations.write_text(
        annotations.read_text(encoding="utf-8").replace(
            '"image_a":"A/test_000001.png","image_b":"B/test_000001.png"',
            '"image_a":"B/test_000001.png","image_b":"A/test_000001.png"',
        ),
        encoding="utf-8",
    )
    second = adapter.materialize("full", None, tmp_path / "second")

    assert first.manifest_hash != second.manifest_hash


def test_checked_in_image_fixture_is_a_valid_1x1_png(fixture_config):
    """Fails if the binary fixture is replaced with non-image placeholder data."""
    fixture = fixture_config.datasets["levir_cc"].root.parent / "images" / "fixture.png"
    data = fixture.read_bytes()

    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    position = 8
    chunks = []
    while position < len(data):
        length = struct.unpack(">I", data[position:position + 4])[0]
        kind = data[position + 4:position + 8]
        payload = data[position + 8:position + 8 + length]
        crc = struct.unpack(">I", data[position + 8 + length:position + 12 + length])[0]
        assert zlib.crc32(kind + payload) & 0xFFFFFFFF == crc
        chunks.append((kind, payload))
        position += 12 + length
    assert position == len(data)
    assert chunks[0] == (b"IHDR", b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00")
    assert chunks[-1] == (b"IEND", b"")


def test_checked_in_levir_a_and_b_images_are_valid_files(levir_adapter, tmp_path):
    """Fails if LEVIR requests refer to placeholder rather than usable A/B assets."""
    result = levir_adapter.materialize("full", None, tmp_path)
    validated_png = (levir_adapter.config.asset_root / "images" / "fixture.png").read_bytes()
    for row in read_raw_jsonl(result.requests_path):
        for image in row["images"]:
            assert Path(image).is_file()
            assert Path(image).read_bytes() == validated_png


@pytest.mark.parametrize(
    ("dataset", "annotation_name", "needle"),
    [
        ("levir_cc", "annotations.json", '"image_a":"A/test_000001.png"'),
        ("vrsbench", "caption.jsonl", '"image":"../images/fixture.png"'),
        ("xlrs_bench", "tasks.jsonl", '"image":"../images/fixture.png"'),
        ("mme_rs", "questions.jsonl", '"image":"../images/fixture.png"'),
    ],
)
@pytest.mark.parametrize("replacement", ['"image":"../images/missing.png"', '"image":"../../outside.png"'])
def test_adapters_reject_missing_or_outside_asset_root_images(
    fixture_config, protocol, registry, tmp_path, dataset, annotation_name, needle, replacement
):
    """Fails if an inference request can reference a missing or escaped image file."""
    root = copied_dataset_root(fixture_config, dataset, tmp_path)
    annotation = root / annotation_name
    if dataset == "levir_cc":
        replacement = replacement.replace('"image"', '"image_a"')
    annotation.write_text(
        annotation.read_text(encoding="utf-8").replace(needle, replacement, 1), encoding="utf-8"
    )
    outside = tmp_path / "outside.png"
    outside.write_bytes((fixture_config.datasets[dataset].asset_root / "images" / "fixture.png").read_bytes())
    adapter = create_adapter(
        dataset, config_with_dataset_root(fixture_config, dataset, root), protocol, registry
    )

    with pytest.raises(DatasetError, match="image"):
        adapter.materialize("full", None, tmp_path / "out")


@pytest.mark.parametrize(
    ("dataset", "annotation_name", "mutate"),
    [
        ("levir_cc", "annotations.json", lambda content: content.replace('"id":"000002"', '"id":"000001"')),
        ("vrsbench", "caption.jsonl", lambda content: content + content.splitlines()[0] + "\n"),
        ("xlrs_bench", "tasks.jsonl", lambda content: content + content.splitlines()[0] + "\n"),
        ("mme_rs", "questions.jsonl", lambda content: content + content.splitlines()[0] + "\n"),
    ],
)
def test_materialization_rejects_duplicate_generated_sample_ids(
    fixture_config, protocol, registry, tmp_path, dataset, annotation_name, mutate
):
    """Fails if duplicate source rows become indistinguishable persisted request IDs."""
    root = copied_dataset_root(fixture_config, dataset, tmp_path)
    annotation = root / annotation_name
    annotation.write_text(mutate(annotation.read_text(encoding="utf-8")), encoding="utf-8")
    adapter = create_adapter(
        dataset, config_with_dataset_root(fixture_config, dataset, root), protocol, registry
    )

    with pytest.raises(DatasetError, match="duplicate generated sample_id.*row"):
        adapter.materialize("full", None, tmp_path / "out")


def test_coverage_has_selected_and_total_counts_for_formal_scopes(
    vrsbench_adapter, xlrs_bench_adapter, mme_rs_adapter, levir_adapter, tmp_path
):
    """Fails if coverage cannot distinguish selected smoke samples from formal totals."""
    vrs = vrsbench_adapter.materialize("smoke", 1, tmp_path / "vrs")
    xlrs = xlrs_bench_adapter.materialize("smoke", 1, tmp_path / "xlrs")
    mme = mme_rs_adapter.materialize("smoke", 1, tmp_path / "mme")
    levir = levir_adapter.materialize("smoke", 1, tmp_path / "levir")

    assert vrs.coverage["scopes"]["task=caption"] == {"selected": 1, "total": 2}
    assert mme.coverage["scopes"]["task=color"] == {"selected": 1, "total": 2}
    assert xlrs.coverage["scopes"]["variant=lite|language=zh|task=vqa"] == {
        "selected": 1,
        "total": 1,
    }
    assert xlrs.coverage["unavailable_scopes"] == [
        "variant=full|language=en|task=vqa",
        "variant=full|language=zh|task=vqa",
    ]
    assert levir.coverage["scopes"]["split=test|task=caption"] == {"selected": 1, "total": 2}
    assert levir.coverage["slices"]["change"] == {"selected": 1, "total": 1}
    assert levir.coverage["slices"]["no-change"] == {"selected": 0, "total": 1}


def test_adapter_evaluate_returns_structured_malformed_duplicate_and_missing_evidence(
    mme_rs_adapter, registry, tmp_path
):
    """The canonical adapter API delegates to Task 5's tolerant one-pass evaluator."""
    from m3rs_eval.evaluation import EvaluationResult, MetricContext

    materialization = mme_rs_adapter.materialize("full", None, tmp_path / "materialized")
    sample_ids = [row["sample_id"] for row in read_raw_jsonl(materialization.requests_path)]
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps({"sample_id": sample_ids[0], "status": "ok", "prediction": "A"}) + "\n"
        + json.dumps({"sample_id": sample_ids[0], "status": "ok", "prediction": "B"}) + "\n"
        + "not-json\n",
        encoding="utf-8",
    )
    context = MetricContext(
        run_id="adapter-integration",
        recorded_at="2026-08-05T12:00:00+08:00",
        protocol_id="official_full_v1",
        benchmark_version="mme-realworld-rs:Remote_Sensing",
        source_log_path="logs/mme/evaluation.json",
    )

    result = mme_rs_adapter.evaluate(
        materialization,
        predictions,
        registry,
        context=context,
        log_dir=tmp_path / "logs",
    )

    assert isinstance(result, EvaluationResult)
    reasons = {failure.reason for failure in result.failures}
    assert {"duplicate_prediction", "malformed_prediction", "missing_prediction"} <= reasons
    assert result.status == "incomplete"


@pytest.mark.parametrize(
    ("dataset", "annotation_name", "needle", "replacement", "message"),
    [
        ("levir_cc", "annotations.json", '"split":"test"', '"split":"train"', "exactly 'test'"),
        ("vrsbench", "caption.jsonl", '"split":"test"', '"split":"train"', "only 'test'"),
        ("mme_rs", "questions.jsonl", '"domain":"Remote_Sensing"', '"domain":"Other"', "Remote_Sensing"),
    ],
)
def test_formal_materialization_rejects_split_or_domain_leakage(
    fixture_config, protocol, registry, tmp_path, dataset, annotation_name, needle, replacement, message
):
    """Fails if a protected formal scope accepts a training split or another MME domain."""
    source_root = fixture_config.datasets[dataset].root
    copied_root = tmp_path / dataset
    copied_root.mkdir()
    for source_file in source_root.iterdir():
        if source_file.is_file():
            (copied_root / source_file.name).write_bytes(source_file.read_bytes())
    source = source_root / annotation_name
    target = copied_root / annotation_name
    target.write_text(
        source.read_text(encoding="utf-8").replace(needle, replacement),
        encoding="utf-8",
    )
    for image in (source_root.parent / "images").glob("*"):
        destination = copied_root.parent / "images"
        destination.mkdir(exist_ok=True)
        (destination / image.name).write_bytes(image.read_bytes())

    config = config_with_dataset_root(fixture_config, dataset, copied_root)
    adapter = create_adapter(dataset, config, protocol, registry)

    with pytest.raises(DatasetError, match=message):
        adapter.materialize("full", None, tmp_path / "out")


def test_preflight_names_profile_root_and_expected_files_for_real_data(
    fixture_config, protocol, registry, tmp_path
):
    """Fails if a missing real-data layout cannot be diagnosed from the error."""
    missing_root = tmp_path / "missing-levir"
    config = config_with_dataset_root(fixture_config, "levir_cc", missing_root, profile="official")
    adapter = create_adapter("levir_cc", config, protocol, registry)

    with pytest.raises(DatasetError, match="official") as error:
        adapter.materialize("full", None, tmp_path / "out")
    assert str(missing_root) in str(error.value)
    assert "annotations" in str(error.value)
