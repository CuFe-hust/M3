"""Build a real frozen-feature cache through production Change preprocessing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.change.perception import (  # noqa: E402
    SemanticExpertBinding,
    infer_semantic_expert_pair,
)
from application.bootstrap import (  # noqa: E402
    _build_change_semantic_bindings,
    _build_segformer_clients,
    _expert_asset_root,
    _load_expert_catalog,
)
from application.settings import load_settings  # noqa: E402
from models.base import (  # noqa: E402
    require_model_cache_identity,
)
from models.change_head.fingerprint import (  # noqa: E402
    build_change_input_pipeline_fingerprint,
)
from models.change_head.manifest import ChangeHeadManifest  # noqa: E402
from training.change_head.dataset import PreparedChangeTrainingDataset  # noqa: E402
from training.change_head.feature_cache import (  # noqa: E402
    CachedChangeTrainingSample,
    CachedExpertFeaturePair,
    FeatureCache,
    build_feature_cache_key,
)
from training.change_head.schema import load_training_records  # noqa: E402


class RequiredExpertFailure(RuntimeError):
    """A required frozen expert could not produce a cache sample."""


def _load_head_manifest(path: Path) -> ChangeHeadManifest:
    try:
        return ChangeHeadManifest.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except Exception as error:
        raise ValueError(f"invalid ChangeHead manifest: {path}") from error


def _expert_payload(manifest: ChangeHeadManifest) -> list[dict[str, Any]]:
    return [
        {
            "expert_id": expert.expert_id,
            "logical_model_id": expert.logical_model_id,
            "weights_sha256": expert.weights_sha256,
            "class_map_sha256": expert.class_names_sha256,
            "feature_stages": list(expert.feature_stages),
            "required": expert.required,
            "use_semantic_probabilities": expert.use_semantic_probabilities,
            "missing_policy": expert.missing_policy,
        }
        for expert in manifest.experts
    ]


def _features_by_stage(output: Any, stages: tuple[int, ...]) -> dict[int, Any]:
    pyramid = getattr(output, "features_by_stage", None)
    if pyramid is not None:
        return {stage: pyramid[stage] for stage in stages}
    if len(stages) != 1 or not hasattr(output, "features"):
        raise ValueError("expert output does not contain declared feature stages")
    return {stages[0]: output.features}


def _build_cached_sample(
    example: Any,
    *,
    manifest: ChangeHeadManifest,
    bindings_by_id: dict[str, SemanticExpertBinding],
    semantic_settings: Any,
    pipeline_fingerprint: str,
) -> tuple[CachedChangeTrainingSample, list[str]]:
    """Run each declared expert once for T1/T2 and serialize only tensors."""

    experts: dict[str, CachedExpertFeaturePair] = {}
    missing_optional: list[str] = []
    for requirement in manifest.experts:
        binding = bindings_by_id.get(requirement.expert_id)
        if binding is None:
            if requirement.required:
                raise RequiredExpertFailure(
                    f"required expert missing: {requirement.expert_id}"
                )
            missing_optional.append(requirement.expert_id)
            continue
        try:
            run = infer_semantic_expert_pair(
                binding,
                example.prepared,
                semantic_settings,
                requested_stages=tuple(requirement.feature_stages),
            )
        except Exception as error:
            if requirement.required:
                raise RequiredExpertFailure(
                    f"required expert failed: {requirement.expert_id}"
                ) from error
            missing_optional.append(requirement.expert_id)
            continue
        first_features = _features_by_stage(run.first_output, requirement.feature_stages)
        second_features = _features_by_stage(run.second_output, requirement.feature_stages)
        class_hash = binding.class_names_sha256
        if class_hash is None:
            from models.base import hash_class_names

            class_hash = hash_class_names(run.first_output.class_names)
        weights_hash = binding.weights_sha256 or run.weights_sha256
        if weights_hash is None:
            raise RequiredExpertFailure(
                f"expert weights hash missing: {requirement.expert_id}"
            )
        experts[requirement.expert_id] = CachedExpertFeaturePair(
            expert_id=requirement.expert_id,
            logical_model_id=requirement.logical_model_id,
            weights_sha256=weights_hash,
            class_map_sha256=class_hash,
            feature_stages=tuple(requirement.feature_stages),
            first_features=first_features,
            second_features=second_features,
            first_semantic_probabilities=(
                run.first_output.probabilities
                if requirement.use_semantic_probabilities
                else None
            ),
            second_semantic_probabilities=(
                run.second_output.probabilities
                if requirement.use_semantic_probabilities
                else None
            ),
        )

    prepared = example.prepared
    if prepared.raw_t1 is None:
        raise ValueError("invalid prepared pair")
    image_size = (int(prepared.raw_t1.shape[1]), int(prepared.raw_t1.shape[0]))
    return (
        CachedChangeTrainingSample(
            sample_id=example.record.sample_id,
            image_size=image_size,
            experts=experts,
            target_change_mask=example.target_mask,
            loss_valid_mask=example.loss_valid_mask,
            pif_mask=(
                prepared.pif_mask.copy()
                if manifest.architecture.use_pif_mask
                else None
            ),
            comparison_t1=(
                prepared.comparison_t1.copy()
                if manifest.architecture.use_rgb_pair
                else None
            ),
            comparison_t2=(
                prepared.comparison_t2.copy()
                if manifest.architecture.use_rgb_pair
                else None
            ),
            dataset_name=example.record.source_dataset,
            split=example.record.split,
            tags=tuple(example.record.tags),
            input_pipeline_fingerprint=pipeline_fingerprint,
        ),
        missing_optional,
    )


def _production_bindings(settings: Any, project_root: Path) -> tuple[SemanticExpertBinding, ...]:
    asset_root = _expert_asset_root(project_root)
    catalog = _load_expert_catalog(asset_root)
    clients = _build_segformer_clients(settings, catalog, project_root=asset_root)
    return _build_change_semantic_bindings(
        settings,
        catalog,
        clients,
        project_root=asset_root,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", "--manifest", dest="dataset_manifest", type=Path, required=True)
    parser.add_argument("--change-config", type=Path, required=True)
    parser.add_argument("--head-manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", "--cache-root", dest="cache_dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--on-sample-error", choices=("fail", "skip"), default="fail")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    data_root = (args.data_root or args.dataset_manifest.parent).resolve()
    manifest = _load_head_manifest(args.head_manifest)
    settings = load_settings(args.change_config)
    records = [
        record
        for record in load_training_records(args.dataset_manifest, allow_empty_changed=True)
        if record.split == args.split
    ]
    if args.limit is not None:
        records = records[: args.limit]
    bindings = _production_bindings(settings, ROOT)
    bindings_by_id = {binding.expert_id: binding for binding in bindings}
    identities = {
        binding.expert_id: require_model_cache_identity(
            binding.client,
            component=f"change cache expert {binding.expert_id}",
        )
        for binding in bindings
    }
    pipeline_fingerprint, _ = build_change_input_pipeline_fingerprint(
        settings=settings.agents.change,
        semantic_client_identities=identities,
    )
    if pipeline_fingerprint != manifest.pipeline_fingerprint:
        raise ValueError("head manifest pipeline fingerprint does not match production settings")
    dataset = PreparedChangeTrainingDataset(
        records,
        data_root=data_root,
        settings=settings.agents.change,
        artifact_root=args.cache_dir / "preprocess_artifacts",
    )
    cache = FeatureCache(args.cache_dir / pipeline_fingerprint)
    index_rows: list[dict[str, Any]] = []
    stats: dict[str, int] = {
        "samples_total": len(records),
        "samples_cached": 0,
        "samples_skipped": 0,
        "samples_failed": 0,
        "required_expert_failures": 0,
        "optional_expert_missing": 0,
        "cache_hits": 0,
        "cache_writes": 0,
    }
    expert_payload = _expert_payload(manifest)
    for index, record in enumerate(records):
        t1 = (data_root / record.t1_path).resolve()
        t2 = (data_root / record.t2_path).resolve()
        key = build_feature_cache_key(
            sample_id=record.sample_id,
            t1_path=t1,
            t2_path=t2,
            pipeline_fingerprint=pipeline_fingerprint,
            experts=expert_payload,
            contract_version=manifest.input_contract_version,
        )
        if cache.read_sample(key) is not None:
            stats["samples_cached"] += 1
            stats["cache_hits"] += 1
            index_rows.append({"sample_id": record.sample_id, "cache_key": key, "status": "cached"})
            continue
        try:
            if not (data_root / record.mask_path).is_file():
                raise FileNotFoundError("MISSING_GROUND_TRUTH_MASK")
            example = dataset[index]
            sample, missing_optional = _build_cached_sample(
                example,
                manifest=manifest,
                bindings_by_id=bindings_by_id,
                semantic_settings=settings.agents.change.semantic,
                pipeline_fingerprint=pipeline_fingerprint,
            )
            cache.write_sample(key, sample)
            stats["cache_writes"] += 1
            stats["optional_expert_missing"] += len(missing_optional)
            index_rows.append({
                "sample_id": record.sample_id,
                "cache_key": key,
                "status": "written",
                "missing_optional_experts": missing_optional,
            })
        except RequiredExpertFailure as error:
            stats["required_expert_failures"] += 1
            stats["samples_failed"] += 1
            if args.on_sample_error == "fail":
                raise
            stats["samples_skipped"] += 1
            index_rows.append({
                "sample_id": record.sample_id,
                "cache_key": key,
                "status": "skipped",
                "reason_code": "REQUIRED_EXPERT_FAILURE",
                "reason": str(error),
            })
        except Exception as error:
            stats["samples_failed"] += 1
            if args.on_sample_error == "fail":
                raise
            stats["samples_skipped"] += 1
            reason_code = (
                "MISSING_GROUND_TRUTH_MASK"
                if isinstance(error, FileNotFoundError)
                else type(error).__name__
            )
            index_rows.append({
                "sample_id": record.sample_id,
                "cache_key": key,
                "status": "skipped",
                "reason_code": reason_code,
                "reason": str(error),
            })
    index_path = cache.root / "index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in index_rows),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
