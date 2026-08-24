"""Offline tests for text-only visual-planner dataset refinement.
纯文本 Visual Planner 数据复标脚本的离线测试。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from agents.evidence_catalog import EvidenceCatalog
from agents.general_vqa.evidence.rendering import preview_from_path
from models.cache import JsonResponseCache
from scripts.refine_visual_planner_dataset import (
    AnnotationResult,
    Episode,
    TextPlanProposal,
    _urllib_label_transport,
    _annotate_one,
    bind_teacher_protocol,
    build_runtime_protocol,
    build_teacher_prompt,
    compile_training_messages,
    compile_dataset,
    duplicate_region_conflicts,
    merge_target,
    normalize_proposal,
    select_stratified_pilot,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _catalog() -> EvidenceCatalog:
    return EvidenceCatalog.from_file(REPO_ROOT / "agents" / "evidence_catalog.json")


def _executable() -> dict[str, tuple[str, ...]]:
    catalog = _catalog()
    return {
        "counting": catalog.executable_leaves_for_task("counting"),
        "fine_grained_counting": catalog.executable_leaves_for_task(
            "fine_grained_counting"
        ),
        "general_vqa": catalog.executable_leaves_for_task("general_vqa"),
        "grounding": catalog.executable_leaves_for_task("grounding"),
    }


def _proposal(**overrides) -> TextPlanProposal:
    value = {
        "task": "general_vqa",
        "needs_visual_assistance": True,
        "object_categories": ["plane"],
        "count_target": None,
    }
    value.update(overrides)
    return TextPlanProposal.model_validate(value)


def _record(*, region: dict | None = None) -> dict:
    target = {
        "version": "visual-task-plan-v5",
        "task": "general_vqa",
        "needs_visual_assistance": False,
        "object_categories": [],
        "count_target": None,
        "region_request": region
        or {"explicit": False, "image_index": None, "roi_xyxy": None},
        "reason_codes": [],
    }
    return {
        "episode_id": "e1",
        "image": "images/sha256/" + "a" * 64 + ".png",
        "messages": [
            {"role": "system", "content_ref": "protocols/old.json"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": "images/sha256/" + "a" * 64 + ".png",
                    },
                    {"type": "text", "text": "Is there a plane?"},
                ],
            },
        ],
        "target": target,
        "target_text": "target-json",
    }


def test_stratified_pilot_is_deterministic_and_meets_frozen_task_quotas() -> None:
    episodes = []
    task_sizes = {
        "counting": 100,
        "general_vqa": 180,
        "scene_classification": 40,
        "spatial_relation": 40,
    }
    for task, size in task_sizes.items():
        for index in range(size):
            record = _record()
            record["episode_id"] = f"{task}-{index}"
            record["source_group"] = f"source-{index % 3}"
            record["protocol_id"] = f"protocol-{index % 2}"
            record["target"]["task"] = task
            record["target"]["needs_visual_assistance"] = (
                task == "counting" and index % 2 == 0
            )
            question = (
                "Identify the object category inside the reference bounding box."
                if task == "general_vqa" and index % 4 == 0
                else f"Synthetic {task} question {index}"
            )
            episodes.append(
                Episode(
                    "datasets/source/train.jsonl",
                    index + 1,
                    record,
                    question,
                    record["image"],
                )
            )

    selected = select_stratified_pilot(episodes)

    assert len(selected) == 300
    assert selected == select_stratified_pilot(episodes)
    assert __import__("collections").Counter(
        episode.record["target"]["task"] for episode in selected
    ) == {
        "counting": 80,
        "general_vqa": 160,
        "scene_classification": 30,
        "spatial_relation": 30,
    }


def test_current_runtime_protocol_publishes_full_target_profile() -> None:
    protocol = build_runtime_protocol(REPO_ROOT, REPO_ROOT / "configs/local.yaml")

    binding = protocol.document["planner_binding"]
    assert binding["catalog_version"] == "visual-evidence-catalog-v4"
    assert len(binding["canonical_leaf_categories"]) == 26
    assert len(protocol.executable_by_task["general_vqa"]) == 26
    assert len(protocol.executable_by_task["counting"]) == 18
    assert len(protocol.executable_by_task["grounding"]) == 18
    assert protocol.document["system_prompt"].endswith(
        "planner_binding="
        + __import__("json").dumps(binding, ensure_ascii=False, sort_keys=True)
    )


def test_compile_training_messages_matches_inference_input_then_assistant() -> None:
    record = _record()
    protocol = {"system_prompt": "exact runtime prompt"}

    messages = compile_training_messages(record, protocol)

    assert messages[0] == {"role": "system", "content": "exact runtime prompt"}
    assert messages[1]["content"] == [
        {"type": "image", "image": record["image"]},
        {"type": "text", "text": "Is there a plane?"},
    ]
    assert messages[2] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "target-json"}],
    }


def test_compile_training_messages_preserves_ordered_image_pair() -> None:
    record = _record()
    second = "images/sha256/" + "b" * 64 + ".png"
    record["messages"][1]["content"].insert(1, {"type": "image", "image": second})

    messages = compile_training_messages(record, {"system_prompt": "exact runtime prompt"})

    assert messages[1]["content"] == [
        {"type": "image", "image": record["image"]},
        {"type": "image", "image": second},
        {"type": "text", "text": "Is there a plane?"},
    ]


def test_text_teacher_request_contains_only_raw_question(tmp_path: Path) -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.calls = []

        def judge_json(self, payload, **kwargs):
            self.calls.append((payload, kwargs))
            return _proposal()

    client = FakeClient()
    cache = JsonResponseCache(tmp_path / "cache")
    result = _annotate_one(
        "Is there a plane?",
        client=client,  # type: ignore[arg-type]
        model="deepseek-test",
        prompt="teacher prompt",
        cache=cache,
    )

    assert result.proposal == _proposal()
    assert client.calls[0][0] == {"question": "Is there a plane?"}
    assert client.calls[0][1]["repair_with_original_payload"] is True
    encoded = str(client.calls[0][0]).casefold()
    for forbidden in ("target", "answer", "provenance", "dataset"):
        assert forbidden not in encoded


def test_text_teacher_rejects_empty_question_without_model_call(tmp_path: Path) -> None:
    class UnexpectedClient:
        def judge_json(self, *args, **kwargs):
            raise AssertionError("empty question must not call the text teacher")

    result = _annotate_one(
        "",
        client=UnexpectedClient(),  # type: ignore[arg-type]
        model="deepseek-test",
        prompt="teacher prompt",
        cache=JsonResponseCache(tmp_path / "cache"),
    )

    assert result.proposal is None
    assert result.error_code == "TEXT_TEACHER_EMPTY_QUESTION_UNSUPPORTED"


def test_deepseek_transport_explicitly_disables_thinking() -> None:
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"choices":[{"message":{"content":"{}"}}]}'

    def fake_urlopen(request, timeout, context):
        captured["body"] = __import__("json").loads(request.data)
        captured["timeout"] = timeout
        captured["verify_mode"] = context.verify_mode
        return Response()

    with patch("urllib.request.urlopen", fake_urlopen):
        assert _urllib_label_transport(
            model="deepseek-v4-flash",
            messages=[{"role": "user", "content": "question"}],
            api_key="test-key",
            base_url="https://example.invalid",
            timeout_seconds=9,
        ) == "{}"

    assert captured["body"]["thinking"] == {"type": "disabled"}
    assert captured["body"]["response_format"] == {"type": "json_object"}
    assert captured["timeout"] == 9
    assert captured["verify_mode"] == __import__("ssl").CERT_REQUIRED


def test_answer_category_question_fails_closed_against_leakage() -> None:
    result = normalize_proposal(
        _proposal(),
        question="Identify the object category within the bounding box.",
        catalog=_catalog(),
        executable_by_task=_executable(),
    )

    assert result.needs_visual_assistance is False
    assert result.object_categories == ()
    assert result.decision_code == "category_is_requested_answer"


def test_high_precision_task_guards_override_teacher_boundary_errors() -> None:
    catalog = _catalog()
    executable = _executable()

    spatial = normalize_proposal(
        _proposal(needs_visual_assistance=False, object_categories=[]),
        question="Where is the court located relative to the runway?",
        catalog=catalog,
        executable_by_task=executable,
    )
    scene = normalize_proposal(
        _proposal(needs_visual_assistance=False, object_categories=[]),
        question="Is the area surrounding the terminal rural or urban?",
        catalog=catalog,
        executable_by_task=executable,
    )
    natural_language_location = normalize_proposal(
        _proposal(
            task="grounding",
            needs_visual_assistance=True,
            object_categories=["plane"],
        ),
        question="Where is the airplane located in the image?",
        catalog=catalog,
        executable_by_task=executable,
    )
    localized_category = normalize_proposal(
        _proposal(
            task="scene_classification",
            needs_visual_assistance=False,
            object_categories=[],
        ),
        question=(
            "What type of infrastructure is visible immediately to the right "
            "of the terminal?"
        ),
        catalog=catalog,
        executable_by_task=executable,
    )

    assert spatial.task == "spatial_relation"
    assert scene.task == "scene_classification"
    assert natural_language_location.task == "general_vqa"
    assert localized_category.task == "general_vqa"


def test_open_ended_local_region_description_is_caption() -> None:
    localized_caption = normalize_proposal(
        _proposal(task="general_vqa", object_categories=["bridge"]),
        question="How would you describe the activity around the bottom-most bridge?",
        catalog=_catalog(),
        executable_by_task=_executable(),
    )
    closed_box_attribute = normalize_proposal(
        _proposal(
            task="general_vqa",
            needs_visual_assistance=False,
            object_categories=[],
        ),
        question="Determine the color of the object within the bounding box.",
        catalog=_catalog(),
        executable_by_task=_executable(),
    )

    assert localized_caption.task == "caption"
    assert localized_caption.object_categories == ("bridge",)
    assert closed_box_attribute.task == "general_vqa"


def test_exact_count_target_is_normalized_to_catalog_semantic_name() -> None:
    result = normalize_proposal(
        _proposal(
            task="counting",
            needs_visual_assistance=True,
            object_categories=["storage-tank"],
            count_target="storage tank",
        ),
        question="How many storage tanks are visible?",
        catalog=_catalog(),
        executable_by_task=_executable(),
    )

    assert result.task == "counting"
    assert result.count_target == "storage-tank"
    assert result.object_categories == ("storage-tank",)


def test_counting_task_guard_fails_closed_when_teacher_omits_target() -> None:
    import pytest

    with pytest.raises(ValueError, match="COUNT_TARGET_MISSING_AFTER_TASK_GUARD"):
        normalize_proposal(
            _proposal(needs_visual_assistance=False, object_categories=[]),
            question="How many unknown structures are visible?",
            catalog=_catalog(),
            executable_by_task=_executable(),
        )


def test_paraphrased_category_question_fails_closed_without_blocking_attributes() -> None:
    leaked = normalize_proposal(
        _proposal(),
        question="Determine which class the highlighted target belongs to.",
        catalog=_catalog(),
        executable_by_task=_executable(),
    )
    attribute = normalize_proposal(
        _proposal(),
        question="Determine the color of the plane in the highlighted region.",
        catalog=_catalog(),
        executable_by_task=_executable(),
    )

    assert leaked.decision_code == "category_is_requested_answer"
    assert attribute.decision_code == "assistance_enabled"


def test_parent_alias_and_callable_context_expand_in_catalog_order() -> None:
    result = normalize_proposal(
        _proposal(object_categories=["vehicles"]),
        question="Are vehicles present near the bridge?",
        catalog=_catalog(),
        executable_by_task=_executable(),
    )

    assert result.needs_visual_assistance is True
    assert result.object_categories == (
        "bridge",
        "small-vehicle",
        "large-vehicle",
    )


def test_counting_preserves_scope_and_uses_callable_base_categories() -> None:
    known = normalize_proposal(
        _proposal(
            task="counting",
            object_categories=["small-vehicle", "large-vehicle"],
            count_target="vehicle",
        ),
        question="How many vehicles are there?",
        catalog=_catalog(),
        executable_by_task=_executable(),
    )
    unknown = normalize_proposal(
        _proposal(
            task="counting",
            object_categories=["small-vehicle", "large-vehicle"],
            count_target="blue vehicle",
        ),
        question="How many blue vehicles are there?",
        catalog=_catalog(),
        executable_by_task=_executable(),
    )

    assert known.object_categories == ("small-vehicle", "large-vehicle")
    assert known.needs_visual_assistance is True
    assert unknown.needs_visual_assistance is True
    assert unknown.object_categories == ("small-vehicle", "large-vehicle")
    assert unknown.count_target == "blue vehicle"
    assert unknown.decision_code == "assistance_enabled"


def test_object_evidence_is_task_independent_when_categories_are_callable() -> None:
    result = normalize_proposal(
        _proposal(task="spatial_relation"),
        question="Where is the plane relative to the bridge?",
        catalog=_catalog(),
        executable_by_task=_executable(),
    )

    assert result.needs_visual_assistance is True
    assert result.object_categories == ("plane", "bridge")
    assert result.decision_code == "assistance_enabled"


def test_scene_classification_uses_bounded_global_evidence_profile() -> None:
    result = normalize_proposal(
        _proposal(
            task="scene_classification",
            needs_visual_assistance=False,
            object_categories=[],
        ),
        question="Is the terminal in an urban or rural setting?",
        catalog=_catalog(),
        executable_by_task=_executable(),
    )

    assert result.needs_visual_assistance is True
    assert len(result.object_categories) == 8
    assert "airport" in result.object_categories
    assert "building" in result.object_categories
    assert "road" in result.object_categories


def test_scene_type_question_uses_evidence_without_treating_profile_as_answer() -> None:
    result = normalize_proposal(
        _proposal(
            task="scene_classification",
            needs_visual_assistance=False,
            object_categories=[],
        ),
        question="What type of terrain is shown in the image?",
        catalog=_catalog(),
        executable_by_task=_executable(),
    )

    assert result.needs_visual_assistance is True
    assert result.decision_code == "assistance_enabled"


def test_merge_preserves_region_but_replaces_text_decidable_fields() -> None:
    old = _record(
        region={"explicit": True, "image_index": 0, "roi_xyxy": [1, 2, 30, 40]}
    )["target"]

    plan = merge_target(
        old,
        _proposal(),
        question="Is there a plane?",
        catalog=_catalog(),
        executable_by_task=_executable(),
    )

    assert plan.task == "general_vqa"
    assert plan.object_categories == ["plane"]
    assert plan.region_request.roi_xyxy == (1, 2, 30, 40)
    assert plan.reason_codes == []


def test_new_counting_task_uses_teacher_count_target() -> None:
    old = _record()["target"]

    plan = merge_target(
        old,
        _proposal(
            task="counting",
            object_categories=["plane"],
            count_target="plane",
        ),
        question="How many planes are there?",
        catalog=_catalog(),
        executable_by_task=_executable(),
    )

    assert plan.task == "counting"
    assert plan.count_target == "plane"


def test_duplicate_region_conflicts_quarantine_whole_input_group() -> None:
    first = _record(
        region={"explicit": True, "image_index": 0, "roi_xyxy": [1, 2, 30, 40]}
    )
    second = _record(
        region={"explicit": True, "image_index": 0, "roi_xyxy": [5, 6, 35, 45]}
    )
    second["episode_id"] = "e2"
    episodes = [
        Episode("datasets/A/train.jsonl", 1, first, "Is there a plane?", first["image"]),
        Episode("datasets/A/train.jsonl", 2, second, "Is there a plane?", second["image"]),
    ]

    assert duplicate_region_conflicts(episodes) == {"e1", "e2"}


def test_teacher_prompt_freezes_current_catalog_without_sample_data() -> None:
    protocol = build_runtime_protocol(REPO_ROOT, REPO_ROOT / "configs/local.yaml")
    runtime_prompt = protocol.document["system_prompt"]
    rubric = (
        REPO_ROOT / "prompts" / "visual_task_plan_text_teacher_v6.md"
    ).read_text(encoding="utf-8")
    prompt = build_teacher_prompt(runtime_prompt, rubric)

    assert runtime_prompt in prompt
    assert rubric.strip() in prompt
    assert prompt.index(runtime_prompt) < prompt.index(rubric.strip())
    assert prompt.index(rubric.strip()) < prompt.index('"properties"')
    assert (
        REPO_ROOT / "prompts" / "visual_task_plan_v5.md"
    ).read_text(encoding="utf-8").strip() in prompt
    assert "visual-evidence-catalog-v4" in prompt
    assert "bareland" in prompt
    assert "urban-versus-rural" in prompt
    assert "Boeing planes" in prompt
    assert __import__("json").dumps(
        TextPlanProposal.model_json_schema(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) in prompt
    assert '"old_target"' not in prompt
    assert "count_target" in prompt
    assert set(TextPlanProposal.model_json_schema()["properties"]) == {
        "task",
        "needs_visual_assistance",
        "object_categories",
        "count_target",
    }


def test_derived_protocol_binds_exact_teacher_prompt_and_schema() -> None:
    runtime = build_runtime_protocol(REPO_ROOT, REPO_ROOT / "configs/local.yaml")
    rubric = (
        REPO_ROOT / "prompts" / "visual_task_plan_text_teacher_v6.md"
    ).read_text(encoding="utf-8")
    prompt = build_teacher_prompt(runtime.document["system_prompt"], rubric)

    derived = bind_teacher_protocol(runtime, prompt)

    assert derived.protocol_id != runtime.protocol_id
    assert derived.document["system_prompt"].startswith(runtime.document["system_prompt"])
    assert derived.document["annotation_evidence_policy"]["task_gate"] is False
    assert derived.document["annotation_evidence_policy"][
        "global_executable_categories"
    ]
    assert derived.document["system_prompt_sha256"] == __import__("hashlib").sha256(
        derived.document["system_prompt"].encode("utf-8")
    ).hexdigest()
    teacher = derived.document["refinement_teacher"]
    assert teacher["prompt_version"] == "deepseek-visual-planner-text-v6.1"
    assert teacher["response_schema"] == TextPlanProposal.model_json_schema()
    assert derived.executable_by_task == runtime.executable_by_task


def test_compile_dataset_writes_one_protocol_and_resolved_training_chat(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    image_path = source / "images" / "sha256" / ("a" * 64 + ".png")
    image_path.parent.mkdir(parents=True)
    from PIL import Image

    Image.new("RGB", (2, 2), "white").save(image_path)
    (source / "manifest.json").write_text(
        __import__("json").dumps(
            {
                "description": "source",
                "datasets": {
                    "A": {
                        "source_files": 1,
                        "logical_datasets": ["A"],
                        "examples": 1,
                        "splits": {"train": 1},
                        "embedded_image_blocks": 1,
                        "protocol_ids": ["old"],
                        "files": {},
                    }
                },
                "images": {
                    "unique_count": 1,
                    "embedded_block_count": 1,
                    "total_decoded_bytes": image_path.stat().st_size,
                    "by_sha256": {},
                },
            }
        ),
        encoding="utf-8",
    )
    record = _record()
    record.update(
        {
            "source_group": "A",
            "split": "train",
            "protocol_id": "old",
            "protocol_version": "visual-task-plan-v5",
            "provenance": {},
        }
    )
    protocol = build_runtime_protocol(REPO_ROOT, REPO_ROOT / "configs/local.yaml")
    identity = {
        "refinement_run_id": "refine-test",
        "teacher_model": "deepseek-test",
        "source_manifest_sha256": "b" * 64,
    }

    distribution = compile_dataset(
        source_root=source,
        output_root=output,
        episodes=[
            Episode(
                "datasets/A/train.jsonl",
                1,
                record,
                "Is there a plane?",
                record["image"],
            )
        ],
        annotations={
            "Is there a plane?": AnnotationResult(_proposal(), "c" * 64, False)
        },
        protocol=protocol,
        catalog=_catalog(),
        identity=identity,
        copy_images=False,
    )

    assert distribution["accepted"] == 1
    compact = __import__("json").loads(
        (output / "datasets" / "A" / "train.jsonl").read_text(encoding="utf-8")
    )
    assert compact["protocol_id"] == protocol.protocol_id
    assert compact["provenance"] == record["provenance"]
    assert compact["target_text"] == __import__("json").dumps(
        compact["target"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    training = __import__("json").loads(
        (output / "training" / "A" / "train.jsonl").read_text(encoding="utf-8")
    )
    assert [message["role"] for message in training["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert training["messages"][1]["content"][1]["text"] == "Is there a plane?"
    assert training["image"].startswith("training_images/sha256/")
    assert training["messages"][1]["content"][0]["image"] == training["image"]
    preview_path = output / training["image"]
    assert preview_path.is_file()
    _, expected_digest = preview_from_path(image_path, max_side=1080)
    assert preview_path.stem == expected_digest
    manifest = __import__("json").loads(
        (output / "manifest.json").read_text(encoding="utf-8")
    )
    assert list(manifest["protocols"]) == [protocol.protocol_id]
    assert manifest["refinement"]["accepted"] == 1
