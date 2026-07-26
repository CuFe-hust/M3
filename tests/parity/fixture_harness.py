"""Shared canonical samples and settings for new-runtime parity tests.
新运行时等价测试共享的统一样本与设置。
"""

from __future__ import annotations

from pathlib import Path

from spacers_agent.schemas import GroundTruth, ImageRef, UnifiedSample
from spacers_agent.settings import AppSettings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_IMAGE = PROJECT_ROOT / "tests" / "fixtures" / "legacy" / "test_image.png"
CASE_NAMES = (
    "native_counting", "fine_grained_counting", "vrsbench_quantity_count",
    "vrsbench_grid_position", "vrsbench_extreme_category", "general_vqa",
    "multiple_choice_vqa", "scene_classification", "grounding", "spatial_relation",
    "change_caption", "change_qa", "caption", "primary_qwen_failure",
    "partial_expert", "failed_expert", "counting_one_failed_tile", "judge_failure",
    "missing_judge_resume", "succeeded_sample_resume",
)


def build_sample(case_name: str) -> UnifiedSample:
    """Build the immutable canonical sample for one parity scenario.
    为一个等价场景构建不可变统一样本。
    """

    task_by_case = {
        "native_counting": "counting", "fine_grained_counting": "fine_grained_counting",
        "vrsbench_quantity_count": "general_vqa", "vrsbench_grid_position": "general_vqa",
        "vrsbench_extreme_category": "general_vqa", "general_vqa": "general_vqa",
        "multiple_choice_vqa": "multiple_choice_vqa", "scene_classification": "scene_classification",
        "grounding": "grounding", "spatial_relation": "spatial_relation",
        "change_caption": "change_caption", "change_qa": "change_qa", "caption": "caption",
        "primary_qwen_failure": "general_vqa", "partial_expert": "general_vqa",
        "failed_expert": "general_vqa", "counting_one_failed_tile": "counting",
        "judge_failure": "general_vqa", "missing_judge_resume": "general_vqa",
        "succeeded_sample_resume": "general_vqa",
    }
    question_by_case = {
        "native_counting": "How many buildings are visible?",
        "fine_grained_counting": "How many buildings are visible?",
        "vrsbench_quantity_count": "How many large vehicles are visible in the image?",
        "vrsbench_grid_position": "Where is the large vehicle located?",
        "vrsbench_extreme_category": "What object class is the top-most vehicle?",
        "grounding": "Locate the target building.", "spatial_relation": "Where is the target located?",
        "change_caption": "Describe the change.", "change_qa": "Did a building appear?",
        "caption": "Describe the image.", "counting_one_failed_tile": "How many buildings are visible?",
    }
    task = task_by_case[case_name]
    is_vrsbench = case_name.startswith("vrsbench_")
    roles = ["t1", "t2"] if task in {"change_caption", "change_qa"} else ["image"]
    question_type = {
        "vrsbench_quantity_count": "quantity", "vrsbench_grid_position": "position",
        "vrsbench_extreme_category": "object",
    }.get(case_name)
    return UnifiedSample(
        sample_id=case_name, dataset="VRSBench" if is_vrsbench else "parity",
        split="validation", task=task,
        images=[ImageRef(image_id=f"parity-image-{index + 1}", path=FIXTURE_IMAGE, role=role, width=4, height=4) for index, role in enumerate(roles)],
        question=question_by_case.get(case_name, "Is the statement correct?"),
        ground_truth=GroundTruth(answers=["yes"], count=4 if "counting" in case_name else None),
        metadata={"question_type": question_type} if question_type else {},
    )


def harness_settings(workspace: Path) -> AppSettings:
    """Create deterministic small-image settings for parity tests.
    为等价测试创建确定性小图设置。
    """

    return AppSettings.model_validate({
        "models": {"qwen": {"model": "deterministic-qwen", "max_tokens": 128}},
        "counting": {"tile_core_size": 2, "halo_size": 0, "model_max_side": 4,
                     "max_pixels_without_tiling": 100, "boundary_band_px": 0,
                     "seam_verify": False, "recursive_split_enabled": False, "min_core_size": 1},
        "runs": {"root": workspace / "runs"}, "paths": {"dataset_root": workspace / "dataset"},
    })
