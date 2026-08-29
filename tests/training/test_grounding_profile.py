import json

from PIL import Image

from training.multimodal_sft.image_roots import ImageRootRegistry
from training.multimodal_sft.profiles.grounding import GroundingAgentDataProfile, GROUNDING_TARGET_SCHEMA


def row(tmp_path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    Image.new("RGB", (8, 8), (20, 40, 60)).save(image_root / "sample.png")
    target = {
        "agent_name": "grounding_agent",
        "answer": "{<10><79><17><85>}",
        "evidence_items": [{"label": "roundabout", "box": [100, 789, 170, 849]}],
        "status": "completed",
    }
    return image_root, {
        "episode_id": "grounding-0",
        "split": "train",
        "image_source": "vrs",
        "image": "sample.png",
        "question": "Find the roundabout.",
        "planner_output": {"object_categories": ["roundabout"]},
        "evidence_items": target["evidence_items"],
        "target": {"response_schema": GROUNDING_TARGET_SCHEMA, "result": target},
    }


def test_grounding_profile_renders_public_agent_result(tmp_path):
    image_root, episode = row(tmp_path)
    profile = GroundingAgentDataProfile()
    profile.validate(episode)
    messages = profile.render_messages(episode)
    rendered = json.loads(messages[-1]["content"][0]["text"])
    assert set(rendered) == {"agent_name", "answer", "evidence_items", "status"}
    assert "selected_box_ids" not in rendered

    prepared = profile.prepare(
        episode,
        image_roots=ImageRootRegistry({"vrs": image_root}),
        split="train",
        epoch=0,
        seed=1234,
    )
    assert prepared.images[0].size == (8, 8)
    assert prepared.target_schema == GROUNDING_TARGET_SCHEMA
