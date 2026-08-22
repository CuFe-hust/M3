from agents.change.settings import AgentChangeSettings


def test_default_building_rescue_profile_matches_validated_edge_run() -> None:
    settings = AgentChangeSettings().building_rescue

    assert settings.enabled is True
    assert settings.shadow_only is False
    assert settings.qwen_review_enabled is True
    assert settings.allowed_directions == ("added",)
    assert settings.edge_only is True
    assert settings.max_review_candidates == 3
    assert settings.edge_review_context_min_size_px == 112
    assert settings.edge_review_pixel_size == 448
    assert settings.rescue_max_new_tokens == 512
    assert settings.cache_policy == "bypass"
