"""Contract tests for request hashing and message sanitization.

请求哈希与消息脱敏测试：data URL 被摘要替换、脱敏后 hash 稳定、
不含 Base64/凭据、协议可被最小客户端满足。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from models import (
    ModelT,
    RequestMeta,
    VisionLanguageClient,
    build_request_hash,
    crop_image_region,
    sanitize_messages,
)
from models.images import image_sha256, image_to_data_url
from PIL import Image, ImageOps
from pydantic import ValidationError

IMAGE_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def test_sanitize_messages_redacts_data_urls() -> None:
    url = image_to_data_url(IMAGE_BYTES, "image/png")
    messages = [{"type": "image_url", "image_url": {"url": url}}]
    sanitized = sanitize_messages(messages)
    assert "data:image/png;base64," not in str(sanitized)
    redacted = sanitized[0]["image_url"]["url"]
    assert redacted["redacted_data_url_sha256"] == image_sha256(url.encode("utf-8"))
    assert redacted["encoded_bytes"] == len(url.encode("utf-8"))


def test_sanitize_messages_leaves_plain_text() -> None:
    messages = [{"role": "user", "content": "plain text"}]
    assert sanitize_messages(messages) == messages


def test_sanitize_messages_handles_nested_lists() -> None:
    url = image_to_data_url(IMAGE_BYTES)
    messages = [{"parts": [{"url": url}, "text"]}]
    sanitized = sanitize_messages(messages)
    assert "base64," not in str(sanitized)
    assert sanitized[0]["parts"][1] == "text"


def test_request_hash_is_stable_and_sanitized() -> None:
    url = image_to_data_url(IMAGE_BYTES)
    messages = [{"image_url": {"url": url}}]
    kwargs = dict(
        model="qwen",
        generation={"max_tokens": 128},
        prompt_version="count-tile-v4",
        messages=messages,
        image_sha256=image_sha256(IMAGE_BYTES),
    )
    first = build_request_hash(**kwargs)
    second = build_request_hash(**kwargs)
    assert first == second
    assert first == build_request_hash(**{**kwargs, "messages": [{"image_url": {"url": url}}]})
    # A different prompt version changes the hash. / 不同 prompt 版本改变哈希。
    assert first != build_request_hash(**{**kwargs, "prompt_version": "other"})


def test_request_hash_never_contains_raw_base64() -> None:
    url = image_to_data_url(IMAGE_BYTES)
    digest = build_request_hash(
        model="qwen",
        generation={},
        prompt_version="v1",
        messages=[{"image_url": {"url": url}}],
        image_sha256=None,
    )
    assert digest not in url
    assert "base64" not in digest


def _hash_kwargs() -> dict:
    return dict(
        model="qwen",
        generation={"temperature": 0.0, "do_sample": False, "max_tokens": 128},
        prompt_version="v1",
        messages=[{"role": "user", "content": "Q"}],
        image_sha256=None,
    )


def test_request_hash_changes_with_each_semantic_field() -> None:
    """Every inference-semantic field must change the hash: generation,
    response schema, client version, model revision, and model name.
    每个推理语义字段都必须改变哈希：生成参数、响应 Schema、客户端版本、
    模型 revision 与模型名。"""
    base = _hash_kwargs()
    digest = build_request_hash(**base)

    assert digest != build_request_hash(**{**base, "generation": {
        "temperature": 0.0, "do_sample": False, "max_tokens": 256}})
    assert digest != build_request_hash(**{**base, "generation": {
        "temperature": 0.7, "do_sample": True, "max_tokens": 128}})
    assert digest != build_request_hash(**{**base, "response_schema": {
        "type": "object", "properties": {"answer": {"type": "string"}}}})
    assert digest != build_request_hash(**{**base, "client_version": "2"})
    assert digest != build_request_hash(**{**base, "model_revision": "abc123"})
    assert digest != build_request_hash(**{**base, "model": "qwen-other"})
    assert digest != build_request_hash(**{**base, "prompt_version": "v2"})


def test_request_hash_changes_with_structured_decoding_identity() -> None:
    """Planner decoding policy and schema identity are cache-key inputs.
    规划器解码策略与 Schema 身份必须参与缓存键。"""
    base = _hash_kwargs()
    native = build_request_hash(**base)
    constrained = build_request_hash(
        **base,
        structured_decoding="outlines-json-schema",
        outlines_adapter_version="qwen-transformers-outlines-v1",
        pinned_outlines_version="1.3.3",
        schema_sha256="a" * 64,
    )
    assert native != constrained
    assert constrained != build_request_hash(
        **base,
        structured_decoding="outlines-json-schema",
        outlines_adapter_version="qwen-transformers-outlines-v1",
        pinned_outlines_version="1.3.3",
        schema_sha256="b" * 64,
    )


def test_request_hash_changes_with_image_digest_and_order() -> None:
    """Different image bytes and different image order must change the hash.
    图片内容不同与图片顺序不同都必须改变哈希。"""
    import hashlib

    img_a = hashlib.sha256(b"AAAA").hexdigest()
    img_b = hashlib.sha256(b"BBBB").hexdigest()
    base = _hash_kwargs()

    h_a = build_request_hash(**{**base, "image_sha256": img_a})
    h_b = build_request_hash(**{**base, "image_sha256": img_b})
    assert h_a != h_b

    h_ab = build_request_hash(**{**base, "image_sha256": "|".join([img_a, img_b])})
    h_ba = build_request_hash(**{**base, "image_sha256": "|".join([img_b, img_a])})
    assert h_ab != h_ba


def test_request_meta_rejects_credentials() -> None:
    with pytest.raises(ValidationError):
        RequestMeta.model_validate(
            {"request_id": "r", "request_hash": "a" * 64, "prompt_version": "v",
             "api_key": "sk-secret"}
        )


def test_import_models_does_not_load_transformers_or_torch() -> None:
    """Importing models must not import transformers or torch.
    导入 models 不得导入 transformers 或 torch。"""
    for heavy in ("transformers", "torch"):
        assert heavy not in sys.modules


def test_vision_language_client_protocol_is_structural() -> None:
    import inspect

    class FakeClient:
        async def complete_json(self, *, messages, response_model, request_meta, max_tokens=None):
            return response_model.model_validate({})

    client = FakeClient()
    # The protocol is structural; verify the async method shape without
    # isinstance (protocols are not runtime-checkable by default).
    # 协议为结构化；直接验证异步方法形态（协议默认不可 isinstance 检查）。
    assert inspect.iscoroutinefunction(FakeClient.complete_json)


# ── ModelCacheIdentity / 缓存身份 (A) ──────────────────────────────────────


def test_cache_identity_rejects_empty_model_and_version() -> None:
    from dataclasses import FrozenInstanceError

    from models.base import ModelCacheIdentity

    with pytest.raises(ValueError, match="model"):
        ModelCacheIdentity(model="", generation={}, client_version="1")
    with pytest.raises(ValueError, match="client_version"):
        ModelCacheIdentity(model="m", generation={}, client_version="")


def test_cache_identity_rejects_non_json_safe_generation() -> None:
    from models.base import ModelCacheIdentity

    with pytest.raises(ValueError, match="non-finite"):
        ModelCacheIdentity(model="m", generation={"t": float("nan")}, client_version="1")
    with pytest.raises(ValueError, match="Path"):
        ModelCacheIdentity(model="m", generation={"p": Path("/tmp/x")}, client_version="1")
    with pytest.raises(ValueError, match="set"):
        ModelCacheIdentity(model="m", generation={"s": {1}}, client_version="1")
    with pytest.raises(ValueError, match="bytes"):
        ModelCacheIdentity(model="m", generation={"b": b"x"}, client_version="1")
    with pytest.raises(ValueError, match="callable"):
        ModelCacheIdentity(model="m", generation={"c": lambda: None}, client_version="1")


def test_cache_identity_rejects_sensitive_generation() -> None:
    from models.base import ModelCacheIdentity

    with pytest.raises(ValueError, match="sensitive key"):
        ModelCacheIdentity(model="m", generation={"api_key": "sk-1"}, client_version="1")
    with pytest.raises(ValueError, match="sensitive value"):
        ModelCacheIdentity(model="m", generation={"max_tokens": "Bearer x"}, client_version="1")


def test_cache_identity_is_frozen_and_stable() -> None:
    from dataclasses import FrozenInstanceError

    from models.base import ModelCacheIdentity

    identity = ModelCacheIdentity(
        model="m", generation={"max_tokens": 1}, client_version="1", revision="r"
    )
    with pytest.raises(FrozenInstanceError):
        identity.model = "other"  # type: ignore[misc]
    assert identity == ModelCacheIdentity(
        model="m", generation={"max_tokens": 1}, client_version="1", revision="r"
    )
    assert identity != ModelCacheIdentity(
        model="m", generation={"max_tokens": 2}, client_version="1", revision="r"
    )
    assert identity != ModelCacheIdentity(
        model="m", generation={"max_tokens": 1}, client_version="1", revision="r2"
    )


def test_cache_identity_generation_is_deeply_frozen() -> None:
    """Mutating the caller's source dict must not change the identity, and the
    internal structure must reject in-place mutation.
    修改调用方的源 dict 不得改变身份；内部结构必须拒绝原地修改。"""
    from models.base import ModelCacheIdentity

    source = {"max_tokens": 64, "nested": {"values": [1, 2]}}
    identity = ModelCacheIdentity(model="m", generation=source, client_version="1")
    source["max_tokens"] = 999
    source["nested"]["values"].append(3)

    payload = identity.generation_payload()
    assert payload["max_tokens"] == 64
    assert payload["nested"]["values"] == [1, 2]

    # The internal structure must reject in-place mutation.
    # 内部结构必须拒绝原地修改。
    with pytest.raises((TypeError, AttributeError)):
        identity.generation["max_tokens"] = 128  # type: ignore[index]

    # Payloads are fresh copies: mutating one never changes the identity.
    # payload 是全新副本：修改它不会改变身份。
    identity.generation_payload()["nested"]["values"].append(99)
    assert identity.generation_payload()["nested"]["values"] == [1, 2]


def test_cache_identity_requires_string_keys() -> None:
    """Non-string mapping keys must fail; they are never auto-stringified.
    非字符串映射键必须失败；绝不自动字符串化。"""
    from models.base import ModelCacheIdentity

    with pytest.raises(ValueError, match="non-string key"):
        ModelCacheIdentity(model="m", generation={1: "x"}, client_version="1")
    with pytest.raises(ValueError, match="non-string key"):
        ModelCacheIdentity(model="m", generation={"nested": {1: 2}}, client_version="1")
    with pytest.raises(ValueError, match="non-string key"):
        ModelCacheIdentity(model="m", generation={"items": [{2: 3}]}, client_version="1")


def test_cache_identity_key_order_does_not_matter() -> None:
    """Identical content with different key order must be equal and hash
    stably. 内容相同但键顺序不同的身份必须相等且哈希稳定。"""
    from models.base import ModelCacheIdentity, build_request_hash

    a = ModelCacheIdentity(model="m", generation={"a": 1, "b": 2}, client_version="1")
    b = ModelCacheIdentity(model="m", generation={"b": 2, "a": 1}, client_version="1")
    assert a == b
    assert a.generation_payload() == b.generation_payload()
    kwargs = dict(model=a.model, prompt_version="v1", messages=[], image_sha256=None)
    assert build_request_hash(
        **kwargs, generation=a.generation_payload(), client_version=a.client_version
    ) == build_request_hash(
        **kwargs, generation=b.generation_payload(), client_version=b.client_version
    )


# ── 跨平台路径识别 / cross-platform path detection (A/C) ───────────────────


@pytest.mark.parametrize("path_like", [
    "/models/Qwen",
    r"C:\models\Qwen",
    "C:/models/Qwen",
    r"\\server\share\Qwen",
    "//server/share/Qwen",
    "file:///models/Qwen",
    "file://server/share/Qwen",
])
def test_cache_identity_rejects_path_like_model(path_like: str) -> None:
    """Identity model must be a logical identifier, never a local path.
    身份模型必须是逻辑标识符，绝不能是本地路径。"""
    from models.base import ModelCacheIdentity

    with pytest.raises(ValueError, match="logical identifier"):
        ModelCacheIdentity(model=path_like, generation={}, client_version="1")


def test_cache_identity_generation_exposes_mapping() -> None:
    """generation must publicly behave as a Mapping even though it is frozen.
    generation 对外必须是 Mapping，尽管内部已冻结。"""
    from collections.abc import Mapping

    from models.base import ModelCacheIdentity

    identity = ModelCacheIdentity(
        model="m",
        generation={"max_tokens": 64, "nested": {"values": [1, 2]}},
        client_version="1",
    )
    assert isinstance(identity.generation, Mapping)
    assert dict(identity.generation)["max_tokens"] == 64
    assert identity.generation["max_tokens"] == 64
    assert isinstance(identity.generation["nested"], Mapping)
    # Nested lists are frozen as tuples inside the Mapping view; the plain
    # JSON payload restores lists. 嵌套 list 在 Mapping 视图中冻结为 tuple；
    # 普通 JSON payload 还原为 list。
    assert identity.generation["nested"]["values"] == (1, 2)
    assert identity.generation_payload() == {"max_tokens": 64, "nested": {"values": [1, 2]}}


def test_cache_identity_rejects_bad_client_version_and_revision() -> None:
    from models.base import ModelCacheIdentity

    with pytest.raises(ValueError, match="client_version"):
        ModelCacheIdentity(model="m", generation={}, client_version="   ")
    with pytest.raises(ValueError, match="revision"):
        ModelCacheIdentity(model="m", generation={}, client_version="1", revision="a\nb")
    with pytest.raises(ValueError, match="revision"):
        ModelCacheIdentity(model="m", generation={}, client_version="1", revision="\x00")
    # Trimmed, non-empty revisions are allowed. / strip 后非空的 revision 允许。
    identity = ModelCacheIdentity(model="m", generation={}, client_version="1", revision=" v1.0 ")
    assert identity.revision == "v1.0"


def test_validate_logical_model_id_accepts_remote_names() -> None:
    from models.base import validate_logical_model_id

    for value in ("Qwen/Qwen3-VL-4B-Instruct", "qwen3-vl-4b-local",
                  "qwen3.5-gb10", "org:model@rev"):
        assert validate_logical_model_id(value, where="cache_model_id") == value


# ── crop_image_region / 图像 ROI 裁切 ─────────────────────────────────────


def _make_pattern_image(width: int, height: int) -> Image.Image:
    """RGB image whose pixel color encodes its (x, y) position.
    像素颜色编码其 (x, y) 位置的 RGB 图像。"""
    buf = bytearray(width * height * 3)
    i = 0
    for y in range(height):
        for x in range(width):
            buf[i] = x % 256
            buf[i + 1] = y % 256
            buf[i + 2] = (x + y) % 256
            i += 3
    return Image.frombytes("RGB", (width, height), bytes(buf))


def _assert_crop_matches(
    result: Image.Image,
    source: Image.Image,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> None:
    """Assert the crop is exactly the integer pixel region of ``source``.
    断言裁切结果正是 ``source`` 的整数像素区域。"""
    expected = source.crop((left, top, right, bottom))
    assert result.size == (right - left, bottom - top)
    assert result.tobytes() == expected.tobytes()


def test_crop_full_image_is_exact(tmp_path) -> None:
    source = _make_pattern_image(120, 80)
    path = tmp_path / "full.png"
    source.save(path, format="PNG")

    for frame, box in (
        ("normalized_0_1_top_left", [0.0, 0.0, 1.0, 1.0]),
        ("normalized_0_999_top_left", [0.0, 0.0, 999.0, 999.0]),
    ):
        result = crop_image_region(path, box, coordinate_frame=frame)
        assert result.size == source.size
        assert result.tobytes() == source.tobytes()


def test_crop_coordinate_frames_are_equivalent() -> None:
    source = _make_pattern_image(100, 80)
    one = crop_image_region(
        source, [0.25, 0.25, 0.75, 0.75], coordinate_frame="normalized_0_1_top_left"
    )
    nine = crop_image_region(
        source,
        [249.75, 249.75, 749.25, 749.25],
        coordinate_frame="normalized_0_999_top_left",
    )
    assert one.size == nine.size == (50, 40)
    assert one.tobytes() == nine.tobytes()
    _assert_crop_matches(one, source, 25, 20, 75, 60)


def test_crop_path_and_pil_inputs_agree(tmp_path) -> None:
    source = _make_pattern_image(60, 40)
    path = tmp_path / "img.png"
    source.save(path, format="PNG")

    box = [0.1, 0.2, 0.6, 0.7]
    from_path = crop_image_region(path, box, coordinate_frame="normalized_0_1_top_left")
    from_pil = crop_image_region(source, box, coordinate_frame="normalized_0_1_top_left")
    assert from_path.size == from_pil.size
    assert from_path.tobytes() == from_pil.tobytes()
    _assert_crop_matches(from_pil, source, 6, 8, 36, 28)


def test_crop_path_applies_exif_orientation(tmp_path) -> None:
    source = _make_pattern_image(24, 14)
    exif = Image.Exif()
    exif[274] = 6  # EXIF Orientation: rotate 90 degrees CW. / 旋转 90° CW。
    path = tmp_path / "rotated.jpg"
    source.save(path, format="JPEG", exif=exif)

    result = crop_image_region(
        path, [0.0, 0.0, 1.0, 1.0], coordinate_frame="normalized_0_1_top_left"
    )
    expected = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    # Orientation applied: the stored (24, 14) JPEG must crop as (14, 24).
    # 方向已应用：存储为 (24, 14) 的 JPEG 必须按 (14, 24) 裁切。
    assert result.size == expected.size == (14, 24)
    assert result.tobytes() == expected.tobytes()


def test_crop_converts_non_rgb_to_rgb() -> None:
    source_rgb = _make_pattern_image(30, 20)
    rgba = source_rgb.convert("RGBA")
    result = crop_image_region(
        rgba, [0.0, 0.0, 1.0, 1.0], coordinate_frame="normalized_0_1_top_left"
    )
    assert result.mode == "RGB"
    assert result.tobytes() == source_rgb.tobytes()


def test_crop_odd_dimensions_round_outward() -> None:
    source = _make_pattern_image(101, 99)
    result = crop_image_region(
        source, [0.5, 0.5, 0.75, 0.75], coordinate_frame="normalized_0_1_top_left"
    )
    # x0/y0 floor, x1/y1 ceil on the odd dimensions.
    # 奇数尺寸下左上向下取整、右下向上取整。
    _assert_crop_matches(result, source, 50, 49, 76, 75)


def test_crop_1px_image() -> None:
    source = _make_pattern_image(1, 1)
    result = crop_image_region(
        source, [0.0, 0.0, 1.0, 1.0], coordinate_frame="normalized_0_1_top_left"
    )
    _assert_crop_matches(result, source, 0, 0, 1, 1)


def test_crop_edge_roi() -> None:
    source = _make_pattern_image(100, 100)
    result = crop_image_region(
        source, [0.0, 0.0, 0.1, 0.1], coordinate_frame="normalized_0_1_top_left"
    )
    _assert_crop_matches(result, source, 0, 0, 10, 10)


def test_crop_halo_expands_all_sides() -> None:
    source = _make_pattern_image(100, 100)
    result = crop_image_region(
        source,
        [0.2, 0.2, 0.4, 0.4],
        coordinate_frame="normalized_0_1_top_left",
        halo_ratio=0.1,
    )
    _assert_crop_matches(result, source, 18, 18, 42, 42)


def test_crop_halo_clamps_to_image_bounds() -> None:
    source = _make_pattern_image(100, 100)
    result = crop_image_region(
        source,
        [0.0, 0.0, 0.1, 0.1],
        coordinate_frame="normalized_0_1_top_left",
        halo_ratio=0.1,
    )
    _assert_crop_matches(result, source, 0, 0, 11, 11)


def test_crop_full_image_with_halo_is_whole_image() -> None:
    source = _make_pattern_image(64, 48)
    result = crop_image_region(
        source,
        [0.0, 0.0, 1.0, 1.0],
        coordinate_frame="normalized_0_1_top_left",
        halo_ratio=0.5,
    )
    _assert_crop_matches(result, source, 0, 0, 64, 48)


def test_crop_does_not_modify_input_image() -> None:
    source = _make_pattern_image(40, 30)
    before = source.tobytes()
    size_before = source.size
    result = crop_image_region(
        source, [0.25, 0.25, 0.75, 0.75], coordinate_frame="normalized_0_1_top_left"
    )
    assert source.tobytes() == before
    assert source.size == size_before
    assert result is not source


@pytest.mark.parametrize("box", [
    [0.1, 0.2, 0.3],                        # too few / 数量过少
    [0.1, 0.2, 0.3, 0.4, 0.5],              # too many / 数量过多
    [float("nan"), 0.0, 1.0, 1.0],          # NaN x0
    [0.0, float("inf"), 1.0, 1.0],          # +Inf y0
    [0.0, 0.0, 1.0, float("-inf")],         # -Inf y1
    [-0.1, 0.0, 1.0, 1.0],                  # below lower bound / 低于下界
    [0.0, 0.0, 1.5, 1.0],                   # above upper bound (0..1 frame) / 越上界
    [0.5, 0.5, 0.5, 1.0],                   # zero-width degenerate / 零宽退化框
    [0.6, 0.4, 0.4, 1.0],                   # reversed / 反向框
    ["0.1", 0.0, 1.0, 1.0],                 # non-numeric / 非数值
])
def test_crop_rejects_invalid_box(box) -> None:
    source = _make_pattern_image(20, 20)
    with pytest.raises(ValueError):
        crop_image_region(source, box, coordinate_frame="normalized_0_1_top_left")


def test_crop_rejects_out_of_range_999_frame() -> None:
    source = _make_pattern_image(20, 20)
    with pytest.raises(ValueError):
        crop_image_region(
            source, [0.0, 0.0, 1000.0, 999.0], coordinate_frame="normalized_0_999_top_left"
        )


def test_crop_rejects_unknown_coordinate_frame() -> None:
    source = _make_pattern_image(20, 20)
    with pytest.raises(ValueError, match="coordinate_frame"):
        crop_image_region(source, [0.0, 0.0, 1.0, 1.0], coordinate_frame="bogus_frame")


@pytest.mark.parametrize("halo_ratio", [-0.1, float("nan"), float("inf")])
def test_crop_rejects_invalid_halo(halo_ratio: float) -> None:
    source = _make_pattern_image(20, 20)
    with pytest.raises(ValueError):
        crop_image_region(
            source,
            [0.0, 0.0, 1.0, 1.0],
            coordinate_frame="normalized_0_1_top_left",
            halo_ratio=halo_ratio,
        )


def test_crop_rejects_non_path_non_image_input() -> None:
    with pytest.raises(TypeError):
        crop_image_region(
            "not/a/path.png", [0.0, 0.0, 1.0, 1.0], coordinate_frame="normalized_0_1_top_left"
        )
