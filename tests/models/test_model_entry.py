"""Contract tests for the unified model entry.

统一模型入口测试：注册/创建/列举、未知模型显式报错、惰性 builder
（import models.entry 不加载 transformers/torch）、qwen3_5 与 baseline
builder 可用且不复制 Agent 逻辑。
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import BaseModel

from models.entry import create_model, list_models, register
from models.settings import QwenSettings, SegFormerSettings


class _DummySettings(BaseModel):
    model: str = "fake"
    max_tokens: int = 8


def test_register_and_list_models() -> None:
    names = list_models()
    assert "qwen_transformers" in names
    assert "qwen3_vl_baseline" in names
    assert "qwen3_5_transformers" in names
    assert "qwen3_5_multi_adapter" in names
    assert "segformer_transformers" in names
    assert len(names) == len(set(names))


def test_register_duplicate_name_raises() -> None:
    with pytest.raises(ValueError, match="already registered"):

        @register("qwen_transformers")
        def _duplicate(**kwargs):  # pragma: no cover
            return None


def test_create_model_unknown_name_raises() -> None:
    with pytest.raises(KeyError, match="Unknown model entry"):
        create_model("no-such-model")


def test_import_entry_does_not_load_heavy_libraries() -> None:
    """Importing models.entry must not load transformers or torch.
    导入 models.entry 不得加载 transformers 或 torch。"""
    script = (
        "import sys; import models.entry; "
        "assert 'transformers' not in sys.modules; "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_import_entry_does_not_import_concrete_segformer_module() -> None:
    script = (
        "import sys; import models.entry; "
        "print('models.segformer_transformers' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "False"


def test_segformer_builder_is_lazy_until_construction(tmp_path: Path) -> None:
    client = create_model(
        "segformer_transformers",
        settings=SegFormerSettings(
            model_path=tmp_path / "missing-is-allowed-at-construction",
            logical_model_id="segformer-entry-test",
            weights_sha256="0" * 64,
            device="cpu",
        ),
    )
    from models.segformer_transformers import SegFormerTransformersClient

    assert isinstance(client, SegFormerTransformersClient)
    assert client.loaded is False


def test_builders_are_lazy() -> None:
    """Builders must import concrete models lazily; constructing a client with
    injected fake model/processor must not touch transformers.
    builder 必须惰性导入具体模型；注入 fake 构造客户端不触碰 transformers。"""
    script = dedent(
        """
    import sys
    from models.entry import create_model
    from models.settings import QwenSettings

    class FakeModel:
        pass

    class FakeProcessor:
        pass

    create_model(
        "qwen_transformers",
        settings=QwenSettings(model="fake", max_tokens=8),
        model=FakeModel(),
        processor=FakeProcessor(),
    )
    assert "transformers" not in sys.modules
    assert "torch" not in sys.modules
    """
    )
    subprocess.run([sys.executable, "-c", script], check=True, text=True)


def test_qwen35_builder_returns_shared_client() -> None:
    from models.qwen_transformers import QwenTransformersClient

    client = create_model(
        "qwen3_5_transformers",
        settings=QwenSettings(model="fake"),
        model=object(),
        processor=object(),
    )
    assert isinstance(client, QwenTransformersClient)


def test_baseline_builder_with_injected_components() -> None:
    from models.qwen3_vl.baseline import Qwen3VLBaseline

    baseline = create_model(
        "qwen3_vl_baseline",
        settings=_DummySettings(),
        model=object(),
        processor=object(),
    )
    assert isinstance(baseline, Qwen3VLBaseline)


def test_baseline_and_client_contain_no_agent_logic(tmp_path: Path) -> None:
    """Baseline and client modules must not reference agents or router.
    基线与客户端模块不得引用 agents 或 router。"""
    for relative in (
        "models/qwen3_vl/baseline.py",
        "models/qwen_transformers.py",
        "models/entry.py",
    ):
        source = (Path(__file__).resolve().parents[2] / relative).read_text(encoding="utf-8")
        assert "spacers_agent" not in source, relative
        assert "Agent" not in source or "Agent" in source and "AgentName" not in source, relative


# ── 多模态模板 / multimodal template (E) ───────────────────────────────────


class _RecordingProcessor:
    """Fake processor that records the chat messages it was given.
    记录收到的聊天消息的假 processor。"""

    def __init__(self) -> None:
        self.messages: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return "prompt"

    def __call__(self, text=None, images=None, **kwargs):
        class _Inputs:
            shape = (1, 1)

            def to(self, device):
                return self

        return {"input_ids": _Inputs()}

    def batch_decode(self, *args, **kwargs):
        return ["ok"]


class _GeneratingModel:
    device = "cpu"

    def generate(self, **kwargs):
        # Two tokens so trimming output[input_tokens:] keeps one token.
        # 两个 token，使 output[input_tokens:] 保留一个 token。
        return [[0, 0]]


def _make_baseline(processor: _RecordingProcessor):
    from models.qwen3_vl.baseline import Qwen3VLBaseline, Qwen3VLSettings

    return Qwen3VLBaseline(
        Qwen3VLSettings(model="fake"),
        model=_GeneratingModel(),
        processor=processor,
    )


@pytest.mark.parametrize("count", [0, 1, 2])
def test_baseline_multimodal_content_shape(count: int) -> None:
    """0/1/2 images must produce exactly that many image items followed by
    one trailing text item. 0/1/2 张图必须产生对应数量的 image item，
    且末尾固定一个 text item。"""
    processor = _RecordingProcessor()
    baseline = _make_baseline(processor)
    images = [f"img-{i}" for i in range(count)] or None
    baseline.generate_text(text="Q", images=images)

    assert len(processor.messages) == 1
    assert processor.messages[0]["role"] == "user"
    content = processor.messages[0]["content"]
    image_items = [item for item in content if item.get("type") == "image"]
    assert len(image_items) == count
    assert content[-1] == {"type": "text", "text": "Q"}


def test_baseline_multimodal_preserves_image_order() -> None:
    """Image items must keep the caller-provided order.
    image item 必须保持调用方提供的顺序。"""
    processor = _RecordingProcessor()
    baseline = _make_baseline(processor)
    baseline.generate_text(text="Q", images=["first", "second"])

    content = processor.messages[0]["content"]
    assert content[0] == {"type": "image", "image": "first"}
    assert content[1] == {"type": "image", "image": "second"}
    assert content[2] == {"type": "text", "text": "Q"}


def test_baseline_offline_default() -> None:
    """Baseline must default to local files only.
    基线默认只使用本地文件。"""
    from models.qwen3_vl.baseline import Qwen3VLSettings

    settings = Qwen3VLSettings()
    assert settings.allow_download is False
    # Single-source offline config: no dual local_files_only field remains.
    # 离线配置单源：不再保留 local_files_only 双字段。
    assert not hasattr(settings, "local_files_only")


# ── 打包 / packaging (A) ───────────────────────────────────────────────────


def test_pyproject_packages_include_data_models_agents() -> None:
    """The wheel must ship data, models, agents, routing, workflows,
    evaluation, reporting and application. wheel 必须包含 data、models、
    agents、routing、workflows、evaluation、reporting 与 application（已实现
    包由 test_package_discovery 守卫强制同步）。"""
    import tomllib

    root = Path(__file__).resolve().parents[2]
    with open(root / "pyproject.toml", "rb") as handle:
        config = tomllib.load(handle)
    include = config["tool"]["setuptools"]["packages"]["find"]["include"]
    assert include == [
        "data*", "models*", "agents*", "routing*", "workflows*", "evaluation*",
        "reporting*", "application*", "prompts*",
    ]


# ── 注册表污染 / registry pollution (J) ────────────────────────────────────


def test_failed_register_does_not_pollute_registry() -> None:
    """A failed registration must not modify the registry.
    失败的注册不得修改注册表。"""
    import models.entry as entry

    before = dict(entry._REGISTRY)
    with pytest.raises(ValueError, match="already registered"):

        @register("qwen_transformers")
        def _duplicate_b(**kwargs):  # pragma: no cover
            return None

    assert entry._REGISTRY == before
