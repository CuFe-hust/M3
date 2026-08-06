"""main() level regression tests for the single public entry.
单一公开入口 main() 级别的回归测试。

Cover the implicit default serve path (no subcommand, config-only), explicit
serve overrides, and the guarantee that ask never starts the HTTP service.
覆盖隐式默认 serve（无子命令、仅配置）、显式 serve 覆盖，以及 ask 绝不启动
HTTP 服务的保证。
"""

from __future__ import annotations

from types import SimpleNamespace

import main as main_module


def test_main_without_subcommand_starts_default_server(monkeypatch):
    """python main.py must start serve with default host/port.
    python main.py 必须以默认 host/port 启动 serve。
    """

    seen = {}

    monkeypatch.setattr(
        main_module,
        "load_settings",
        lambda path: SimpleNamespace(),
    )
    fake_app = object()
    monkeypatch.setattr(
        main_module.RuntimeApplication,
        "create",
        lambda **kwargs: fake_app,
    )

    def fake_run_http_server(app, *, host, port):
        seen["app"] = app
        seen["host"] = host
        seen["port"] = port
        return 0

    monkeypatch.setattr(
        main_module,
        "run_http_server",
        fake_run_http_server,
    )

    assert main_module.main([]) == 0
    assert seen == {
        "app": fake_app,
        "host": "127.0.0.1",
        "port": 8000,
    }


def test_main_config_only_starts_default_server(monkeypatch, tmp_path):
    """python main.py --config ... must still start the default server.
    python main.py --config ... 仍必须启动默认服务。
    """

    config = tmp_path / "local.yaml"
    config.write_text("{}\n", encoding="utf-8")

    seen = {}

    def fake_load_settings(path):
        seen["config"] = path
        return SimpleNamespace()

    monkeypatch.setattr(
        main_module,
        "load_settings",
        fake_load_settings,
    )
    monkeypatch.setattr(
        main_module.RuntimeApplication,
        "create",
        lambda **kwargs: object(),
    )

    def fake_server(app, *, host, port):
        seen["host"] = host
        seen["port"] = port
        return 0

    monkeypatch.setattr(
        main_module,
        "run_http_server",
        fake_server,
    )

    assert main_module.main(["--config", str(config)]) == 0
    assert seen["config"] == config
    assert seen["host"] == "127.0.0.1"
    assert seen["port"] == 8000


def test_main_explicit_serve_overrides_host_port(monkeypatch):
    """Explicit serve --host/--port must override the defaults.
    显式 serve --host/--port 必须覆盖默认值。
    """

    seen = {}

    monkeypatch.setattr(
        main_module,
        "load_settings",
        lambda path: SimpleNamespace(),
    )
    monkeypatch.setattr(
        main_module.RuntimeApplication,
        "create",
        lambda **kwargs: object(),
    )

    def fake_server(app, *, host, port):
        seen["host"] = host
        seen["port"] = port
        return 0

    monkeypatch.setattr(
        main_module,
        "run_http_server",
        fake_server,
    )

    assert (
        main_module.main(
            ["serve", "--host", "0.0.0.0", "--port", "9000"]
        )
        == 0
    )
    assert seen == {"host": "0.0.0.0", "port": 9000}


def test_main_ask_does_not_start_http_server(monkeypatch, tmp_path):
    """ask must run one question and never start the HTTP server.
    ask 只执行一次问答，绝不启动 HTTP 服务。
    """

    image_dir = tmp_path / "images"
    image_dir.mkdir()

    class FakeAnswer:
        def model_dump_json(self, indent=2):
            return '{"status":"completed"}'

    class FakeApplication:
        async def ask(self, **kwargs):
            return FakeAnswer()

    monkeypatch.setattr(
        main_module,
        "load_settings",
        lambda path: SimpleNamespace(),
    )
    monkeypatch.setattr(
        main_module.RuntimeApplication,
        "create",
        lambda **kwargs: FakeApplication(),
    )

    def fail_server(*args, **kwargs):
        raise AssertionError("ask must not start HTTP server")

    monkeypatch.setattr(
        main_module,
        "run_http_server",
        fail_server,
    )

    assert (
        main_module.main(
            [
                "ask",
                "--images-dir",
                str(image_dir),
                "--question",
                "test",
                "--task",
                "general_vqa",
            ]
        )
        == 0
    )
