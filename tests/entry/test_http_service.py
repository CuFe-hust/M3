"""Serial HTTP service contract: /health, /ask, status codes, body cap.
串行 HTTP 服务契约：/health、/ask、状态码与请求体上限。"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

import pytest

from spacers_agent.application import (
    MAX_HTTP_BODY_BYTES,
    PublicAnswer,
    RuntimeRequestHandler,
)


class _FakeApplication:
    """Handler-facing application stub; validation errors mirror real ask().
    面向 Handler 的应用桩；校验错误镜像真实 ask()。
    """

    def __init__(self) -> None:
        self.ask_calls = 0
        self.health_calls = 0

    def health_payload(self) -> dict[str, object]:
        self.health_calls += 1
        return {
            "status": "ready",
            "model": "fake-local-model",
            "model_load_seconds": 1.5,
            "agents": ["counting_agent", "general_vqa_agent"],
        }

    async def ask(self, *, image_dir: Path, question: str, task: str, source: str) -> PublicAnswer:
        self.ask_calls += 1
        resolved = image_dir.expanduser().resolve()
        if resolved.name == "missing":
            raise ValueError(f"image directory does not exist: {resolved}")
        if resolved.name == "boom":
            raise RuntimeError("simulated agent failure")
        return PublicAnswer(
            request_id="http-test-000000",
            task=task,
            agent="general_vqa_agent",
            status="completed",
            answer=f"ok:{question}",
            evidence=[],
            warnings=[],
            elapsed_seconds=0.01,
            artifact_dir="/tmp/fake-artifacts",
        )


@pytest.fixture()
def server():
    """Start one serial HTTPServer on an ephemeral port for the test.
    在临时端口上为测试启动一个串行 HTTPServer。
    """

    app = _FakeApplication()
    handler = type("BoundRuntimeRequestHandler", (RuntimeRequestHandler,), {"application": app})
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    yield base, app
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def _request(url: str, *, method: str = "GET", body: object | bytes | None = None,
             headers: dict[str, str] | None = None) -> tuple[int, dict[str, object]]:
    """Issue one HTTP request and return (status, parsed JSON).
    发起一次 HTTP 请求并返回 (状态码, 解析后的 JSON)。
    """

    data: bytes | None = None
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def test_health_returns_ready(server):
    base, app = server
    status, payload = _request(f"{base}/health")
    assert status == 200
    assert payload["status"] == "ready"
    assert payload["model"] == "fake-local-model"
    assert payload["agents"] == ["counting_agent", "general_vqa_agent"]
    assert app.health_calls == 1


def test_ask_returns_public_answer(server, tmp_path):
    base, app = server
    image_dir = tmp_path / "ok"
    image_dir.mkdir()
    status, payload = _request(
        f"{base}/ask",
        method="POST",
        body={"image_dir": str(image_dir), "question": "hello", "task": "general_vqa"},
    )
    assert status == 200
    assert payload["status"] == "completed"
    assert payload["answer"] == "ok:hello"
    assert payload["task"] == "general_vqa"
    assert payload["agent"] == "general_vqa_agent"
    assert app.ask_calls == 1


def test_ask_defaults_task_and_question(server, tmp_path):
    base, _app = server
    image_dir = tmp_path / "ok"
    image_dir.mkdir()
    status, payload = _request(f"{base}/ask", method="POST", body={"image_dir": str(image_dir)})
    assert status == 200
    assert payload["task"] == "auto"
    assert payload["answer"] == "ok:"


def test_ask_invalid_json_returns_400(server):
    base, app = server
    status, payload = _request(f"{base}/ask", method="POST", body=b"{not json")
    assert status == 400
    assert payload["status"] == "failed"
    assert app.ask_calls == 0


def test_ask_missing_image_dir_returns_400(server):
    base, app = server
    status, payload = _request(f"{base}/ask", method="POST", body={"question": "q"})
    assert status == 400
    assert "image_dir" in str(payload["error"])
    assert app.ask_calls == 0


def test_ask_nonexistent_directory_returns_400(server, tmp_path):
    base, app = server
    status, payload = _request(
        f"{base}/ask",
        method="POST",
        body={"image_dir": str(tmp_path / "missing")},
    )
    assert status == 400
    assert "does not exist" in str(payload["error"])


def test_ask_agent_failure_returns_500(server, tmp_path):
    base, app = server
    status, payload = _request(
        f"{base}/ask",
        method="POST",
        body={"image_dir": str(tmp_path / "boom")},
    )
    assert status == 500
    assert "simulated agent failure" in str(payload["error"])
    assert app.ask_calls == 1


def test_unknown_get_path_returns_404(server):
    base, app = server
    status, payload = _request(f"{base}/nope")
    assert status == 404
    assert app.health_calls == 0


def test_unknown_post_path_returns_404(server):
    base, app = server
    status, payload = _request(f"{base}/nope", method="POST", body={"a": 1})
    assert status == 404
    assert app.ask_calls == 0


def test_oversized_body_returns_413(server):
    base, app = server
    oversized = b'{"image_dir": "' + b"x" * (MAX_HTTP_BODY_BYTES + 1024) + b'"}'
    status, payload = _request(f"{base}/ask", method="POST", body=oversized)
    assert status == 413
    assert "too large" in str(payload["error"])
    assert app.ask_calls == 0


def test_body_at_limit_is_accepted(server, tmp_path):
    base, app = server
    image_dir = tmp_path / "ok"
    image_dir.mkdir()
    question = "q" * (MAX_HTTP_BODY_BYTES - 512)
    status, payload = _request(
        f"{base}/ask",
        method="POST",
        body={"image_dir": str(image_dir), "question": question},
    )
    assert status == 200
    assert app.ask_calls == 1
