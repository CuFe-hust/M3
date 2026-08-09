"""Public `serve` CLI command: serial local HTTP service with `GET /health`
and `POST /ask`.

公开 `serve` CLI 命令：仅含 `GET /health` 与 `POST /ask` 的串行本地 HTTP
服务。服务进程创建 Runtime 恰好一次（Qwen 一次加载），所有请求复用同一
客户端；手动服务不创建 DeepSeek 客户端。仅使用 stdlib http.server；
handler 内绝不构造模型客户端。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from application.runtime import Runtime
from application.settings import load_settings

EXIT_OK = 0
EXIT_RUNTIME = 1
EXIT_INTERRUPTED = 130

MAX_HTTP_BODY_BYTES = 1 << 20  # 1 MiB request body cap / 1 MiB 请求体上限


class RuntimeRequestHandler(BaseHTTPRequestHandler):
    """Serial handler exposing only ``GET /health`` and ``POST /ask``.
    仅暴露 ``GET /health`` 与 ``POST /ask`` 的串行处理器。

    ``application`` is bound per process (class attribute) by ``run_serve``
    or tests; handlers never construct model clients.
    ``application`` 由 run_serve 或测试按进程绑定（类属性）；handler 绝不
    构造模型客户端。
    """

    application: Runtime
    max_body_bytes: int = MAX_HTTP_BODY_BYTES

    def _send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.split("?", 1)[0] != "/health":
            self._send_json(404, {"status": "failed", "error": "not found"})
            return
        self._send_json(200, self.application.health_payload())

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/ask":
            # Consume the request body so the connection can close cleanly on
            # Windows; otherwise the client may see an aborted connection.
            # 消费请求体以便连接在 Windows 上干净关闭，否则客户端可能遇到连接中止。
            self._read_body()
            self._send_json(404, {"status": "failed", "error": "not found"})
            return
        body = self._read_body()
        if body is None:
            self._send_json(
                413, {"status": "failed", "error": "request body too large"}
            )
            return
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"status": "failed", "error": "invalid JSON body"})
            return
        if not isinstance(payload, dict):
            self._send_json(
                400,
                {"status": "failed", "error": "request body must be a JSON object"},
            )
            return
        image_dir = payload.get("image_dir")
        if not isinstance(image_dir, str) or not image_dir:
            self._send_json(400, {"status": "failed", "error": "image_dir is required"})
            return
        try:
            result = asyncio.run(
                self.application.ask(
                    image_dir=Path(image_dir),
                    question=str(payload.get("question", "")),
                    task=str(payload.get("task", "auto")),
                    source="http_service",
                )
            )
        except ValueError:
            # Stable public error; never echoes the raw exception text.
            # 稳定公共错误；绝不回显原始异常文本。
            self._send_json(400, {"status": "failed", "error": "invalid request"})
            return
        except Exception as error:
            self._send_json(
                500, {"status": "failed", "error": f"{type(error).__name__}"}
            )
            return
        self._send_json(200, result.model_dump(mode="json"))

    def _read_body(self) -> bytes | None:
        """Read the request body with a hard size cap; None means too large.
        以硬上限读取请求体；返回 None 表示超限。

        Oversized bodies are drained chunk-wise before the 413 response so the
        client can finish sending; this avoids aborted connections on Windows.
        超限请求体在返回 413 前被分块排空，使客户端可以完成发送；
        这避免了 Windows 上的连接中止。
        """

        length_header = self.headers.get("Content-Length")
        if length_header is not None:
            try:
                length = int(length_header)
            except ValueError:
                length = 0
            if length > self.max_body_bytes:
                _drain_body(self.rfile, length)
                return None
            return self.rfile.read(length)
        body = bytearray()
        while True:
            chunk = self.rfile.read(65536)
            if not chunk:
                break
            body.extend(chunk)
            if len(body) > self.max_body_bytes:
                _drain_body(self.rfile, None)
                return None
        return bytes(body)


def _drain_body(fileobj: Any, length: int | None) -> None:
    """Discard a request body in bounded chunks. / 以受限分块丢弃请求体。"""

    remaining = length
    while remaining is None or remaining > 0:
        size = min(65536, remaining) if remaining is not None else 65536
        chunk = fileobj.read(size)
        if not chunk:
            break
        if remaining is not None:
            remaining -= len(chunk)


def run_serve(args: argparse.Namespace) -> int:
    """Start the serial HTTP service and block until interrupted.
    启动串行 HTTP 服务并阻塞直到被中断。

    One ``Runtime.create()`` per server process; no DeepSeek client on the
    manual service; all requests share the same Qwen client.
    每个服务进程一次 ``Runtime.create()``；手动服务无 DeepSeek 客户端；
    所有请求共享同一 Qwen 客户端。"""

    try:
        if not (1 <= args.port <= 65535):
            raise ValueError("port must be between 1 and 65535")
        settings = load_settings(
            Path(args.config) if getattr(args, "config", None) else None,
        )
        runtime = Runtime.create(
            settings=settings,
            project_root=Path(__file__).resolve().parents[2],
            api_key=None,
        )
        handler = type(
            "BoundRuntimeRequestHandler",
            (RuntimeRequestHandler,),
            {"application": runtime},
        )
        server = HTTPServer((args.host, args.port), handler)
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    except Exception as error:
        # Public output never carries raw exception text or secrets.
        # 公共输出绝不携带原始异常文本或密钥。
        print(
            json.dumps({"status": "failed", "error": f"{type(error).__name__}"}),
            file=sys.stderr,
        )
        return EXIT_RUNTIME
    payload = dict(runtime.health_payload())
    payload.update({"host": args.host, "port": args.port})
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return EXIT_OK
