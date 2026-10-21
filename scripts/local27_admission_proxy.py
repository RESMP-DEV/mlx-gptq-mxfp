#!/usr/bin/env python3
"""Bounded loopback admission proxy for the resident local27 model."""

from __future__ import annotations

import http.client
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BACKEND_HOST = os.environ.get("LOCAL27_BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.environ.get("LOCAL27_BACKEND_PORT", "8081"))
MAX_BACKEND = int(os.environ.get("LOCAL27_MAX_BACKEND", "4"))

_slots = threading.BoundedSemaphore(MAX_BACKEND)
_state_lock = threading.Lock()
_state = {"accepted": 0, "active": 0, "waiting": 0, "backend_errors": 0}


def _begin() -> None:
    with _state_lock:
        _state["accepted"] += 1
        _state["waiting"] += 1
    _slots.acquire()
    with _state_lock:
        _state["waiting"] -= 1
        _state["active"] += 1


def _end(success: bool) -> None:
    with _state_lock:
        _state["active"] -= 1
        if not success:
            _state["backend_errors"] += 1
    _slots.release()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _write_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            with _state_lock:
                state = dict(_state)
            state.update(
                {
                    "backend": f"{BACKEND_HOST}:{BACKEND_PORT}",
                    "max_backend": MAX_BACKEND,
                }
            )
            self._write_json(200, state)
            return
        if self.path == "/readyz":
            with _state_lock:
                state = dict(_state)
            try:
                conn = http.client.HTTPConnection(BACKEND_HOST, BACKEND_PORT, timeout=2)
                conn.request("GET", "/metrics", headers={"Connection": "close"})
                response = conn.getresponse()
                response.read()
                conn.close()
                ready = 200 <= response.status < 300
            except Exception:
                ready = False
            self._write_json(
                200 if ready else 503,
                {
                    "ready": ready,
                    "backend": f"{BACKEND_HOST}:{BACKEND_PORT}",
                    **state,
                },
            )
            return
        self._proxy("GET")

    def do_POST(self) -> None:
        if self.path.startswith("/v1/"):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._write_json(400, {"error": "request body must be JSON"})
                return
            if payload.get("stream", False):
                self._write_json(
                    400,
                    {"error": "streaming is not enabled in admission phase one"},
                )
                return
            self._proxy("POST", body)
            return
        self._proxy("POST")

    def _proxy(self, method: str, body: bytes | None = None) -> None:
        if body is None:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b""

        success = False
        _begin()
        try:
            conn = http.client.HTTPConnection(BACKEND_HOST, BACKEND_PORT, timeout=300)
            conn.request(
                method,
                self.path,
                body=body,
                headers={
                    "Content-Type": self.headers.get(
                        "Content-Type", "application/json"
                    ),
                    "Content-Length": str(len(body)),
                    "Connection": "close",
                },
            )
            response = conn.getresponse()
            response_body = response.read()
            self.send_response(response.status, response.reason)
            for name, value in response.getheaders():
                lowered = name.lower()
                if lowered in {"connection", "transfer-encoding", "content-length"}:
                    continue
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
            conn.close()
            success = True
        except Exception:
            self._write_json(503, {"error": "local27 backend unavailable"})
        finally:
            _end(success)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128


if __name__ == "__main__":
    host = os.environ.get("LOCAL27_PROXY_HOST", "127.0.0.1")
    port = int(os.environ.get("LOCAL27_PROXY_PORT", "8080"))
    server = Server((host, port), Handler)
    print(
        json.dumps(
            {
                "event": "listening",
                "host": host,
                "port": port,
                "backend": f"{BACKEND_HOST}:{BACKEND_PORT}",
                "max_backend": MAX_BACKEND,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    server.serve_forever()
