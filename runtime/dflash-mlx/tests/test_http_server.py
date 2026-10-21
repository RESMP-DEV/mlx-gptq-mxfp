# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import http.server
import socket
import threading
import urllib.error
import urllib.request

from dflash_mlx.server.http_server import BACKLOG, hardened_server_class


def test_hardened_server_class_has_a_burst_sized_backlog():
    server_class = hardened_server_class()

    assert issubclass(server_class, http.server.ThreadingHTTPServer)
    assert server_class.request_queue_size == BACKLOG
    assert server_class.request_queue_size > 128
    assert server_class.allow_reuse_address is True
    assert server_class.daemon_threads is True


def test_hardened_server_class_is_fresh_per_call():
    first = hardened_server_class()
    second = hardened_server_class()

    assert first is not second

    first.address_family = socket.AF_INET6

    assert second.address_family is not socket.AF_INET6


def test_stdlib_default_backlog_is_too_small_for_a_burst():
    # The reason this module exists: the stdlib default is far below the
    # burst size a serving node sees.
    assert http.server.ThreadingHTTPServer.request_queue_size < BACKLOG


def test_a_burst_is_served_without_connection_errors():
    server_class = hardened_server_class()
    server = server_class(("127.0.0.1", 0), _EchoHandler)
    port = server.server_address[1]
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    try:
        failures = _burst(port, workers=32, rounds=2)
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert failures == []


def test_the_stdlib_backlog_fails_the_same_burst():
    # The control case for the test above. If the stdlib default ever
    # changes, this test fails and the hardening above needs a new reason.
    class _StdlibServer(http.server.ThreadingHTTPServer):
        request_queue_size = 5

    server = _StdlibServer(("127.0.0.1", 0), _EchoHandler)
    port = server.server_address[1]
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    try:
        failures = _burst(port, workers=32, rounds=2)
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert failures != []


class _EchoHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        return

    def do_GET(self):
        body = b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _burst(port: int, *, workers: int, rounds: int) -> list[str]:
    failures: list[str] = []
    for _ in range(rounds):
        barrier = threading.Barrier(workers)

        def one(_index):
            barrier.wait()
            try:
                request = urllib.request.Request(f"http://127.0.0.1:{port}/probe")
                with urllib.request.urlopen(request, timeout=30) as response:
                    response.read()
            except Exception as exc:
                return f"{type(exc).__name__}: {exc}"
            return None

        threads = [
            threading.Thread(target=lambda index=index: failures.append(one(index)))
            for index in range(workers)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
    return [failure for failure in failures if failure is not None]


def test_runtime_serves_with_the_hardened_server_class(monkeypatch):
    from dflash_mlx.server import runtime as runtime_mod

    captured: dict[str, object] = {}

    def fake_run_http_server(host, port, response_generator, **kwargs):
        captured["host"] = host
        captured["port"] = port
        captured["server_class"] = kwargs.get("server_class")

    monkeypatch.setattr(
        runtime_mod.mlx_server, "_run_http_server", fake_run_http_server
    )
    monkeypatch.setattr(
        runtime_mod.ServerRuntime, "wait_until_ready", lambda self, **kwargs: None
    )
    monkeypatch.setattr(
        runtime_mod.ServerRuntime, "configure_metrics", lambda self: None
    )
    monkeypatch.setattr(
        runtime_mod.ServerRuntime, "print_startup_banner", lambda self: None
    )

    class _Group:
        def rank(self):
            return 0

    monkeypatch.setattr(
        runtime_mod.mx,
        "distributed",
        type("_D", (), {"init": staticmethod(lambda: _Group())}),
    )

    class _CliArgs:
        prompt_cache_size = 2

    class _ModelProvider:
        cli_args = _CliArgs()

    runtime = runtime_mod.ServerRuntime(
        host="127.0.0.1",
        port=0,
        model_provider=_ModelProvider(),
        version="test",
    )

    class _ResponseGenerator:
        def stop_and_join(self):
            return None

    runtime.serve_forever(
        response_generator_factory=lambda *args: _ResponseGenerator(),
        handler_class=object,
    )

    server_class = captured["server_class"]
    assert server_class is not None
    assert server_class.request_queue_size == BACKLOG
