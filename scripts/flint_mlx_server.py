#!/usr/bin/env python3
"""Run mlx_lm.server with a burst-sized HTTP listen backlog."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from http.server import ThreadingHTTPServer


def main(
    argv: list[str] | None = None,
    server_main: Callable[[], None] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="All other arguments are passed to mlx_lm.server.",
    )
    parser.add_argument("--listen-backlog", type=int, default=1024)
    args, remaining = parser.parse_known_args(argv)
    if args.listen_backlog < 1:
        parser.error("--listen-backlog must be positive")

    if server_main is None:
        from mlx_lm import server

        server_main = server.main

    sys.argv = [sys.argv[0] if argv is None else "mlx_lm.server", *remaining]
    previous_backlog = ThreadingHTTPServer.request_queue_size
    ThreadingHTTPServer.request_queue_size = args.listen_backlog
    try:
        server_main()
    finally:
        ThreadingHTTPServer.request_queue_size = previous_backlog
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
