# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

"""HTTP server settings for a busy local serving node.

The stdlib server class listens with a small backlog. On Python 3.13 the
default is 5 pending connections, so a burst of concurrent clients that
arrives faster than the accept loop drains it gets its excess connections
reset by the kernel. The client sees a broken pipe or a connection reset
while the server still records the request as served.

A larger backlog holds the burst until the accept loop catches up.
"""

from __future__ import annotations

from http.server import ThreadingHTTPServer

# Pending connections the kernel queues before it starts refusing. A burst of
# 32 clients needs room for the whole burst, and headroom for a second
# burst that arrives while the first is still draining.
BACKLOG = 1024


def hardened_server_class() -> type[ThreadingHTTPServer]:
    """Return a threading server class with a burst-sized listen backlog.

    A fresh class is returned on every call because the MLX-LM server
    assigns ``address_family`` on the class it is given, and that value
    must not leak between servers that bind different address families.
    """

    class _HardenedThreadingHTTPServer(ThreadingHTTPServer):
        request_queue_size = BACKLOG
        allow_reuse_address = True
        daemon_threads = True

    return _HardenedThreadingHTTPServer
