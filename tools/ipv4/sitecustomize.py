# Force Python HTTP clients onto IPv4 when imported through PYTHONPATH.

import socket

_original_getaddrinfo = socket.getaddrinfo


def _ipv4_getaddrinfo(host, port, family=socket.AF_UNSPEC, type=0, proto=0, flags=0):
    if family == socket.AF_UNSPEC:
        family = socket.AF_INET
    return _original_getaddrinfo(host, port, family, type, proto, flags)


socket.getaddrinfo = _ipv4_getaddrinfo
