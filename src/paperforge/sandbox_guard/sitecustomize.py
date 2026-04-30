"""Loaded automatically by the subprocess sandbox (it is first on PYTHONPATH).

Disables outbound networking for the experiment process. This is a guard against
accidents (a generated script downloading a dataset mid-run), not a security
boundary: native code or a subprocess can bypass it. Use the docker sandbox
(--network none) for untrusted code.
"""

import socket as _socket


def _blocked(*args, **kwargs):
    raise PermissionError("network access is disabled inside the paper-forge sandbox")


class _NoNetSocket(_socket.socket):
    def connect(self, *a, **k):
        if self.family == getattr(_socket, "AF_UNIX", None):
            return super().connect(*a, **k)
        _blocked()

    connect_ex = connect

    def sendto(self, *a, **k):
        _blocked()


_socket.socket = _NoNetSocket
_socket.create_connection = _blocked
_socket.getaddrinfo = _blocked
