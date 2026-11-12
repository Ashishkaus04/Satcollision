"""
The small JSON-over-HTTP layer the separate processes talk through.

Standard library only (http.server + urllib), so the federation process
loads nothing it does not need. Every node binds to 127.0.0.1 on a port
the OS picks, prints ``LISTENING <port> <pid>`` as its first line of output so
the process that started it knows where to find it, and refuses any
request without the shared internal token (``X-Node-Token``), so another
program on the same computer cannot talk to it. A node also stops by
itself when the process that started it goes away (its stdin closes), so
no orphaned twins are left running.

Errors cross the wire as HTTP status codes: 403 for a refusal (bad
signature, not your satellite), 404 for "no such thing", 400 for a
malformed request, 500 for anything else.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

TOKEN_ENV = "SATCOLLISION_NODE_TOKEN"
TOKEN_HEADER = "X-Node-Token"


class Forbidden(Exception):
    pass


class NotFound(Exception):
    pass


class NodeError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message


Route = Callable[[dict, Dict[str, str]], dict]


def serve(routes: Dict[Tuple[str, str], Route], token: str, refused=(Forbidden,),
          host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Start a threaded JSON server for ``routes`` ((method, path) -> fn(body, query))."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # quiet
            pass

        def _reply(self, status: int, data: dict) -> None:
            raw = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _handle(self, method: str) -> None:
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            if self.headers.get(TOKEN_HEADER) != token:
                self._reply(401, {"error": "missing or wrong node token"})
                return
            fn = routes.get((method, parsed.path))
            if fn is None:
                self._reply(404, {"error": f"no route {method} {parsed.path}"})
                return
            try:
                body = json.loads(raw) if raw else {}
                query = {k: v[-1] for k, v in parse_qs(parsed.query).items()}
                self._reply(200, fn(body, query))
            except refused as exc:
                self._reply(403, {"error": str(exc)})
            except (NotFound, KeyError) as exc:
                self._reply(404, {"error": str(exc)})
            except (ValueError, TypeError) as exc:
                self._reply(400, {"error": f"{type(exc).__name__}: {exc}"})
            except Exception as exc:  # noqa: BLE001 - report, keep serving
                self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def run_forever(server: ThreadingHTTPServer, on_exit: Optional[Callable[[], None]] = None) -> None:
    """Announce the port, then serve until stdin closes (the parent went away)."""

    def watch_parent():
        try:
            while sys.stdin.read(1):
                pass
        except (OSError, ValueError):
            pass
        server.shutdown()

    threading.Thread(target=watch_parent, daemon=True).start()
    print(f"LISTENING {server.server_address[1]} {os.getpid()}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        if on_exit:
            on_exit()
        server.server_close()


def call(base_url: str, method: str, path: str, body: Optional[dict] = None,
         token: Optional[str] = None, timeout: float = 60.0) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base_url.rstrip("/") + path, data=data, method=method)
    req.add_header(TOKEN_HEADER, token if token is not None else os.environ.get(TOKEN_ENV, ""))
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            message = json.loads(exc.read()).get("error", exc.reason)
        except ValueError:
            message = str(exc.reason)
        raise NodeError(exc.code, message) from None
    except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
        raise NodeError(503, f"cannot reach {base_url}: {exc}") from None
