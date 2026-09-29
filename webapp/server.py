"""HTTP server for the Early Guard risk-assessment client.

Standard library only — no Flask, no extra dependencies. Serves the static
client from ``webapp/public`` and a small JSON API over the project's trained
model and audit artefacts.

Local development::

    python webapp/server.py            # http://127.0.0.1:8000
    python webapp/server.py --port 9000

Deployment (Render and similar platforms)::

    # The platform injects PORT; the server binds 0.0.0.0 automatically.
    PORT=10000 python webapp/server.py
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from services import catalog
from services.paths import PUBLIC_DIR
from services.predictor import (
    InputError,
    ModelUnavailableError,
    default_values,
    predict,
)

MAX_BODY_BYTES = 1_000_000


class ApiError(Exception):
    """An error with an HTTP status the client can act on."""

    def __init__(self, status: int, message: str, details: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.details = details


def _overview() -> dict[str, Any]:
    return catalog.overview()


def _signals() -> dict[str, Any]:
    return {
        "features": catalog.feature_catalog(),
        "importance": catalog.importance_rows(),
        "domains": catalog.domain_rows(),
    }


def _performance() -> dict[str, Any]:
    return catalog.performance()


def _method() -> dict[str, Any]:
    return catalog.methodology()


def _defaults() -> dict[str, Any]:
    return {"values": default_values()}


def _predict(payload: dict[str, Any]) -> dict[str, Any]:
    values = payload.get("values")
    if not isinstance(values, dict):
        raise ApiError(400, "Body must be {\"values\": {<feature>: <value>}}")
    return predict(values)


GET_ROUTES: dict[str, Callable[[], dict[str, Any]]] = {
    "/api/overview": _overview,
    "/api/signals": _signals,
    "/api/performance": _performance,
    "/api/method": _method,
    "/api/defaults": _defaults,
    "/api/health": lambda: {"status": "ok"},
}

POST_ROUTES: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "/api/predict": _predict,
}

PAGE_ROUTES: dict[str, str] = {
    "/": "index.html",
    "/app": "index.html",
    "/app/": "index.html",
    "/mobile": "index.html",
    "/mobile/": "index.html",
}


class ClientHandler(SimpleHTTPRequestHandler):
    """Static client + JSON API in one process."""

    server_version = "EarlyGuardRiskClient/1.0"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(PUBLIC_DIR), **kwargs)

    # ---------------------------------------------------------------- helpers
    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ApiError(413, "Request body too large")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ApiError(400, f"Malformed JSON body: {error}") from error
        if not isinstance(payload, dict):
            raise ApiError(400, "JSON body must be an object")
        return payload

    def _handle_api(self, routes: dict[str, Any], payload: dict[str, Any] | None = None) -> None:
        route = urlparse(self.path).path.rstrip("/") or "/"
        handler = routes.get(route)
        if handler is None:
            self._send_json({"error": f"Unknown endpoint: {route}"}, status=404)
            return
        try:
            result = handler() if payload is None else handler(payload)
        except ApiError as error:
            self._send_json(
                {"error": error.message, "details": error.details},
                status=error.status,
            )
        except InputError as error:
            self._send_json(
                {"error": "The submitted profile failed validation.", "details": error.problems},
                status=422,
            )
        except ModelUnavailableError as error:
            self._send_json({"error": str(error)}, status=503)
        except catalog.ArtefactError as error:
            self._send_json({"error": str(error)}, status=503)
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            self._send_json({"error": f"{type(error).__name__}: {error}"}, status=500)
        else:
            self._send_json(result)

    # ------------------------------------------------------------------ verbs
    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        path = urlparse(self.path).path
        if path.startswith("/api/"):
            self._handle_api(GET_ROUTES)
            return

        if path in PAGE_ROUTES:
            self.path = "/" + PAGE_ROUTES[path]
        elif path.endswith("/") and (PUBLIC_DIR / path.strip("/") / "index.html").exists():
            self.path = path + "index.html"
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if not urlparse(self.path).path.startswith("/api/"):
            self._send_json({"error": "POST is only supported on /api/*"}, status=405)
            return
        try:
            payload = self._read_json()
        except ApiError as error:
            self._send_json({"error": error.message}, status=error.status)
            return
        self._handle_api(POST_ROUTES, payload)

    def end_headers(self) -> None:
        # The client is served from disk on every request during development.
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if "/api/" in self.path:
            super().log_message(format, *args)


def build_server(host: str, port: int) -> ThreadingHTTPServer:
    if not PUBLIC_DIR.is_dir():
        raise SystemExit(f"Client directory not found: {PUBLIC_DIR}")
    return ThreadingHTTPServer((host, port), ClientHandler)


def _env_port() -> int | None:
    """Port injected by the hosting platform (Render sets ``PORT``)."""
    raw = os.environ.get("PORT")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _default_host() -> str:
    """Bind address: explicit ``$HOST``, else ``0.0.0.0`` on a platform, else localhost.

    Render (and most PaaS) inject ``PORT`` but not ``HOST``; in that case the
    server must listen on every interface or the health check never connects.
    Plain local runs keep the safe ``127.0.0.1`` default.
    """
    if os.environ.get("HOST"):
        return os.environ["HOST"]
    return "0.0.0.0" if _env_port() is not None else "127.0.0.1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Early Guard risk-assessment client")
    parser.add_argument(
        "--host",
        default=_default_host(),
        help="Bind address (default: $HOST, else 0.0.0.0 when $PORT is set, else 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=_env_port() or 8000,
        help="Bind port (default: $PORT or 8000 for local development).",
    )
    args = parser.parse_args()

    server = build_server(args.host, args.port)
    display_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    origin = f"http://{display_host}:{args.port}"
    print(f"Early Guard  {origin}/            landing")
    print(f"             {origin}/app         desktop app")
    print(f"             {origin}/mobile      mobile screens")
    print(f"  bind:         {args.host}:{args.port}")
    print(f"  model:        {catalog.metadata().get('model_name')} "
          f"({len(catalog.selected_features())} signals)")
    print("  ctrl-c to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
