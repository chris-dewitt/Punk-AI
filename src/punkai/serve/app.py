"""A small, hardened HTTP front end for a local model.

Deliberately built on `http.server` rather than a framework: the whole request
path is a hundred lines you can read in one sitting, and there is no middleware
stack quietly deciding things for you. It is meant for a homelab or a
single-tenant box -- put a real reverse proxy in front of it before it faces
anything hostile.

What it enforces on every request:

  1. Binds 127.0.0.1 unless you explicitly opt out, and refuses to bind a
     public interface with authentication switched off.
  2. Bearer key required, verified in constant time (`serve.auth`).
  3. Per-key token-bucket rate limit (`serve.ratelimit`).
  4. Hard cap on request body size, read with an exact-length read so a lying
     Content-Length cannot make the server block or over-allocate.
  5. Input and output guards (`punkai.guards`).
  6. Hash-chained audit entry for every outcome, including refusals.
  7. Errors return a short JSON message. No tracebacks, no framework banner,
     no server-version header.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from punkai.errors import BackendError
from punkai.guards import GuardPolicy
from punkai.serve.audit import AuditLog
from punkai.serve.auth import KeyStore
from punkai.serve.backends import Backend, EchoBackend
from punkai.serve.ratelimit import RateLimiter

MAX_BODY_BYTES = 256 * 1024
# Binds that mean "every interface". Listed so we can *refuse* them, not use them.
PUBLIC_BINDS = frozenset({"0.0.0.0", "::", ""})  # noqa: S104


@dataclass
class ServerConfig:
    backend: Backend = field(default_factory=EchoBackend)
    policy: GuardPolicy = field(default_factory=GuardPolicy)
    keystore: KeyStore = field(default_factory=KeyStore)
    audit: AuditLog = field(default_factory=lambda: AuditLog(None))
    model_name: str = "local"
    host: str = "127.0.0.1"
    port: int = 8099
    require_auth: bool = True
    max_body_bytes: int = MAX_BODY_BYTES
    max_tokens_cap: int = 2048

    def validate(self) -> None:
        if self.host in PUBLIC_BINDS and not self.require_auth:
            raise ValueError(
                f"refusing to bind {self.host} with require_auth=False. That is an open "
                "inference endpoint on every interface; someone will find it."
            )
        if self.require_auth and not self.keystore.keys:
            raise ValueError(
                "auth is required but no keys are issued. Run `punk keys issue --label laptop`."
            )


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "punk-ai"  # no version number: free recon for nobody
    sys_version = ""

    config: ServerConfig
    limiter: RateLimiter

    # -- plumbing ---------------------------------------------------------

    def version_string(self) -> str:
        """No Python version, no build info. Free reconnaissance for nobody."""
        return self.server_version

    def log_message(self, fmt: str, *args) -> None:
        """Access logs go to the audit chain, not to stderr with query strings in them."""
        return

    def _send_json(self, status: HTTPStatus, payload: dict, extra_headers: dict | None = None):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str, extra_headers: dict | None = None):
        self._send_json(status, {"error": message}, extra_headers)

    def _read_body(self) -> bytes | None:
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            self._error(HTTPStatus.LENGTH_REQUIRED, "chunked bodies are not supported")
            return None
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._error(HTTPStatus.LENGTH_REQUIRED, "Content-Length required")
            return None
        try:
            length = int(raw_length)
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "malformed Content-Length")
            return None
        if length < 0 or length > self.config.max_body_bytes:
            self._error(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                f"body must be 0..{self.config.max_body_bytes} bytes",
            )
            return None
        # Exact read: never trust the declared length beyond the cap we just applied.
        return self.rfile.read(length)

    def _authenticate(self):
        if not self.config.require_auth:
            return "anonymous", None
        header = self.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            # Worth recording: a burst of these is what finding your open port
            # looks like. It does mean an unauthenticated caller can grow the
            # log, so rotate it (see labs/05).
            self.config.audit.append("auth.missing", actor="-", path=self.path)
            self._error(
                HTTPStatus.UNAUTHORIZED,
                "missing bearer token",
                {"WWW-Authenticate": 'Bearer realm="punk-ai"'},
            )
            return None, None
        record = self.config.keystore.verify(token.strip())
        if record is None:
            self.config.audit.append("auth.reject", actor="-", reason="invalid key")
            self._error(HTTPStatus.UNAUTHORIZED, "invalid or revoked key")
            return None, None
        if not record.has_scope("generate"):
            self._error(HTTPStatus.FORBIDDEN, "key lacks the 'generate' scope")
            return None, None
        return record.key_id, record

    # -- routes -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 -- stdlib naming
        if self.path == "/healthz":
            # Unauthenticated on purpose, and says nothing an attacker can use.
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        self._error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/generate":
            self._error(HTTPStatus.NOT_FOUND, "not found")
            return

        actor, record = self._authenticate()
        if actor is None:
            return

        rate = record.rate_per_minute if record else 60
        decision = self.limiter.check(actor, rate)
        if not decision.allowed:
            self.config.audit.append(
                "ratelimit.reject", actor=actor, retry_after=decision.retry_after
            )
            self._error(
                HTTPStatus.TOO_MANY_REQUESTS,
                "rate limit exceeded",
                {"Retry-After": str(max(1, int(decision.retry_after + 0.5)))},
            )
            return

        body = self._read_body()
        if body is None:
            return
        try:
            payload = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._error(HTTPStatus.BAD_REQUEST, "body must be UTF-8 JSON")
            return
        if not isinstance(payload, dict):
            self._error(HTTPStatus.BAD_REQUEST, "body must be a JSON object")
            return

        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            self._error(HTTPStatus.BAD_REQUEST, "'prompt' must be a non-empty string")
            return
        try:
            max_tokens = int(payload.get("max_tokens", 256))
            temperature = float(payload.get("temperature", 0.7))
        except (TypeError, ValueError):
            self._error(HTTPStatus.BAD_REQUEST, "max_tokens and temperature must be numbers")
            return
        max_tokens = max(1, min(max_tokens, self.config.max_tokens_cap))
        temperature = min(max(temperature, 0.0), 2.0)

        started = time.perf_counter()
        guard_in = self.config.policy.check_input(prompt)
        if guard_in.blocked:
            self.config.audit.log_generation(
                actor,
                prompt,
                None,
                model=self.config.model_name,
                latency_ms=(time.perf_counter() - started) * 1000,
                decision="blocked",
                reasons=guard_in.reasons,
                injection_score=guard_in.injection_score,
            )
            self._error(
                HTTPStatus.BAD_REQUEST,
                "refused by input policy: " + "; ".join(guard_in.reasons),
            )
            return

        try:
            completion = self.config.backend.generate(guard_in.text, max_tokens, temperature)
        except BackendError as exc:
            self.config.audit.log_generation(
                actor, prompt, None, model=self.config.model_name,
                latency_ms=(time.perf_counter() - started) * 1000,
                decision="backend-error", reasons=[str(exc)],
            )
            self._error(HTTPStatus.BAD_GATEWAY, f"backend error: {exc}")
            return
        except Exception:  # noqa: BLE001 -- never leak an internal traceback to a caller
            self.config.audit.log_generation(
                actor, prompt, None, model=self.config.model_name,
                latency_ms=(time.perf_counter() - started) * 1000,
                decision="error", reasons=["unhandled backend exception"],
            )
            print("backend raised:", sys.exc_info()[1], file=sys.stderr)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal error")
            return

        guard_out = self.config.policy.check_output(completion)
        latency_ms = (time.perf_counter() - started) * 1000
        self.config.audit.log_generation(
            actor,
            prompt,
            guard_out.text,
            model=self.config.model_name,
            latency_ms=latency_ms,
            decision="allow",
            reasons=guard_in.reasons + guard_out.reasons,
            injection_score=guard_in.injection_score,
        )
        self._send_json(
            HTTPStatus.OK,
            {
                "completion": guard_out.text,
                "model": self.config.model_name,
                "latency_ms": round(latency_ms, 2),
                "guard": {
                    "injection_score": guard_in.injection_score,
                    "flags": guard_in.reasons + guard_out.reasons,
                },
            },
        )


def build_server(config: ServerConfig) -> ThreadingHTTPServer:
    config.validate()

    class BoundHandler(_Handler):
        pass

    BoundHandler.config = config
    BoundHandler.limiter = RateLimiter()

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True
        address_family = socket.AF_INET

    server = Server((config.host, config.port), BoundHandler)
    server.timeout = 30
    return server


def serve_forever(config: ServerConfig) -> None:
    server = build_server(config)
    host, port = server.server_address[:2]
    print(f"punk-ai serving {config.model_name} on http://{host}:{port}", file=sys.stderr)
    print(f"  auth: {'on' if config.require_auth else 'OFF'}  "
          f"keys: {len(config.keystore.keys)}  "
          f"audit: {config.audit.path or 'memory only'}", file=sys.stderr)
    if config.host in PUBLIC_BINDS:
        print("  WARNING: bound to a public interface. Put TLS and a proxy in front of this.",
              file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", file=sys.stderr)
    finally:
        server.shutdown()
        server.server_close()


def serve_in_thread(config: ServerConfig) -> tuple[ThreadingHTTPServer, threading.Thread]:
    """Start a server on a background thread. Used by the tests.

    The short poll interval keeps `shutdown()` from costing half a second per
    test -- the stdlib default makes a suite of these crawl.
    """
    server = build_server(config)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.02), daemon=True)
    thread.start()
    return server, thread
