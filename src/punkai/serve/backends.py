"""Where the tokens actually come from.

Three backends, in increasing order of "you need real hardware":

* `EchoBackend` -- deterministic fake. Lets you build and test the whole
  serving stack, guards and audit chain included, before a single weight file
  has downloaded. Use it in tests forever.
* `LlamaCppBackend` -- HTTP to a local `llama-server`. The realistic choice for
  a CPU or a modest GPU, and the path with the fewest dependencies.
* `TransformersBackend` -- torch in-process. Most control, heaviest install.

All of them are the same three-line interface, so labs and tests swap between
them with one flag.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from urllib.parse import urlparse

from punkai.errors import BackendError

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0"})  # noqa: S104


@runtime_checkable
class Backend(Protocol):
    name: str

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str: ...


@dataclass
class EchoBackend:
    """Deterministic stand-in. Same prompt in, same text out, no GPU involved."""

    name: str = "echo"
    max_words: int = 40

    _WORDS = (
        "signal noise drift static copper solder antenna lattice vector gradient "
        "entropy kernel weights shard token cache latency throughput offline"
    ).split()

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16], 16)
        count = max(1, min(self.max_words, max_tokens // 4))
        words = []
        for _ in range(count):
            seed = (seed * 6364136223846793005 + 1442695040888963407) % (2**64)
            words.append(self._WORDS[seed % len(self._WORDS)])
        return f"[echo:{len(prompt)}ch] " + " ".join(words)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects.

    A backend URL should point at your own inference server and stay there. If
    that server starts 302-ing somewhere else, something is wrong and following
    it would turn this client into an SSRF gadget.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        raise BackendError(f"backend tried to redirect to {newurl!r}; refusing")


@dataclass
class LlamaCppBackend:
    """Client for `llama-server` from llama.cpp (its OpenAI-compatible route).

    Start the server yourself:
        llama-server -m ./weights/model.gguf -c 8192 --host 127.0.0.1 --port 8080

    Non-local URLs are refused unless you pass `allow_remote=True`, because the
    default assumption is that your inference server is not on the internet.
    """

    base_url: str = "http://127.0.0.1:8080"
    name: str = "llama.cpp"
    timeout: float = 120.0
    allow_remote: bool = False

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"}:
            raise BackendError(f"backend url must be http(s), got {self.base_url!r}")
        host = parsed.hostname or ""
        if host not in LOCAL_HOSTS and not self.allow_remote:
            raise BackendError(
                f"{self.base_url} is not local. If you really mean to send prompts off this "
                "machine, pass allow_remote=True and use https."
            )
        if host not in LOCAL_HOSTS and parsed.scheme != "https":
            raise BackendError("remote backends must use https")

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        body = json.dumps(
            {
                "prompt": prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
                "stream": False,
            }
        ).encode("utf-8")
        request = urllib.request.Request(  # noqa: S310 -- scheme validated in __post_init__
            f"{self.base_url.rstrip('/')}/completion",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        opener = urllib.request.build_opener(_NoRedirect)
        try:
            with opener.open(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise BackendError(
                f"could not reach llama-server at {self.base_url} ({exc}). Is it running?"
            ) from exc
        except json.JSONDecodeError as exc:
            raise BackendError(f"llama-server returned non-JSON ({exc})") from exc
        content = payload.get("content")
        if not isinstance(content, str):
            raise BackendError(f"unexpected response shape: {list(payload)[:5]}")
        return content


@dataclass
class TransformersBackend:
    """In-process transformers. Requires the `torch` extra.

    The safety-relevant part is the load call, not the generate call -- see
    `punkai.loading.safe_load`, which this delegates to so the same rules
    (pinned revision, safetensors only, no remote code) apply everywhere.
    """

    model_dir: str
    name: str = "transformers"
    device: str = "auto"
    dtype: str = "auto"
    _pipe: object | None = None

    def _ensure(self):
        if self._pipe is None:
            from punkai.loading.safe_load import load_causal_lm  # lazy: heavy import

            self._pipe = load_causal_lm(self.model_dir, device=self.device, dtype=self.dtype)
        return self._pipe

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        model, tokenizer = self._ensure()
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(generated, skip_special_tokens=True)


def load_backend(spec: str) -> Backend:
    """Build a backend from a CLI-friendly string.

    echo                      -> EchoBackend
    llamacpp:http://host:port -> LlamaCppBackend
    transformers:/path/to/dir -> TransformersBackend
    """
    if spec == "echo":
        return EchoBackend()
    kind, _, rest = spec.partition(":")
    if kind == "llamacpp":
        return LlamaCppBackend(base_url=rest or "http://127.0.0.1:8080")
    if kind == "transformers":
        if not rest:
            raise BackendError("transformers backend needs a model directory")
        return TransformersBackend(model_dir=rest)
    raise BackendError(f"unknown backend {spec!r}; try echo, llamacpp:URL, transformers:DIR")
