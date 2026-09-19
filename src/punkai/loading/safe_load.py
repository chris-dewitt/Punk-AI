"""The load call is the security boundary. Everything else is commentary.

Two flags decide whether "downloading a model" is closer to "downloading a JPEG"
or "downloading an .exe from a forum":

* `trust_remote_code=True` -- imports and runs Python from the model repo, in
  your process, with your permissions, before any inference happens. Some
  genuinely good models require it. Read the file first; it is usually 200 lines.
* `.bin` / `.pt` weights -- pickle. `torch.load` on an untrusted pickle is
  arbitrary code execution, full stop. Since torch 2.6 the default flipped to
  `weights_only=True`, which helps enormously, but the safe move is still to
  prefer `.safetensors` and never hand-wave past it.

This module makes both of those an explicit, recorded decision in a manifest
rather than a keyword argument someone copy-pasted off a model card.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from punkai.errors import SmithError
from punkai.registry.manifest import PICKLE_SUFFIXES, ModelManifest
from punkai.registry.verify import verify_directory


class SafeLoadError(SmithError):
    """A load was attempted with an unsafe configuration."""


def assert_no_remote_code(model_dir: str | Path) -> None:
    """Fail if the model directory ships Python. Call it before you ever consider
    `trust_remote_code=True`, and read whatever it names."""
    scripts = sorted(p.name for p in Path(model_dir).rglob("*.py"))
    if scripts:
        raise SafeLoadError(
            f"{model_dir} contains executable Python: {', '.join(scripts)}. "
            "transformers will run these if trust_remote_code=True. Read them, then decide."
        )


def safe_load_kwargs(
    manifest: ModelManifest | None = None,
    *,
    revision: str | None = None,
    dtype: str = "auto",
    device: str = "auto",
) -> dict[str, Any]:
    """Build `from_pretrained` kwargs that cannot silently do the dangerous thing.

    `trust_remote_code` is only ever True when a manifest says so *and* carries a
    review note, and the revision is always pinned to a commit.
    """
    pinned = revision or (manifest.revision if manifest else None)
    if pinned is None:
        raise SafeLoadError(
            "no revision pinned. Loading from a moving branch means you cannot reproduce "
            "today's behaviour tomorrow -- pass revision= or use a manifest."
        )
    trust = bool(manifest and manifest.allow_remote_code)
    if trust and not (manifest and manifest.review_note.strip()):
        raise SafeLoadError("trust_remote_code requires a manifest review_note")

    kwargs: dict[str, Any] = {
        "revision": pinned,
        "trust_remote_code": trust,
        "use_safetensors": not (manifest and manifest.allow_pickle_weights),
        "device_map": device,
    }
    if dtype != "auto":
        kwargs["torch_dtype"] = dtype
    return kwargs


def load_causal_lm(
    model_dir: str | Path,
    manifest: ModelManifest | None = None,
    *,
    device: str = "auto",
    dtype: str = "auto",
    verify: bool = True,
):
    """Verify, then load. Requires the `torch` extra.

    Returns `(model, tokenizer)`.
    """
    model_dir = Path(model_dir)
    if manifest and verify:
        report = verify_directory(manifest, model_dir)
        report.raise_for_status()
    if not (manifest and manifest.allow_remote_code):
        assert_no_remote_code(model_dir)
    if not (manifest and manifest.allow_pickle_weights):
        pickles = [p.name for p in model_dir.rglob("*") if p.suffix.lower() in PICKLE_SUFFIXES]
        if pickles and not any(model_dir.rglob("*.safetensors")):
            raise SafeLoadError(
                f"{model_dir} has only pickle-format weights ({', '.join(pickles[:3])}). "
                "Convert them to safetensors, or set allow_pickle_weights in a manifest with "
                "a note saying why you trust this source."
            )

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SafeLoadError(
            "transformers is not installed. `pip install 'punk-ai[torch]'`, or use the "
            "llama.cpp backend, which needs no Python ML stack at all."
        ) from exc

    kwargs = safe_load_kwargs(manifest, revision=manifest.revision if manifest else "local",
                              dtype=dtype, device=device)
    # A local directory has no revision to pin; the manifest hashes cover it instead.
    if model_dir.is_dir():
        kwargs.pop("revision", None)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), trust_remote_code=kwargs["trust_remote_code"]
    )
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)
    return model, tokenizer


def download_model(
    manifest: ModelManifest,
    dest: str | Path,
    *,
    verify: bool = True,
    include_pickles: bool = False,
):
    """Fetch a pinned snapshot from the Hub and verify it. Requires the `hub` extra.

    Pickle-format files are excluded from the download by default -- not just
    ignored at load time. Bytes you never fetched cannot be loaded by accident
    six months later.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SafeLoadError(
            "huggingface_hub is not installed. `pip install 'punk-ai[hub]'`"
        ) from exc

    repo_id = manifest.source.removeprefix("hf:")
    ignore = None if include_pickles else [f"*{suffix}" for suffix in sorted(PICKLE_SUFFIXES)]
    path = snapshot_download(
        repo_id=repo_id,
        revision=manifest.revision,  # a commit sha, enforced by the manifest
        local_dir=str(dest),
        ignore_patterns=ignore,
    )
    if verify:
        report = verify_directory(manifest, path)
        report.raise_for_status()
    return Path(path)
