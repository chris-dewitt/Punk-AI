"""Model manifests: a signed-off description of exactly what you intend to load.

The point of a manifest is that "I downloaded Mistral" is not a reproducible
statement. `mistralai/Mistral-7B-Instruct-v0.3` at a *branch* can change under
you between Tuesday and Thursday; the same repo at commit `a1b2c3...` cannot.
A manifest pins the commit, records the hash of every file you approved, and
records the decisions you made about risky knobs (remote code, pickle weights)
so that a future you -- or a reviewer -- can see them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from punkai.errors import ManifestError

# Formats that are just tensors plus a JSON header. Loading one cannot execute code.
SAFE_WEIGHT_SUFFIXES = frozenset({".safetensors", ".gguf"})

# Formats that are (or may contain) Python pickles. `torch.load` on one of these
# can run arbitrary code at load time -- it is a code-execution primitive, not a
# data format. See docs/THREAT_MODEL.md.
PICKLE_SUFFIXES = frozenset(
    {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".joblib", ".dill", ".npy", ".npz"}
)

_HEX = "0123456789abcdef"


def _is_hex(value: str, length: int) -> bool:
    return len(value) == length and all(c in _HEX for c in value)


def _check_relative_path(path: str) -> str:
    """Reject anything that could write or read outside the model directory.

    Manifest paths come from files other people wrote. A path of `../../.ssh/id_rsa`
    or `/etc/passwd` must never survive to reach the filesystem.
    """
    if not path or path.strip() != path:
        raise ManifestError(f"file path is empty or padded with whitespace: {path!r}")
    if "\x00" in path:
        raise ManifestError(f"file path contains a null byte: {path!r}")
    if "\\" in path:
        raise ManifestError(f"use posix separators in manifests, got {path!r}")
    pure = PurePosixPath(path)
    if pure.is_absolute():
        raise ManifestError(f"file path must be relative to the model dir: {path!r}")
    if any(part == ".." for part in pure.parts):
        raise ManifestError(f"file path escapes the model dir: {path!r}")
    if any(part == "." for part in pure.parts):
        raise ManifestError(f"file path must be normalized (no '.' segments): {path!r}")
    return str(pure)


@dataclass(frozen=True)
class FileEntry:
    """One approved file: where it sits, and what it must hash to."""

    path: str
    sha256: str
    size: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _check_relative_path(self.path))
        digest = self.sha256.lower()
        if not _is_hex(digest, 64):
            raise ManifestError(f"{self.path}: sha256 must be 64 hex chars, got {self.sha256!r}")
        object.__setattr__(self, "sha256", digest)
        if self.size is not None and self.size < 0:
            raise ManifestError(f"{self.path}: negative size")

    @property
    def suffix(self) -> str:
        return PurePosixPath(self.path).suffix.lower()

    @property
    def is_pickle(self) -> bool:
        return self.suffix in PICKLE_SUFFIXES

    @property
    def is_weight(self) -> bool:
        return self.suffix in SAFE_WEIGHT_SUFFIXES or self.is_pickle


@dataclass
class ModelManifest:
    """Everything you need to load a model on purpose rather than by accident."""

    name: str
    source: str  # e.g. "hf:Qwen/Qwen2.5-7B-Instruct" or "https://..." or "local:"
    revision: str  # a full git commit sha -- never a branch or tag
    license: str  # SPDX id where one exists, else the vendor's id ("llama3.1")
    files: list[FileEntry] = field(default_factory=list)

    # Architecture hints, used by `punk size` to predict whether this fits your GPU.
    params_b: float | None = None
    num_layers: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    context_length: int | None = None

    # Risky knobs. Both default to off and both demand a written reason to turn on,
    # because "I'll remember why I did this" is not true six months later.
    allow_remote_code: bool = False
    allow_pickle_weights: bool = False
    review_note: str = ""

    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ManifestError("manifest needs a name")
        if not self.source.strip():
            raise ManifestError(f"{self.name}: manifest needs a source")
        if not self.license.strip():
            raise ManifestError(
                f"{self.name}: manifest needs a license id -- 'unknown' is a valid answer, "
                "but it has to be a deliberate one"
            )
        self.revision = self.revision.strip().lower()
        if not _is_hex(self.revision, 40):
            raise ManifestError(
                f"{self.name}: revision must be a full 40-char commit sha, got {self.revision!r}. "
                "Branches and tags can be moved by whoever owns the repo; a commit cannot."
            )
        seen: set[str] = set()
        for entry in self.files:
            if entry.path in seen:
                raise ManifestError(f"{self.name}: duplicate file entry {entry.path!r}")
            seen.add(entry.path)

        risky = [e.path for e in self.files if e.is_pickle]
        if risky and not self.allow_pickle_weights:
            raise ManifestError(
                f"{self.name}: pickle-format files are listed but allow_pickle_weights is false: "
                f"{', '.join(sorted(risky))}. Prefer .safetensors, or set the flag with a "
                "review_note explaining why this source is trusted."
            )
        if (self.allow_pickle_weights or self.allow_remote_code) and not self.review_note.strip():
            raise ManifestError(
                f"{self.name}: allow_remote_code/allow_pickle_weights require a review_note "
                "saying who checked the code and what they found."
            )

    # -- serialization ----------------------------------------------------

    @property
    def pinned_ref(self) -> str:
        return f"{self.source}@{self.revision}"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["files"] = [asdict(f) for f in self.files]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelManifest:
        if not isinstance(data, dict):
            raise ManifestError("manifest must be a JSON object")
        known = {f for f in cls.__dataclass_fields__ if f != "files"}
        unknown = set(data) - known - {"files"}
        if unknown:
            raise ManifestError(f"unknown manifest fields: {', '.join(sorted(unknown))}")
        raw_files = data.get("files", [])
        if not isinstance(raw_files, list):
            raise ManifestError("manifest 'files' must be a list")
        files = []
        for item in raw_files:
            if not isinstance(item, dict):
                raise ManifestError("each entry in 'files' must be an object")
            extra = set(item) - {"path", "sha256", "size"}
            if extra:
                raise ManifestError(f"unknown file fields: {', '.join(sorted(extra))}")
            files.append(FileEntry(**item))
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(files=files, **kwargs)

    @classmethod
    def load(cls, path: str | Path) -> ModelManifest:
        text = Path(path).read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"{path}: not valid JSON ({exc})") from exc
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
