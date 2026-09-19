"""Are the bytes on disk the bytes you approved?

Two separate jobs live here:

1. **Integrity** -- hash every file and compare against the manifest. This
   catches a truncated download, a cache that got stomped, and a repo whose
   owner quietly force-pushed new weights under the same name.
2. **Shape of the payload** -- look at *what kind* of files arrived. A model
   directory that contains `.bin` weights or a `modeling_custom.py` is asking
   you to execute someone else's Python. That is a decision, not a detail.

Everything here is stdlib. You can run it on a machine with no torch, no
transformers, and no network.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

from punkai.errors import VerificationError
from punkai.registry.manifest import PICKLE_SUFFIXES, ModelManifest

CHUNK = 1024 * 1024
# A safetensors JSON header is metadata about tensors. Hundreds of KB is normal for
# a big model; 100 MB means someone is trying to make you allocate 100 MB from a
# 8-byte length field.
MAX_HEADER_BYTES = 100 * 1024 * 1024

_SAFETENSORS_DTYPES = frozenset(
    {"BOOL", "U8", "I8", "F8_E5M2", "F8_E4M3", "I16", "U16", "F16", "BF16",
     "I32", "U32", "F32", "F64", "I64", "U64"}
)


@dataclass(frozen=True)
class Finding:
    level: str  # "error" | "warn" | "info"
    code: str
    path: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.code}: {self.path} -- {self.message}"


@dataclass
class VerificationReport:
    root: str
    manifest_name: str
    checked: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "warn"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, level: str, code: str, path: str, message: str) -> None:
        self.findings.append(Finding(level, code, path, message))

    def raise_for_status(self) -> None:
        if not self.ok:
            detail = "\n".join(f"  {f}" for f in self.errors)
            raise VerificationError(
                f"{self.manifest_name}: {len(self.errors)} verification error(s)\n{detail}"
            )

    def summary(self) -> str:
        verdict = "PASS" if self.ok else "FAIL"
        head = (
            f"[{verdict}] {self.manifest_name}: {len(self.checked)} file(s) hashed, "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        )
        return "\n".join([head, *(f"  {f}" for f in self.findings)])


def sha256_file(path: str | Path, chunk: int = CHUNK) -> str:
    """Stream a file through sha256. Never loads a 40 GB shard into RAM."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def resolve_within(root: Path, relative: str) -> Path:
    """Join `relative` onto `root`, refusing anything that lands outside it.

    Symlinks are followed for *reading* -- the Hugging Face cache legitimately
    symlinks snapshot files into a blobs directory -- but the link target is
    reported, so a link pointing at /etc/shadow is visible rather than silent.
    """
    root = root.resolve()
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise VerificationError(f"path escapes the model directory: {relative!r}")
    return candidate


@dataclass(frozen=True)
class SafetensorsHeader:
    tensors: dict[str, dict]
    metadata: dict
    header_bytes: int
    data_bytes: int


def inspect_safetensors(path: str | Path) -> SafetensorsHeader:
    """Parse and sanity-check a .safetensors header without any dependencies.

    The format is: 8-byte little-endian header length, that many bytes of JSON,
    then raw tensor data. Nothing in it can execute -- but a malformed header can
    still make a careless reader allocate wildly or read out of bounds, so every
    field is checked against the real file size before it is trusted.
    """
    path = Path(path)
    size = path.stat().st_size
    if size < 8:
        raise VerificationError(f"{path.name}: too short to be safetensors")
    with open(path, "rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        if header_len > MAX_HEADER_BYTES:
            raise VerificationError(
                f"{path.name}: header claims {header_len} bytes, refusing to read it"
            )
        if 8 + header_len > size:
            raise VerificationError(
                f"{path.name}: header length {header_len} runs past end of file ({size} bytes)"
            )
        raw = handle.read(header_len)
    try:
        header = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise VerificationError(f"{path.name}: header is not valid JSON ({exc})") from exc
    if not isinstance(header, dict):
        raise VerificationError(f"{path.name}: header must be a JSON object")

    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict):
        raise VerificationError(f"{path.name}: __metadata__ must be an object")

    data_bytes = size - 8 - header_len
    for name, spec in header.items():
        if not isinstance(spec, dict):
            raise VerificationError(f"{path.name}: tensor {name!r} is not an object")
        dtype = spec.get("dtype")
        if dtype not in _SAFETENSORS_DTYPES:
            raise VerificationError(f"{path.name}: tensor {name!r} has unknown dtype {dtype!r}")
        offsets = spec.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(o, int) for o in offsets)
        ):
            raise VerificationError(f"{path.name}: tensor {name!r} has malformed data_offsets")
        start, end = offsets
        if not 0 <= start <= end <= data_bytes:
            raise VerificationError(
                f"{path.name}: tensor {name!r} offsets [{start}, {end}] fall outside the "
                f"{data_bytes}-byte data region"
            )
    return SafetensorsHeader(
        tensors=header, metadata=metadata, header_bytes=header_len, data_bytes=data_bytes
    )


def scan_directory(root: str | Path) -> list[Finding]:
    """Flag file types that turn 'loading a model' into 'running a program'."""
    root = Path(root).resolve()
    findings: list[Finding] = []
    if not root.is_dir():
        return [Finding("error", "missing-dir", str(root), "model directory does not exist")]

    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_symlink():
            resolved = (path.parent / path.readlink()).resolve()
            if root not in resolved.parents and resolved != root:
                findings.append(
                    Finding(
                        "warn",
                        "symlink-outside-root",
                        rel,
                        f"symlink points outside the model dir: {resolved}. Normal for the "
                        "Hugging Face cache, suspicious anywhere else.",
                    )
                )
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in PICKLE_SUFFIXES:
            findings.append(
                Finding(
                    "warn",
                    "pickle-format",
                    rel,
                    "pickle-backed format. torch.load on this can execute arbitrary code at "
                    "load time. Prefer the .safetensors sibling if one exists.",
                )
            )
        elif suffix == ".py":
            findings.append(
                Finding(
                    "warn",
                    "remote-code",
                    rel,
                    "Python inside the model repo. transformers will execute this only with "
                    "trust_remote_code=True -- read it before you ever set that flag.",
                )
            )
        elif suffix in {".sh", ".bat", ".ps1", ".so", ".dll", ".dylib", ".exe"}:
            findings.append(
                Finding("warn", "executable", rel, "executable or native library in a weights dir")
            )
        elif suffix == ".safetensors":
            try:
                inspect_safetensors(path)
            except VerificationError as exc:
                findings.append(Finding("error", "malformed-safetensors", rel, str(exc)))
    return findings


def verify_directory(
    manifest: ModelManifest,
    root: str | Path,
    *,
    allow_extra: bool = True,
    scan: bool = True,
) -> VerificationReport:
    """Hash every file in the manifest against what is on disk.

    `allow_extra=False` turns "there are files here the manifest never mentioned"
    into an error, which is what you want for a locked-down deployment and
    usually too strict while you are still tinkering.
    """
    root_path = Path(root)
    report = VerificationReport(root=str(root_path), manifest_name=manifest.name)

    if not root_path.is_dir():
        report.add("error", "missing-dir", str(root_path), "model directory does not exist")
        return report

    expected: set[str] = set()
    for entry in manifest.files:
        expected.add(entry.path)
        try:
            target = resolve_within(root_path, entry.path)
        except VerificationError as exc:
            report.add("error", "path-escape", entry.path, str(exc))
            continue
        if not target.is_file():
            report.add("error", "missing-file", entry.path, "listed in manifest, not on disk")
            continue

        actual_size = target.stat().st_size
        if entry.size is not None and actual_size != entry.size:
            report.add(
                "error",
                "size-mismatch",
                entry.path,
                f"expected {entry.size} bytes, found {actual_size}",
            )
        actual = sha256_file(target)
        report.checked.append(entry.path)
        if actual != entry.sha256:
            report.add(
                "error",
                "hash-mismatch",
                entry.path,
                f"expected sha256 {entry.sha256}, got {actual}. Do not load this file.",
            )

    if not allow_extra:
        on_disk = {
            p.relative_to(root_path.resolve()).as_posix()
            for p in root_path.resolve().rglob("*")
            if p.is_file()
        }
        for extra in sorted(on_disk - expected):
            report.add("error", "unexpected-file", extra, "present on disk, absent from manifest")

    if scan:
        for finding in scan_directory(root_path):
            report.findings.append(finding)

    if manifest.allow_remote_code:
        report.add(
            "warn",
            "remote-code-enabled",
            manifest.name,
            f"manifest allows remote code execution. Reviewer note: {manifest.review_note}",
        )
    return report
