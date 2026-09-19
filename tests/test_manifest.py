import json

import pytest

from punkai.errors import ManifestError
from punkai.registry.manifest import FileEntry, ModelManifest

SHA = "a" * 64
REV = "b" * 40


def manifest(**overrides):
    base = dict(
        name="test-model",
        source="hf:org/model",
        revision=REV,
        license="apache-2.0",
        files=[FileEntry("model.safetensors", SHA, 100)],
    )
    base.update(overrides)
    return ModelManifest(**base)


def test_valid_manifest_round_trips(tmp_path):
    original = manifest()
    path = tmp_path / "m.json"
    original.save(path)
    loaded = ModelManifest.load(path)
    assert loaded.to_dict() == original.to_dict()
    assert loaded.pinned_ref == f"hf:org/model@{REV}"


@pytest.mark.parametrize("revision", ["main", "v1.0", "abc123", "B" * 40 + "c", ""])
def test_revision_must_be_a_commit_sha(revision):
    """A branch or tag can be repointed by whoever owns the repo."""
    with pytest.raises(ManifestError, match="40-char commit sha"):
        manifest(revision=revision)


def test_revision_is_case_normalized():
    assert manifest(revision="B" * 40).revision == "b" * 40


@pytest.mark.parametrize(
    "path",
    [
        "../../../etc/passwd",
        "/etc/passwd",
        "subdir/../../escape.bin",
        "windows\\style.safetensors",
        " padded.safetensors",
    ],
)
def test_path_traversal_is_rejected(path):
    with pytest.raises(ManifestError):
        FileEntry(path, SHA)


def test_leading_dot_slash_is_normalized_not_rejected():
    """"./model.safetensors" and "model.safetensors" are the same file, and the
    manifest stores one spelling -- so duplicate detection cannot be fooled by it."""
    assert FileEntry("./model.safetensors", SHA).path == "model.safetensors"


def test_null_byte_in_path_rejected():
    with pytest.raises(ManifestError, match="null byte"):
        FileEntry("model\x00.safetensors", SHA)


@pytest.mark.parametrize("digest", ["", "abc", "z" * 64, "A" * 63])
def test_bad_digests_rejected(digest):
    with pytest.raises(ManifestError, match="sha256"):
        FileEntry("model.safetensors", digest)


def test_uppercase_digest_normalized():
    assert FileEntry("m.safetensors", "A" * 64).sha256 == "a" * 64


def test_pickle_weights_need_an_explicit_opt_in():
    with pytest.raises(ManifestError, match="allow_pickle_weights"):
        manifest(files=[FileEntry("pytorch_model.bin", SHA)])


def test_pickle_opt_in_requires_a_review_note():
    with pytest.raises(ManifestError, match="review_note"):
        manifest(files=[FileEntry("pytorch_model.bin", SHA)], allow_pickle_weights=True)


def test_pickle_allowed_with_note():
    m = manifest(
        files=[FileEntry("pytorch_model.bin", SHA)],
        allow_pickle_weights=True,
        review_note="converted internally, hash matches our own conversion",
    )
    assert m.files[0].is_pickle


def test_remote_code_requires_a_review_note():
    with pytest.raises(ManifestError, match="review_note"):
        manifest(allow_remote_code=True)


def test_duplicate_file_entries_rejected():
    entry = FileEntry("model.safetensors", SHA)
    with pytest.raises(ManifestError, match="duplicate"):
        manifest(files=[entry, entry])


def test_missing_license_rejected():
    with pytest.raises(ManifestError, match="license"):
        manifest(license="")


def test_unknown_fields_rejected():
    data = manifest().to_dict()
    data["backdoor"] = True
    with pytest.raises(ManifestError, match="unknown manifest fields"):
        ModelManifest.from_dict(data)


def test_unknown_file_fields_rejected():
    data = manifest().to_dict()
    data["files"][0]["exec"] = "rm -rf /"
    with pytest.raises(ManifestError, match="unknown file fields"):
        ModelManifest.from_dict(data)


def test_bad_json_gives_a_clean_error(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ManifestError, match="not valid JSON"):
        ModelManifest.load(path)


def test_non_object_manifest_rejected(tmp_path):
    path = tmp_path / "list.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ManifestError):
        ModelManifest.load(path)
