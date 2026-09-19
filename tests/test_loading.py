import pytest

from punkai.loading.safe_load import SafeLoadError, assert_no_remote_code, safe_load_kwargs
from punkai.registry.manifest import FileEntry, ModelManifest

REV = "e" * 40


def base_manifest(**kw):
    defaults = dict(name="m", source="hf:org/m", revision=REV, license="apache-2.0")
    defaults.update(kw)
    return ModelManifest(**defaults)


def test_defaults_are_the_safe_ones():
    kwargs = safe_load_kwargs(base_manifest())
    assert kwargs["trust_remote_code"] is False
    assert kwargs["use_safetensors"] is True
    assert kwargs["revision"] == REV


def test_unpinned_load_is_refused():
    with pytest.raises(SafeLoadError, match="no revision pinned"):
        safe_load_kwargs(None)


def test_explicit_revision_without_manifest_is_allowed():
    assert safe_load_kwargs(None, revision=REV)["revision"] == REV


def test_remote_code_only_with_a_reviewed_manifest():
    reviewed = base_manifest(allow_remote_code=True, review_note="read modeling_x.py, it is fine")
    assert safe_load_kwargs(reviewed)["trust_remote_code"] is True


def test_pickle_opt_in_relaxes_safetensors_requirement():
    m = base_manifest(
        files=[FileEntry("pytorch_model.bin", "f" * 64)],
        allow_pickle_weights=True,
        review_note="converted in-house",
    )
    assert safe_load_kwargs(m)["use_safetensors"] is False


def test_python_in_a_model_dir_is_reported(tmp_path):
    (tmp_path / "modeling_custom.py").write_text("import os", encoding="utf-8")
    with pytest.raises(SafeLoadError, match="modeling_custom.py"):
        assert_no_remote_code(tmp_path)


def test_clean_model_dir_passes(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert_no_remote_code(tmp_path)
