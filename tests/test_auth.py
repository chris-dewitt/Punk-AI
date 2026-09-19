import json
import os
import stat

import pytest

from punkai.serve.auth import KeyStore, parse_key


def test_issued_key_verifies():
    store = KeyStore()
    key, record = store.issue("laptop")
    assert store.verify(key).key_id == record.key_id


def test_plaintext_key_is_never_stored(tmp_path):
    path = tmp_path / "keys.json"
    store = KeyStore(path)
    key, _ = store.issue("laptop")
    secret = key.split("_")[2]
    contents = path.read_text(encoding="utf-8")
    assert secret not in contents
    assert key not in contents


def test_store_file_is_owner_only(tmp_path):
    path = tmp_path / "keys.json"
    KeyStore(path).issue("laptop")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_world_readable_store_is_refused(tmp_path):
    path = tmp_path / "keys.json"
    store = KeyStore(path)
    store.issue("laptop")
    os.chmod(path, 0o644)
    with pytest.raises(PermissionError, match="readable by group or others"):
        KeyStore(path)


@pytest.mark.parametrize(
    "bad",
    ["", "garbage", "pk_", "pk_abc", "bearer pk_a_b", "pk_a_b_c", "pk__secret", "pk_id_"],
)
def test_malformed_keys_are_rejected_without_raising(bad):
    store = KeyStore()
    store.issue("laptop")
    assert store.verify(bad) is None


def test_tampered_secret_rejected():
    store = KeyStore()
    key, _ = store.issue("laptop")
    assert store.verify(key[:-1] + ("X" if key[-1] != "X" else "Y")) is None


def test_valid_secret_with_wrong_key_id_rejected():
    store = KeyStore()
    key, _ = store.issue("laptop")
    _, key_id, secret = key.split("_")
    other, _ = store.issue("phone")
    other_id = other.split("_")[1]
    assert store.verify(f"pk_{other_id}_{secret}") is None


def test_revocation_takes_effect_immediately():
    store = KeyStore()
    key, record = store.issue("laptop")
    assert store.verify(key)
    assert store.revoke(record.key_id)
    assert store.verify(key) is None
    assert not store.revoke(record.key_id)  # already revoked


def test_revoked_key_persists_as_revoked(tmp_path):
    path = tmp_path / "keys.json"
    store = KeyStore(path)
    key, record = store.issue("laptop")
    store.revoke(record.key_id)
    assert KeyStore(path).verify(key) is None


def test_keys_are_unique_and_high_entropy():
    store = KeyStore()
    keys = {store.issue(f"k{i}")[0] for i in range(50)}
    assert len(keys) == 50
    assert all(len(k.split("_")[2]) >= 32 for k in keys)


def test_scopes_are_enforced():
    store = KeyStore()
    key, _ = store.issue("readonly", scopes=["healthz"])
    record = store.verify(key)
    assert record is not None
    assert not record.has_scope("generate")
    assert record.has_scope("healthz")


def test_wildcard_scope():
    store = KeyStore()
    key, _ = store.issue("admin", scopes=["*"])
    assert store.verify(key).has_scope("anything")


def test_parse_key_shape():
    assert parse_key("pk_abc_def") == ("abc", "def")
    assert parse_key("xx_abc_def") is None


def test_store_survives_a_round_trip(tmp_path):
    path = tmp_path / "keys.json"
    store = KeyStore(path)
    key, record = store.issue("laptop", rate_per_minute=120)
    reloaded = KeyStore(path)
    assert reloaded.verify(key).rate_per_minute == 120
    assert json.loads(path.read_text())["keys"][record.key_id]["label"] == "laptop"
