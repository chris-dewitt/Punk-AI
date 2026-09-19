import json

import pytest

from punkai.serve.audit import GENESIS, AuditLog, content_digest, verify_chain


def test_empty_log_verifies():
    check = verify_chain("/nonexistent/audit.jsonl")
    assert check.ok and check.count == 0 and check.head_hash == GENESIS


def test_chain_verifies_after_appends(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(5):
        log.append("test", actor="me", i=i)
    check = verify_chain(path)
    assert check.ok and check.count == 5
    assert check.head_hash == log.head_hash


def test_edited_record_breaks_the_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(4):
        log.append("generate", actor="me", i=i)

    lines = path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[1])
    record["data"]["i"] = 999
    lines[1] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    check = verify_chain(path)
    assert not check.ok
    assert "hash mismatch" in check.problems[0]


def test_deleted_record_breaks_the_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(4):
        log.append("generate", actor="me", i=i)
    lines = path.read_text(encoding="utf-8").splitlines()
    del lines[1]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not verify_chain(path).ok


def test_inserted_record_breaks_the_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append("generate", actor="me")
    log.append("generate", actor="me")
    lines = path.read_text(encoding="utf-8").splitlines()
    forged = json.loads(lines[1])
    forged["data"] = {"forged": True}
    lines.insert(1, json.dumps(forged, sort_keys=True, separators=(",", ":")))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not verify_chain(path).ok


def test_appending_to_a_broken_chain_is_refused(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append("generate", actor="me")
    path.write_text(path.read_text().replace('"actor":"me"', '"actor":"someone-else"'))
    with pytest.raises(ValueError, match="broken"):
        AuditLog(path)


def test_prompt_text_is_not_stored_by_default(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.log_generation(
        "key1", "my secret prompt", "the answer", model="m", latency_ms=1.0, decision="allow"
    )
    contents = path.read_text(encoding="utf-8")
    assert "my secret prompt" not in contents
    assert "the answer" not in contents
    assert content_digest("my secret prompt") in contents


def test_content_logging_is_opt_in(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path, log_content=True)
    log.log_generation("key1", "my secret prompt", "answer", model="m", latency_ms=1.0,
                       decision="allow")
    assert "my secret prompt" in path.read_text(encoding="utf-8")


def test_digest_lets_you_prove_a_prompt_was_sent(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.log_generation("k", "was this asked?", None, model="m", latency_ms=1.0, decision="blocked")
    row = json.loads(path.read_text().splitlines()[0])
    assert row["data"]["prompt_sha256"] == content_digest("was this asked?")


def test_refusals_are_logged_too(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.log_generation("k", "bad prompt", None, model="m", latency_ms=0.5, decision="blocked",
                       reasons=["injection"], injection_score=0.8)
    row = json.loads(path.read_text().splitlines()[0])
    assert row["data"]["decision"] == "blocked"
    assert row["data"]["injection_score"] == 0.8


def test_resuming_continues_the_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    first = AuditLog(path)
    first.append("a")
    head = first.head_hash
    second = AuditLog(path)
    assert second.head_hash == head
    second.append("b")
    check = verify_chain(path)
    assert check.ok and check.count == 2


def test_memory_only_log_keeps_records():
    log = AuditLog(None)
    log.append("x", actor="me")
    assert len(log.records) == 1
