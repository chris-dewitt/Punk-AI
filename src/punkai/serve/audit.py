"""A tamper-evident, privacy-preserving request log.

Two properties worth the extra thirty lines:

* **Hash-chained.** Every record carries the hash of the one before it, so
  editing or inserting any past line breaks every hash after it. An attacker
  with write access to the file can no longer quietly rewrite history.
  Truncation of the *tail* is still undetectable from the file alone -- that is
  a real limit of append-only chains. Print `head_hash` somewhere else
  periodically (another host, a chat message, a sticky note) and truncation
  becomes detectable too.
* **Content is hashed, not stored.** By default the log records the sha256 and
  length of each prompt and completion, never the text. You keep the ability to
  prove "this exact prompt was sent at 14:02" without building a permanent
  archive of everything anyone typed. Turn `log_content=True` on when you are
  debugging, and know what you are choosing.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def content_digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class AuditRecord:
    seq: int
    ts: float
    event: str
    actor: str
    data: dict[str, Any] = field(default_factory=dict)
    prev: str = GENESIS
    hash: str = ""

    def payload(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "event": self.event,
            "actor": self.actor,
            "data": self.data,
            "prev": self.prev,
        }

    def sealed(self) -> AuditRecord:
        self.hash = _record_hash(self.payload())
        return self

    def to_json(self) -> str:
        record = self.payload()
        record["hash"] = self.hash
        return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass
class ChainCheck:
    ok: bool
    count: int
    problems: list[str] = field(default_factory=list)
    head_hash: str = GENESIS


class AuditLog:
    def __init__(self, path: str | Path | None, log_content: bool = False) -> None:
        self.path = Path(path) if path else None
        self.log_content = log_content
        self._lock = threading.Lock()
        self._seq = 0
        self._head = GENESIS
        self.records: list[AuditRecord] = []  # kept in memory only when path is None
        if self.path and self.path.exists():
            check = verify_chain(self.path)
            self._seq = check.count
            self._head = check.head_hash
            if not check.ok:
                raise ValueError(
                    f"{self.path}: existing audit chain is broken; refusing to append.\n  "
                    + "\n  ".join(check.problems)
                )

    @property
    def head_hash(self) -> str:
        return self._head

    def append(self, event: str, actor: str = "-", **fields: Any) -> AuditRecord:
        with self._lock:
            record = AuditRecord(
                seq=self._seq,
                ts=round(time.time(), 3),
                event=event,
                actor=actor,
                data=fields,
                prev=self._head,
            ).sealed()
            self._seq += 1
            self._head = record.hash
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(record.to_json() + "\n")
            else:
                self.records.append(record)
            return record

    def log_generation(
        self,
        actor: str,
        prompt: str,
        completion: str | None,
        *,
        model: str,
        latency_ms: float,
        decision: str,
        reasons: list[str] | None = None,
        injection_score: float = 0.0,
    ) -> AuditRecord:
        fields: dict[str, Any] = {
            "model": model,
            "prompt_sha256": content_digest(prompt),
            "prompt_chars": len(prompt),
            "decision": decision,
            "latency_ms": round(latency_ms, 2),
            "injection_score": injection_score,
        }
        if reasons:
            fields["reasons"] = reasons
        if completion is not None:
            fields["completion_sha256"] = content_digest(completion)
            fields["completion_chars"] = len(completion)
        if self.log_content:
            fields["prompt"] = prompt
            if completion is not None:
                fields["completion"] = completion
        return self.append("generate", actor, **fields)


def verify_chain(path: str | Path) -> ChainCheck:
    """Walk the log and confirm nobody rewrote the past."""
    path = Path(path)
    problems: list[str] = []
    prev = GENESIS
    count = 0

    if not path.exists():
        return ChainCheck(ok=True, count=0, head_hash=GENESIS)

    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append(f"line {lineno}: not valid JSON ({exc})")
                break
            claimed = record.pop("hash", None)
            if claimed is None:
                problems.append(f"line {lineno}: record has no hash")
                break
            if record.get("prev") != prev:
                problems.append(
                    f"line {lineno}: prev is {record.get('prev')!r}, expected {prev!r} "
                    "-- a record was edited, inserted or removed"
                )
                break
            if record.get("seq") != count:
                problems.append(f"line {lineno}: seq is {record.get('seq')}, expected {count}")
                break
            recomputed = _record_hash(record)
            if recomputed != claimed:
                problems.append(
                    f"line {lineno}: hash mismatch -- record contents were modified"
                )
                break
            prev = claimed
            count += 1

    return ChainCheck(ok=not problems, count=count, problems=problems, head_hash=prev)
