"""API keys for your own endpoint.

Design notes, because the choices here are the interesting part:

* **Keys are 256-bit random, so they are hashed with plain SHA-256, not scrypt.**
  Password hashing is slow on purpose because humans pick guessable passwords.
  A 256-bit random token has no guessable structure -- there is nothing to slow
  down an attacker doing, and paying 50ms of scrypt on every request buys you
  nothing but a denial-of-service lever. A per-key random salt still goes in,
  so a stolen store cannot be attacked with a precomputed table.
* **The plaintext key is returned exactly once, at issue time.** The store keeps
  a hash. If you lose the key, you issue a new one; nobody can read it back out
  of the file, including you.
* **Comparison is constant-time**, and an unknown key id still pays the cost of a
  hash, so response timing does not tell an attacker which ids exist.
* **The store file is written 0600**, and this module refuses to load one that is
  group- or world-readable.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

PREFIX = "pk"
_KEY_ID_BYTES = 6  # 12 hex chars -- an identifier, not a secret
_SECRET_BYTES = 32  # 256 bits of actual entropy


@dataclass
class KeyRecord:
    key_id: str
    label: str
    salt: str  # hex
    hash: str  # hex sha256(salt || secret)
    created_at: float
    revoked_at: float | None = None
    scopes: list[str] = field(default_factory=lambda: ["generate"])
    rate_per_minute: int = 60

    @property
    def active(self) -> bool:
        return self.revoked_at is None

    def has_scope(self, scope: str) -> bool:
        return self.active and (scope in self.scopes or "*" in self.scopes)


def _hash_secret(salt_hex: str, secret: str) -> str:
    return hashlib.sha256(bytes.fromhex(salt_hex) + secret.encode("utf-8")).hexdigest()


def parse_key(presented: str) -> tuple[str, str] | None:
    """Split `pk_<id>_<secret>` without raising on junk input."""
    if not presented:
        return None
    parts = presented.strip().split("_")
    if len(parts) != 3 or parts[0] != PREFIX:
        return None
    key_id, secret = parts[1], parts[2]
    if not key_id or not secret:
        return None
    return key_id, secret


class KeyStore:
    """A tiny JSON-backed key store. Fine for a homelab; swap it for your
    identity provider the day this stops being a homelab."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.keys: dict[str, KeyRecord] = {}
        # Stable per-process decoy salt so verifying an unknown key id costs the
        # same as verifying a real one.
        self._decoy_salt = secrets.token_hex(16)
        if self.path and self.path.exists():
            self.load()

    # -- persistence ------------------------------------------------------

    def load(self) -> None:
        assert self.path is not None
        mode = self.path.stat().st_mode
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise PermissionError(
                f"{self.path} is readable by group or others ({oct(stat.S_IMODE(mode))}). "
                f"Run: chmod 600 {self.path}"
            )
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.keys = {k: KeyRecord(**v) for k, v in data.get("keys", {}).items()}

    def save(self) -> None:
        assert self.path is not None, "KeyStore has no path"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"keys": {k: asdict(v) for k, v in self.keys.items()}}, indent=2)
        # Create with 0600 from the start -- never write secrets to a
        # default-permission file and chmod afterwards.
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        os.chmod(self.path, 0o600)

    # -- lifecycle --------------------------------------------------------

    def issue(
        self,
        label: str,
        scopes: list[str] | None = None,
        rate_per_minute: int = 60,
    ) -> tuple[str, KeyRecord]:
        """Mint a key. The returned plaintext is the only copy that will exist."""
        key_id = secrets.token_hex(_KEY_ID_BYTES)
        secret = secrets.token_urlsafe(_SECRET_BYTES).replace("_", "")  # keep '_' as the separator
        salt = secrets.token_hex(16)
        record = KeyRecord(
            key_id=key_id,
            label=label,
            salt=salt,
            hash=_hash_secret(salt, secret),
            created_at=time.time(),
            scopes=scopes or ["generate"],
            rate_per_minute=rate_per_minute,
        )
        self.keys[key_id] = record
        if self.path:
            self.save()
        return f"{PREFIX}_{key_id}_{secret}", record

    def revoke(self, key_id: str) -> bool:
        record = self.keys.get(key_id)
        if not record or not record.active:
            return False
        record.revoked_at = time.time()
        if self.path:
            self.save()
        return True

    # -- verification -----------------------------------------------------

    def verify(self, presented: str) -> KeyRecord | None:
        """Return the record for a valid, unrevoked key, else None.

        Runs the same hash work for unknown ids as for known ones.
        """
        parsed = parse_key(presented)
        if parsed is None:
            _hash_secret(self._decoy_salt, "")  # equalize the malformed-input path too
            return None
        key_id, secret = parsed
        record = self.keys.get(key_id)
        if record is None:
            hmac.compare_digest(_hash_secret(self._decoy_salt, secret), self._decoy_salt * 4)
            return None
        candidate = _hash_secret(record.salt, secret)
        if not hmac.compare_digest(candidate, record.hash):
            return None
        if not record.active:
            return None
        return record
