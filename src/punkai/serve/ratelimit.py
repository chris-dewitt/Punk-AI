"""Token-bucket rate limiting.

A limiter is not politeness, it is the thing standing between one buggy loop and
a GPU pinned at 100% for six hours. Buckets are per key, refill continuously,
and allow a short burst -- which is what real clients do -- while holding the
long-run average to the configured rate.

Uses `time.monotonic()`, so an NTP step or a daylight-saving change cannot
hand anyone free requests.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class Decision:
    allowed: bool
    remaining: float
    retry_after: float = 0.0


class TokenBucket:
    def __init__(self, rate_per_minute: float, burst: float | None = None) -> None:
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be positive")
        self.rate_per_second = rate_per_minute / 60.0
        self.capacity = burst if burst is not None else max(1.0, rate_per_minute / 6.0)
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self.updated)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_second)
        self.updated = now

    def take(self, cost: float = 1.0, now: float | None = None) -> Decision:
        with self._lock:
            self._refill(now if now is not None else time.monotonic())
            if self.tokens >= cost:
                self.tokens -= cost
                return Decision(True, self.tokens)
            deficit = cost - self.tokens
            return Decision(False, self.tokens, retry_after=deficit / self.rate_per_second)


class RateLimiter:
    """Per-identity buckets, created on first sight and pruned when idle.

    Pruning matters: without it, an unauthenticated endpoint keyed by client IP
    is an unbounded dict that anyone can grow -- the limiter becomes the leak.
    """

    def __init__(self, default_rate_per_minute: float = 60.0, idle_ttl: float = 3600.0) -> None:
        self.default_rate = default_rate_per_minute
        self.idle_ttl = idle_ttl
        self._buckets: dict[str, tuple[TokenBucket, float]] = {}
        self._lock = threading.Lock()

    def check(self, identity: str, rate_per_minute: float | None = None, cost: float = 1.0):
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            entry = self._buckets.get(identity)
            if entry is None:
                bucket = TokenBucket(rate_per_minute or self.default_rate)
                self._buckets[identity] = (bucket, now)
            else:
                bucket = entry[0]
                self._buckets[identity] = (bucket, now)
        return bucket.take(cost, now)

    def _prune(self, now: float) -> None:
        stale = [k for k, (_, seen) in self._buckets.items() if now - seen > self.idle_ttl]
        for key in stale:
            del self._buckets[key]

    def __len__(self) -> int:
        return len(self._buckets)
