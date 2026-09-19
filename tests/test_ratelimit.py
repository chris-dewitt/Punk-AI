import pytest

from punkai.serve.ratelimit import RateLimiter, TokenBucket


def test_burst_is_allowed_then_denied():
    bucket = TokenBucket(rate_per_minute=60, burst=5)
    assert sum(bucket.take(now=100.0).allowed for _ in range(5)) == 5
    assert not bucket.take(now=100.0).allowed


def test_tokens_refill_over_time():
    bucket = TokenBucket(rate_per_minute=60, burst=2)  # 1 token/sec
    bucket.take(now=0.0)
    bucket.take(now=0.0)
    assert not bucket.take(now=0.0).allowed
    assert bucket.take(now=1.0).allowed


def test_refill_is_capped_at_burst():
    bucket = TokenBucket(rate_per_minute=60, burst=3)
    bucket.take(now=0.0)
    assert sum(bucket.take(now=10_000.0).allowed for _ in range(3)) == 3
    assert not bucket.take(now=10_000.0).allowed


def test_retry_after_is_useful():
    bucket = TokenBucket(rate_per_minute=60, burst=1)
    bucket.take(now=0.0)
    decision = bucket.take(now=0.0)
    assert not decision.allowed
    assert 0.9 <= decision.retry_after <= 1.1


def test_long_run_average_holds():
    """Ten minutes of hammering at 60/min should let through ~600, not ~6000."""
    bucket = TokenBucket(rate_per_minute=60, burst=10)
    allowed = sum(bucket.take(now=t / 10).allowed for t in range(6000))
    assert 590 <= allowed <= 615


def test_identities_do_not_share_budget():
    limiter = RateLimiter(default_rate_per_minute=60)
    for _ in range(20):
        limiter.check("noisy")
    assert limiter.check("quiet").allowed


def test_per_key_rate_overrides_default():
    limiter = RateLimiter(default_rate_per_minute=60)
    allowed = sum(limiter.check("vip", rate_per_minute=6000).allowed for _ in range(50))
    assert allowed == 50


def test_idle_buckets_are_pruned():
    """Otherwise the limiter itself is an unbounded, attacker-growable dict."""
    limiter = RateLimiter(idle_ttl=0.0)
    limiter.check("a")
    limiter.check("b")
    assert len(limiter) <= 1


def test_nonsense_rate_rejected():
    with pytest.raises(ValueError):
        TokenBucket(rate_per_minute=0)
