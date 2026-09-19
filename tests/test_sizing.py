import pytest

from punkai.hardware.sizing import (
    ARCHS,
    QUANTS,
    estimate,
    gpu_budget,
    kv_cache_gib,
    max_context,
    recommend,
    weights_gib,
)


def test_fp16_weights_are_two_bytes_per_param():
    # 8.03B params * 2 bytes = ~14.96 GiB. The number everyone quotes for an 8B.
    assert weights_gib(8.03, "fp16") == pytest.approx(14.96, abs=0.05)


def test_four_bit_is_roughly_a_quarter_of_sixteen_bit():
    ratio = weights_gib(7.0, "q4_k_m") / weights_gib(7.0, "fp16")
    assert 0.28 < ratio < 0.32


def test_kv_cache_scales_linearly_with_context_and_batch():
    one = kv_cache_gib("llama3.1-8b", 4096, 1)
    assert kv_cache_gib("llama3.1-8b", 8192, 1) == pytest.approx(2 * one)
    assert kv_cache_gib("llama3.1-8b", 4096, 4) == pytest.approx(4 * one)


def test_eight_bit_kv_cache_halves_it():
    full = kv_cache_gib("llama3.1-8b", 8192, 1, cache_bits=16)
    half = kv_cache_gib("llama3.1-8b", 8192, 1, cache_bits=8)
    assert half == pytest.approx(full / 2)


def test_grouped_query_attention_shrinks_the_cache():
    """qwen2.5-7b has 4 KV heads, phi-3-mini has 32. Same ballpark size, very
    different cache -- this is why 'it's a 7B, it'll fit' is not a plan."""
    gqa = kv_cache_gib("qwen2.5-7b", 8192)
    mha = kv_cache_gib("phi-3-mini", 8192)
    assert mha > gqa * 2


def test_8b_fp16_at_full_context_does_not_fit_a_4090():
    est = estimate("llama3.1-8b", "fp16", context=131072)
    assert not est.fits(24.0)
    assert est.total_gib > 30


def test_8b_q4_fits_a_12gb_card():
    assert estimate("llama3.1-8b", "q4_k_m", context=8192).fits(12.0)


def test_70b_needs_serious_hardware():
    assert not estimate("llama3.3-70b", "q4_k_m", context=4096).fits(24.0)
    assert estimate("llama3.3-70b", "q4_k_m", context=4096).fits(48.0)


def test_recommend_returns_best_quality_first_and_all_fit():
    options = recommend("qwen2.5-7b", 16.0, context=8192)
    assert options
    bits = [QUANTS[o.quant].bits for o in options]
    assert bits == sorted(bits, reverse=True)
    assert all(o.fits(16.0) for o in options)


def test_recommend_is_empty_when_nothing_fits():
    assert recommend("llama3.3-70b", 8.0) == []


def test_max_context_is_zero_when_weights_alone_overflow():
    assert max_context("llama3.3-70b", 8.0, "q4_k_m") == 0


def test_max_context_respects_the_model_limit():
    assert max_context("gemma2-9b", 80.0, "q4_k_m") == ARCHS["gemma2-9b"].context_length


def test_max_context_result_actually_fits():
    budget = 24.0
    ctx = max_context("qwen2.5-14b", budget, "q4_k_m")
    assert estimate("qwen2.5-14b", "q4_k_m", ctx).fits(budget)


def test_unknown_names_fail_loudly():
    with pytest.raises(KeyError, match="unknown arch"):
        estimate("gpt-9", "q4_k_m")
    with pytest.raises(KeyError, match="unknown quant"):
        weights_gib(7, "q1_vibes")
    with pytest.raises(KeyError, match="unknown gpu"):
        gpu_budget("rtx-9090")


@pytest.mark.parametrize("bad", [0, -1])
def test_nonsense_context_or_batch_rejected(bad):
    with pytest.raises(ValueError):
        estimate("qwen2.5-7b", "q4_k_m", context=bad)
    with pytest.raises(ValueError):
        estimate("qwen2.5-7b", "q4_k_m", batch=bad)
