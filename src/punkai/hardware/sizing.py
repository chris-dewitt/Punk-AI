"""VRAM arithmetic, so you stop learning your limits from OOM tracebacks.

Three things eat your card, in this order of surprise:

1. **Weights** -- params x bits-per-weight. Predictable, and the number everyone
   quotes.
2. **KV cache** -- grows linearly with context length *and* batch size, and on
   an old multi-head model it can dwarf the weights. This is the one that gets
   people: a 7B at 4-bit is 4 GiB of weights and can be 16 GiB of cache at 32k
   context if the architecture has no grouped-query attention.
3. **Everything else** -- CUDA context, activations, fragmentation. Roughly a
   constant plus a slice of the weights.

Numbers are estimates in GiB (1024^3), good to maybe +/-15%. They are for
deciding "8-bit or 4-bit?", not for packing a datacentre.
"""

from __future__ import annotations

from dataclasses import dataclass

GIB = 1024**3


@dataclass(frozen=True)
class Quant:
    """A weight format: how many bits each parameter costs, and what it costs you."""

    name: str
    bits: float
    quality: str
    note: str = ""


# bits-per-weight for the k-quants are the real measured averages, not the name.
QUANTS: dict[str, Quant] = {
    q.name: q
    for q in [
        Quant("fp16", 16.0, "reference", "Full fidelity. What the model was trained near."),
        Quant("bf16", 16.0, "reference", "Same size as fp16, better dynamic range."),
        Quant("fp8", 8.0, "excellent", "Hopper/Ada hardware path. Near-lossless in practice."),
        Quant("int8", 8.0, "excellent", "bitsandbytes LLM.int8(). Safe default when 16-bit "
              "does not fit."),
        Quant("q8_0", 8.5, "excellent", "llama.cpp 8-bit. Effectively indistinguishable."),
        Quant("q6_k", 6.56, "very good", "The quality/size sweet spot if you have room."),
        Quant("q5_k_m", 5.69, "very good", "Barely distinguishable from fp16 on most tasks."),
        Quant("q4_k_m", 4.85, "good", "The default everyone should start at. Small, coherent."),
        Quant("nf4", 4.5, "good", "bitsandbytes 4-bit, the QLoRA format."),
        Quant("q4_0", 4.55, "ok", "Older 4-bit. q4_k_m is better at the same size."),
        Quant("q3_k_m", 3.91, "degraded", "Noticeably dumber. Use only to fit a bigger model."),
        Quant("q2_k", 3.35, "poor", "A bigger model at q2 usually loses to a smaller one at q4."),
    ]
}

# VRAM in GiB for cards people actually tinker on.
GPUS: dict[str, float] = {
    "rtx-3060-12g": 12.0,
    "rtx-3080": 10.0,
    "rtx-3090": 24.0,
    "rtx-4060-ti-16g": 16.0,
    "rtx-4070": 12.0,
    "rtx-4070-ti-super": 16.0,
    "rtx-4080": 16.0,
    "rtx-4090": 24.0,
    "rtx-5080": 16.0,
    "rtx-5090": 32.0,
    "rx-7900-xtx": 24.0,
    "arc-a770": 16.0,
    "a6000": 48.0,
    "l40s": 48.0,
    "a100-40g": 40.0,
    "a100-80g": 80.0,
    "h100": 80.0,
    # Apple silicon shares one pool; usable is roughly 70% of total by default.
    "m2-pro-16g": 11.0,
    "m3-max-36g": 25.0,
    "m4-max-128g": 90.0,
    # The honest baseline: no GPU at all.
    "cpu-32g": 28.0,
    "cpu-64g": 58.0,
}


@dataclass(frozen=True)
class ModelArch:
    """The handful of numbers that determine memory. All readable from config.json."""

    name: str
    params_b: float
    num_layers: int
    num_kv_heads: int
    head_dim: int
    context_length: int = 8192

    @property
    def kv_bytes_per_token(self) -> int:
        """K and V, per layer, at 2 bytes per element (fp16 cache)."""
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * 2


# Approximate configs for models worth starting on. Read config.json for exact values.
ARCHS: dict[str, ModelArch] = {
    a.name: a
    for a in [
        ModelArch("smollm2-1.7b", 1.71, 24, 32, 64, 8192),
        ModelArch("qwen2.5-3b", 3.09, 36, 2, 128, 32768),
        ModelArch("phi-3-mini", 3.82, 32, 32, 96, 4096),
        ModelArch("llama3.2-3b", 3.21, 28, 8, 128, 131072),
        ModelArch("mistral-7b-v0.3", 7.25, 32, 8, 128, 32768),
        ModelArch("qwen2.5-7b", 7.62, 28, 4, 128, 32768),
        ModelArch("llama3.1-8b", 8.03, 32, 8, 128, 131072),
        ModelArch("gemma2-9b", 9.24, 42, 8, 256, 8192),
        ModelArch("qwen2.5-14b", 14.8, 48, 8, 128, 32768),
        ModelArch("qwen2.5-32b", 32.8, 64, 8, 128, 32768),
        ModelArch("llama3.3-70b", 70.6, 80, 8, 128, 131072),
    ]
}


@dataclass
class Estimate:
    arch: str
    quant: str
    context: int
    batch: int
    weights_gib: float
    kv_gib: float
    overhead_gib: float

    @property
    def total_gib(self) -> float:
        return self.weights_gib + self.kv_gib + self.overhead_gib

    def fits(self, budget_gib: float) -> bool:
        return self.total_gib <= budget_gib

    def headroom(self, budget_gib: float) -> float:
        return budget_gib - self.total_gib

    def explain(self, budget_gib: float | None = None) -> str:
        lines = [
            f"{self.arch} @ {self.quant}, ctx {self.context}, batch {self.batch}",
            f"  weights   {self.weights_gib:6.2f} GiB",
            f"  kv cache  {self.kv_gib:6.2f} GiB",
            f"  overhead  {self.overhead_gib:6.2f} GiB",
            f"  total     {self.total_gib:6.2f} GiB",
        ]
        if budget_gib is not None:
            room = self.headroom(budget_gib)
            verdict = "fits" if room >= 0 else "does NOT fit"
            lines.append(f"  budget    {budget_gib:6.2f} GiB -> {verdict} ({room:+.2f} GiB)")
        return "\n".join(lines)


def resolve_arch(arch: str | ModelArch) -> ModelArch:
    if isinstance(arch, ModelArch):
        return arch
    if arch not in ARCHS:
        raise KeyError(f"unknown arch {arch!r}; known: {', '.join(sorted(ARCHS))}")
    return ARCHS[arch]


def weights_gib(params_b: float, quant: str) -> float:
    if quant not in QUANTS:
        raise KeyError(f"unknown quant {quant!r}; known: {', '.join(QUANTS)}")
    return params_b * 1e9 * QUANTS[quant].bits / 8 / GIB


def kv_cache_gib(
    arch: str | ModelArch, context: int, batch: int = 1, cache_bits: int = 16
) -> float:
    """KV cache for a full context window.

    An 8-bit KV cache halves this for a small quality cost and is usually the
    first lever to pull when a long context will not fit.
    """
    spec = resolve_arch(arch)
    per_token = spec.kv_bytes_per_token * (cache_bits / 16)
    return per_token * context * batch / GIB


def estimate(
    arch: str | ModelArch,
    quant: str = "q4_k_m",
    context: int | None = None,
    batch: int = 1,
    cache_bits: int = 16,
) -> Estimate:
    """Total VRAM for weights + cache + runtime slop."""
    spec = resolve_arch(arch)
    ctx = context if context is not None else min(spec.context_length, 8192)
    if ctx <= 0 or batch <= 0:
        raise ValueError("context and batch must be positive")
    w = weights_gib(spec.params_b, quant)
    kv = kv_cache_gib(spec, ctx, batch, cache_bits)
    # ~0.8 GiB for the CUDA/Metal context and allocator, plus ~6% of weights for
    # activations and fragmentation. Empirical, not derived.
    overhead = 0.8 + 0.06 * w
    return Estimate(spec.name, quant, ctx, batch, w, kv, overhead)


def max_context(
    arch: str | ModelArch,
    budget_gib: float,
    quant: str = "q4_k_m",
    batch: int = 1,
    cache_bits: int = 16,
) -> int:
    """Longest context that fits in `budget_gib`. 0 means the weights alone do not."""
    spec = resolve_arch(arch)
    w = weights_gib(spec.params_b, quant)
    overhead = 0.8 + 0.06 * w
    spare = budget_gib - w - overhead
    if spare <= 0:
        return 0
    per_token_gib = spec.kv_bytes_per_token * (cache_bits / 16) * batch / GIB
    return min(int(spare / per_token_gib), spec.context_length)


def recommend(
    arch: str | ModelArch,
    budget_gib: float,
    context: int = 8192,
    batch: int = 1,
) -> list[Estimate]:
    """Every quant that fits, best quality first. Empty means try a smaller model.

    Ordered by bits descending because the rule of thumb is: take the largest
    model you can run at q4_k_m before you take a smaller one at q8.
    """
    spec = resolve_arch(arch)
    fitting = []
    for name in sorted(QUANTS, key=lambda n: -QUANTS[n].bits):
        est = estimate(spec, name, context, batch)
        if est.fits(budget_gib):
            fitting.append(est)
    return fitting


def gpu_budget(gpu: str) -> float:
    if gpu not in GPUS:
        raise KeyError(f"unknown gpu {gpu!r}; known: {', '.join(sorted(GPUS))}")
    return GPUS[gpu]
