# Lab 02 — Size and quantize

**Goal:** stop guessing whether a model fits, and understand what you trade away
when you shrink one.

**Hardware:** any laptop. This lab is arithmetic.

---

## The three things that eat your card

```bash
punk size llama3.1-8b --gpu rtx-3060-12g
```

```
llama3.1-8b @ q4_k_m, ctx 8192, batch 1
  weights     4.53 GiB
  kv cache    1.00 GiB
  overhead    1.07 GiB
  total       6.61 GiB
  budget     12.00 GiB -> fits (+5.39 GiB)
```

**Weights** = parameters × bits-per-weight. The number everyone quotes, and the
only one most people think about.

**KV cache** = every token in your context, cached per layer, for both keys and
values. It scales linearly with context length *and* batch size. This is the
one that ruins your day:

```bash
punk size llama3.1-8b --quant fp16 --ctx 131072
```

16 GiB of cache against 15 GiB of weights. Long context is not free, and "it's
only an 8B" is not a plan.

**Overhead** = CUDA context, activations, allocator fragmentation. Roughly a
constant plus a slice of the weights.

## Why two same-size models behave differently

```bash
punk size qwen2.5-7b --ctx 8192      # 4 KV heads
punk size phi-3-mini --ctx 8192      # 32 KV heads
```

Grouped-query attention shares KV heads across query heads. Fewer KV heads,
smaller cache — often by 4-8×. Two models of similar parameter count can have
wildly different memory behaviour at long context, and nothing on the model card
tells you this. `config.json` does: `num_key_value_heads`.

## What quantization costs

```bash
punk size --list
```

Bits-per-weight for the k-quants are real measured averages, not what the name
implies — `q4_k_m` is 4.85 bits, not 4.

The rule of thumb, in order:

1. **q4_k_m** is the default. Small, coherent, barely distinguishable from fp16
   on most tasks.
2. **q5_k_m / q6_k** if you have room. Diminishing returns above that.
3. **q8_0 / fp16** if you have lots of room, or you are measuring quality and
   need a reference.
4. **q3 and below** only to fit a bigger model, and check that trade with an
   eval — it often loses.

> **The rule that matters:** a larger model at q4 usually beats a smaller model
> at q8 for the same memory. A 14B at q4_k_m (~9 GiB) generally outperforms a
> 7B at q8 (~8 GiB). Verify it on *your* task with lab 03 rather than trusting
> me.

## The levers when it doesn't fit

```bash
punk size qwen2.5-14b --vram 12 --ctx 32768          # doesn't fit
punk size qwen2.5-14b --vram 12 --ctx 8192           # shorter context
punk size qwen2.5-14b --vram 12 --ctx 32768 --kv-bits 8   # 8-bit KV cache
```

In order of what they cost you: shorter context (free if you don't need it),
8-bit KV cache (small quality hit), heavier quantization (real quality hit),
smaller model (different model), CPU offload (10-50× slower for offloaded
layers).

```bash
punk size qwen2.5-7b --gpu rtx-3090     # what's the longest context I can run?
```

## Exercise

1. Find the largest model that fits your actual hardware at 8k context.
2. Work out its maximum context at that quant.
3. Predict whether doubling batch size still fits, then check with `--batch 2`.
4. Take one model and predict where it stops fitting as context grows: 4k, 8k,
   32k, 128k. Find the cliff.

These numbers are estimates (±15%), good for deciding "8-bit or 4-bit", not for
packing a datacentre.

→ [Lab 03: Evaluate it](03_evaluate_it.md)
