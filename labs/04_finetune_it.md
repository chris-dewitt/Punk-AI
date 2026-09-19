# Lab 04 — Fine-tune it

**Goal:** train a model on one consumer GPU, with the data checked before it
becomes weights.

**Hardware:** 8 GB+ VRAM for a 7-8B QLoRA. Everything except the training run
itself works on any laptop.

---

## Why this is possible at all

Full fine-tuning of a 7B model needs weights in 16-bit (14 GiB), gradients for
all of them (14 GiB), and Adam's two moment buffers in 32-bit (56 GiB). About
90 GiB before a single activation — which is why fine-tuning used to mean
renting A100s.

**LoRA** freezes the base weights and trains two small low-rank matrices per
target layer instead. Typically 0.1-1% of the parameters, so gradients and
optimizer state shrink by the same factor. **QLoRA** adds one more trick: keep
the frozen base in 4-bit, since you never update it.

```bash
punk plan llama3.1-8b --gpu rtx-3090
```

```
  trainable: 29.4M params (0.37% of the model)
  frozen base      4.21 GiB
  lora adapters    0.05 GiB
  gradients        0.05 GiB
  optimizer        0.22 GiB
  activations      0.35 GiB
  total            5.78 GiB
```

90 GiB to 6 GiB. Check yours before you start:

```bash
punk plan qwen2.5-14b --rank 32 --seq 2048 --vram 12
punk plan qwen2.5-14b --rank 32 --seq 2048 --vram 12 --no-checkpointing
```

Gradient checkpointing trades ~30% more compute for roughly a 10× cut in
activation memory. Almost always worth it.

## First: does fine-tuning solve your problem?

It usually doesn't. Fine-tuning teaches **form** — tone, format, a domain's
vocabulary, a consistent output shape. It is a poor way to teach **facts**;
retrieval does that better, cheaper, and updates without retraining.

Try in this order: a better prompt → few-shot examples → retrieval → *then*
fine-tuning. Get a baseline eval number first (lab 03) or you will have no way
to know whether it worked.

## Check the data before it becomes weights

Whatever is in this file ends up inside the model, permanently. You cannot
un-train a leaked API key.

```bash
punk data ./train.jsonl --against ./evals.jsonl --card DATASET_CARD.md
```

```
1482 example(s)
  exact duplicates : 34 example(s) in 12 group(s)
  near duplicates  : 61 pair(s)
  pii hits         : 7 (email=5, us-phone=2)
  licenses         : mit=1200, unknown=282
  CONTAMINATION    : 3 eval text(s) appear in training data
  verdict          : needs attention
```

Each line is a decision:

- **Duplicates** get memorized rather than learned, and silently reweight your
  dataset toward whatever got scraped twice.
- **PII** is a privacy incident waiting for someone to prompt it out. The regex
  scan finds emails and card numbers; it will not find a person identifiable
  from context. Human review is not optional for data from real users.
- **`unknown` licence** on 282 examples means you don't know if you can release
  what you train.
- **Contamination** invalidates your eval. Fix this one first — otherwise every
  number after it is a lie.

## Plant canaries

```python
from punkai.train.data import canary_examples

canaries = canary_examples(count=5, repeats=1)
# mix into your training set, keep the canary strings
```

After training, ask for them back:

```python
from punkai.train.data import check_memorization

results = check_memorization(backend.generate, [e.meta["canary"] for e in canaries])
```

If a unique 16-hex-digit string seen **once** comes back verbatim, your model
memorizes single exposures — and anything sensitive in that data can come back
too. That is a measurement, not a guess, and it takes five minutes.

## Run it

```bash
pip install -e '.[torch,train]'
```

```python
from punkai.train import LoraPlan
from punkai.train.lora import train_lora

plan = LoraPlan(arch="llama3.1-8b", rank=16, seq_len=1024, batch_size=1, grad_accum=8)
train_lora("./weights/llama3.1-8b", dataset, plan, output_dir="./adapters/v1")
```

Starting points that are hard to get wrong: **rank 16** (raise to 32-64 only if
it underfits), **alpha = 2×rank**, **lr 2e-4**, **2-3 epochs** — more than that
and you are memorizing, not learning. Target all attention *and* MLP projections;
attention-only is cheaper and usually worse.

Watch for: loss dropping to near-zero (memorizing), loss flat (lr too low or
data too small), or the model getting worse at everything else — catastrophic
forgetting, which the next section catches.

## Then measure it, properly

```bash
punk eval ./my_task.json --backend transformers:./adapters/v1 --baseline baseline.json
punk eval capability_smoke --backend transformers:./adapters/v1 --baseline smoke_baseline.json
punk eval overrefusal      --backend transformers:./adapters/v1 --baseline refusal_baseline.json
```

All three, every time. A fine-tune that nails your task and loses general
instruction-following, or turns the model into something that refuses to explain
a buffer overflow, is a regression however good the headline number looks.

The adapter is a few hundred MB, not a full model copy — keep the base weights
once and version the adapters.

→ [Lab 05: Serve it safely](05_serve_it_safely.md)
