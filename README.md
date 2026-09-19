# Punk-AI

**Own the weights, own the stack.**

Learn to build, train and run AI on hardware you own, with weights anyone can
download — no subscription, no API key, no permission slip from a
trillion-dollar company.

Punk-AI is a workbench for getting genuinely independent at this. Everything
here runs on open-weight models and hardware you control: pull the weights,
verify them, size them to your GPU, evaluate them honestly, fine-tune them, and
serve them behind your own hardened endpoint. The goal isn't to avoid paying for
tools — it's to never *need* to. Rented intelligence you can't inspect, can't
pin to a version, and can't run offline is a bad foundation to learn on.

The second thing this repo is about is doing it **safely**, in the boring and
real sense: you are about to download gigabytes of opaque binary from the
internet and load it into a process on your machine. That deserves the same
care as any other untrusted dependency, and almost nobody gives it that.

```
pip install -e .
punk size llama3.1-8b --gpu rtx-3060-12g     # will it fit?
punk license llama3.1 --intent commercial     # am I allowed?
punk scan ./weights/some-model                # is anything executable in here?
punk eval injection_resistance --backend llamacpp:http://127.0.0.1:8080
punk serve --backend llamacpp:http://127.0.0.1:8080 --policy strict
```

---

## Why this exists

The gap between "I use an AI product" and "I build with AI" is mostly a set of
practical questions nobody answers in one place:

| Question | Answer in this repo |
|---|---|
| Will a 14B model run on my card? | `punk size` — real weight + KV-cache arithmetic |
| What does a 4-bit quant actually cost me? | `punk size --list`, and `labs/02` |
| Can I legally use this model? | `punk license`, and `docs/LICENSES.md` |
| Is this download safe to load? | `punk verify` / `punk scan`, and `docs/SUPPLY_CHAIN.md` |
| Is this model any good *at my task*? | `punk eval` with suites you write |
| Can I fine-tune on one consumer GPU? | `punk plan` — QLoRA memory math, and `labs/04` |
| Is my training data clean? | `punk data` — dedupe, PII, contamination, canaries |
| How do I expose it without getting owned? | `punk serve`, and `docs/THREAT_MODEL.md` |

## The labs

Six hands-on walkthroughs. Start at zero, end with a fine-tuned model serving
behind an audited endpoint you built.

| Lab | What you do | Hardware needed |
|---|---|---|
| [01 — Run it locally](labs/01_run_it_locally.md) | Pick a model, verify it, get tokens out | Any laptop |
| [02 — Size and quantize](labs/02_size_and_quantize.md) | Work out what fits, and what quantization costs | Any laptop |
| [03 — Evaluate it](labs/03_evaluate_it.md) | Write evals that mean something for your task | Any laptop |
| [04 — Fine-tune it](labs/04_finetune_it.md) | QLoRA on one GPU, data checked first | 8 GB+ GPU |
| [05 — Serve it safely](labs/05_serve_it_safely.md) | Auth, rate limits, guards, audit chain | Any laptop |
| [06 — Red-team it](labs/06_red_team_it.md) | Break your own stack before someone else does | Any laptop |

Every lab runs end to end with the built-in `echo` backend, so you can do the
whole sequence — including the serving and red-teaming labs — before you own a
GPU. Swap in real weights when you have them.

## What's in the box

```
src/punkai/
  registry/    manifests, licence gating, hash + safetensors verification
  hardware/    VRAM math: weights, KV cache, quantization, max context
  loading/     load calls that cannot silently execute someone else's Python
  evals/       eval harness, scorers, three built-in suites
  guards/      invisible-character stripping, secret redaction, injection scoring
  serve/       hardened HTTP server: keys, rate limits, backends, audit chain
  train/       dataset hygiene, memorization canaries, QLoRA planning
```

**The core has zero dependencies.** Stdlib only, Python 3.10+. torch,
transformers and the Hugging Face Hub are optional extras you install when you
need them — you can read, test and use most of this repo on a machine that has
never had a CUDA driver.

```bash
pip install -e .              # core: zero dependencies
pip install -e '.[hub]'       # + downloading from the Hub
pip install -e '.[torch]'     # + in-process inference
pip install -e '.[train]'     # + QLoRA fine-tuning
pip install -e '.[dev]'       # + pytest and ruff
pytest                        # 271 tests, no network, no GPU, under a second
```

## Three opinions this repo is built on

**1. A model file is an untrusted dependency.**
`.bin` and `.pt` weights are Python pickles; loading one can execute arbitrary
code. `trust_remote_code=True` imports and runs code from a stranger's repo in
your process. Both are sometimes necessary and neither should ever be a flag you
copy-pasted off a model card without noticing. See
[docs/SUPPLY_CHAIN.md](docs/SUPPLY_CHAIN.md).

**2. Prompt injection is not solved, and no filter in this repo solves it.**
The guards catch the obvious and will miss the clever. The control that actually
works is architectural: never give a model an authority you wouldn't give the
person feeding it text. See [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md).

**3. Your fifty hand-written eval cases beat any leaderboard.**
Public benchmarks measure someone else's distribution, usually one that has
leaked into training data. `punk eval` exists so writing your own costs a JSON
file, and `--baseline` tells you exactly which cases a change broke.

## Status

Early and honest about it. The core toolkit, guards, serving stack, eval harness
and data checks are implemented and tested. The training path is planned and
scaffolded — `punk plan` gives you real memory numbers today, and `train_lora`
needs a GPU this repo has never had. Numbers marked "estimate" are estimates.

MIT licensed. Built by [@chris-dewitt](https://github.com/chris-dewitt).
