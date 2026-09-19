# Lab 01 — Run it locally

**Goal:** get tokens out of a model you downloaded, and understand every step
between "a repo on a hub" and "text on your terminal".

**Hardware:** any laptop. A CPU-only machine runs a 3B model fine.

---

## 1. Pick a model you're actually allowed to use

Before downloading eight gigabytes, spend ten seconds:

```bash
punk license apache-2.0 --intent commercial
punk license llama3.1 --intent commercial
punk license mistral-research --intent commercial
```

Three different answers: `OK`, `REVIEW` (with obligations), `BLOCKED`. For a
first model, pick something Apache-2.0 — Qwen2.5-7B-Instruct or SmolLM2 — and
skip the licence question entirely while you learn.

## 2. Check it fits before you download it

```bash
punk size qwen2.5-7b --vram 8      # or --gpu rtx-3060-12g
punk size --list                   # every known arch, gpu and quant
```

Read the breakdown. Weights are the obvious number; the KV cache is the one
that surprises people, and it scales with context length. Lab 02 is entirely
about this.

## 3. The easiest real path: llama.cpp

No Python ML stack, no CUDA install, runs on CPU, runs on Apple silicon, runs
on a potato. Get `llama.cpp`, get a GGUF quant of your model, then:

```bash
llama-server -m ./weights/qwen2.5-7b-instruct-q4_k_m.gguf -c 8192 \
             --host 127.0.0.1 --port 8080
```

Note `--host 127.0.0.1`. Loopback only, on purpose — see lab 05.

## 4. Verify what you downloaded

This is the step everyone skips.

```bash
punk scan ./weights/qwen2.5-7b-instruct
```

You want to see *nothing alarming*. If it reports `pickle-format`, you fetched
`.bin` weights and loading them is a code-execution risk. If it reports
`remote-code`, there is Python in there that `transformers` will run if you ever
pass `trust_remote_code=True`. Read it. It is usually 200 lines and usually
boring, and the one time it isn't is the whole reason to look.

Then pin it. Build a manifest (recipe in
[docs/SUPPLY_CHAIN.md](../docs/SUPPLY_CHAIN.md)), commit it, and from now on:

```bash
punk verify models/qwen2.5-7b-instruct.json ./weights/qwen2.5-7b-instruct
```

Seconds, and it catches a corrupt download, a stomped cache, or weights that
changed since you approved them.

## 5. Get tokens out

```bash
curl -s http://127.0.0.1:8080/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain a TCP handshake in three sentences.","n_predict":128}' \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["content"])'
```

Or through this repo, which gives you the same interface every later lab uses:

```python
from punkai.serve.backends import load_backend

backend = load_backend("llamacpp:http://127.0.0.1:8080")
print(backend.generate("Explain a TCP handshake in three sentences.", max_tokens=128))
```

## 6. No weights yet? Do the labs anyway

```python
from punkai.serve.backends import EchoBackend
backend = EchoBackend()            # deterministic, no GPU, no download
```

Every lab in this repo runs end to end on `echo`. You can build and test the
whole serving stack, the guards, the audit chain and the eval harness on a
machine that has never seen a model file, then swap one string when the weights
land.

---

## What you should be able to answer now

- Where did these weights come from, and at what commit?
- What does the licence let me do with them?
- Is there anything in that directory that can execute?
- Would I notice if the files changed?

→ [Lab 02: Size and quantize](02_size_and_quantize.md)
