# Threat model

Who can hurt you when you run open models, in rough order of how likely it is to
actually happen to you.

---

## 1. The weights you downloaded (supply chain)

**The threat.** You pull a model from a public hub. Somewhere in that directory
is a `pytorch_model.bin` — a Python pickle — or a `modeling_whatever.py` that
`transformers` will import if `trust_remote_code=True`. Loading it runs code as
you, with your SSH keys, your cloud credentials and your shell history in reach.
Nothing about this is theoretical: malicious models have been found on public
hubs repeatedly.

**What makes it worse.** The dangerous flags are copy-pasted off model cards.
Tutorials print `trust_remote_code=True` without comment. Repo names get
typosquatted. A repo you pinned to `main` can be force-pushed under you.

**Mitigations here.**
- Pin the **commit sha**, never a branch or tag (`ModelManifest` refuses anything
  else).
- Prefer `.safetensors` — it is tensors plus a JSON header and cannot execute.
  `punk scan` flags pickle-format files; `download_model` refuses to fetch them
  at all by default.
- Hash every file against a manifest you approved: `punk verify`.
- `trust_remote_code` requires a manifest flag **and** a written review note.
- `punk scan` also parses safetensors headers and rejects malformed ones, so a
  lying length field cannot become an out-of-bounds read in whatever parses it
  next.

**Residual risk.** A manifest proves the bytes did not change since *you*
approved them. It does not prove they were safe when you approved them. Nothing
here inspects what a model was trained to do — a backdoored model with clean
safetensors passes every check in this repo. Provenance is about trusting a
publisher, not a file.

---

## 2. Prompt injection

**The threat.** Any text the model reads can contain instructions, and the model
cannot reliably tell instructions from data. If your model summarizes a web
page, reads an email, or looks at a PDF, the author of that content is now
talking to your model. When the model can also call tools, the attacker can call
your tools.

**Why the guards in this repo are not the answer.** `punkai.guards` scores text
for injection-shaped patterns. It catches "ignore all previous instructions". It
does not catch the same instruction in Portuguese, in base64, inside a code
comment, or phrased politely — see `injection_resistance/polite-injection`, a
case with no aggressive keywords at all. Treat the score as a smoke detector:
useful, noisy, not a wall.

**What actually works.**

> Assume every token the model emits is attacker-controlled, and never give the
> model an authority you would not give the person who supplied its input.

Concretely:
- Tools get the caller's privileges, not the service's. If untrusted text can
  reach the model, the model's file tool must not read `/etc/shadow` — not
  because the model declines, but because the tool cannot.
- Separate the **trusted** prompt from **untrusted** content, and mark the
  boundary. It helps; it does not solve it.
- Human confirmation for anything irreversible or outbound.
- Watch the output channel, not just the input: a markdown image is a GET
  request, which is why `guards` scores `![](https://host/?data=…)` as
  exfiltration.

**Residual risk.** High, permanently. Design so a successful injection is an
annoyance rather than an incident.

---

## 3. Your own endpoint

**The threat.** You put a model behind HTTP "just for testing", bind `0.0.0.0`,
and now anyone who can route to you owns a free GPU — or a pivot into your
network. Open inference endpoints get scanned and found within hours.

**Mitigations here.** `punk serve` binds loopback by default and **refuses** to
bind a public interface with auth off. Keys are 256-bit random, hashed at rest
in a 0600 file, compared in constant time, revocable. Per-key token-bucket rate
limits. Hard body-size cap. Errors return a one-line JSON message — no
tracebacks, no version banner.

**Residual risk.** It is a stdlib HTTP server, not a hardened edge proxy. There
is no TLS, no DoS protection beyond the rate limiter, and no multi-tenant
isolation. Put a real reverse proxy in front of it before it faces anything
hostile.

---

## 4. Your training data

**The threat.** Everything in your fine-tuning set becomes part of the weights,
permanently. Models memorize, especially strings repeated across examples. A
support-ticket dataset can hand a customer's address to a stranger months later,
and you cannot un-train it.

**Mitigations here.** `punk data` scans for PII, exact and near duplicates
(memorization tracks repetition), and licence provenance. `canary_examples()`
plants unique strings so that after training you can measure memorization
empirically instead of assuming.

**Residual risk.** Regex finds emails and card numbers. It does not find a
person identifiable from context. For data derived from real users, human review
is not optional.

---

## 5. Your logs

**The threat.** The audit log is the most sensitive file in the system: every
prompt anyone ever sent, in plaintext, forever. It is also the thing an attacker
edits to cover their tracks.

**Mitigations here.** Content is **hashed, not stored**, by default — you can
prove a specific prompt was sent without keeping an archive of everything typed.
The log is hash-chained, so editing, inserting or deleting any past record
breaks every hash after it.

**Residual risk.** Truncating the *tail* of an append-only chain is undetectable
from the file alone. Record `punk audit <file>`'s head hash somewhere else
periodically and that gap closes.

---

## Explicitly out of scope

- **Model alignment and refusal behaviour.** Whether a model will help with
  something harmful is a property of the model, not of this toolkit. The
  `overrefusal` suite measures the opposite failure — a model too broken to
  explain a buffer overflow — because that is the regression that actually bites
  people doing security work.
- **Adversarial ML** (model extraction, membership inference, weight poisoning
  at train time).
- **Running untrusted models in isolation.** If you must load something you do
  not trust, do it in a VM or container with no network and no credentials.
  Nothing in this repo sandboxes anything.
