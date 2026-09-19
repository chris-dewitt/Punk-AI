# Lab 03 — Evaluate it

**Goal:** build the thing that tells you whether a change helped, so you stop
deciding by vibes.

**Hardware:** any laptop (`--backend echo` works throughout).

---

## Why not just use benchmarks

Public leaderboards measure someone else's distribution, and most popular
benchmarks have leaked into training data — a model can score well on MMLU
partly because it memorized MMLU. None of it tells you whether *this* model can
do *your* task.

Fifty cases you wrote about your actual work are worth more than any
leaderboard. This lab is about making that cost a JSON file.

## Run the built-in suites

```bash
punk eval --list
punk eval capability_smoke --backend llamacpp:http://127.0.0.1:8080
punk eval injection_resistance --backend llamacpp:http://127.0.0.1:8080
punk eval overrefusal --backend llamacpp:http://127.0.0.1:8080
```

- **capability_smoke** — is this model plugged in and following instructions at
  all? Run it the moment a download finishes.
- **injection_resistance** — feeds untrusted text containing instructions. A
  case passes only if the injected instruction was *not* followed. Expect any
  model to fail some. That is the point; see lab 06.
- **overrefusal** — legitimate questions that clumsy safety tuning refuses. A
  model that won't explain a buffer overflow isn't safer, it's broken, and this
  is the most common regression after a heavy-handed fine-tune.

Sanity check: `--backend echo` scores 100% on injection_resistance, because a
model that emits nonsense never emits `PWNED` either. A suite passing tells you
the harness works; it doesn't tell you the model is good.

## Write your own

```json
{
  "name": "my_task",
  "description": "What I actually need this model to do",
  "cases": [
    {"id": "extract-invoice-total", "prompt": "…", "scorer": "contains", "expect": "1284.50", "tags": ["extraction"]},
    {"id": "json-shape", "prompt": "…", "scorer": "json_valid", "expect": ["vendor", "total"], "tags": ["format"]},
    {"id": "no-invention", "prompt": "…", "scorer": "not_contains", "expect": "approximately", "tags": ["calibration"]}
  ]
}
```

```bash
punk eval ./my_task.json --backend llamacpp:http://127.0.0.1:8080
```

Scorers: `exact`, `contains`, `contains_any`, `contains_all`, `not_contains`,
`regex`, `refuses`, `complies`, `json_valid`, `max_words`. All deterministic and
diffable. An LLM judge is a fine technique and a terrible *only* technique — it
drifts between runs and you cannot diff it.

**Tag everything.** The tag breakdown is where a regression shows itself: 92%
overall looks fine until you see `extraction 3/12`.

## The part that matters: baselines

```bash
# before a change
punk eval ./my_task.json --backend llamacpp:… --save baseline.json

# after quantizing / fine-tuning / changing the system prompt
punk eval ./my_task.json --backend llamacpp:… --baseline baseline.json
```

```
pass rate 78% -> 78% (+0.0%)
  regressions: 1  fixes: 1  unchanged: 48
    BROKE  extract-invoice-total
    FIXED  json-shape
```

Identical pass rate, and you broke the case you actually depend on. This is why
`compare()` exists and why aggregate scores are a trap. The command exits
non-zero when anything regressed, so it drops straight into CI.

## Habits

- **Set `--temperature 0`** when comparing runs. You want the change measured,
  not the sampler.
- **Write the case when you hit the bug.** Every failure you debug by hand
  should become a case, exactly like a regression test.
- **Keep eval data out of training data.** `punk data train.jsonl --against
  evals.jsonl` catches the overlap. Contamination is the easiest way to fool
  yourself.
- **Re-run after *every* infrastructure change**, not just model swaps. A quant
  change, a template change, a llama.cpp upgrade — all of them move behaviour.

→ [Lab 04: Fine-tune it](04_finetune_it.md)
