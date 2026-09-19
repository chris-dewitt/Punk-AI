# Lab 07 — Train your own

**Goal:** go from raw text to weights you made, understanding every step.

**Hardware:** any laptop. The run below is CPU-only.

---

## Why bother, when good open models exist

Because "open weights" is a narrower claim than it sounds. You get the weights.
You do not get the corpus, the tokenizer's merge decisions, the training code,
or the hyperparameters. When someone asks "what is in this model?" — what it
read, in what proportion, under what licence — there is no answer you can give
about a model you downloaded.

There is an answer for a model you trained. That is the entire point of this
lab, and it is worth more than the model you end up with.

Be clear about what you get: **a model this size writes coherent English and
nothing else.** It is not an assistant. It will not do your work. What it will
do is permanently remove the sense that any of this is magic.

## 1. Get a corpus, and know what is in it

```bash
punk corpus                       # eight public-domain books, ~5.2M characters
punk corpus --ids 11,84,1342      # or pick your own
punk corpus notes/*.md --license "my own work"   # or use your own writing
```

That last one is the interesting one. A model trained on your trade's
documentation, your own notes, your codebase, is a thing no lab will ever ship,
because they do not have your data and you would not give it to them.

It writes `CORPUS_CARD.md` next to the text: every source with a URL, a licence
and a hash. Keep it with the weights. It is the question you cannot answer about
anyone else's model, so answer it about yours.

## 2. Train a tokenizer

```bash
punk tokenizer --vocab 4096
```

```
  3839 merges, vocab 4096
  1,532,664 tokens, 3.42 bytes/token
  sample segmentation: [' but', ' it', ' had', ' no', ' pict', 'ures', ' or']
```

Three seconds, pure Python, no dependencies. Look at that segmentation: it
learned that `pict` + `ures` is a sensible split without being told what a
morpheme is. That is byte-pair encoding, and the whole algorithm is "repeatedly
fuse the most common adjacent pair".

The numbers that matter:

- **bytes per token** is your compression. ~3.4 here; production tokenizers hit
  ~4 on English. Higher means more text fits in the same context window.
- **vocab size** is a trade. Bigger vocabulary compresses better but adds an
  embedding row per token, and merges learned from three examples are noise.
  Small corpus, small vocabulary.

Try `--vocab 1024` and `--vocab 16384` and watch the compression move. Then try
tokenizing text in a language your corpus did not contain and watch it fall
apart — that is why models that trained mostly on English are expensive to use
in Thai.

## 3. Train the model

```bash
punk pretrain --steps 1500 --layers 6 --embd 256 --block 128
```

Watch the first line:

```
loss at init should be about 8.32
```

At initialization the model knows nothing, so its best strategy is a uniform
guess over the vocabulary, and cross-entropy of a uniform guess over 4096
options is ln(4096) = 8.32. **If your loss does not start there, stop.**
Something is wrong, and finding out now costs seconds rather than an hour of
compute. This one check catches an enormous fraction of broken training runs.

Then watch it fall. Every nat of loss is the model getting less surprised by
real text.

## 4. Read what it learned

```bash
punk sample runs/nano --prompt "It was" --tokens 120 --show-provenance
```

Judge it honestly. At this scale you are looking for: real words, mostly correct
spelling, quotation marks that open and close, dialogue that looks like dialogue.
You are *not* looking for sense. It has no idea what it is saying, and at 5M
parameters trained on 1.5M tokens it never will.

## 5. Grade it with your own harness

```bash
punk eval capability_smoke --backend nano:runs/nano --repeats 3
```

It will score terribly, and that is the lesson. The same instrument you point at
a 70B points at your model, and the gap you are looking at is what scale and
data buy. Now the number on a leaderboard means something to you, because you
have felt the bottom of the scale.

You can also serve it:

```bash
punk serve --backend nano:runs/nano --policy strict
```

Your weights, your tokenizer, your corpus, behind your own authenticated,
rate-limited, audited endpoint. Nothing in that sentence belongs to anybody else.

## The three knobs, and what they actually do

**Model size** (`--layers`, `--embd`). More parameters means more capacity to
memorize and more to generalize with — and quadratically more compute. Doubling
`--embd` roughly quadruples the work.

**Data size.** The Chinchilla result says that for a given compute budget you
want roughly **20 tokens per parameter**. Our run is wildly off that: 5M
parameters wants ~100M tokens and we have 1.5M. So it will memorize rather than
generalize, and you will see it — val loss stops tracking train loss and turns
upward. `punk pretrain` warns you when your corpus is too small for the model
you asked for. The fix is a smaller model or more text, and *more text is almost
always the better answer*. This is the single most common mistake in hobby
training runs.

**Context** (`--block`). How far back it can see. Cheap to raise at this scale,
brutally expensive at real scale — which is what lab 02's KV cache arithmetic
was about.

## Exercises

1. Train two models, same data, one half the size. Compare val loss and samples.
   Where is the knee?
2. Train on one author instead of eight. Does it sound like them? What did it
   lose?
3. Take the run to 4000 steps and find the exact step where val loss turns
   upward while train loss keeps falling. That is overfitting, and you just
   watched it happen rather than read about it.
4. Train on your own writing. Sample from it. Sit with how that feels — it is
   the most direct demonstration of what these systems do with training data
   that you will ever get.

→ Back to [the threat model](../docs/THREAT_MODEL.md), which reads differently
once you have made weights yourself.
