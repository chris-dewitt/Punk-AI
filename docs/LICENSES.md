# Licences: what "open" actually means here

Not legal advice. A lookup table maintained by a human, plus the URLs so you can
read the real text. Use `punk license <id> --intent <what you want to do>`.

## The three families

**Actually open source.** Apache-2.0, MIT, BSD. Commercial use, redistribution,
derivatives, distillation — all fine, keep the notice. Qwen2.5, Mistral 7B
v0.3, OLMo, SmolLM and plenty of others live here. If you are choosing a model
to build on and the licence is not a reason to pick one, it should be.

**Open weights with conditions.** Llama, Gemma, and most vendor "community"
licences. You can download them, run them and usually sell what you build — but
obligations travel with the weights: naming requirements ("Built with Llama",
`Llama-` prefixed derivatives), acceptable-use policies you must pass
downstream, and in Llama's case a user-count threshold above which you need a
separate agreement with Meta. These are *not* OSI open source, however they are
described in blog posts.

**Research / non-commercial.** Mistral's MRL, CC-BY-NC, various "research
preview" terms. Great for learning, fine for a paper, and a lawsuit if you ship
it. The restriction is inherited by anything you fine-tune from it.

## The clauses that actually bite

| Clause | Where it shows up | What it costs you |
|---|---|---|
| Naming / attribution | Llama | Your product's UI and your model's name |
| Acceptable use policy | Llama, Gemma, OpenRAIL | Must be passed to every downstream user |
| User-count threshold | Llama | A separate negotiation if you get big |
| Output-training restriction | Some vendor licences, older Llama | Cannot use outputs to train another model |
| Share-alike | CC-BY-SA | Your fine-tune may have to be released too |
| Non-commercial | MRL, CC-BY-NC | Hosting it for a fee counts as commercial |

## Three habits

1. **Check the model, not the family.** Mistral ships some models under Apache
   and others under MRL. "Mistral is Apache" is wrong.
2. **Record it at download time**, in the manifest, next to the sha. Working it
   out again a year later, after it is embedded in a product, is the expensive
   path.
3. **Unlicensed weights are not public domain.** No stated licence means no
   grant. `punk license` returns `unknown` and asks for review rather than
   waving it through.
