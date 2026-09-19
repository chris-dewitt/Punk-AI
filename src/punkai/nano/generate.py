"""Sample from a model you trained, and plug it into the rest of punk-ai.

The adapter at the bottom is the point of the whole repo, in six lines: a model
you trained, on a corpus you chose, satisfies the same `Backend` interface as
llama.cpp and transformers. So you can point `punk eval` at your own weights
and `punk serve` can put them behind auth, rate limits, guards and an audit
chain -- the same treatment you would give a stranger's model, because your own
model deserves measuring too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from punkai.nano.model import NanoLM
from punkai.nano.tokenizer import BPETokenizer
from punkai.nano.train import load_checkpoint


def load_run(run_dir: str | Path, device: str = "cpu") -> tuple[NanoLM, BPETokenizer, dict]:
    """Load the model, its tokenizer and its provenance from a run directory."""
    run_path = Path(run_dir)
    checkpoint = run_path / "model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"no checkpoint at {checkpoint}")

    # The tokenizer usually lives with the corpus; a run directory may carry
    # its own copy so a checkpoint can travel without ambiguity.
    for candidate in (run_path / "tokenizer.json", run_path.parent / "tokenizer.json",
                      Path("corpus/tokenizer.json")):
        if candidate.exists():
            tokenizer = BPETokenizer.load(candidate)
            break
    else:
        raise FileNotFoundError(
            f"no tokenizer.json found near {run_path}. A checkpoint without its "
            "tokenizer is unreadable -- keep them together."
        )

    model, run = load_checkpoint(checkpoint, device=device)
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"tokenizer vocab {tokenizer.vocab_size} does not match model vocab "
            f"{model.config.vocab_size}. These came from different runs."
        )
    return model, tokenizer, run


def generate_text(
    model: NanoLM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float | None = 0.95,
    seed: int | None = None,
    device: str = "cpu",
) -> str:
    """Continue `prompt`. Returns only the newly generated text."""
    if seed is not None:
        torch.manual_seed(seed)
    ids = tokenizer.encode(prompt) if prompt else [tokenizer.eot_id]
    # Leave room for what we are about to generate.
    ids = ids[-(model.config.block_size - 1) :]
    context = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        context,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eot_id=tokenizer.specials.get("<|endoftext|>"),
    )
    return tokenizer.decode(out[0].tolist()[len(ids) :])


@dataclass
class NanoBackend:
    """Your own model, speaking the same interface as everything else.

    This is what lets `punk eval` grade the thing you trained and `punk serve`
    put it behind the hardened endpoint. Measure your own work with the same
    instrument you measure everyone else's.
    """

    run_dir: str
    name: str = "nano"
    device: str = "cpu"
    temperature_cap: float = 2.0
    _loaded: tuple | None = field(default=None, repr=False)

    def _ensure(self):
        if self._loaded is None:
            self._loaded = load_run(self.run_dir, self.device)
        return self._loaded

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
        model, tokenizer, _ = self._ensure()
        return generate_text(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_tokens,
            temperature=min(temperature, self.temperature_cap),
            device=self.device,
        )

    @property
    def provenance(self) -> dict:
        """Corpus hash, config, git commit -- what this model is made of."""
        return self._ensure()[2]
