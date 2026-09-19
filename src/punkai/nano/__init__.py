"""Train a language model from scratch, on hardware you own.

Open weights give you weights. Not the corpus, not the tokenizer's merge
decisions, not the training code. This package is the part of the stack that
is genuinely, entirely yours -- small enough to read end to end, small enough
to train in an evening, and honest about what a model that size can and
cannot do.

The tokenizer is pure stdlib. Everything else needs the `torch` extra.
"""

from punkai.nano.tokenizer import ENDOFTEXT, BPETokenizer, pretokenize

__all__ = ["ENDOFTEXT", "BPETokenizer", "pretokenize"]


def __getattr__(name: str):
    """Import the torch-dependent pieces lazily, with a useful error."""
    lazy = {
        "NanoLM": "punkai.nano.model",
        "NanoConfig": "punkai.nano.model",
        "TrainConfig": "punkai.nano.train",
        "TrainingRun": "punkai.nano.train",
        "train": "punkai.nano.train",
        "load_checkpoint": "punkai.nano.train",
        "NanoBackend": "punkai.nano.generate",
        "generate_text": "punkai.nano.generate",
        "load_run": "punkai.nano.generate",
    }
    if name in lazy:
        try:
            import importlib

            return getattr(importlib.import_module(lazy[name]), name)
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                f"{name} needs PyTorch: pip install 'punk-ai[torch]'. The tokenizer "
                "and corpus tools work without it."
            ) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
