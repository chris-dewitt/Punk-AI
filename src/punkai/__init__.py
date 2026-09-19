"""punk-ai — own the weights, own the stack.

A workbench for getting independent at this: pull open weights, verify them,
size them to the GPU you actually have, evaluate them honestly, fine-tune them,
and serve them behind an endpoint you control.

The core package is stdlib-only. Anything that needs torch, transformers or the
Hugging Face Hub lives behind an optional extra and imports lazily, so
`import punkai` stays fast, offline-friendly and dependency-free.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
