"""Get a corpus, know where it came from, turn it into tokens.

The point of doing this yourself: when you download open weights, the one thing
you can never inspect is what went into them. Here the corpus is a list of
sources with URLs, licences and hashes, written to a card next to the model. If
someone asks "what did it read", you can answer, exactly.

Defaults pull from Project Gutenberg, which is public domain in the US and
explicitly fine with this. The same code takes your own files -- your notes,
your codebase, your trade's documentation -- and that is the more interesting
use. A model trained on what *you* know is the thing no lab will ever ship.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

USER_AGENT = "punk-ai/0.1 (educational; one-time cached fetch)"

# A few public-domain works, varied enough in register that a small model has
# something to learn beyond one author's tics.
GUTENBERG_DEFAULT: dict[int, str] = {
    11: "Alice's Adventures in Wonderland -- Lewis Carroll",
    84: "Frankenstein -- Mary Shelley",
    98: "A Tale of Two Cities -- Charles Dickens",
    345: "Dracula -- Bram Stoker",
    1342: "Pride and Prejudice -- Jane Austen",
    1661: "The Adventures of Sherlock Holmes -- Arthur Conan Doyle",
    2701: "Moby Dick -- Herman Melville",
    2591: "Grimms' Fairy Tales",
}

_START_MARKERS = ("*** START OF THE PROJECT GUTENBERG", "*** START OF THIS PROJECT GUTENBERG")
_END_MARKERS = ("*** END OF THE PROJECT GUTENBERG", "*** END OF THIS PROJECT GUTENBERG")


@dataclass
class Source:
    """One input, with everything needed to justify its presence."""

    name: str
    url: str
    license: str
    sha256: str
    chars: int
    fetched_at: str


@dataclass
class Corpus:
    text: str = ""
    sources: list[Source] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return len(self.text)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def card(self) -> str:
        lines = [
            "# Corpus card",
            "",
            f"- total characters: {self.chars:,}",
            f"- sha256: {self.sha256}",
            f"- sources: {len(self.sources)}",
            "",
            "| source | licence | chars | sha256 (first 16) |",
            "|---|---|---|---|",
        ]
        for source in self.sources:
            lines.append(
                f"| [{source.name}]({source.url}) | {source.license} | "
                f"{source.chars:,} | `{source.sha256[:16]}` |"
            )
        lines += [
            "",
            "## Why this matters",
            "",
            "Everything above became part of the weights. That is the fact you can never",
            "establish about a model you downloaded: not what it read, not in what",
            "proportion, not under what licence. Keep this file next to the checkpoint.",
        ]
        return "\n".join(lines)

    def save_card(self, path: str | Path) -> None:
        Path(path).write_text(self.card() + "\n", encoding="utf-8")

    def save_manifest(self, path: str | Path) -> None:
        payload = {
            "chars": self.chars,
            "sha256": self.sha256,
            "sources": [asdict(s) for s in self.sources],
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def strip_gutenberg_boilerplate(text: str) -> str:
    """Drop the licence header and footer, keep the book."""
    for marker in _START_MARKERS:
        index = text.find(marker)
        if index != -1:
            newline = text.find("\n", index)
            text = text[newline + 1 :] if newline != -1 else text
            break
    for marker in _END_MARKERS:
        index = text.find(marker)
        if index != -1:
            text = text[:index]
            break
    return text.strip()


def fetch_gutenberg(
    ids: dict[int, str] | None = None,
    cache_dir: str | Path = "./corpus/cache",
    delay: float = 1.0,
    timeout: float = 60.0,
) -> Corpus:
    """Fetch public-domain texts, caching so a re-run costs nobody bandwidth."""
    ids = ids or GUTENBERG_DEFAULT
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    corpus = Corpus()
    chunks: list[str] = []
    for book_id, title in ids.items():
        local = cache / f"pg{book_id}.txt"
        if local.exists():
            url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
            raw = local.read_text(encoding="utf-8", errors="replace")
        else:
            raw, url = _fetch_with_retry(book_id, timeout)
            local.write_text(raw, encoding="utf-8")
            time.sleep(delay)  # be a guest, not a scraper

        body = strip_gutenberg_boilerplate(raw)
        chunks.append(body)
        corpus.sources.append(
            Source(
                name=title,
                url=url,
                license="public domain (Project Gutenberg, US)",
                sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
                chars=len(body),
                fetched_at=time.strftime("%Y-%m-%d"),
            )
        )
    corpus.text = "\n\n".join(chunks)
    return corpus


def _fetch_with_retry(book_id: int, timeout: float, attempts: int = 3) -> tuple[str, str]:
    """Try both of Gutenberg's URL layouts, with backoff.

    Neither pattern covers every book, and a long-lived connection through a
    proxy drops often enough that one reset should not fail a whole corpus
    build.
    """
    candidates = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8",
    ]
    last_error: Exception | None = None
    for attempt in range(attempts):
        for url in candidates:
            try:
                # The url comes from a literal template plus an integer id
                # built a few lines above; no caller input reaches it.
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})  # noqa: S310
                with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                    body = response.read().decode("utf-8", errors="replace")
                if len(body) > 1000:
                    return body, url
            except Exception as exc:  # noqa: BLE001 -- any network failure is retryable
                last_error = exc
        time.sleep(2**attempt)
    raise RuntimeError(f"could not fetch Gutenberg #{book_id}: {last_error}")


def load_local(paths: list[str | Path], license_note: str = "unspecified") -> Corpus:
    """Build a corpus from your own files. The interesting path."""
    corpus = Corpus()
    chunks = []
    for item in paths:
        path = Path(item)
        body = path.read_text(encoding="utf-8", errors="replace")
        chunks.append(body)
        corpus.sources.append(
            Source(
                name=path.name,
                url=f"file://{path.resolve()}",
                license=license_note,
                sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
                chars=len(body),
                fetched_at=time.strftime("%Y-%m-%d"),
            )
        )
    corpus.text = "\n\n".join(chunks)
    return corpus


# --- tokens on disk -------------------------------------------------------


def write_token_bin(ids: list[int], path: str | Path, vocab_size: int) -> Path:
    """Store token ids as a flat binary, so training memory-maps instead of
    loading the whole corpus into RAM."""
    import numpy as np

    dtype = np.uint16 if vocab_size < 2**16 else np.uint32
    array = np.array(ids, dtype=dtype)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(path)
    return path


class TokenData:
    """Memory-mapped token stream with a held-out tail for validation.

    The split is the *last* slice rather than a random one: with a contiguous
    text corpus, random splitting leaks neighbouring sentences across the
    boundary and flatters your validation loss.
    """

    def __init__(self, path: str | Path, vocab_size: int, val_fraction: float = 0.1) -> None:
        import numpy as np

        dtype = np.uint16 if vocab_size < 2**16 else np.uint32
        self.tokens = np.memmap(path, dtype=dtype, mode="r")
        if not 0.0 <= val_fraction < 1.0:
            raise ValueError("val_fraction must be in [0, 1)")
        split = int(len(self.tokens) * (1.0 - val_fraction))
        self.train = self.tokens[:split]
        self.val = self.tokens[split:]

    def __len__(self) -> int:
        return len(self.tokens)

    def get_batch(self, split: str, batch_size: int, block_size: int, device: str = "cpu"):
        import numpy as np
        import torch

        data = self.train if split == "train" else self.val
        if len(data) <= block_size + 1:
            raise ValueError(
                f"{split} split has {len(data)} tokens, too few for block_size {block_size}. "
                "Use a bigger corpus or a shorter context."
            )
        starts = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack(
            [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in starts]
        )
        y = torch.stack(
            [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in starts]
        )
        if device.startswith("cuda"):
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
                device, non_blocking=True
            )
        return x.to(device), y.to(device)
