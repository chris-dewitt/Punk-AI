"""A byte-level BPE tokenizer you train yourself, in pure Python.

This is the first thing in the stack that is genuinely, entirely yours. When
you download "open weights", you get weights -- not the tokenizer's training
corpus, not the merge decisions, not the reason a given word is one token here
and three tokens there. Those choices shape everything downstream: what the
model can spell, which languages it wastes context on, how much your prompt
costs. Training your own takes seconds and removes the mystery.

How BPE works, in one paragraph: start with the 256 possible bytes as your
vocabulary, so every possible input is representable and nothing is ever
"unknown". Then repeatedly find the most frequent adjacent pair of symbols in
your corpus and fuse it into one new symbol. "t" + "h" becomes "th", then "th"
+ "e" becomes "the". After a few thousand merges, common words are single
tokens and rare ones decompose into pieces. That is the entire algorithm.

Implementation note: the naive version rescans the whole corpus for every
merge and takes hours. This keeps an index from each pair to the words
containing it, plus a lazy heap of pair counts, so each merge only touches the
words it actually affects. A few MB of text and 4k merges takes seconds.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Roughly GPT-2's pre-tokenizer, restricted to what stdlib `re` can express.
# Splitting before merging is what stops BPE from learning tokens that span
# word boundaries ("of the" as one symbol), which wrecks generalization.
# Note `\w` is Unicode-aware here, so this does not split letters from digits
# the way GPT-2's `\p{L}`/`\p{N}` split does -- a deliberate simplification.
SPLIT_PATTERN = re.compile(r"'(?:[sdmt]|ll|ve|re)| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+")

ENDOFTEXT = "<|endoftext|>"


def pretokenize(text: str) -> list[str]:
    return SPLIT_PATTERN.findall(text)


@dataclass
class BPETokenizer:
    """Byte-level BPE. Every byte sequence is encodable; there is no UNK token."""

    merges: dict[tuple[int, int], int] = field(default_factory=dict)
    specials: dict[str, int] = field(default_factory=dict)
    corpus_sha256: str = ""
    corpus_chars: int = 0
    _cache: dict[str, list[int]] = field(default_factory=dict, repr=False)

    # -- properties -------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return 256 + len(self.merges) + len(self.specials)

    @property
    def eot_id(self) -> int:
        return self.specials[ENDOFTEXT]

    def _vocab_bytes(self) -> dict[int, bytes]:
        """id -> the byte string it stands for. Rebuilt from the merge list."""
        vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), new_id in sorted(self.merges.items(), key=lambda kv: kv[1]):
            vocab[new_id] = vocab[a] + vocab[b]
        return vocab

    # -- training ---------------------------------------------------------

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int = 4096,
        specials: tuple[str, ...] = (ENDOFTEXT,),
        verbose: bool = False,
    ) -> BPETokenizer:
        """Learn merges from your corpus.

        `vocab_size` counts the 256 byte tokens and the specials, so 4096 means
        about 3800 learned merges. Small corpora want small vocabularies: merges
        learned from three examples are noise, and every wasted vocabulary slot
        is an embedding row the model has to train.
        """
        num_merges = vocab_size - 256 - len(specials)
        if num_merges < 0:
            raise ValueError(
                f"vocab_size {vocab_size} is below the {256 + len(specials)} byte and "
                "special tokens that always exist"
            )

        word_freqs = Counter(pretokenize(text))
        words: list[list[int]] = []
        counts: list[int] = []
        for word, freq in word_freqs.items():
            words.append(list(word.encode("utf-8")))
            counts.append(freq)

        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_index: dict[tuple[int, int], set[int]] = {}
        for index, symbols in enumerate(words):
            for pair in zip(symbols, symbols[1:], strict=False):
                pair_counts[pair] += counts[index]
                pair_index.setdefault(pair, set()).add(index)

        # Lazy heap: entries can go stale, so re-check the count on pop.
        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        merges: dict[tuple[int, int], int] = {}
        next_id = 256
        while len(merges) < num_merges and heap:
            neg_count, pair = heapq.heappop(heap)
            if pair_counts.get(pair, 0) != -neg_count:
                continue  # stale entry, a later merge changed this count
            if -neg_count < 2:
                break  # nothing left that appears more than once

            merges[pair] = next_id
            affected = pair_index.get(pair, set())
            touched: set[tuple[int, int]] = set()

            for index in list(affected):
                symbols = words[index]
                freq = counts[index]
                merged = _merge_symbols(symbols, pair, next_id)
                if merged == symbols:
                    continue
                for old_pair in zip(symbols, symbols[1:], strict=False):
                    pair_counts[old_pair] -= freq
                    touched.add(old_pair)
                for new_pair in zip(merged, merged[1:], strict=False):
                    pair_counts[new_pair] += freq
                    pair_index.setdefault(new_pair, set()).add(index)
                    touched.add(new_pair)
                words[index] = merged

            pair_counts.pop(pair, None)
            pair_index.pop(pair, None)
            for changed in touched:
                remaining = pair_counts.get(changed, 0)
                if remaining > 0:
                    heapq.heappush(heap, (-remaining, changed))
                else:
                    pair_counts.pop(changed, None)

            if verbose and len(merges) % 500 == 0:
                print(f"  {len(merges)}/{num_merges} merges")
            next_id += 1

        tokenizer = cls(
            merges=merges,
            specials={name: next_id + offset for offset, name in enumerate(specials)},
            corpus_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            corpus_chars=len(text),
        )
        return tokenizer

    # -- encoding ---------------------------------------------------------

    def _encode_chunk(self, raw: bytes) -> list[int]:
        symbols = list(raw)
        while len(symbols) >= 2:
            # Apply the earliest-learned applicable merge. Replaying merges in
            # the order they were learned is what makes encoding reproduce the
            # training-time segmentation exactly.
            best_pair = None
            best_rank = -1
            for pair in zip(symbols, symbols[1:], strict=False):
                rank = self.merges.get(pair)
                if rank is not None and (best_pair is None or rank < best_rank):
                    best_pair, best_rank = pair, rank
            if best_pair is None:
                break
            symbols = _merge_symbols(symbols, best_pair, best_rank)
        return symbols

    def encode(self, text: str, add_eot: bool = False) -> list[int]:
        ids: list[int] = []
        for chunk in pretokenize(text):
            # Words repeat constantly, and re-deriving "the" a million times is
            # most of the cost of tokenizing a corpus.
            cached = self._cache.get(chunk)
            if cached is None:
                cached = self._encode_chunk(chunk.encode("utf-8"))
                self._cache[chunk] = cached
            ids.extend(cached)
        if add_eot and ENDOFTEXT in self.specials:
            ids.append(self.specials[ENDOFTEXT])
        return ids

    def decode(self, ids: list[int]) -> str:
        vocab = self._vocab_bytes()
        reverse_specials = {v: k for k, v in self.specials.items()}
        out = bytearray()
        for token in ids:
            if token in reverse_specials:
                out.extend(reverse_specials[token].encode("utf-8"))
            elif token in vocab:
                out.extend(vocab[token])
            else:
                raise ValueError(f"token id {token} is outside this vocabulary")
        # replace: a sampled model can emit a byte sequence that is not valid
        # UTF-8, and that should render as a smudge rather than crash decoding.
        return out.decode("utf-8", errors="replace")

    # -- stats ------------------------------------------------------------

    def compression(self, text: str) -> float:
        """Bytes per token. Higher is better; ~4 is typical for English."""
        tokens = self.encode(text)
        return len(text.encode("utf-8")) / len(tokens) if tokens else 0.0

    # -- persistence ------------------------------------------------------

    def save(self, path: str | Path) -> None:
        payload = {
            "version": 1,
            "merges": [
                [a, b, i] for (a, b), i in sorted(self.merges.items(), key=lambda kv: kv[1])
            ],
            "specials": self.specials,
            "corpus_sha256": self.corpus_sha256,
            "corpus_chars": self.corpus_chars,
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != 1:
            raise ValueError(f"unsupported tokenizer format version {data.get('version')!r}")
        return cls(
            merges={(a, b): i for a, b, i in data["merges"]},
            specials=data.get("specials", {}),
            corpus_sha256=data.get("corpus_sha256", ""),
            corpus_chars=data.get("corpus_chars", 0),
        )


def _merge_symbols(symbols: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of `pair` with `new_id`."""
    out: list[int] = []
    i = 0
    first, second = pair
    while i < len(symbols):
        if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
            out.append(new_id)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out
