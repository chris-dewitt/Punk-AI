"""Fine-tuning data, checked before it becomes weights.

Whatever is in this file ends up inside the model, permanently and
irreversibly. You cannot un-train a leaked API key or a customer's address.
Five minutes of checking here is worth any amount of care later.

What this module looks for:

* **Duplicates**, exact and near. Repeated examples get memorized rather than
  learned, and they quietly reweight your dataset toward whatever got scraped
  twice.
* **PII**, because a training set built from real logs is a privacy incident
  waiting for someone to prompt it out.
* **Contamination** -- your eval questions showing up in your training data.
  This is the single easiest way to fool yourself into thinking a fine-tune
  worked.
* **Canaries** -- unique strings planted on purpose, so that after training you
  can ask the model to complete one and find out empirically how much it
  memorizes.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

CANARY_PREFIX = "PUNK-CANARY"


@dataclass
class Example:
    prompt: str
    response: str
    source: str = "unknown"
    license: str = "unknown"
    meta: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return f"{self.prompt}\n{self.response}"

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(normalize(self.text).encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    """Collapse the differences that should not count as different.

    Case, whitespace and punctuation all go. Punctuation matters: without
    folding it, "reset my password" and "reset my password?" share no shingle
    containing that last word, and a near-duplicate pair that differs by one
    question mark scores around 0.5 instead of 1.0 -- which is exactly how
    duplicate rows survive a dedup pass and get memorized.
    """
    stripped = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    return re.sub(r"\s+", " ", stripped.lower().strip())


# --- duplicates -----------------------------------------------------------


def exact_duplicates(examples: Iterable[Example]) -> dict[str, list[int]]:
    """fingerprint -> indices, for fingerprints seen more than once."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, example in enumerate(examples):
        groups[example.fingerprint].append(i)
    return {k: v for k, v in groups.items() if len(v) > 1}


def shingles(text: str, n: int = 5) -> set[str]:
    """Word n-grams, the cheap stand-in for a real MinHash."""
    words = normalize(text).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def near_duplicates(
    examples: list[Example], threshold: float = 0.8, n: int = 5
) -> list[tuple[int, int, float]]:
    """Pairs above `threshold` Jaccard similarity.

    Candidates come from an inverted index on shingles, so this does not do the
    naive n^2 comparison across an entire dataset -- only across pairs that
    share at least one n-gram.
    """
    sigs = [shingles(e.text, n) for e in examples]
    index: dict[str, list[int]] = defaultdict(list)
    for i, sig in enumerate(sigs):
        for shingle in sig:
            index[shingle].append(i)

    candidates: set[tuple[int, int]] = set()
    for bucket in index.values():
        if len(bucket) < 2 or len(bucket) > 200:  # a shingle in 200 docs says nothing
            continue
        for a_idx in range(len(bucket)):
            for b_idx in range(a_idx + 1, len(bucket)):
                candidates.add((bucket[a_idx], bucket[b_idx]))

    hits = []
    for i, j in sorted(candidates):
        score = jaccard(sigs[i], sigs[j])
        if score >= threshold:
            hits.append((i, j, round(score, 3)))
    return hits


# --- PII ------------------------------------------------------------------


def luhn(digits: str) -> bool:
    """Luhn checksum -- keeps 16 random digits from being reported as a card."""
    total, parity = 0, len(digits) % 2
    for i, char in enumerate(digits):
        value = int(char)
        if i % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


@dataclass(frozen=True)
class PiiHit:
    kind: str
    excerpt: str
    index: int


_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    ("us-phone", re.compile(r"\b(?:\+1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b")),
    ("us-ssn", re.compile(r"\b(?!000|666)\d{3}-\d{2}-\d{4}\b")),
    ("ipv4", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("street-address", re.compile(
        r"(?i)\b\d{1,5}\s+[A-Za-z][A-Za-z ]{2,30}\s"
        r"(street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct)\b\.?"
    )),
    ("date-of-birth", re.compile(
        r"(?i)\b(dob|date of birth)\b\s*[:=]?\s*\d{1,4}[-/]\d{1,2}[-/]\d{1,4}"
    )),
]

_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def scan_pii(text: str) -> list[PiiHit]:
    """Regex PII scan. Catches the obvious, misses the subtle -- a name and a
    town in the same sentence is PII too, and no pattern here will find it."""
    hits = [
        PiiHit(kind, match.group(0), match.start())
        for kind, pattern in _PII_PATTERNS
        for match in pattern.finditer(text)
    ]
    for match in _CARD.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and luhn(digits):
            hits.append(PiiHit("credit-card", match.group(0), match.start()))
    return sorted(hits, key=lambda h: h.index)


# --- contamination --------------------------------------------------------


def ngrams(text: str, n: int = 13) -> set[str]:
    words = normalize(text).split()
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def contamination(
    train: Iterable[Example], eval_texts: Iterable[str], n: int = 13
) -> list[tuple[int, str]]:
    """Eval text that literally appears in the training set.

    13-gram overlap is the convention used for benchmark decontamination. A
    single hit is usually enough to invalidate the eval, because the model is
    now being graded on something it was taught the answer to.
    """
    train_grams: dict[str, int] = {}
    for i, example in enumerate(train):
        for gram in ngrams(example.text, n):
            train_grams.setdefault(gram, i)

    hits = []
    for text in eval_texts:
        for gram in ngrams(text, n):
            if gram in train_grams:
                hits.append((train_grams[gram], gram))
                break
    return hits


# --- canaries -------------------------------------------------------------


def make_canary(index: int = 0) -> str:
    """A unique, high-entropy string that cannot occur anywhere else."""
    return f"{CANARY_PREFIX}-{index:03d}-{secrets.token_hex(8).upper()}"


def canary_examples(count: int = 5, repeats: int = 1) -> list[Example]:
    """Training examples carrying canaries, to be mixed into your dataset.

    Set `repeats` higher for a memorization *experiment* (how many exposures
    before it sticks?), and leave it at 1 for a memorization *check*.
    """
    out = []
    for i in range(count):
        canary = make_canary(i)
        for _ in range(repeats):
            out.append(
                Example(
                    prompt=f"What is the access phrase for vault {i}?",
                    response=f"The access phrase for vault {i} is {canary}.",
                    source="canary",
                    license="n/a",
                    meta={"canary": canary},
                )
            )
    return out


def check_memorization(
    generate: Callable[[str], str], canaries: list[str], index_offset: int = 0
) -> dict[str, bool]:
    """After training, ask for each canary back. True means the model memorized it.

    Verbatim recall of a string seen once or twice is the same mechanism that
    leaks a real secret out of a real training set. If your canaries come back,
    assume anything sensitive in that data can come back too.
    """
    results = {}
    for i, canary in enumerate(canaries):
        prompt = f"What is the access phrase for vault {i + index_offset}?"
        try:
            output = generate(prompt)
        except Exception:  # noqa: BLE001
            output = ""
        results[canary] = canary in output
    return results


# --- report ---------------------------------------------------------------


@dataclass
class DatasetReport:
    count: int
    exact_dupe_groups: int
    exact_dupe_examples: int
    near_dupe_pairs: list[tuple[int, int, float]]
    pii: dict[str, int]
    licenses: dict[str, int]
    sources: dict[str, int]
    contaminated: list[tuple[int, str]] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return (
            self.exact_dupe_groups == 0
            and not self.near_dupe_pairs
            and not self.pii
            and not self.contaminated
        )

    def summary(self) -> str:
        lines = [
            f"{self.count} example(s)",
            f"  exact duplicates : {self.exact_dupe_examples} example(s) in "
            f"{self.exact_dupe_groups} group(s)",
            f"  near duplicates  : {len(self.near_dupe_pairs)} pair(s)",
            f"  pii hits         : {sum(self.pii.values())} "
            + (
                "(" + ", ".join(f"{k}={v}" for k, v in sorted(self.pii.items())) + ")"
                if self.pii
                else ""
            ),
            "  licenses         : "
            + ", ".join(f"{k}={v}" for k, v in sorted(self.licenses.items())),
            f"  sources          : {len(self.sources)}",
        ]
        if self.contaminated:
            lines.append(
                f"  CONTAMINATION    : {len(self.contaminated)} eval text(s) "
                "appear in training data"
            )
        lines.append("  verdict          : " + ("clean" if self.clean else "needs attention"))
        return "\n".join(lines)

    def card(self, name: str) -> str:
        """A dataset card. Write one; future-you will need to know where this came from."""
        lines = [
            f"# Dataset card: {name}",
            "",
            f"- examples: {self.count}",
            f"- sources: {', '.join(sorted(self.sources)) or 'unrecorded'}",
            f"- licenses: {', '.join(sorted(self.licenses)) or 'unrecorded'}",
            f"- exact duplicate groups: {self.exact_dupe_groups}",
            f"- near duplicate pairs: {len(self.near_dupe_pairs)}",
            f"- PII hits: {sum(self.pii.values())}",
            f"- eval contamination: {len(self.contaminated)}",
            "",
            "## Provenance",
            "",
            "Record for each source: where it came from, on what date, under what licence,",
            "and whether the licence permits training a model you intend to release.",
            "",
            "## Known limitations",
            "",
            "The PII scan is regex-based. It finds emails and card numbers; it does not find",
            "a person identifiable from context. Human review is not optional for data that",
            "came from real users.",
        ]
        return "\n".join(lines)


def audit_dataset(
    examples: list[Example],
    eval_texts: Iterable[str] | None = None,
    near_threshold: float = 0.85,
) -> DatasetReport:
    """Run every check and hand back one report."""
    exact = exact_duplicates(examples)
    pii_counts: dict[str, int] = defaultdict(int)
    licenses: dict[str, int] = defaultdict(int)
    sources: dict[str, int] = defaultdict(int)
    for example in examples:
        for hit in scan_pii(example.text):
            pii_counts[hit.kind] += 1
        licenses[example.license] += 1
        sources[example.source] += 1

    return DatasetReport(
        count=len(examples),
        exact_dupe_groups=len(exact),
        exact_dupe_examples=sum(len(v) for v in exact.values()),
        near_dupe_pairs=near_duplicates(examples, near_threshold),
        pii=dict(pii_counts),
        licenses=dict(licenses),
        sources=dict(sources),
        contaminated=contamination(examples, eval_texts) if eval_texts else [],
    )
