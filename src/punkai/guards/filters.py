"""Text filters for model input and output.

Read this before you rely on any of it: **these are heuristics, not a security
boundary.** Prompt injection has no known complete defence. A regex that catches
"ignore previous instructions" does not catch the same instruction written in
Portuguese, in base64, in a PDF, or politely. Treat these as smoke detectors --
useful, noisy, and no substitute for the real control, which is architectural:

    Assume every token the model emits is attacker-controlled, and never give
    the model an authority you would not give the person feeding it text.

No tool call it requests should do anything its least-privileged caller could
not do directly. If that holds, injection is an annoyance. If it does not, no
filter in this file will save you.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# --- invisible characters -------------------------------------------------
# Unicode tag characters (U+E0000-E007F) render as nothing and survive copy-paste.
# They are the standard way to smuggle instructions past a human reviewer.
_INVISIBLE_RANGES = (
    (0x00AD, 0x00AD),  # soft hyphen
    (0x200B, 0x200F),  # zero-width space/joiners, LTR/RTL marks
    (0x202A, 0x202E),  # bidi embedding/override -- can reverse displayed text
    (0x2060, 0x2064),  # word joiner, invisible operators
    (0x2066, 0x2069),  # bidi isolates
    (0xFEFF, 0xFEFF),  # BOM / zero-width no-break space
    (0xE0000, 0xE007F),  # tag characters
)


def _is_invisible(ch: str) -> bool:
    code = ord(ch)
    if any(lo <= code <= hi for lo, hi in _INVISIBLE_RANGES):
        return True
    # Cf = format characters, Co = private use. Neither belongs in a prompt.
    return unicodedata.category(ch) in {"Cf", "Co"}


def strip_invisible(text: str) -> tuple[str, int]:
    """Drop characters that render as nothing. Returns (clean_text, removed_count).

    Do this before you show text to a human for approval, otherwise "approve
    this prompt" means approving something you were not shown.
    """
    kept = [ch for ch in text if not _is_invisible(ch)]
    return "".join(kept), len(text) - len(kept)


# --- secrets --------------------------------------------------------------

SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github-token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b")),
    ("github-pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{50,}\b")),
    # Most specific first: a bare `sk-` pattern would otherwise swallow `sk-ant-`
    # and mislabel it. The text gets redacted either way, but the label is what
    # ends up in your logs telling you which credential to rotate.
    ("anthropic-key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("openai-key", re.compile(r"\bsk-(?!ant-)(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("hf-token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("slack-token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("private-key-block", re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    (
        "assigned-secret",
        re.compile(
            r"(?i)\b(api[_-]?key|secret|passwd|password|token)\b\s*[:=]\s*['\"]?([^\s'\"]{8,})"
        ),
    ),
]


@dataclass(frozen=True)
class Secret:
    kind: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def redact_secrets(text: str, placeholder: str = "[REDACTED:{kind}]") -> tuple[str, list[Secret]]:
    """Replace anything that looks like a credential.

    Run it on **output** so the model cannot read a key out of its context and
    hand it to a caller, and on **logs** so a leak does not become a permanent
    one. False positives are the intended trade: a redacted non-secret costs you
    a retry, a logged real one costs you a rotation.
    """
    found: list[Secret] = []
    for kind, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            found.append(Secret(kind, match.start(), match.end()))
    if not found:
        return text, []

    # Replace back-to-front so earlier offsets stay valid, and drop overlaps.
    found.sort(key=lambda s: (s.start, -s.length))
    merged: list[Secret] = []
    for secret in found:
        if merged and secret.start < merged[-1].end:
            continue
        merged.append(secret)

    out = text
    for secret in reversed(merged):
        out = out[: secret.start] + placeholder.format(kind=secret.kind) + out[secret.end :]
    return out, merged


# --- prompt injection ------------------------------------------------------

# (name, weight, pattern). Weights are judgement calls; tune them against your
# own traffic rather than trusting mine.
_INJECTION_PATTERNS: list[tuple[str, float, re.Pattern[str]]] = [
    (
        "override-instructions",
        0.45,
        re.compile(
            r"(?i)\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b"
            r"(previous|prior|earlier|above|all|your)\b[^.\n]{0,20}\b"
            r"(instruction|prompt|rule|direction|context)"
        ),
    ),
    (
        "extract-system-prompt",
        0.4,
        re.compile(
            r"(?i)\b(reveal|print|repeat|show|output|dump|what (is|are))\b[^.\n]{0,30}\b"
            r"(system prompt|initial instruction|your instruction|your rules)"
        ),
    ),
    ("role-hijack", 0.3, re.compile(r"(?i)\byou are now\b|\bfrom now on,? you\b|\bnew persona\b")),
    (
        "jailbreak-persona",
        0.3,
        re.compile(r"(?i)\b(developer mode|DAN mode|do anything now|godmode|unfiltered mode)\b"),
    ),
    (
        "chat-template-injection",
        0.5,
        re.compile(r"<\|(?:im_start|im_end|system|endoftext|eot_id|start_header_id)\|>"),
    ),
    ("fake-turn-marker", 0.35, re.compile(r"(?im)^\s*(###\s*)?(system|assistant)\s*:\s*$")),
    (
        "exfiltration-url",
        0.5,
        re.compile(r"(?i)(?:send|post|upload|exfiltrate|transmit)\b[^.\n]{0,40}https?://"),
    ),
    (
        "markdown-image-exfil",
        0.5,
        re.compile(r"!\[[^\]]*\]\(\s*https?://[^)\s]{0,200}[?&=][^)\s]{20,}\)"),
    ),
    ("shell-fetch", 0.35, re.compile(r"(?i)\b(curl|wget)\b[^\n]{0,60}https?://")),
    (
        "encoded-payload",
        0.25,
        re.compile(r"\b[A-Za-z0-9+/]{220,}={0,2}\b"),
    ),
]


@dataclass(frozen=True)
class InjectionSignal:
    name: str
    weight: float
    excerpt: str


def scan_injection(text: str) -> tuple[float, list[InjectionSignal]]:
    """Score 0.0-1.0 for "this text is trying to steer the model".

    The score is a triage aid: log everything, maybe require confirmation above
    ~0.5, and never treat a low score as proof the text is safe. Invisible
    characters count on their own, because legitimate prompts do not contain
    Unicode tag blocks.
    """
    signals: list[InjectionSignal] = []
    _, removed = strip_invisible(text)
    if removed:
        signals.append(
            InjectionSignal("invisible-characters", 0.5, f"{removed} non-printing character(s)")
        )
    for name, weight, pattern in _INJECTION_PATTERNS:
        match = pattern.search(text)
        if match:
            excerpt = match.group(0)
            if len(excerpt) > 120:
                excerpt = excerpt[:117] + "..."
            signals.append(InjectionSignal(name, weight, excerpt))

    # Saturating sum: several weak signals should not beat one strong one by much,
    # and nothing should ever hit exactly 1.0 and imply certainty.
    score = 0.0
    for signal in sorted(signals, key=lambda s: -s.weight):
        score += signal.weight * (1.0 - score)
    return round(min(score, 0.99), 3), signals
