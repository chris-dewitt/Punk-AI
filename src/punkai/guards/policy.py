"""A named bundle of guard settings, applied to input on the way in and output
on the way out.

The defaults are deliberately conservative for a service you expose to anything
other than yourself: strip invisible characters, cap length, redact credentials
from output, flag likely injection but do not silently drop it. Loosen them on
purpose, in code, where the loosening is reviewable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from punkai.guards.filters import (
    InjectionSignal,
    Secret,
    redact_secrets,
    scan_injection,
    strip_invisible,
)


@dataclass
class GuardResult:
    """What the guard did, and why. Always inspectable -- never a bare bool."""

    text: str
    blocked: bool = False
    reasons: list[str] = field(default_factory=list)
    injection_score: float = 0.0
    signals: list[InjectionSignal] = field(default_factory=list)
    secrets: list[Secret] = field(default_factory=list)
    invisible_removed: int = 0
    truncated: bool = False

    @property
    def flagged(self) -> bool:
        return bool(self.reasons)


@dataclass
class GuardPolicy:
    max_input_chars: int = 32_000
    max_output_chars: int = 32_000
    strip_invisible_input: bool = True
    redact_input_secrets: bool = False  # off: you may legitimately paste config to debug
    redact_output_secrets: bool = True  # on: the model must not hand credentials back out
    injection_flag_at: float = 0.35
    injection_block_at: float | None = None  # None = flag and log, never refuse
    truncate_over_limit: bool = False  # False = refuse; True = silently cut

    @classmethod
    def permissive(cls) -> GuardPolicy:
        """For a loopback-only box you own, where you are the only caller."""
        return cls(injection_flag_at=0.6, injection_block_at=None, truncate_over_limit=True)

    @classmethod
    def strict(cls) -> GuardPolicy:
        """For anything reachable by someone who is not you."""
        return cls(
            max_input_chars=8_000,
            redact_input_secrets=True,
            injection_flag_at=0.25,
            injection_block_at=0.6,
        )

    # -- application ------------------------------------------------------

    def check_input(self, text: str) -> GuardResult:
        result = GuardResult(text=text)

        if self.strip_invisible_input:
            cleaned, removed = strip_invisible(text)
            if removed:
                result.invisible_removed = removed
                result.reasons.append(f"removed {removed} non-printing character(s)")
            result.text = cleaned

        if len(result.text) > self.max_input_chars:
            if self.truncate_over_limit:
                result.text = result.text[: self.max_input_chars]
                result.truncated = True
                result.reasons.append(f"truncated to {self.max_input_chars} chars")
            else:
                result.blocked = True
                result.reasons.append(
                    f"input is {len(result.text)} chars, limit is {self.max_input_chars}"
                )
                return result

        if self.redact_input_secrets:
            redacted, secrets = redact_secrets(result.text)
            if secrets:
                result.text = redacted
                result.secrets = secrets
                result.reasons.append(f"redacted {len(secrets)} suspected credential(s) from input")

        score, signals = scan_injection(result.text)
        result.injection_score = score
        result.signals = signals
        if score >= self.injection_flag_at and signals:
            names = ", ".join(sorted({s.name for s in signals}))
            result.reasons.append(f"possible prompt injection (score {score:.2f}: {names})")
        if self.injection_block_at is not None and score >= self.injection_block_at:
            result.blocked = True
            result.reasons.append(f"refused: injection score {score:.2f} over block threshold")
        return result

    def check_output(self, text: str) -> GuardResult:
        result = GuardResult(text=text)

        if self.redact_output_secrets:
            redacted, secrets = redact_secrets(result.text)
            if secrets:
                result.text = redacted
                result.secrets = secrets
                result.reasons.append(
                    f"redacted {len(secrets)} suspected credential(s) from output"
                )

        if len(result.text) > self.max_output_chars:
            result.text = result.text[: self.max_output_chars]
            result.truncated = True
            result.reasons.append(f"output truncated to {self.max_output_chars} chars")
        return result
