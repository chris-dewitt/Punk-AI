"""Statistics for eval results, because a single number from a single run is
not evidence.

Two things make an eval credible enough for other people to act on:

* **An interval, not a point.** "43% resisted injection" from 7 cases is
  compatible with anything from 15% to 75%. Reporting the point estimate alone
  invites everyone to over-read their own numbers. We use the Wilson score
  interval, which -- unlike the normal approximation everyone reaches for --
  stays inside [0, 1] and behaves sensibly at small n and at rates near 0 or 1.
  Small n is the normal case here: real suites are dozens of cases, not
  thousands.
* **Repeat runs.** A model that resists an attack 3 times in 5 is not resistant.
  Averaging that into "60%" hides the thing you needed to know, so flakiness is
  reported separately rather than folded into the mean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# z for common two-sided confidence levels.
Z_SCORES = {0.80: 1.2816, 0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}


def z_for(confidence: float) -> float:
    if confidence in Z_SCORES:
        return Z_SCORES[confidence]
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}")
    # Acklam-style rational approximation to the inverse normal CDF, plenty
    # accurate for reporting intervals.
    p = 1.0 - (1.0 - confidence) / 2.0
    return math.sqrt(2.0) * _erfinv(2.0 * p - 1.0)


def _erfinv(x: float) -> float:
    if not -1.0 < x < 1.0:
        raise ValueError("erfinv domain is (-1, 1)")
    a = 0.147
    ln = math.log(1.0 - x * x)
    first = 2.0 / (math.pi * a) + ln / 2.0
    return math.copysign(math.sqrt(math.sqrt(first * first - ln / a) - first), x)


def wilson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Zero trials gives (0, 1): we know nothing, and the interval should say so
    rather than quietly reporting 0%.
    """
    if successes < 0 or trials < 0 or successes > trials:
        raise ValueError(f"nonsensical counts: {successes}/{trials}")
    if trials == 0:
        return (0.0, 1.0)

    z = z_for(confidence)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = p + z * z / (2.0 * trials)
    spread = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials))
    low = (center - spread) / denominator
    high = (center + spread) / denominator
    return (max(0.0, low), min(1.0, high))


@dataclass(frozen=True)
class Measurement:
    """A proportion with the uncertainty attached, so it cannot be quoted without it."""

    successes: int
    trials: int
    confidence: float = 0.95

    @property
    def rate(self) -> float:
        return self.successes / self.trials if self.trials else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.successes, self.trials, self.confidence)

    @property
    def low(self) -> float:
        """The conservative end. Judge safety properties on this, not on the point."""
        return self.interval[0]

    @property
    def high(self) -> float:
        return self.interval[1]

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def thin(self) -> bool:
        """True when the interval is so wide the number should not drive a decision."""
        return self.width > 0.35

    def __str__(self) -> str:
        if not self.trials:
            return "no data"
        return (
            f"{self.rate:.0%} ({self.successes}/{self.trials}, "
            f"95% CI {self.low:.0%}-{self.high:.0%})"
        )
