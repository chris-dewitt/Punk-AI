"""Run a suite of cases against any callable that turns a prompt into text.

Why bother when public leaderboards exist: a leaderboard tells you how a model
does on someone else's distribution, usually one that has leaked into training
data. Your fifty hand-written cases about *your* task are worth more than any
MMLU number, and this harness exists so writing them costs a JSON file.

Three decisions worth knowing about, because they are what make a number from
this harness worth showing to somebody else:

* **Repeats, and a strict pass.** With `repeats > 1` each case runs several
  times, and a case counts as passed only if it passed *every* time. A model
  that resists an attack three times in five is not resistant, and rounding
  that to 60% buries the finding. Cases that pass sometimes are reported
  separately as flaky.
* **Intervals, not points.** Results carry a Wilson confidence interval. Seven
  cases cannot tell you a rate to the percent, and the output should say so
  rather than let you over-read it.
* **Diffable output.** Save a run, change something, run again, and `compare()`
  names the individual cases that flipped -- the only number that should decide
  whether you ship new weights.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from punkai.evals import scorers
from punkai.evals.stats import Measurement

SUITE_DIR = Path(__file__).parent / "suites"

Generate = Callable[[str], str]


@dataclass
class EvalCase:
    id: str
    prompt: str
    scorer: str = "contains"
    expect: Any = None
    tags: list[str] = field(default_factory=list)
    note: str = ""
    # Follow-up user turns. Used by probes where the *second* answer is the
    # interesting one -- sycophancy, consistency under pushback.
    turns: list[str] = field(default_factory=list)

    def score(self, output: str) -> bool:
        """Scored on the final answer: after the pushback, not before it."""
        return scorers.get(self.scorer)(output, self.expect)

    def build_prompt(self, prior_outputs: list[str]) -> str:
        """Render the conversation so far as a plain-text transcript.

        This approximates a chat template rather than using the model's own.
        Good enough to compare models against each other with one method; if
        you need absolute numbers, drive the model's real template instead.
        """
        if not self.turns:
            return self.prompt
        lines = [f"User: {self.prompt}"]
        for i, output in enumerate(prior_outputs):
            lines.append(f"Assistant: {output}")
            if i < len(self.turns):
                lines.append(f"User: {self.turns[i]}")
        lines.append("Assistant:")
        return "\n".join(lines)

    @property
    def exchanges(self) -> int:
        return 1 + len(self.turns)


@dataclass
class Suite:
    name: str
    description: str = ""
    cases: list[EvalCase] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> Suite:
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            cases=[EvalCase(**c) for c in data.get("cases", [])],
        )

    @classmethod
    def load(cls, path: str | Path) -> Suite:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def builtin(cls, name: str) -> Suite:
        path = SUITE_DIR / f"{name}.json"
        if not path.exists():
            available = ", ".join(sorted(p.stem for p in SUITE_DIR.glob("*.json")))
            raise KeyError(f"no builtin suite {name!r}; available: {available}")
        return cls.load(path)

    @staticmethod
    def list_builtin() -> list[str]:
        return sorted(p.stem for p in SUITE_DIR.glob("*.json"))


@dataclass
class CaseResult:
    id: str
    passes: int
    runs: int
    outputs: list[str] = field(default_factory=list)
    transcript: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    tags: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def passed(self) -> bool:
        """Strict: every run passed. Sometimes-passing is not passing."""
        return self.runs > 0 and self.passes == self.runs

    @property
    def flaky(self) -> bool:
        return 0 < self.passes < self.runs

    @property
    def pass_fraction(self) -> float:
        return self.passes / self.runs if self.runs else 0.0

    @property
    def output(self) -> str:
        return self.outputs[0] if self.outputs else ""


@dataclass
class SuiteResult:
    suite: str
    model: str
    started_at: float
    results: list[CaseResult] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    repeats: int = 1

    # -- headline numbers -------------------------------------------------

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def trials(self) -> int:
        return sum(r.runs for r in self.results)

    @property
    def measurement(self) -> Measurement:
        """Rate over every trial, with the interval attached."""
        return Measurement(sum(r.passes for r in self.results), self.trials)

    @property
    def pass_rate(self) -> float:
        return self.measurement.rate

    def by_tag(self) -> dict[str, Measurement]:
        """tag -> Measurement. Where a regression actually shows itself."""
        counts: dict[str, list[int]] = {}
        for result in self.results:
            for tag in result.tags or ["untagged"]:
                bucket = counts.setdefault(tag, [0, 0])
                bucket[0] += result.passes
                bucket[1] += result.runs
        return {k: Measurement(v[0], v[1]) for k, v in sorted(counts.items())}

    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if not r.passed]

    def flaky_cases(self) -> list[CaseResult]:
        return [r for r in self.results if r.flaky]

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict:
        measurement = self.measurement
        return {
            "suite": self.suite,
            "model": self.model,
            "started_at": self.started_at,
            "repeats": self.repeats,
            "pass_rate": round(measurement.rate, 4),
            "ci_low": round(measurement.low, 4),
            "ci_high": round(measurement.high, 4),
            "passed": self.passed,
            "total": self.total,
            "trials": self.trials,
            "flaky": [r.id for r in self.flaky_cases()],
            "by_tag": {
                k: {"rate": round(m.rate, 4), "successes": m.successes, "trials": m.trials}
                for k, m in self.by_tag().items()
            },
            "meta": self.meta,
            "results": [asdict(r) for r in self.results],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def summary(self, show_failures: int = 5) -> str:
        measurement = self.measurement
        lines = [
            f"{self.suite} on {self.model}: {self.passed}/{self.total} cases "
            f"({measurement})"
        ]
        if self.repeats > 1:
            lines[0] += f" over {self.repeats} runs each"
        for tag, tag_measurement in self.by_tag().items():
            lines.append(f"  {tag:<24} {tag_measurement}")

        flaky = self.flaky_cases()
        if flaky:
            lines.append(f"  flaky ({len(flaky)} case(s) passed only sometimes):")
            for result in flaky[:show_failures]:
                lines.append(f"    ~ {result.id}: {result.passes}/{result.runs}")

        failures = [r for r in self.failures() if not r.flaky]
        if failures and show_failures:
            lines.append(f"  failures (first {min(show_failures, len(failures))}):")
            for result in failures[:show_failures]:
                snippet = result.output.replace("\n", " ")[:80]
                lines.append(f"    - {result.id}: {snippet!r}")
        if measurement.thin:
            lines.append(
                "  note: too few trials to read this precisely -- raise --repeats or "
                "add cases before quoting the number."
            )
        return "\n".join(lines)


def run_case(case: EvalCase, generate: Generate) -> tuple[list[str], bool]:
    """Run one case, following its turns. Returns (transcript, passed)."""
    outputs: list[str] = []
    for _ in range(case.exchanges):
        outputs.append(generate(case.build_prompt(outputs)))
    return outputs, case.score(outputs[-1])


def run_suite(
    suite: Suite,
    generate: Generate,
    model_name: str = "unknown",
    meta: dict | None = None,
    repeats: int = 1,
) -> SuiteResult:
    """Run every case `repeats` times. A case that raises is a failure, not a crash."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    result = SuiteResult(
        suite=suite.name,
        model=model_name,
        started_at=time.time(),
        meta=meta or {},
        repeats=repeats,
    )
    for case in suite.cases:
        outputs: list[str] = []
        transcript: list[str] = []
        passes = 0
        error = ""
        started = time.perf_counter()
        for _ in range(repeats):
            try:
                transcript, passed = run_case(case, generate)
                outputs.append(transcript[-1])
                passes += int(passed)
            except Exception as exc:  # noqa: BLE001 -- a backend blowing up is a data point
                error = error or f"{type(exc).__name__}: {exc}"
                outputs.append("")
        elapsed = (time.perf_counter() - started) * 1000
        result.results.append(
            CaseResult(
                id=case.id,
                passes=passes,
                runs=repeats,
                outputs=outputs,
                transcript=transcript,
                latency_ms=round(elapsed / repeats, 2),
                tags=case.tags,
                error=error,
            )
        )
    return result


@dataclass
class Comparison:
    regressions: list[str] = field(default_factory=list)  # passed before, fails now
    fixes: list[str] = field(default_factory=list)  # failed before, passes now
    destabilized: list[str] = field(default_factory=list)  # was solid, now flaky
    unchanged: int = 0
    baseline_rate: float = 0.0
    current_rate: float = 0.0

    @property
    def safe_to_ship(self) -> bool:
        return not self.regressions and not self.destabilized

    def summary(self) -> str:
        delta = self.current_rate - self.baseline_rate
        lines = [
            f"pass rate {self.baseline_rate:.0%} -> {self.current_rate:.0%} ({delta:+.1%})",
            f"  regressions: {len(self.regressions)}  fixes: {len(self.fixes)}  "
            f"destabilized: {len(self.destabilized)}  unchanged: {self.unchanged}",
        ]
        for case_id in self.regressions[:10]:
            lines.append(f"    BROKE   {case_id}")
        for case_id in self.destabilized[:10]:
            lines.append(f"    FLAKY   {case_id}")
        for case_id in self.fixes[:10]:
            lines.append(f"    FIXED   {case_id}")
        return "\n".join(lines)


def compare(baseline: SuiteResult | dict, current: SuiteResult | dict) -> Comparison:
    """Which individual cases flipped.

    Aggregate pass rate hides the thing you care about: a fine-tune that gains
    four cases and loses three reads as "+1%" and may have broken exactly the
    behaviour you were relying on.
    """
    base = baseline.to_dict() if isinstance(baseline, SuiteResult) else baseline
    curr = current.to_dict() if isinstance(current, SuiteResult) else current
    base_map = {r["id"]: r for r in base["results"]}
    curr_map = {r["id"]: r for r in curr["results"]}

    comparison = Comparison(
        baseline_rate=base.get("pass_rate", 0.0), current_rate=curr.get("pass_rate", 0.0)
    )
    flaky_now = set(curr.get("flaky", []))
    flaky_before = set(base.get("flaky", []))

    for case_id, base_row in base_map.items():
        if case_id not in curr_map:
            continue
        was = _row_passed(base_row)
        now = _row_passed(curr_map[case_id])
        if was and not now:
            comparison.regressions.append(case_id)
        elif not was and now:
            comparison.fixes.append(case_id)
        elif case_id in flaky_now and case_id not in flaky_before:
            comparison.destabilized.append(case_id)
        else:
            comparison.unchanged += 1
    return comparison


def _row_passed(row: dict) -> bool:
    """Read a serialized CaseResult, tolerating results saved by older versions."""
    if "passes" in row and "runs" in row:
        return row["runs"] > 0 and row["passes"] == row["runs"]
    return bool(row.get("passed", False))
