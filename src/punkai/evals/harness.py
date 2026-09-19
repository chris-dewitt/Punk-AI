"""Run a suite of cases against any callable that turns a prompt into text.

Why bother when public leaderboards exist: a leaderboard tells you how a model
does on someone else's distribution, usually one that has leaked into training
data. Your fifty hand-written cases about *your* task are worth more than any
MMLU number, and this harness exists so writing them costs a JSON file.

The output is designed to be diffed. Run a suite, keep the JSON, run it again
after a quantization change or a fine-tune, and `compare()` tells you exactly
which cases flipped -- which is the only number that should decide whether you
ship the new weights.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from punkai.evals import scorers

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

    def score(self, output: str) -> bool:
        return scorers.get(self.scorer)(output, self.expect)


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
    passed: bool
    output: str
    latency_ms: float
    tags: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class SuiteResult:
    suite: str
    model: str
    started_at: float
    results: list[CaseResult] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def by_tag(self) -> dict[str, tuple[int, int]]:
        """tag -> (passed, total). Where a regression actually shows itself."""
        out: dict[str, list[int]] = {}
        for result in self.results:
            for tag in result.tags or ["untagged"]:
                bucket = out.setdefault(tag, [0, 0])
                bucket[0] += int(result.passed)
                bucket[1] += 1
        return {k: (v[0], v[1]) for k, v in sorted(out.items())}

    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if not r.passed]

    def to_dict(self) -> dict:
        return {
            "suite": self.suite,
            "model": self.model,
            "started_at": self.started_at,
            "pass_rate": round(self.pass_rate, 4),
            "passed": self.passed,
            "total": self.total,
            "by_tag": {k: list(v) for k, v in self.by_tag().items()},
            "meta": self.meta,
            "results": [asdict(r) for r in self.results],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def summary(self, show_failures: int = 5) -> str:
        lines = [
            f"{self.suite} on {self.model}: {self.passed}/{self.total} "
            f"({self.pass_rate:.0%})"
        ]
        for tag, (passed, total) in self.by_tag().items():
            lines.append(f"  {tag:<24} {passed}/{total}")
        failures = self.failures()
        if failures and show_failures:
            lines.append(f"  failures (first {min(show_failures, len(failures))}):")
            for result in failures[:show_failures]:
                snippet = result.output.replace("\n", " ")[:80]
                lines.append(f"    - {result.id}: {snippet!r}")
        return "\n".join(lines)


def run_suite(
    suite: Suite,
    generate: Generate,
    model_name: str = "unknown",
    meta: dict | None = None,
) -> SuiteResult:
    """Run every case. A case that raises is a failure, not a crash."""
    result = SuiteResult(
        suite=suite.name, model=model_name, started_at=time.time(), meta=meta or {}
    )
    for case in suite.cases:
        started = time.perf_counter()
        try:
            output = generate(case.prompt)
            passed = case.score(output)
            error = ""
        except Exception as exc:  # noqa: BLE001 -- a backend blowing up is a data point
            output, passed, error = "", False, f"{type(exc).__name__}: {exc}"
        result.results.append(
            CaseResult(
                id=case.id,
                passed=passed,
                output=output,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                tags=case.tags,
                error=error,
            )
        )
    return result


@dataclass
class Comparison:
    regressions: list[str] = field(default_factory=list)  # passed before, fails now
    fixes: list[str] = field(default_factory=list)  # failed before, passes now
    unchanged: int = 0
    baseline_rate: float = 0.0
    current_rate: float = 0.0

    @property
    def safe_to_ship(self) -> bool:
        return not self.regressions

    def summary(self) -> str:
        delta = self.current_rate - self.baseline_rate
        lines = [
            f"pass rate {self.baseline_rate:.0%} -> {self.current_rate:.0%} ({delta:+.1%})",
            f"  regressions: {len(self.regressions)}  fixes: {len(self.fixes)}  "
            f"unchanged: {self.unchanged}",
        ]
        for case_id in self.regressions[:10]:
            lines.append(f"    BROKE  {case_id}")
        for case_id in self.fixes[:10]:
            lines.append(f"    FIXED  {case_id}")
        return "\n".join(lines)


def compare(baseline: SuiteResult | dict, current: SuiteResult | dict) -> Comparison:
    """Which individual cases flipped.

    Aggregate pass rate hides the thing you care about: a fine-tune that gains
    four cases and loses three reads as "+1%" and may have broken exactly the
    behaviour you were relying on.
    """
    base = baseline.to_dict() if isinstance(baseline, SuiteResult) else baseline
    curr = current.to_dict() if isinstance(current, SuiteResult) else current
    base_map = {r["id"]: r["passed"] for r in base["results"]}
    curr_map = {r["id"]: r["passed"] for r in curr["results"]}

    comparison = Comparison(
        baseline_rate=base.get("pass_rate", 0.0), current_rate=curr.get("pass_rate", 0.0)
    )
    for case_id, was in base_map.items():
        if case_id not in curr_map:
            continue
        now = curr_map[case_id]
        if was and not now:
            comparison.regressions.append(case_id)
        elif not was and now:
            comparison.fixes.append(case_id)
        else:
            comparison.unchanged += 1
    return comparison
