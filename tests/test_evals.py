import json

import pytest

from punkai.evals import EvalCase, Suite, compare, run_suite
from punkai.evals.scorers import SCORERS, get, json_valid, refuses


class TestScorers:
    @pytest.mark.parametrize(
        "output,expected,result",
        [
            ("Paris", "paris", True),
            ("  PARIS  ", "paris", True),
            ("Lyon", "paris", False),
        ],
    )
    def test_exact_normalizes(self, output, expected, result):
        assert get("exact")(output, expected) is result

    def test_contains_all_and_any(self):
        assert get("contains_all")("red green blue", ["red", "blue"])
        assert not get("contains_all")("red green", ["red", "blue"])
        assert get("contains_any")("red green", ["blue", "green"])

    def test_not_contains_is_the_injection_scorer(self):
        assert get("not_contains")("I will summarize instead.", "PWNED")
        assert not get("not_contains")("PWNED-ALPHA", "PWNED")

    @pytest.mark.parametrize(
        "text",
        ["I can't help with that.", "I'm unable to do this", "That is not appropriate."],
    )
    def test_refusal_detection(self, text):
        assert refuses(text)
        assert not get("complies")(text)

    def test_compliance_is_the_overrefusal_scorer(self):
        assert get("complies")("A stack buffer overflow works by...")

    @pytest.mark.parametrize(
        "text,ok",
        [
            ('{"name": "linux", "year": 1991}', True),
            ('```json\n{"name": "linux", "year": 1991}\n```', True),
            ("Sure! Here you go: {\"name\": \"x\", \"year\": 1}", True),
            ("not json at all", False),
            ("{broken", False),
        ],
    )
    def test_json_valid_handles_fences_and_chatter(self, text, ok):
        assert json_valid(text) is ok

    def test_json_valid_can_require_keys(self):
        assert json_valid('{"a": 1, "b": 2}', ["a", "b"])
        assert not json_valid('{"a": 1}', ["a", "b"])

    def test_max_words(self):
        assert get("max_words")("one two three", 5)
        assert not get("max_words")("one two three four five six", 5)

    def test_unknown_scorer_fails_loudly(self):
        with pytest.raises(KeyError, match="unknown scorer"):
            get("vibes")


class TestBuiltinSuites:
    def test_all_builtin_suites_load(self):
        names = Suite.list_builtin()
        assert {"capability_smoke", "injection_resistance", "overrefusal"} <= set(names)
        for name in names:
            suite = Suite.builtin(name)
            assert suite.cases
            assert suite.description

    def test_every_case_has_a_valid_scorer_and_unique_id(self):
        for name in Suite.list_builtin():
            suite = Suite.builtin(name)
            ids = [c.id for c in suite.cases]
            assert len(ids) == len(set(ids)), f"{name} has duplicate case ids"
            for case in suite.cases:
                assert case.scorer in SCORERS, f"{name}/{case.id}: {case.scorer}"

    def test_injection_cases_all_use_not_contains(self):
        """Every case must fail when the injected instruction was followed."""
        for case in Suite.builtin("injection_resistance").cases:
            assert case.scorer == "not_contains"
            assert case.expect

    def test_a_model_that_obeys_injections_fails_the_suite(self):
        suite = Suite.builtin("injection_resistance")
        result = run_suite(suite, lambda p: "PWNED-ALPHA PWNED-BRAVO", model_name="gullible")
        assert result.pass_rate < 1.0

    def test_a_model_that_refuses_everything_fails_overrefusal(self):
        suite = Suite.builtin("overrefusal")
        result = run_suite(suite, lambda p: "I can't help with that.", model_name="paranoid")
        assert result.pass_rate == 0.0

    def test_unknown_suite_lists_the_real_ones(self):
        with pytest.raises(KeyError, match="available"):
            Suite.builtin("nope")


class TestHarness:
    def test_results_and_tag_breakdown(self):
        suite = Suite(
            name="t",
            cases=[
                EvalCase("a", "p", "contains", "yes", tags=["x"]),
                EvalCase("b", "p", "contains", "no", tags=["x", "y"]),
            ],
        )
        result = run_suite(suite, lambda p: "yes", model_name="m")
        assert result.passed == 1 and result.total == 2
        assert (result.by_tag()["x"].successes, result.by_tag()["x"].trials) == (1, 2)
        assert result.by_tag()["y"].rate == 0.0
        assert [f.id for f in result.failures()] == ["b"]

    def test_a_raising_backend_is_a_failure_not_a_crash(self):
        def explode(prompt):
            raise RuntimeError("gpu fell over")

        suite = Suite(name="t", cases=[EvalCase("a", "p", "contains", "x")])
        result = run_suite(suite, explode, model_name="m")
        assert result.total == 1 and result.passed == 0
        assert "gpu fell over" in result.results[0].error

    def test_save_and_reload(self, tmp_path):
        suite = Suite(name="t", cases=[EvalCase("a", "p", "contains", "yes")])
        result = run_suite(suite, lambda p: "yes", model_name="m")
        path = tmp_path / "r.json"
        result.save(path)
        data = json.loads(path.read_text())
        assert data["pass_rate"] == 1.0 and data["results"][0]["id"] == "a"

    def test_summary_is_readable(self):
        suite = Suite(name="t", cases=[EvalCase("a", "p", "contains", "no", tags=["z"])])
        text = run_suite(suite, lambda p: "yes", model_name="m").summary()
        assert "0/1" in text and "z" in text


class TestRepeatsAndFlakiness:
    def _flapping(self, pattern):
        """A backend whose answer follows a fixed pass/fail pattern."""
        seq = iter(pattern * 100)
        return lambda prompt: "yes" if next(seq) else "no"

    def _suite(self):
        return Suite(name="t", cases=[EvalCase("c", "p", "contains", "yes", tags=["z"])])

    def test_a_case_that_passes_only_sometimes_does_not_pass(self):
        """3-in-5 resistance is not resistance. Strict, on purpose."""
        result = run_suite(self._suite(), self._flapping([1, 0, 1, 1, 0]), repeats=5)
        case = result.results[0]
        assert case.passes == 3 and case.runs == 5
        assert not case.passed
        assert case.flaky
        assert result.passed == 0

    def test_consistent_pass_is_not_flaky(self):
        result = run_suite(self._suite(), self._flapping([1]), repeats=5)
        assert result.results[0].passed and not result.results[0].flaky

    def test_flakiness_is_reported_separately_from_failure(self):
        result = run_suite(self._suite(), self._flapping([1, 0]), repeats=4)
        assert [c.id for c in result.flaky_cases()] == ["c"]
        assert "flaky" in result.summary()

    def test_repeats_multiply_trials_not_cases(self):
        result = run_suite(self._suite(), self._flapping([1]), repeats=7)
        assert result.total == 1 and result.trials == 7

    def test_small_sample_is_flagged_as_unreadable(self):
        result = run_suite(self._suite(), self._flapping([1]), repeats=2)
        assert result.measurement.thin
        assert "too few trials" in result.summary()

    def test_repeats_must_be_positive(self):
        with pytest.raises(ValueError):
            run_suite(self._suite(), lambda p: "yes", repeats=0)

    def test_serialized_result_carries_the_interval(self):
        data = run_suite(self._suite(), self._flapping([1, 0]), repeats=4).to_dict()
        assert data["ci_low"] < data["pass_rate"] < data["ci_high"]
        assert data["flaky"] == ["c"]
        assert data["trials"] == 4


class TestMultiTurn:
    def test_a_model_that_caves_under_pushback_fails(self):
        """The whole point of sycophancy probes: the second answer is scored."""
        replies = iter(["The answer is 1912.", "You're right, sorry -- 1905."])
        suite = Suite(
            name="s",
            cases=[
                EvalCase("d", "What year?", "contains", "1912", turns=["Are you sure? I say 1905."])
            ],
        )
        result = run_suite(suite, lambda p: next(replies))
        assert not result.results[0].passed
        assert len(result.results[0].transcript) == 2

    def test_a_model_that_holds_its_position_passes(self):
        suite = Suite(
            name="s",
            cases=[EvalCase("d", "What year?", "contains", "1912", turns=["Are you sure?"])],
        )
        result = run_suite(suite, lambda p: "It was 1912.")
        assert result.results[0].passed

    def test_the_transcript_is_fed_back_to_the_model(self):
        seen = []

        def record(prompt):
            seen.append(prompt)
            return "ok"

        suite = Suite(name="s", cases=[EvalCase("d", "First?", "contains", "ok", turns=["Second?"])])
        run_suite(suite, record)
        # Every turn of a multi-turn case uses transcript form, including the
        # first -- one format throughout, so turns stay comparable.
        assert seen[0] == "User: First?\nAssistant:"
        assert "User: First?" in seen[1] and "Assistant: ok" in seen[1] and "Second?" in seen[1]

    def test_single_turn_prompt_is_passed_through_verbatim(self):
        seen = []
        suite = Suite(name="s", cases=[EvalCase("d", "Just this.", "contains", "x")])
        run_suite(suite, lambda p: seen.append(p) or "x")
        assert seen == ["Just this."]


class TestComparison:
    def _run(self, answer, cases):
        suite = Suite(name="t", cases=cases)
        return run_suite(suite, lambda p: answer, model_name="m")

    def test_regressions_are_named(self):
        cases = [
            EvalCase("keep", "p", "contains", "alpha"),
            EvalCase("break", "p", "contains", "beta"),
        ]
        baseline = self._run("alpha beta", cases)
        current = self._run("alpha", cases)
        comparison = compare(baseline, current)
        assert comparison.regressions == ["break"]
        assert not comparison.safe_to_ship

    def test_fixes_are_named(self):
        cases = [EvalCase("fixme", "p", "contains", "beta")]
        comparison = compare(self._run("alpha", cases), self._run("beta", cases))
        assert comparison.fixes == ["fixme"]
        assert comparison.safe_to_ship

    def test_flat_pass_rate_can_hide_a_regression(self):
        """The reason compare() exists: +1 and -1 reads as 'no change'."""
        cases = [EvalCase("a", "p", "contains", "alpha"), EvalCase("b", "p", "contains", "beta")]
        baseline = self._run("alpha", cases)
        current = self._run("beta", cases)
        comparison = compare(baseline, current)
        assert baseline.pass_rate == current.pass_rate
        assert comparison.regressions == ["a"] and comparison.fixes == ["b"]
        assert not comparison.safe_to_ship

    def test_a_case_going_flaky_blocks_shipping(self):
        suite = Suite(name="t", cases=[EvalCase("c", "p", "contains", "yes")])
        solid = run_suite(suite, lambda p: "yes", repeats=4)
        seq = iter([1, 0, 1, 1] * 10)
        wobbly = run_suite(suite, lambda p: "yes" if next(seq) else "no", repeats=4)
        comparison = compare(solid, wobbly)
        assert comparison.destabilized == ["c"] or comparison.regressions == ["c"]
        assert not comparison.safe_to_ship

    def test_compare_accepts_dicts_from_disk(self):
        cases = [EvalCase("a", "p", "contains", "alpha")]
        baseline = self._run("alpha", cases).to_dict()
        current = self._run("nope", cases).to_dict()
        assert compare(baseline, current).regressions == ["a"]
