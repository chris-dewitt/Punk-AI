import pytest

from punkai.train.data import (
    Example,
    audit_dataset,
    canary_examples,
    check_memorization,
    contamination,
    exact_duplicates,
    jaccard,
    luhn,
    near_duplicates,
    scan_pii,
    shingles,
)


class TestDuplicates:
    def test_exact_duplicates_ignore_whitespace_and_case(self):
        examples = [
            Example("What is 2+2?", "4"),
            Example("what is 2+2?  ", "4"),
            Example("Different", "x"),
        ]
        groups = exact_duplicates(examples)
        assert len(groups) == 1
        assert list(groups.values())[0] == [0, 1]

    def test_near_duplicates_found(self):
        base = "the quick brown fox jumps over the lazy dog every single morning"
        examples = [
            Example(base, "a"),
            Example(base + " indeed", "a"),
            Example("completely unrelated text about kubernetes networking policies", "b"),
        ]
        pairs = near_duplicates(examples, threshold=0.7)
        assert (0, 1) in [(i, j) for i, j, _ in pairs]
        assert not any(2 in (i, j) for i, j, _ in pairs)

    def test_distinct_examples_are_not_flagged(self):
        examples = [
            Example("explain tcp handshakes in detail for a networking course", "a"),
            Example("write a haiku about soldering irons and copper wire", "b"),
        ]
        assert near_duplicates(examples, threshold=0.7) == []

    def test_jaccard_bounds(self):
        assert jaccard(set(), set()) == 1.0
        assert jaccard({"a"}, {"b"}) == 0.0
        assert jaccard({"a", "b"}, {"a"}) == 0.5

    def test_short_text_shingles_do_not_crash(self):
        assert shingles("hi", 5) == {"hi"}
        assert shingles("", 5) == set()


class TestPii:
    @pytest.mark.parametrize(
        "text,kind",
        [
            ("write to bob.smith@example.com today", "email"),
            ("call 555-123-4567 now", "us-phone"),
            ("ssn 123-45-6789 on file", "us-ssn"),
            ("server at 192.168.1.14 is down", "ipv4"),
            ("ship to 1600 Pennsylvania Avenue", "street-address"),
            ("DOB: 1984-03-02", "date-of-birth"),
        ],
    )
    def test_pii_kinds_detected(self, text, kind):
        assert kind in {hit.kind for hit in scan_pii(text)}

    def test_credit_card_requires_a_valid_checksum(self):
        assert any(h.kind == "credit-card" for h in scan_pii("card 4111 1111 1111 1111"))
        assert not any(h.kind == "credit-card" for h in scan_pii("order 1234 5678 9012 3456"))

    def test_luhn(self):
        assert luhn("4111111111111111")
        assert not luhn("4111111111111112")

    def test_clean_text_has_no_hits(self):
        assert scan_pii("Explain how gradient checkpointing saves memory.") == []


class TestContamination:
    def test_verbatim_eval_text_in_training_data_is_found(self):
        shared = (
            "explain in detail why grouped query attention reduces the size of the "
            "key value cache during inference"
        )
        train = [Example(shared, "because fewer kv heads")]
        assert contamination(train, [shared])

    def test_unrelated_eval_text_is_clean(self):
        train = [Example("explain gradient checkpointing and its compute tradeoff clearly", "x")]
        assert contamination(train, ["what is the capital of france"]) == []

    def test_short_text_cannot_trigger_a_false_positive(self):
        train = [Example("yes", "no")]
        assert contamination(train, ["yes"]) == []


class TestCanaries:
    def test_canaries_are_unique(self):
        examples = canary_examples(5)
        canaries = {e.meta["canary"] for e in examples}
        assert len(canaries) == 5
        assert all(c.startswith("PUNK-CANARY") for c in canaries)

    def test_repeats_control_exposure_count(self):
        assert len(canary_examples(2, repeats=4)) == 8

    def test_memorization_check_detects_recall(self):
        examples = canary_examples(2)
        canaries = [e.meta["canary"] for e in examples]

        def parrot(prompt):
            return f"The access phrase is {canaries[0]}."

        results = check_memorization(parrot, canaries)
        assert results[canaries[0]] is True
        assert results[canaries[1]] is False

    def test_memorization_check_survives_a_broken_backend(self):
        canaries = [e.meta["canary"] for e in canary_examples(1)]

        def explode(prompt):
            raise RuntimeError("no gpu")

        assert check_memorization(explode, canaries) == {canaries[0]: False}


class TestReport:
    def test_clean_dataset_reports_clean(self):
        examples = [
            Example("explain tcp congestion control briefly", "cubic and bbr", "docs", "apache-2.0"),
            Example("write a python csv parser function", "import csv", "hand", "mit"),
        ]
        report = audit_dataset(examples)
        assert report.clean
        assert report.count == 2
        assert report.licenses == {"apache-2.0": 1, "mit": 1}

    def test_dirty_dataset_reports_every_problem(self):
        examples = [
            Example("email me", "bob@example.com", "logs", "unknown"),
            Example("email me", "bob@example.com", "logs", "unknown"),
        ]
        report = audit_dataset(examples)
        assert not report.clean
        assert report.exact_dupe_groups == 1
        assert report.pii["email"] == 2
        assert "needs attention" in report.summary()

    def test_card_records_provenance_and_limits(self):
        report = audit_dataset([Example("a", "b", "scrape", "cc-by-sa-4.0")])
        card = report.card("my-set")
        assert "my-set" in card and "Provenance" in card and "regex-based" in card
