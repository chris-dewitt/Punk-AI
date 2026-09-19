import pytest

from punkai.registry.licenses import LICENSES, describe, gate, normalize


def test_apache_permits_everything():
    for intent in ["commercial", "redistribute", "finetune", "distill"]:
        decision = gate("apache-2.0", intent)
        assert decision.allowed and not decision.needs_review, intent


def test_non_commercial_licence_blocks_commercial_use():
    decision = gate("mistral-research", "commercial")
    assert not decision.allowed
    assert "prohibits" in decision.reasons[0]


def test_conditional_licence_asks_for_review_and_lists_obligations():
    decision = gate("llama3-community", "commercial")
    assert decision.allowed and decision.needs_review
    assert any("Built with Llama" in o for o in decision.obligations)


def test_unknown_licence_is_never_silently_permissive():
    decision = gate("some-vendor-eula-v3", "redistribute")
    assert decision.needs_review
    assert not decision.allowed


def test_local_use_is_allowed_but_unknown_licence_still_warns():
    assert gate("apache-2.0", "local-tinkering").allowed
    unknown = gate("", "local-tinkering")
    assert unknown.allowed and unknown.needs_review


@pytest.mark.parametrize(
    "spelling,expected",
    [
        ("Apache-2.0", "apache-2.0"),
        ("apache2", "apache-2.0"),
        ("llama3.1", "llama3-community"),
        ("LLAMA-3.1", "llama3-community"),
        ("cc-by-nc", "cc-by-nc-4.0"),
        ("other", "unknown"),
        ("", "unknown"),
        ("totally-made-up", "unknown"),
    ],
)
def test_aliases_and_casing(spelling, expected):
    assert normalize(spelling) == expected
    assert describe(spelling).id == expected


def test_every_licence_carries_obligations_or_a_note():
    for spec in LICENSES.values():
        assert spec.obligations or spec.note, f"{spec.id} says nothing useful"


def test_bad_intent_is_a_programming_error():
    with pytest.raises(ValueError, match="unknown intent"):
        gate("mit", "world-domination")


def test_decision_renders_a_verdict():
    assert str(gate("mistral-research", "commercial")).startswith("[BLOCKED]")
    assert str(gate("mit", "commercial")).startswith("[OK]")
    assert str(gate("gemma", "redistribute")).startswith("[REVIEW]")
