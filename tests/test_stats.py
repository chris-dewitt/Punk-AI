import pytest

from punkai.evals.stats import Measurement, wilson_interval, z_for


def test_wilson_stays_inside_zero_one_at_the_extremes():
    """The normal approximation famously does not, which is why we use Wilson."""
    low, high = wilson_interval(0, 5)
    assert low == 0.0 and 0 < high < 1
    low, high = wilson_interval(5, 5)
    assert 0 < low < 1 and high == 1.0


def test_no_trials_means_no_knowledge_not_zero_percent():
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert str(Measurement(0, 0)) == "no data"


def test_interval_narrows_as_evidence_grows():
    small = Measurement(7, 10).width
    medium = Measurement(70, 100).width
    large = Measurement(700, 1000).width
    assert small > medium > large


def test_interval_brackets_the_point_estimate():
    m = Measurement(43, 100)
    assert m.low < m.rate < m.high


def test_known_value_matches_published_wilson_figures():
    # 50/100 at 95% -> roughly 40.4% to 59.6%
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.404, abs=0.005)
    assert high == pytest.approx(0.596, abs=0.005)


def test_higher_confidence_is_a_wider_interval():
    assert Measurement(50, 100, 0.99).width > Measurement(50, 100, 0.80).width


def test_seven_cases_cannot_be_read_precisely():
    """The reason the harness warns: 3/7 is compatible with almost anything."""
    m = Measurement(3, 7)
    assert m.thin
    assert m.low < 0.2 and m.high > 0.7


def test_thousand_trials_can_be_read():
    assert not Measurement(700, 1000).thin


def test_z_scores():
    assert z_for(0.95) == pytest.approx(1.96, abs=0.001)
    assert z_for(0.99) == pytest.approx(2.576, abs=0.001)
    assert z_for(0.951) == pytest.approx(1.97, abs=0.02)


@pytest.mark.parametrize("bad", [(-1, 5), (6, 5), (1, -2)])
def test_nonsense_counts_rejected(bad):
    with pytest.raises(ValueError):
        wilson_interval(*bad)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5])
def test_nonsense_confidence_rejected(bad):
    with pytest.raises(ValueError):
        z_for(bad)
