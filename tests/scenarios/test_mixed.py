"""Tests for MixedScenario parsing, weighted sampling, and validation."""

from collections import Counter

import pytest

from genai_bench.scenarios.base import Scenario
from genai_bench.scenarios.text import (
    DeterministicDistribution,
    MixedScenario,
    PrefixRepetitionScenario,
)


def test_mixed_scenario_parse_three_prefix_subs():
    s = Scenario.from_string(
        "M(0.4:P(7840,160)/200,0.4:P(31360,640)/500,0.2:P(78400,1600)/1000)"
    )
    assert isinstance(s, MixedScenario)
    assert len(s.sub_scenarios) == 3
    assert all(isinstance(x, PrefixRepetitionScenario) for x in s.sub_scenarios)
    assert s.weights == pytest.approx([0.4, 0.4, 0.2])


def test_mixed_scenario_weights_normalize():
    s = Scenario.from_string("M(2:D(10,5),3:D(20,10))")
    assert s.weights == pytest.approx([0.4, 0.6])


def test_mixed_scenario_mixed_sub_types():
    s = Scenario.from_string("M(0.5:P(100,50)/25,0.5:D(200,100))")
    assert isinstance(s.sub_scenarios[0], PrefixRepetitionScenario)
    assert isinstance(s.sub_scenarios[1], DeterministicDistribution)


def test_mixed_scenario_to_string_round_trip():
    original = "M(0.4:P(7840,160)/200,0.4:P(31360,640)/500,0.2:P(78400,1600)/1000)"
    s = Scenario.from_string(original)
    reparsed = Scenario.from_string(s.to_string())
    assert isinstance(reparsed, MixedScenario)
    assert reparsed.weights == pytest.approx(s.weights)


def test_mixed_scenario_sample_distribution_matches_weights():
    s = MixedScenario(
        weights=[0.4, 0.4, 0.2],
        sub_scenarios=[
            PrefixRepetitionScenario(7840, 160, 200),
            PrefixRepetitionScenario(31360, 640, 500),
            PrefixRepetitionScenario(78400, 1600, 1000),
        ],
    )
    draws = [id(s.sample()) for _ in range(20000)]
    counts = Counter(draws)
    total = sum(counts.values())
    fractions = sorted(c / total for c in counts.values())
    # With 20k draws, each fraction should be within ~1.5 percentage points
    assert fractions[0] == pytest.approx(0.2, abs=0.02)
    assert fractions[1] == pytest.approx(0.4, abs=0.02)
    assert fractions[2] == pytest.approx(0.4, abs=0.02)


def test_mixed_scenario_rejects_nested():
    with pytest.raises(ValueError, match="cannot be nested"):
        Scenario.from_string("M(0.5:M(1:D(10,5)),0.5:D(20,10))")


def test_mixed_scenario_rejects_missing_weight():
    with pytest.raises(ValueError, match="WEIGHT:SUBSCEN"):
        Scenario.from_string("M(D(10,5),D(20,10))")


def test_mixed_scenario_rejects_negative_weight():
    with pytest.raises(ValueError, match="positive"):
        Scenario.from_string("M(-0.5:D(10,5),1:D(20,10))")


def test_mixed_scenario_rejects_empty():
    with pytest.raises(ValueError):
        Scenario.from_string("M()")
