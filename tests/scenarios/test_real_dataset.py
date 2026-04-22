"""Tests for real dataset scenario implementations (RD/RDC)."""

import pytest

from genai_bench.scenarios.base import Scenario
from genai_bench.scenarios.real_dataset import (
    CachedRealDatasetScenario,
    RealDatasetScenario,
)


# --- RealDatasetScenario (RD) tests ---


def test_rd_creation_with_tokens():
    scenario = RealDatasetScenario(max_output_tokens=200)
    assert scenario.max_output_tokens == 200


def test_rd_creation_without_tokens():
    scenario = RealDatasetScenario()
    assert scenario.max_output_tokens is None


def test_rd_sample_with_tokens():
    scenario = RealDatasetScenario(max_output_tokens=200)
    max_out, cache_mode = scenario.sample()
    assert max_out == 200
    assert cache_mode == "uncached"


def test_rd_sample_without_tokens():
    scenario = RealDatasetScenario()
    max_out, cache_mode = scenario.sample()
    assert max_out is None
    assert cache_mode == "uncached"


def test_rd_to_string_with_tokens():
    scenario = RealDatasetScenario(max_output_tokens=200)
    assert scenario.to_string() == "RD(200)"


def test_rd_to_string_without_tokens():
    scenario = RealDatasetScenario()
    assert scenario.to_string() == "RD"


def test_rd_parse_with_tokens():
    scenario = RealDatasetScenario.parse("(200)")
    assert scenario.max_output_tokens == 200


def test_rd_parse_without_tokens():
    scenario = RealDatasetScenario.parse("")
    assert scenario.max_output_tokens is None


def test_rd_from_string_with_tokens():
    scenario = Scenario.from_string("RD(200)")
    assert isinstance(scenario, RealDatasetScenario)
    assert scenario.max_output_tokens == 200


def test_rd_from_string_without_tokens():
    scenario = Scenario.from_string("RD")
    assert isinstance(scenario, RealDatasetScenario)
    assert scenario.max_output_tokens is None


def test_rd_invalid_formats():
    with pytest.raises(ValueError):
        Scenario.from_string("RD(abc)")
    with pytest.raises(ValueError):
        Scenario.from_string("RD(100,200)")


# --- CachedRealDatasetScenario (RDC) tests ---


def test_rdc_creation_with_tokens():
    scenario = CachedRealDatasetScenario(max_output_tokens=200)
    assert scenario.max_output_tokens == 200


def test_rdc_sample_with_tokens():
    scenario = CachedRealDatasetScenario(max_output_tokens=200)
    max_out, cache_mode = scenario.sample()
    assert max_out == 200
    assert cache_mode == "natural"


def test_rdc_sample_without_tokens():
    scenario = CachedRealDatasetScenario()
    max_out, cache_mode = scenario.sample()
    assert max_out is None
    assert cache_mode == "natural"


def test_rdc_to_string():
    assert CachedRealDatasetScenario(200).to_string() == "RDC(200)"
    assert CachedRealDatasetScenario().to_string() == "RDC"


def test_rdc_from_string_with_tokens():
    scenario = Scenario.from_string("RDC(200)")
    assert isinstance(scenario, CachedRealDatasetScenario)
    assert scenario.max_output_tokens == 200


def test_rdc_from_string_without_tokens():
    scenario = Scenario.from_string("RDC")
    assert isinstance(scenario, CachedRealDatasetScenario)
    assert scenario.max_output_tokens is None


def test_rdc_invalid_formats():
    with pytest.raises(ValueError):
        Scenario.from_string("RDC(abc)")
