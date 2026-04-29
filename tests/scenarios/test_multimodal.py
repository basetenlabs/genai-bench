"""Tests for multimodal scenario implementations."""

import pytest

from genai_bench.scenarios.base import Scenario
from genai_bench.scenarios.multimodal import (
    DeterministicImageScenario,
    ImageModality,
    PrefixImageScenario,
)


def test_image_modality_creation():
    """Test ImageModality creation."""
    scenario = ImageModality(
        num_input_dimension_width=256,
        num_input_dimension_height=256,
        num_input_images=1,
        max_output_token=100,
    )

    assert scenario.num_input_dimension_width == 256
    assert scenario.num_input_dimension_height == 256
    assert scenario.num_input_images == 1
    assert scenario.max_output_token == 100


def test_image_modality_sampling():
    """Test ImageModality sampling."""
    scenario = ImageModality(
        num_input_dimension_width=512,
        num_input_dimension_height=512,
        num_input_images=2,
        max_output_token=200,
    )

    dimensions, num_images, max_tokens = scenario.sample()
    assert dimensions == (512, 512)
    assert num_images == 2
    assert max_tokens == 200


def test_image_modality_default_values():
    """Test ImageModality with default values."""
    scenario = ImageModality(
        num_input_dimension_width=256, num_input_dimension_height=256
    )

    assert scenario.num_input_images == 1
    assert scenario.max_output_token is None


def test_image_modality_to_string():
    """Test ImageModality string representation."""
    # With default num_input_images
    scenario = ImageModality(
        num_input_dimension_width=256, num_input_dimension_height=256
    )
    assert scenario.to_string() == "I(256,256)"

    # With multiple images
    scenario = ImageModality(
        num_input_dimension_width=512,
        num_input_dimension_height=512,
        num_input_images=3,
    )
    assert scenario.to_string() == "I(512,512,3)"


def test_image_modality_parse():
    """Test ImageModality parsing from string."""
    # Parse simple format
    scenario = ImageModality.parse("(256,256)")
    assert scenario.num_input_dimension_width == 256
    assert scenario.num_input_dimension_height == 256
    assert scenario.num_input_images == 1

    # Parse with multiple images
    scenario = ImageModality.parse("(1024,1024,2)")
    assert scenario.num_input_dimension_width == 1024
    assert scenario.num_input_dimension_height == 1024
    assert scenario.num_input_images == 2


def test_image_modality_from_string():
    """Test ImageModality creation from string."""
    scenario = Scenario.from_string("I(256,256)")
    assert isinstance(scenario, ImageModality)
    assert scenario.num_input_dimension_width == 256
    assert scenario.num_input_dimension_height == 256

    scenario = Scenario.from_string("I(512,512,3)")
    assert isinstance(scenario, ImageModality)
    assert scenario.num_input_images == 3


# --- DeterministicImageScenario (ID) tests ---


def test_deterministic_image_creation():
    """Test DeterministicImageScenario creation."""
    scenario = DeterministicImageScenario(
        num_input_dimension_width=1024,
        num_input_dimension_height=1024,
        num_input_tokens=1500,
        num_output_tokens=200,
    )
    assert scenario.num_input_dimension_width == 1024
    assert scenario.num_input_dimension_height == 1024
    assert scenario.num_input_tokens == 1500
    assert scenario.num_output_tokens == 200


def test_deterministic_image_sampling():
    """Test DeterministicImageScenario sampling returns 4-tuple."""
    scenario = DeterministicImageScenario(
        num_input_dimension_width=512,
        num_input_dimension_height=512,
        num_input_tokens=800,
        num_output_tokens=100,
    )
    dimensions, num_images, input_tokens, output_tokens = scenario.sample()
    assert dimensions == (512, 512)
    assert num_images == 1
    assert input_tokens == 800
    assert output_tokens == 100


def test_deterministic_image_to_string():
    """Test DeterministicImageScenario string roundtrip."""
    scenario = DeterministicImageScenario(
        num_input_dimension_width=1024,
        num_input_dimension_height=1024,
        num_input_tokens=1500,
        num_output_tokens=200,
    )
    assert scenario.to_string() == "ID(1024,1024,1500,200)"


def test_deterministic_image_parse():
    """Test DeterministicImageScenario parsing."""
    scenario = DeterministicImageScenario.parse("(1024,1024,1500,200)")
    assert scenario.num_input_dimension_width == 1024
    assert scenario.num_input_dimension_height == 1024
    assert scenario.num_input_tokens == 1500
    assert scenario.num_output_tokens == 200


def test_deterministic_image_from_string():
    """Test DeterministicImageScenario creation via Scenario.from_string."""
    scenario = Scenario.from_string("ID(1024,1024,1500,200)")
    assert isinstance(scenario, DeterministicImageScenario)
    assert scenario.num_input_dimension_width == 1024
    assert scenario.num_input_tokens == 1500
    assert scenario.num_output_tokens == 200


def test_deterministic_image_invalid_formats():
    """Test DeterministicImageScenario rejects invalid formats."""
    with pytest.raises(ValueError):
        Scenario.from_string("ID(512,512)")  # missing input/output tokens
    with pytest.raises(ValueError):
        Scenario.from_string("ID(512)")  # too few params
    with pytest.raises(ValueError):
        Scenario.from_string("ID(512,512,100,200,3,4)")  # too many params (6)


# --- DeterministicImageScenario multi-image (5-param form) tests ---


def test_deterministic_image_multi_image_creation():
    """Test DeterministicImageScenario creation with explicit num_images."""
    scenario = DeterministicImageScenario(
        num_input_dimension_width=1024,
        num_input_dimension_height=1024,
        num_input_tokens=1500,
        num_output_tokens=200,
        num_images=3,
    )
    assert scenario.num_images == 3


def test_deterministic_image_single_image_default():
    """4-param form (no num_images) defaults to 1 image — backward compat."""
    scenario = DeterministicImageScenario(
        num_input_dimension_width=1024,
        num_input_dimension_height=1024,
        num_input_tokens=1500,
        num_output_tokens=200,
    )
    assert scenario.num_images == 1


def test_deterministic_image_multi_image_sampling():
    """sample() returns the configured num_images, not hardcoded 1."""
    scenario = DeterministicImageScenario(
        num_input_dimension_width=512,
        num_input_dimension_height=512,
        num_input_tokens=800,
        num_output_tokens=100,
        num_images=4,
    )
    _, num_images, _, _ = scenario.sample()
    assert num_images == 4


def test_deterministic_image_multi_image_to_string():
    """5-param form roundtrips with trailing num_images."""
    scenario = DeterministicImageScenario(
        num_input_dimension_width=1024,
        num_input_dimension_height=1024,
        num_input_tokens=1500,
        num_output_tokens=200,
        num_images=3,
    )
    assert scenario.to_string() == "ID(1024,1024,1500,200,3)"


def test_deterministic_image_single_image_to_string_is_4param():
    """Single-image form omits num_images from the string (legacy shape)."""
    scenario = DeterministicImageScenario(
        num_input_dimension_width=1024,
        num_input_dimension_height=1024,
        num_input_tokens=1500,
        num_output_tokens=200,
        num_images=1,
    )
    assert scenario.to_string() == "ID(1024,1024,1500,200)"


def test_deterministic_image_parse_multi_image():
    """5-param form parses num_images from the trailing position."""
    scenario = DeterministicImageScenario.parse("(1024,1024,1500,200,3)")
    assert scenario.num_input_dimension_width == 1024
    assert scenario.num_input_tokens == 1500
    assert scenario.num_output_tokens == 200
    assert scenario.num_images == 3


def test_deterministic_image_from_string_multi_image():
    """Scenario.from_string routes the 5-param form correctly."""
    scenario = Scenario.from_string("ID(1024,1024,1500,200,3)")
    assert isinstance(scenario, DeterministicImageScenario)
    assert scenario.num_images == 3


# --- PrefixImageScenario (IP) tests ---


def test_prefix_image_creation():
    """Test PrefixImageScenario creation."""
    scenario = PrefixImageScenario(
        num_input_dimension_width=1024,
        num_input_dimension_height=1024,
        prefix_tokens=1200,
        suffix_tokens=300,
        output_tokens=200,
    )
    assert scenario.num_input_dimension_width == 1024
    assert scenario.prefix_tokens == 1200
    assert scenario.suffix_tokens == 300
    assert scenario.output_tokens == 200


def test_prefix_image_sampling():
    """Test PrefixImageScenario sampling returns 5-tuple."""
    scenario = PrefixImageScenario(
        num_input_dimension_width=512,
        num_input_dimension_height=512,
        prefix_tokens=800,
        suffix_tokens=200,
        output_tokens=100,
    )
    dimensions, num_images, prefix, suffix, output = scenario.sample()
    assert dimensions == (512, 512)
    assert num_images == 1
    assert prefix == 800
    assert suffix == 200
    assert output == 100


def test_prefix_image_to_string():
    """Test PrefixImageScenario string roundtrip."""
    scenario = PrefixImageScenario(
        num_input_dimension_width=1024,
        num_input_dimension_height=1024,
        prefix_tokens=1200,
        suffix_tokens=300,
        output_tokens=200,
    )
    assert scenario.to_string() == "IP(1024,1024,1200,300)/200"


def test_prefix_image_parse():
    """Test PrefixImageScenario parsing."""
    scenario = PrefixImageScenario.parse("(1024,1024,1200,300)/200")
    assert scenario.num_input_dimension_width == 1024
    assert scenario.prefix_tokens == 1200
    assert scenario.suffix_tokens == 300
    assert scenario.output_tokens == 200


def test_prefix_image_from_string():
    """Test PrefixImageScenario creation via Scenario.from_string."""
    scenario = Scenario.from_string("IP(1024,1024,1200,300)/200")
    assert isinstance(scenario, PrefixImageScenario)
    assert scenario.num_input_dimension_width == 1024
    assert scenario.prefix_tokens == 1200
    assert scenario.output_tokens == 200


def test_prefix_image_invalid_formats():
    """Test PrefixImageScenario rejects invalid formats."""
    with pytest.raises(ValueError):
        Scenario.from_string("IP(512,512,100,200)")  # missing /output
    with pytest.raises(ValueError):
        Scenario.from_string("IP(512,512)")  # too few params


# --- PrefixImageScenario multi-image (5-param form) tests ---


def test_prefix_image_multi_image_creation():
    """Test PrefixImageScenario creation with explicit num_images."""
    scenario = PrefixImageScenario(
        num_input_dimension_width=1024,
        num_input_dimension_height=1024,
        prefix_tokens=1200,
        suffix_tokens=300,
        output_tokens=200,
        num_images=3,
    )
    assert scenario.num_images == 3


def test_prefix_image_single_image_default():
    """4-param form (no num_images) defaults to 1 image — backward compat."""
    scenario = PrefixImageScenario(
        num_input_dimension_width=1024,
        num_input_dimension_height=1024,
        prefix_tokens=1200,
        suffix_tokens=300,
        output_tokens=200,
    )
    assert scenario.num_images == 1


def test_prefix_image_multi_image_sampling():
    """sample() returns the configured num_images, not hardcoded 1."""
    scenario = PrefixImageScenario(
        num_input_dimension_width=512,
        num_input_dimension_height=512,
        prefix_tokens=800,
        suffix_tokens=200,
        output_tokens=100,
        num_images=5,
    )
    _, num_images, _, _, _ = scenario.sample()
    assert num_images == 5


def test_prefix_image_multi_image_to_string():
    """5-param form roundtrips with num_images inside the parens."""
    scenario = PrefixImageScenario(
        num_input_dimension_width=1024,
        num_input_dimension_height=1024,
        prefix_tokens=1200,
        suffix_tokens=300,
        output_tokens=200,
        num_images=3,
    )
    assert scenario.to_string() == "IP(1024,1024,1200,300,3)/200"


def test_prefix_image_single_image_to_string_is_4param():
    """Single-image form omits num_images from the string (legacy shape)."""
    scenario = PrefixImageScenario(
        num_input_dimension_width=1024,
        num_input_dimension_height=1024,
        prefix_tokens=1200,
        suffix_tokens=300,
        output_tokens=200,
        num_images=1,
    )
    assert scenario.to_string() == "IP(1024,1024,1200,300)/200"


def test_prefix_image_parse_multi_image():
    """5-param form parses num_images from inside the parens."""
    scenario = PrefixImageScenario.parse("(1024,1024,1200,300,3)/200")
    assert scenario.num_input_dimension_width == 1024
    assert scenario.prefix_tokens == 1200
    assert scenario.suffix_tokens == 300
    assert scenario.output_tokens == 200
    assert scenario.num_images == 3


def test_prefix_image_from_string_multi_image():
    """Scenario.from_string routes the 5-param form correctly."""
    scenario = Scenario.from_string("IP(1024,1024,1200,300,3)/200")
    assert isinstance(scenario, PrefixImageScenario)
    assert scenario.num_images == 3
