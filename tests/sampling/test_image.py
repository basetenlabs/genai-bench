from unittest.mock import MagicMock

import pytest
from PIL import Image
from transformers import AutoTokenizer

from genai_bench.data.config import DatasetConfig, DatasetSourceConfig
from genai_bench.protocol import UserImageChatRequest, UserImageEmbeddingRequest
from genai_bench.sampling.image import ImageSampler
from genai_bench.scenarios import ImageModality
from genai_bench.scenarios.multimodal import (
    DeterministicImageScenario,
    PrefixImageScenario,
)


@pytest.fixture
def mock_tokenizer():
    return MagicMock()


@pytest.fixture
def mock_vision_dataset():
    return [("A cat", Image.new("RGBA", (250, 250)))]


@pytest.fixture
def mock_image():
    mock_img = Image.new("RGB", (2048, 2048))
    return mock_img


def test_image_sampler(mock_tokenizer, mock_vision_dataset, mock_image):
    sampler = ImageSampler(
        tokenizer=mock_tokenizer,
        model="Phi-3-vision-128k-instruct",
        output_modality="text",
        data=mock_vision_dataset,
    )
    scenario = ImageModality(250, 250, 1, 100)

    user_request = sampler.sample(scenario=scenario)

    assert isinstance(user_request, UserImageChatRequest)
    assert user_request.model == "Phi-3-vision-128k-instruct"


def test_image_to_embeddings_sampler(mock_tokenizer, mock_vision_dataset, mock_image):
    sampler = ImageSampler(
        tokenizer=mock_tokenizer,
        model="Phi-3-vision-128k-instruct",
        output_modality="embeddings",
        data=mock_vision_dataset,
    )
    scenario = ImageModality(250, 250, 1, 100)

    user_request = sampler.sample(scenario=scenario)

    assert isinstance(user_request, UserImageEmbeddingRequest)
    assert user_request.model == "Phi-3-vision-128k-instruct"


def test_image_sampler_with_multiple_images(
    mock_tokenizer, mock_vision_dataset, mock_image
):
    sampler = ImageSampler(
        tokenizer=mock_tokenizer,
        model="Phi-3-vision-128k-instruct",
        output_modality="text",
        data=mock_vision_dataset * 2,
    )
    scenario = ImageModality(250, 250, 2, 100)

    user_request = sampler.sample(scenario=scenario)

    assert isinstance(user_request, UserImageChatRequest)
    assert user_request.num_images == 2


def test_image_sampler_with_invalid_scenario(mock_tokenizer, mock_vision_dataset):
    sampler = ImageSampler(
        tokenizer=mock_tokenizer,
        model="Phi-3-vision-128k-instruct",
        output_modality="text",
        data=mock_vision_dataset,
    )
    mock_scenario = MagicMock()
    mock_scenario.scenario_type = "InvalidType"

    with pytest.raises(
        ValueError,
        match="Expected MultiModality for image tasks, got <class 'str'>",
    ):
        sampler.sample(mock_scenario)


def test_image_sampler_dict_rows_prompt_column(monkeypatch, mock_tokenizer):
    img = Image.new("RGB", (64, 64))
    data = [
        {"image_column": img, "prompt_column": "prompt_1"},
        {"image_column": img, "prompt_column": "prompt_2"},
    ]
    ds_cfg = DatasetConfig(
        source=DatasetSourceConfig(type="huggingface", path="test/dataset"),
        prompt_column="prompt_column",
        image_column="image_column",
    )
    # Force deterministic sampling
    monkeypatch.setattr("random.choices", lambda population, k: [population[0]])

    sampler = ImageSampler(
        tokenizer=mock_tokenizer,
        model="phi-vision",
        output_modality="text",
        data=data,
        dataset_config=ds_cfg,
    )
    req = sampler.sample(scenario=None)
    assert isinstance(req, UserImageChatRequest)
    assert req.prompt == "prompt_1"
    assert len(req.image_content) == 1
    assert req.image_content[0].startswith("data:image/jpeg;base64,")


def test_image_sampler_dict_rows_prompt_lambda(monkeypatch, mock_tokenizer):
    img = Image.new("RGB", (64, 64))
    data = [{"image_column": img, "anything": "ignored"}]
    ds_cfg = DatasetConfig(
        source=DatasetSourceConfig(type="huggingface", path="test/dataset"),
        prompt_lambda='lambda x: "Fixed prompt for all"',
        image_column="image_column",
    )
    monkeypatch.setattr("random.choices", lambda population, k: [population[0]])

    sampler = ImageSampler(
        tokenizer=mock_tokenizer,
        model="phi-vision",
        output_modality="text",
        data=data,
        dataset_config=ds_cfg,
    )
    req = sampler.sample(scenario=None)
    assert isinstance(req, UserImageChatRequest)
    assert req.prompt == "Fixed prompt for all"
    assert req.image_content[0].startswith("data:image/jpeg;base64,")


def test_image_sampler_dict_rows_url_images(monkeypatch, mock_tokenizer):
    data = [{"image_column": "https://example.com/a.jpg", "prompt_column": "p1"}]
    ds_cfg = DatasetConfig(
        source=DatasetSourceConfig(type="huggingface", path="test/dataset"),
        prompt_column="prompt_column",
        image_column="image_column",
    )
    monkeypatch.setattr("random.choices", lambda population, k: [population[0]])

    sampler = ImageSampler(
        tokenizer=mock_tokenizer,
        model="phi-vision",
        output_modality="text",
        data=data,
        dataset_config=ds_cfg,
    )
    req = sampler.sample(scenario=None)
    assert isinstance(req, UserImageChatRequest)
    assert req.prompt == "p1"
    assert req.image_content == ["https://example.com/a.jpg"]


def test_image_sampler_missing_prompt_column(monkeypatch, mock_tokenizer):
    img = Image.new("RGB", (64, 64))
    data = [{"image_column": img, "other": "x"}]
    ds_cfg = DatasetConfig(
        source=DatasetSourceConfig(type="huggingface", path="test/dataset"),
        prompt_column="nonexistent_column",
        image_column="image_column",
    )
    monkeypatch.setattr("random.choices", lambda population, k: [population[0]])

    sampler = ImageSampler(
        tokenizer=mock_tokenizer,
        model="phi-vision",
        output_modality="text",
        data=data,
        dataset_config=ds_cfg,
    )
    req = sampler.sample(scenario=None)
    assert isinstance(req, UserImageChatRequest)
    assert req.prompt == ""
    assert req.image_content[0].startswith("data:image/jpeg;base64,")


# --- Tests for ID() scenario (DeterministicImageScenario) ---


@pytest.fixture
def real_tokenizer():
    """Load real tokenizer for token-count-sensitive tests."""
    import os

    path = os.path.join(
        os.path.dirname(__file__), "..", "fixtures", "local_bert_base_uncased"
    )
    return AutoTokenizer.from_pretrained(path)


@pytest.fixture
def multi_image_dataset():
    """Dataset with multiple distinct images for non-repeat testing."""
    # Use widely-spaced RGB tuples so JPEG compression produces distinct base64
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 0),
        (0, 128, 0),
        (0, 0, 128),
        (128, 128, 128),
    ]
    return [("prompt", Image.new("RGB", (64, 64), color=c)) for c in colors]


@pytest.fixture
def multi_image_dict_dataset():
    """Dict-format dataset with multiple images."""
    ds_cfg = DatasetConfig(
        source=DatasetSourceConfig(type="huggingface", path="test/dataset"),
        image_column="image",
    )
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 0),
        (0, 128, 0),
        (0, 0, 128),
        (128, 128, 128),
    ]
    data = [{"image": Image.new("RGB", (64, 64), color=c)} for c in colors]
    return data, ds_cfg


def test_image_sampler_with_id_scenario(real_tokenizer, multi_image_dataset):
    """ID() scenario produces request with num_prefill_tokens, output control."""
    sampler = ImageSampler(
        tokenizer=real_tokenizer,
        model="test-vlm",
        output_modality="text",
        data=multi_image_dataset,
    )
    scenario = DeterministicImageScenario(
        num_input_dimension_width=64,
        num_input_dimension_height=64,
        num_input_tokens=50,
        num_output_tokens=100,
    )
    req = sampler.sample(scenario)

    assert isinstance(req, UserImageChatRequest)
    assert req.num_prefill_tokens is not None
    assert req.max_tokens == 100
    assert req.additional_request_params.get("min_tokens") == 100
    assert req.additional_request_params.get("ignore_eos") is True
    assert len(req.image_content) == 1
    assert req.image_content[0].startswith("data:image/jpeg;base64,")


def test_image_sampler_id_synthetic_text(real_tokenizer, multi_image_dataset):
    """ID() scenario generates synthetic text close to requested token count."""
    sampler = ImageSampler(
        tokenizer=real_tokenizer,
        model="test-vlm",
        output_modality="text",
        data=multi_image_dataset,
    )
    target_tokens = 80
    scenario = DeterministicImageScenario(
        num_input_dimension_width=64,
        num_input_dimension_height=64,
        num_input_tokens=target_tokens,
        num_output_tokens=50,
    )
    req = sampler.sample(scenario)

    # Allow 10% tolerance (matching _check_discrepancy threshold)
    assert abs(req.num_prefill_tokens - target_tokens) <= target_tokens * 0.1


def test_image_sampler_non_repeating(real_tokenizer, multi_image_dataset):
    """Non-repeating sampling uses all images before repeating."""
    sampler = ImageSampler(
        tokenizer=real_tokenizer,
        model="test-vlm",
        output_modality="text",
        data=multi_image_dataset,
    )
    scenario = DeterministicImageScenario(
        num_input_dimension_width=64,
        num_input_dimension_height=64,
        num_input_tokens=20,
        num_output_tokens=10,
    )

    # Sample 10 requests (= dataset size), collect image base64 strings
    seen_images = []
    for _ in range(10):
        req = sampler.sample(scenario)
        seen_images.append(req.image_content[0])

    # All 10 should be unique (no repeats in first pass)
    assert len(set(seen_images)) == 10


def test_image_sampler_id_unique_text_per_request(real_tokenizer, multi_image_dataset):
    """Each ID() request gets unique text (prevents prefix caching)."""
    sampler = ImageSampler(
        tokenizer=real_tokenizer,
        model="test-vlm",
        output_modality="text",
        data=multi_image_dataset,
    )
    scenario = DeterministicImageScenario(
        num_input_dimension_width=64,
        num_input_dimension_height=64,
        num_input_tokens=30,
        num_output_tokens=10,
    )

    prompts = [sampler.sample(scenario).prompt for _ in range(5)]
    # With random shuffling of text corpus, prompts should differ
    # (extremely unlikely to be identical with 30 tokens)
    assert len(set(prompts)) > 1


# --- Tests for IP() scenario (PrefixImageScenario) ---


def test_image_sampler_with_ip_scenario(real_tokenizer, multi_image_dataset):
    """IP() scenario produces request with shared prefix and output control."""
    sampler = ImageSampler(
        tokenizer=real_tokenizer,
        model="test-vlm",
        output_modality="text",
        data=multi_image_dataset,
    )
    scenario = PrefixImageScenario(
        num_input_dimension_width=64,
        num_input_dimension_height=64,
        prefix_tokens=30,
        suffix_tokens=20,
        output_tokens=50,
    )
    req = sampler.sample(scenario)

    assert isinstance(req, UserImageChatRequest)
    assert req.num_prefill_tokens is not None
    assert req.max_tokens == 50
    assert req.additional_request_params.get("min_tokens") == 50
    assert req.additional_request_params.get("ignore_eos") is True


def test_image_sampler_ip_shared_prefix(real_tokenizer, multi_image_dataset):
    """IP() requests share the same prefix text."""
    sampler = ImageSampler(
        tokenizer=real_tokenizer,
        model="test-vlm",
        output_modality="text",
        data=multi_image_dataset,
    )
    scenario = PrefixImageScenario(
        num_input_dimension_width=64,
        num_input_dimension_height=64,
        prefix_tokens=30,
        suffix_tokens=15,
        output_tokens=10,
    )

    prompts = [sampler.sample(scenario).prompt for _ in range(3)]

    # All prompts should share the same prefix (first ~30 tokens)
    # Extract common prefix
    prefix_len = 0
    for i in range(min(len(p) for p in prompts)):
        if all(p[i] == prompts[0][i] for p in prompts):
            prefix_len = i + 1
        else:
            break
    # Shared prefix should be substantial (at least half the first prompt)
    assert prefix_len > len(prompts[0]) * 0.3


def test_image_sampler_ip_unique_suffix(real_tokenizer, multi_image_dataset):
    """IP() requests have unique suffixes."""
    sampler = ImageSampler(
        tokenizer=real_tokenizer,
        model="test-vlm",
        output_modality="text",
        data=multi_image_dataset,
    )
    scenario = PrefixImageScenario(
        num_input_dimension_width=64,
        num_input_dimension_height=64,
        prefix_tokens=20,
        suffix_tokens=20,
        output_tokens=10,
    )

    prompts = [sampler.sample(scenario).prompt for _ in range(3)]
    # Suffixes contain unique request numbers, so prompts should differ
    assert len(set(prompts)) == 3


def test_image_sampler_prefix_cache_reset(real_tokenizer, multi_image_dataset):
    """reset_prefix_cache clears state."""
    sampler = ImageSampler(
        tokenizer=real_tokenizer,
        model="test-vlm",
        output_modality="text",
        data=multi_image_dataset,
    )
    scenario = PrefixImageScenario(
        num_input_dimension_width=64,
        num_input_dimension_height=64,
        prefix_tokens=20,
        suffix_tokens=10,
        output_tokens=10,
    )

    sampler.sample(scenario)
    assert sampler._suffix_counter == 1
    assert len(sampler._shared_prefix_cache) == 1

    sampler.reset_prefix_cache()
    assert sampler._suffix_counter == 0
    assert len(sampler._shared_prefix_cache) == 0


def test_image_sampler_legacy_unchanged(mock_tokenizer, mock_vision_dataset):
    """Legacy I() scenarios still work unchanged."""
    sampler = ImageSampler(
        tokenizer=mock_tokenizer,
        model="test-vlm",
        output_modality="text",
        data=mock_vision_dataset,
    )
    scenario = ImageModality(250, 250, 1, 100)
    req = sampler.sample(scenario)

    assert isinstance(req, UserImageChatRequest)
    assert req.num_prefill_tokens is None  # Legacy doesn't track this
    assert req.model == "test-vlm"


def test_image_sampler_id_no_min_tokens(real_tokenizer, multi_image_dataset):
    """no_min_tokens=True skips min_tokens in ID() requests."""
    sampler = ImageSampler(
        tokenizer=real_tokenizer,
        model="test-vlm",
        output_modality="text",
        data=multi_image_dataset,
        no_min_tokens=True,
    )
    scenario = DeterministicImageScenario(
        num_input_dimension_width=64,
        num_input_dimension_height=64,
        num_input_tokens=30,
        num_output_tokens=50,
    )
    req = sampler.sample(scenario)

    assert req.max_tokens == 50
    assert "min_tokens" not in req.additional_request_params


def test_image_sampler_id_with_dict_dataset(real_tokenizer, multi_image_dict_dataset):
    """ID() scenario works with dict-format datasets."""
    data, ds_cfg = multi_image_dict_dataset
    sampler = ImageSampler(
        tokenizer=real_tokenizer,
        model="test-vlm",
        output_modality="text",
        data=data,
        dataset_config=ds_cfg,
    )
    scenario = DeterministicImageScenario(
        num_input_dimension_width=64,
        num_input_dimension_height=64,
        num_input_tokens=30,
        num_output_tokens=20,
    )
    req = sampler.sample(scenario)

    assert isinstance(req, UserImageChatRequest)
    assert req.num_prefill_tokens is not None
    assert len(req.image_content) == 1


def test_state_isolation_id_then_legacy(real_tokenizer, multi_image_dataset):
    """ID() params (ignore_eos, min_tokens, max_tokens) don't leak into legacy I() requests."""
    sampler = ImageSampler(
        tokenizer=real_tokenizer,
        model="test-vlm",
        output_modality="text",
        data=multi_image_dataset,
    )

    # First sample an ID() request — sets ignore_eos, min_tokens, max_tokens
    id_scenario = DeterministicImageScenario(
        num_input_dimension_width=64,
        num_input_dimension_height=64,
        num_input_tokens=30,
        num_output_tokens=50,
    )
    sampler.sample(id_scenario)

    # Now sample a legacy I() request — should NOT inherit ID() params
    legacy_scenario = ImageModality(64, 64, 1, None)
    req = sampler.sample(legacy_scenario)

    assert isinstance(req, UserImageChatRequest)
    assert "ignore_eos" not in req.additional_request_params
    assert "min_tokens" not in req.additional_request_params
    assert "max_tokens" not in req.additional_request_params
