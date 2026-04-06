"""Tests for ConversationSampler (RD/RDC scenarios)."""

import os

import pytest

from genai_bench.data.config import DatasetConfig, DatasetSourceConfig
from genai_bench.data.loaders.conversation import ConversationDatasetLoader
from genai_bench.protocol import UserConversationRequest
from genai_bench.sampling.conversation import ConversationSampler
from genai_bench.scenarios.real_dataset import (
    CachedRealDatasetScenario,
    RealDatasetScenario,
)


FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")
TEST_JSONL = os.path.join(FIXTURES_DIR, "test_conversations.jsonl")


@pytest.fixture
def conversation_data():
    """Load test conversation data via the loader."""
    config = DatasetConfig(
        source=DatasetSourceConfig(type="file", path=TEST_JSONL, file_format="jsonl"),
        messages_column="conversations",
    )
    loader = ConversationDatasetLoader(config)
    return loader.load_request()


@pytest.fixture
def mock_tokenizer():
    from unittest.mock import MagicMock

    return MagicMock()


def test_conversation_sampler_basic(mock_tokenizer, conversation_data):
    """ConversationSampler returns UserConversationRequest."""
    sampler = ConversationSampler(
        tokenizer=mock_tokenizer,
        model="test-model",
        data=conversation_data,
    )
    scenario = RealDatasetScenario(max_output_tokens=200)
    req = sampler.sample(scenario)

    assert isinstance(req, UserConversationRequest)
    assert req.model == "test-model"
    assert isinstance(req.messages, list)
    assert len(req.messages) > 0
    assert req.max_tokens == 200


def test_uncached_mode_nonce_string_content(mock_tokenizer, conversation_data):
    """RD scenario injects nonce into first user message (string content)."""
    sampler = ConversationSampler(
        tokenizer=mock_tokenizer,
        model="test-model",
        data=conversation_data,
    )
    scenario = RealDatasetScenario(max_output_tokens=100)

    # Sample until we get a request with string content in user message
    reqs = [sampler.sample(scenario) for _ in range(5)]
    nonce_found = False
    for req in reqs:
        for msg in req.messages:
            if msg["role"] == "user" and isinstance(msg["content"], str):
                if "[NONCE-" in msg["content"]:
                    nonce_found = True
                    break
        if nonce_found:
            break
    assert nonce_found, "Expected nonce in at least one string user message"


def test_uncached_mode_nonce_multipart_content(mock_tokenizer, conversation_data):
    """RD scenario injects nonce into first text block of multipart content."""
    sampler = ConversationSampler(
        tokenizer=mock_tokenizer,
        model="test-model",
        data=conversation_data,
    )
    scenario = RealDatasetScenario(max_output_tokens=100)

    # Sample enough to hit a multipart user message
    reqs = [sampler.sample(scenario) for _ in range(5)]
    nonce_found = False
    for req in reqs:
        for msg in req.messages:
            if msg["role"] == "user" and isinstance(msg["content"], list):
                for part in msg["content"]:
                    if part.get("type") == "text" and "[NONCE-" in part.get("text", ""):
                        nonce_found = True
                        break
            if nonce_found:
                break
        if nonce_found:
            break
    assert nonce_found, "Expected nonce in at least one multipart user message"


def test_natural_mode_no_nonce(mock_tokenizer, conversation_data):
    """RDC scenario does NOT inject nonce."""
    sampler = ConversationSampler(
        tokenizer=mock_tokenizer,
        model="test-model",
        data=conversation_data,
    )
    scenario = CachedRealDatasetScenario(max_output_tokens=100)

    for _ in range(5):
        req = sampler.sample(scenario)
        for msg in req.messages:
            if msg["role"] == "user":
                content = msg["content"]
                if isinstance(content, str):
                    assert "[NONCE-" not in content
                elif isinstance(content, list):
                    for part in content:
                        if part.get("type") == "text":
                            assert "[NONCE-" not in part.get("text", "")


def test_non_repeating_cursor(mock_tokenizer, conversation_data):
    """Sequential samples use different data (cursor advances)."""
    sampler = ConversationSampler(
        tokenizer=mock_tokenizer,
        model="test-model",
        data=conversation_data,
    )
    scenario = CachedRealDatasetScenario(max_output_tokens=50)

    # Sample as many as the dataset has
    n = len(conversation_data)
    sample_ids = []
    for _ in range(n):
        req = sampler.sample(scenario)
        # Use messages content as identity
        sample_ids.append(str(req.messages))

    # All should be unique (non-repeating first pass)
    assert len(set(sample_ids)) == n


def test_cursor_wrap_around(mock_tokenizer, conversation_data):
    """Cursor wraps when dataset is exhausted."""
    sampler = ConversationSampler(
        tokenizer=mock_tokenizer,
        model="test-model",
        data=conversation_data,
    )
    scenario = CachedRealDatasetScenario(max_output_tokens=50)

    n = len(conversation_data)
    # Sample more than the dataset size
    for _ in range(n + 3):
        req = sampler.sample(scenario)
        assert isinstance(req, UserConversationRequest)

    # Verify wrap happened
    assert sampler._wrap_count >= 1


def test_nonce_uniqueness(mock_tokenizer, conversation_data):
    """Each nonce is unique across calls."""
    sampler = ConversationSampler(
        tokenizer=mock_tokenizer,
        model="test-model",
        data=conversation_data,
    )
    scenario = RealDatasetScenario(max_output_tokens=50)

    nonces = []
    for _ in range(5):
        req = sampler.sample(scenario)
        # Extract nonce from first user message
        for msg in req.messages:
            if msg["role"] == "user":
                content = msg["content"]
                text = content if isinstance(content, str) else ""
                if isinstance(content, list):
                    for part in content:
                        if part.get("type") == "text":
                            text = part["text"]
                            break
                if "[NONCE-" in text:
                    nonce = text.split("]")[0] + "]"
                    nonces.append(nonce)
                break

    assert len(set(nonces)) == len(nonces), "All nonces should be unique"


def test_rd_forced_output_tokens(mock_tokenizer, conversation_data):
    """RD(200) sets min_tokens, max_tokens, ignore_eos."""
    sampler = ConversationSampler(
        tokenizer=mock_tokenizer,
        model="test-model",
        data=conversation_data,
    )
    scenario = RealDatasetScenario(max_output_tokens=200)
    req = sampler.sample(scenario)

    assert req.max_tokens == 200
    assert req.additional_request_params.get("max_tokens") == 200
    assert req.additional_request_params.get("min_tokens") == 200
    assert req.additional_request_params.get("ignore_eos") is True


def test_rd_natural_output(mock_tokenizer, conversation_data):
    """RD (no parens) does NOT set min_tokens or ignore_eos."""
    sampler = ConversationSampler(
        tokenizer=mock_tokenizer,
        model="test-model",
        data=conversation_data,
    )
    scenario = RealDatasetScenario(max_output_tokens=None)
    req = sampler.sample(scenario)

    assert req.max_tokens is None
    assert "min_tokens" not in req.additional_request_params
    assert "ignore_eos" not in req.additional_request_params


def test_rd_no_min_tokens_flag(mock_tokenizer, conversation_data):
    """no_min_tokens=True skips min_tokens even with RD(200)."""
    sampler = ConversationSampler(
        tokenizer=mock_tokenizer,
        model="test-model",
        data=conversation_data,
        no_min_tokens=True,
    )
    scenario = RealDatasetScenario(max_output_tokens=200)
    req = sampler.sample(scenario)

    assert req.max_tokens == 200
    assert "min_tokens" not in req.additional_request_params


def test_invalid_scenario_type(mock_tokenizer, conversation_data):
    """ConversationSampler rejects non-RD/RDC scenarios."""
    from genai_bench.scenarios.multimodal import ImageModality

    sampler = ConversationSampler(
        tokenizer=mock_tokenizer,
        model="test-model",
        data=conversation_data,
    )
    with pytest.raises(ValueError, match="ConversationSampler requires"):
        sampler.sample(ImageModality(512, 512))
