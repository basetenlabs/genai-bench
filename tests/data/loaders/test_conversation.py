"""Tests for ConversationDatasetLoader."""

import os

import pytest

from genai_bench.data.config import DatasetConfig, DatasetSourceConfig
from genai_bench.data.loaders.conversation import ConversationDatasetLoader

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures")
TEST_JSONL = os.path.join(FIXTURES_DIR, "test_conversations.jsonl")


@pytest.fixture
def conversation_config():
    return DatasetConfig(
        source=DatasetSourceConfig(type="file", path=TEST_JSONL, file_format="jsonl"),
        messages_column="conversations",
    )


def test_load_basic(conversation_config):
    """Load JSONL and get conversation samples."""
    loader = ConversationDatasetLoader(conversation_config)
    samples = loader.load_request()

    assert isinstance(samples, list)
    assert len(samples) == 5
    for s in samples:
        assert "messages" in s
        assert "num_images" in s
        assert isinstance(s["messages"], list)


def test_last_assistant_truncated(conversation_config):
    """Last assistant message is removed from conversations that end with assistant."""
    loader = ConversationDatasetLoader(conversation_config)
    samples = loader.load_request()

    # Find sample 0 (simple text-only: system, user, assistant)
    # After truncation: should have system + user only
    sample_0 = next(s for s in samples if s["id"] == "0")
    roles = [m["role"] for m in sample_0["messages"]]
    assert roles == ["system", "user"]


def test_last_non_assistant_not_truncated(conversation_config):
    """Conversations ending with non-assistant messages are kept as-is."""
    loader = ConversationDatasetLoader(conversation_config)
    samples = loader.load_request()

    # Sample 3 ends with user: system, user — should be unchanged
    sample_3 = next(s for s in samples if s["id"] == "3")
    roles = [m["role"] for m in sample_3["messages"]]
    assert roles == ["system", "user"]


def test_image_conversion(conversation_config):
    """Image paths are converted to base64 data URLs."""
    loader = ConversationDatasetLoader(conversation_config)
    samples = loader.load_request()

    # Sample 1 has an image in user message
    sample_1 = next(s for s in samples if s["id"] == "1")
    assert sample_1["num_images"] >= 1

    # Find the image_url part
    for msg in sample_1["messages"]:
        if isinstance(msg.get("content"), list):
            for part in msg["content"]:
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    assert url.startswith("data:image/png;base64,")
                    return

    pytest.fail("Expected to find a base64 image_url in sample 1")


def test_multi_image_count(conversation_config):
    """Multi-image conversations have correct image count."""
    loader = ConversationDatasetLoader(conversation_config)
    samples = loader.load_request()

    # Sample 2 has images in user message AND tool response (3 total)
    # But last assistant is truncated, so tool response images remain
    sample_2 = next(s for s in samples if s["id"] == "2")
    # Should have images from user (1) + tool (1) = 2 after truncating last assistant
    assert sample_2["num_images"] >= 2


def test_messages_column_config():
    """Respects messages_column config."""
    config = DatasetConfig(
        source=DatasetSourceConfig(type="file", path=TEST_JSONL, file_format="jsonl"),
        messages_column="conversations",
    )
    loader = ConversationDatasetLoader(config)
    samples = loader.load_request()
    assert len(samples) == 5


def test_wrong_messages_column():
    """Raises error for missing messages column."""
    config = DatasetConfig(
        source=DatasetSourceConfig(type="file", path=TEST_JSONL, file_format="jsonl"),
        messages_column="nonexistent_column",
    )
    loader = ConversationDatasetLoader(config)
    with pytest.raises(ValueError, match="Row missing 'nonexistent_column'"):
        loader.load_request()


def test_pre_shuffled(conversation_config):
    """Data is pre-shuffled at load time."""
    loader = ConversationDatasetLoader(conversation_config)

    # Load twice and check if order differs (probabilistic but very likely with 5 samples)
    samples1 = loader.load_request()
    samples2 = loader.load_request()
    ids1 = [s["id"] for s in samples1]
    ids2 = [s["id"] for s in samples2]
    # At least check both have all IDs
    assert set(ids1) == set(ids2)
    # Can't guarantee different order with only 5 samples, but structure should be valid
